"""Bounded background shape refinement; never publishes historical positions.

Only current, robot-filtered components can accept a template. Reuse is an
optimization of a *current* cover, not a replacement for tracking or sensing.
ROS-free so the worker can be spawned without forking an initialized executor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

import numpy as np

from rmp_camera.dynamic_obstacle_sphere_core import (
    DynamicComponentResult, DynamicSphereParameters, DynamicSphereResult,
    _covered_mask, _state_overlap_constraint_satisfied,
    build_dynamic_sphere_search_state, refine_dynamic_component,
    DynamicCandidateValidator,
)


@dataclass(frozen=True)
class RefinementOptions:
    rate_hz: float = 3.0
    max_age_s: float = 0.5
    validation_budget_ms: float = 4.0
    max_voxels: int = 8192
    max_translation_m: float = 0.30
    min_shape_support: float = 0.65

    def validate(self):
        for value in (self.rate_hz, self.max_age_s, self.validation_budget_ms,
                      self.max_translation_m):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("refinement rates, ages and budgets must be finite and positive")
        if self.max_voxels <= 0 or not 0 < self.min_shape_support <= 1:
            raise ValueError("invalid refinement snapshot/shape limits")


def _shape_translation(current, previous, params, options):
    """Match geometry, never the per-frame component enumeration ID."""
    size = params.voxel_size_m
    shift = np.rint((np.mean(current, axis=0) - np.mean(previous, axis=0)) / size)
    if np.linalg.norm(shift * size) > options.max_translation_m:
        return None
    # Voxel centers can sit on half integers; quantize relative to the actual
    # workspace origin to avoid round-to-even aliasing.
    origin = np.asarray((params.min_x_m, params.min_y_m, params.min_z_m))
    old_keys = {tuple(p) for p in np.rint((previous - origin) / size - .5).astype(np.int64)}
    new_keys = {tuple(p) for p in np.rint((current - origin) / size - .5).astype(np.int64)}
    choices = (shift.astype(np.int64), np.zeros(3, dtype=np.int64))
    best = None
    for candidate in choices:
        moved = {tuple(np.asarray(p) + candidate) for p in old_keys}
        support = len(moved & new_keys) / max(len(old_keys), len(new_keys), 1)
        if support >= options.min_shape_support and (best is None or support > best[0]):
            best = (support, candidate * size)
    return None if best is None else best[1]


def reuse_refinement(current: DynamicComponentResult, template: DynamicComponentResult,
                     params: DynamicSphereParameters, options: RefinementOptions,
                     deadline: float, *, observation_evidence=None):
    """Return a no-worse current cover, or None (keep the fresh fast cover).

    Every previously covered current voxel must remain covered. Current target
    coverage, radius caps, support, empty-space and overlap limits also apply.
    A small deformation can be repaired with current seed spheres, but only if
    the final sphere count is strictly smaller. Nothing extrapolates velocity.
    """
    points = current.voxel_centers
    if (monotonic() >= deadline or not len(points) or not len(template.voxel_centers)
            or len(points) > options.max_voxels or len(template.voxel_centers) > options.max_voxels
            or not np.isfinite(points).all() or not np.isfinite(template.voxel_centers).all()
            or not template.spheres or len(template.spheres) >= len(current.spheres)):
        return None
    shift = _shape_translation(points, template.voxel_centers, params, options)
    if shift is None or monotonic() >= deadline:
        return None
    validator = DynamicCandidateValidator(points, params, observation_evidence)
    candidates = []
    for sphere in template.spheres:
        center = sphere.center + shift
        if not validator(center, sphere.raw_radius, deadline, local_width=sphere.local_width):
            return None
        candidate = replace(sphere, x=float(center[0]), y=float(center[1]), z=float(center[2]),
            output_radius=sphere.raw_radius + params.safety_margin_m,
            component_id=current.component_id, track_id=-1, age=1)
        if not np.any(_covered_mask(points, [candidate], params.coverage_tolerance_m)):
            return None  # No orphan sphere, even if the other spheres cover the object.
        candidates.append(candidate)
    required = _covered_mask(points, current.spheres, params.coverage_tolerance_m)
    covered = _covered_mask(points, candidates, params.coverage_tolerance_m)
    # Preserve individual supported voxels, not just an aggregate percentage
    # that could trade away the moving hand for a better-covered torso.
    for sphere in current.spheres:
        if np.all(covered[required]):
            break
        mask = _covered_mask(points, [sphere], params.coverage_tolerance_m)
        if np.any(required & ~covered & mask):
            candidates.append(sphere)
            covered |= mask
        if len(candidates) >= len(current.spheres) or monotonic() >= deadline:
            return None
    coverage = float(np.mean(covered))
    if (not np.all(covered[required]) or coverage + 1e-12 < params.target_coverage
            or len(candidates) >= len(current.spheres) or monotonic() >= deadline):
        return None
    state = build_dynamic_sphere_search_state(candidates)
    baseline = build_dynamic_sphere_search_state(current.spheres)
    constraint_satisfied = _state_overlap_constraint_satisfied(state, params)
    # The existing solver can output an overlap-infeasible fallback. Do not
    # forbid an improvement to that fallback, but never make its overlap worse.
    allowed = max(params.dynamic_max_allowed_overlap_fraction,
        baseline.max_output_overlap_fraction if params.dynamic_min_k_use_output_overlap
        else baseline.max_raw_overlap_fraction)
    actual = (state.max_output_overlap_fraction if params.dynamic_min_k_use_output_overlap
              else state.max_raw_overlap_fraction)
    if ((params.dynamic_min_k_enable_overlap_constraint and actual > allowed + 1e-12)
            or state.total_raw_overlap > baseline.total_raw_overlap + 1e-12
            or state.total_output_overlap > baseline.total_output_overlap + 1e-12
            or monotonic() >= deadline):
        return None
    confidence = min(1., coverage * min(1., len(points) / max(1, params.min_component_voxels * 2)))
    return replace(current,
        spheres=[replace(s, component_coverage=coverage, confidence=confidence) for s in candidates],
        coverage=coverage, uncovered_voxels=points[~covered],
        termination_reason="validated_async_refinement", final_sphere_count=len(candidates),
        min_k_sphere_count=len(candidates), min_k_search_applied=True,
        min_k_states_explored=template.min_k_states_explored,
        min_k_search_termination="validated_async_refinement",
        max_raw_overlap_fraction=state.max_raw_overlap_fraction,
        max_output_overlap_fraction=state.max_output_overlap_fraction,
        total_raw_overlap=state.total_raw_overlap, total_output_overlap=state.total_output_overlap,
        overlap_constraint_satisfied=constraint_satisfied)


def _worker(requests, replies):
    while True:
        job = requests.get()
        if job is None:
            return
        epoch, stamp_ns, components, params = job
        start = monotonic()
        try:
            deadline = start + params.processing_budget_ms / 1000.
            refined = []
            for component in components:
                if monotonic() >= deadline:
                    break
                refined.append(refine_dynamic_component(component.voxel_centers,
                    component.spheres, params, component.component_id, deadline))
            replies.put((epoch, stamp_ns, refined, (monotonic() - start) * 1000., ""))
        except Exception as exc:
            # A bad job cannot interrupt current-frame publication.
            replies.put((epoch, stamp_ns, [], (monotonic() - start) * 1000.,
                         type(exc).__name__ + ": " + str(exc)))


def _collect_replies(transport, completed, stop):
    """Decode IPC off the ROS thread, including partially delivered messages.

    multiprocessing.Queue.get_nowait() is not truly nonblocking after it
    acquires the message semaphore: receiving a large partial pipe message can
    still wait for its writer. Only this daemon collector ever reads that pipe.
    """
    while not stop.is_set():
        try:
            reply = transport.get(timeout=.1)
        except Empty:
            continue
        except (EOFError, OSError, ValueError):
            return
        while not stop.is_set():
            try:
                completed.put(reply, timeout=.1)
                break
            except Full:
                continue


class AsyncSphereRefiner:
    """One spawned process, one outstanding snapshot, zero queued old frames."""
    def __init__(self, options=RefinementOptions()):
        options.validate()
        self.options = options
        context = mp.get_context("spawn")
        self.requests = context.Queue(maxsize=1)
        self.reply_transport = context.Queue(maxsize=1)
        self.replies = Queue(maxsize=1)
        self.process = context.Process(target=_worker,
            args=(self.requests, self.reply_transport), daemon=True)
        self.process.start()
        self.collector_stop = Event()
        self.collector = Thread(target=_collect_replies,
            args=(self.reply_transport, self.replies, self.collector_stop), daemon=True)
        self.collector.start()
        self.busy = False
        self.epoch = 0
        self.templates = []
        self.template_stamp_ns = 0
        self.last_stamp_ns = None
        self.last_submit_time = -float("inf")
        self.worker_ms = 0.
        self.error = ""
        self.closed = False

    def reset(self):
        self.epoch += 1
        self.templates = []
        self.template_stamp_ns = 0
        self.last_stamp_ns = None
        self.last_submit_time = -float("inf")

    def _poll(self):
        try:
            epoch, stamp, templates, elapsed, error = self.replies.get_nowait()
        except Empty:
            if not self.process.is_alive():
                self.error = "worker_exited"
                self.busy = False
                self.templates = []
            elif getattr(self, "collector", None) is not None and not self.collector.is_alive():
                self.error = "worker_transport_unavailable"
                self.busy = False
                self.templates = []
            return
        self.busy = False
        self.worker_ms = elapsed
        self.error = error
        if epoch == self.epoch and not error:
            self.templates = templates
            self.template_stamp_ns = stamp

    def update(self, result: DynamicSphereResult, source_stamp_ns: int,
               params: DynamicSphereParameters, *, observation_evidence=None):
        """Called only on a NEW current observation, never on a timer snapshot."""
        start = monotonic()
        if self.last_stamp_ns is not None and source_stamp_ns < self.last_stamp_ns:
            self.reset()
        self.last_stamp_ns = source_stamp_ns
        self._poll()
        originals = result.components
        applied = 0
        if not originals:
            self.reset()  # Even an in-flight old result cannot resurrect an empty frame.
        elif (0 <= source_stamp_ns - self.template_stamp_ns <= self.options.max_age_s * 1e9
                and sum(len(c.voxel_centers) for c in originals) <= self.options.max_voxels):
            deadline = start + self.options.validation_budget_ms / 1000.
            available = list(self.templates)
            replacements = []
            for component in originals:
                replacement = None
                for index, template in enumerate(available):
                    replacement = reuse_refinement(component, template, params, self.options, deadline,
                                                   observation_evidence=observation_evidence)
                    if replacement is not None:
                        available.pop(index)  # One old object cannot be reused for two new objects.
                        applied += 1
                        break
                    if monotonic() >= deadline:
                        break
                replacements.append(replacement if replacement is not None else component)
            if applied:
                result.components = replacements
                result.spheres = [s for c in replacements for s in c.spheres]
                result.uncovered_voxels = np.vstack([c.uncovered_voxels for c in replacements])
        submitted = False
        if (not self.closed and not self.busy and not self.error and self.process.is_alive()
                and monotonic() - self.last_submit_time >= 1. / self.options.rate_hz):
            eligible, voxels = [], 0
            for component in originals:
                if ((component.refinement_budget_exhausted
                     or component.min_k_search_termination in ("deadline", "max_states"))
                        and (params.dynamic_enable_min_k_search or params.dynamic_enable_agglomerative_merge)
                        and len(component.spheres) > 1
                        and voxels + len(component.voxel_centers) <= self.options.max_voxels):
                    # Queue serialization is deferred: own immutable snapshots.
                    eligible.append(replace(component, voxel_centers=component.voxel_centers.copy(),
                        uncovered_voxels=component.uncovered_voxels.copy(), spheres=list(component.spheres)))
                    voxels += len(component.voxel_centers)
            if eligible:
                try:
                    self.requests.put_nowait((self.epoch, source_stamp_ns, eligible, params))
                    self.busy = submitted = True
                    self.last_submit_time = monotonic()
                except Full:
                    pass
        return dict(refinement_applied_components=applied, refinement_submitted=submitted,
            refinement_busy=self.busy, refinement_worker_ms=self.worker_ms,
            refinement_validation_ms=(monotonic() - start) * 1000., refinement_error=self.error)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.collector_stop.set()
        # Never wait through a long optimization during ROS shutdown.
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1.)
        self.collector.join(timeout=.2)
        for queue in (self.requests, self.reply_transport):
            queue.cancel_join_thread()
            queue.close()
