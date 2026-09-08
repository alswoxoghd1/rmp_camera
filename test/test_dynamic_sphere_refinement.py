from dataclasses import replace
from queue import Queue
from time import monotonic, sleep
from types import SimpleNamespace
from threading import Event, Thread

import numpy as np
import pytest

from rmp_camera.dynamic_obstacle_sphere_core import (
    DynamicComponentResult, DynamicSphere, DynamicSphereParameters, DynamicSphereResult,
    _covered_mask, generate_dynamic_spheres, voxel_centers,
)
from rmp_camera.dynamic_sphere_refinement import (
    AsyncSphereRefiner, RefinementOptions, reuse_refinement, _collect_replies,
)
import rmp_camera.dynamic_obstacle_sphere_core as core


def fixture_geometry():
    params = DynamicSphereParameters(min_x_m=0., min_y_m=0., min_z_m=0.,
        max_x_m=2., max_y_m=2., max_z_m=2., safety_margin_m=0.,
        coverage_tolerance_m=0., target_coverage=1.,
        dynamic_merge_enable_empty_space_guard=True, dynamic_merge_max_empty_fraction=.7,
        processing_budget_ms=150.)
    indices = np.asarray([(x, y, z) for x in range(4, 8) for y in range(4, 8) for z in range(4, 8)])
    points = voxel_centers(indices, params)
    seeds = [DynamicSphere(*p, .025, .025, 7) for p in points]
    current = DynamicComponentResult(7, points, seeds, 1., np.empty((0, 3)), "fresh")
    sphere = DynamicSphere(*np.mean(points, axis=0), .131, .131, 99)
    template = replace(current, component_id=99, spheres=[sphere])
    return params, current, template


def check(current, template, params, options=RefinementOptions(), budget=1.):
    return reuse_refinement(current, template, params, options, monotonic() + budget)


def test_rigid_translation_reuses_geometry_not_component_id():
    params, current, template = fixture_geometry()
    shift = np.asarray((.1, -.05, .05))
    moved = replace(current, voxel_centers=current.voxel_centers + shift,
        spheres=[replace(s, x=s.x + shift[0], y=s.y + shift[1], z=s.z + shift[2]) for s in current.spheres])
    result = check(moved, template, params)
    assert result is not None and len(result.spheres) == 1
    assert result.spheres[0].component_id == current.component_id
    np.testing.assert_allclose(result.spheres[0].center, template.spheres[0].center + shift)
    assert np.all(_covered_mask(moved.voxel_centers, result.spheres, 0.))
    assert result.coverage == 1.


@pytest.mark.parametrize("change", ["far", "split", "expired_budget", "empty", "nan", "radius", "orphan"])
def test_invalid_templates_cannot_move_or_resurrect_geometry(change):
    params, current, template = fixture_geometry()
    if change == "far":
        template = replace(template, voxel_centers=template.voxel_centers + 1.)
    elif change == "split":
        current = replace(current, voxel_centers=current.voxel_centers[:16])
    elif change == "empty":
        current = replace(current, voxel_centers=np.empty((0, 3)), spheres=[])
    elif change == "nan":
        template = replace(template, spheres=[replace(template.spheres[0], x=float("nan"))])
    elif change == "radius":
        template = replace(template, spheres=[replace(template.spheres[0], raw_radius=1.)])
    elif change == "orphan":
        template = replace(template, spheres=template.spheres + [DynamicSphere(1.5, 1.5, 1.5, .1, .1)])
    assert check(current, template, params, budget=-1. if change == "expired_budget" else 1.) is None


def test_current_hand_voxels_are_not_traded_for_torso_coverage():
    params, current, template = fixture_geometry()
    # A short extension remains shape-compatible. Repair with current seeds,
    # preserving every covered voxel including the extension's tip.
    extra = np.asarray([[.425, .325, .325], [.475, .325, .325]])
    points = np.vstack((current.voxel_centers, extra))
    current = replace(current, voxel_centers=points,
        spheres=current.spheres + [DynamicSphere(*p, .025, .025) for p in extra])
    result = check(current, template, params)
    assert result is not None and len(result.spheres) < len(current.spheres)
    assert np.all(_covered_mask(points, result.spheres, 0.))


def test_reuse_rechecks_empty_space_and_component_radius_cap():
    params, current, template = fixture_geometry()
    assert check(current, template, params) is not None
    assert check(current, template, replace(params, dynamic_merge_max_empty_fraction=.01)) is None
    assert check(current, template, replace(params, component_radius_scale=.1)) is None


def test_reuse_does_not_ignore_overlap_constraint():
    params, current, template = fixture_geometry()
    template = replace(template, spheres=template.spheres * 2)
    assert check(current, template, params) is None


def fake_refiner(template, stamp=1_000_000_000):
    obj = AsyncSphereRefiner.__new__(AsyncSphereRefiner)
    obj.options = RefinementOptions(validation_budget_ms=100.)
    obj.requests, obj.replies = Queue(1), Queue(1)
    obj.process = SimpleNamespace(is_alive=lambda: True)
    obj.busy = True  # Don't submit jobs in cache-only tests.
    obj.epoch = 0
    obj.templates = [template]
    obj.template_stamp_ns = stamp
    obj.last_stamp_ns = stamp
    obj.last_submit_time = -float("inf")
    obj.worker_ms, obj.error, obj.closed = 0., "", False
    return obj


def as_result(current):
    return DynamicSphereResult(components=[current], spheres=list(current.spheres),
        voxel_centers=current.voxel_centers, uncovered_voxels=current.uncovered_voxels)


def test_current_only_one_to_one_reuse_and_no_more_spheres():
    params, current, template = fixture_geometry()
    refiner = fake_refiner(template)
    result = as_result(current)
    result.components.append(replace(current, component_id=8))
    stats = refiner.update(result, 1_100_000_000, params)
    assert stats["refinement_applied_components"] == 1
    assert len(result.components[0].spheres) == 1
    assert result.components[1].spheres == current.spheres


@pytest.mark.parametrize("stamp", [999_999_999, 1_600_000_000])
def test_clock_rewind_or_old_result_not_applied(stamp):
    params, current, template = fixture_geometry()
    refiner = fake_refiner(template)
    result = as_result(current)
    stats = refiner.update(result, stamp, params)
    assert stats["refinement_applied_components"] == 0
    assert result.spheres == current.spheres


def test_empty_frame_invalidates_inflight_epoch():
    params, current, template = fixture_geometry()
    refiner = fake_refiner(template)
    empty = DynamicSphereResult()
    refiner.update(empty, 1_100_000_000, params)
    assert not refiner.templates
    refiner.replies.put((0, 1_000_000_000, [template], 5., ""))
    refiner.closed = True  # Prevent a new submission, not polling.
    stats = refiner.update(as_result(current), 1_200_000_000, params)
    assert stats["refinement_applied_components"] == 0
    assert not refiner.templates


def test_worker_failure_keeps_fresh_result():
    params, current, template = fixture_geometry()
    refiner = fake_refiner(template)
    refiner.process.is_alive = lambda: False
    result = as_result(current)
    stats = refiner.update(result, 1_100_000_000, params)
    assert stats["refinement_error"] == "worker_exited"
    assert result.spheres == current.spheres


def test_spawned_worker_has_one_job_no_backlog_and_closes():
    params, current, _ = fixture_geometry()
    current.min_k_search_termination = "deadline"
    # Smaller search job keeps the spawn/lifecycle test fast.
    params = replace(params, dynamic_min_k_max_states=2, dynamic_min_k_beam_width=1)
    refiner = AsyncSphereRefiner()
    try:
        first = refiner.update(as_result(current), 1_000_000_000, params)
        assert first["refinement_submitted"]
        second = refiner.update(as_result(current), 1_020_000_000, params)
        assert not second["refinement_submitted"]
        deadline = monotonic() + 8.
        while refiner.busy and monotonic() < deadline:
            refiner._poll()
            sleep(.01)
        assert not refiner.busy and not refiner.error
        assert refiner.templates
    finally:
        refiner.close()
        refiner.close()  # Idempotent shutdown.
    assert not refiner.process.is_alive()


def test_oversize_snapshot_keeps_fresh_result():
    params, current, template = fixture_geometry()
    refiner = fake_refiner(template)
    refiner.options = replace(refiner.options, max_voxels=1)
    result = as_result(current)
    stats = refiner.update(result, 1_100_000_000, params)
    assert stats["refinement_applied_components"] == 0
    assert result.spheres == current.spheres


def test_robot_rejected_component_cannot_be_restored_by_template():
    params, current, template = fixture_geometry()
    result = generate_dynamic_spheres(current.voxel_centers, params,
        robot_sphere_centers=np.mean(current.voxel_centers, axis=0)[None, :],
        robot_sphere_radii=np.asarray([.2]), robot_component_overlap_threshold=.02)
    assert result.robot_rejected_component_count == 1
    refiner = fake_refiner(template)
    refiner.update(result, 1_100_000_000, params)
    assert not result.spheres and not refiner.templates


def test_short_search_budget_never_shortens_seed_generation(monkeypatch):
    params, current, _ = fixture_geometry()
    budgets = []
    seeds = []
    original_seed = core._optimize_component_spheres
    original_refine = core.refine_dynamic_component

    def observe_seed(centers, spheres, variant, component_id, deadline):
        assert deadline - monotonic() > .5
        result = original_seed(centers, spheres, variant, component_id, deadline)
        seeds.append(result)
        return result

    def observe_refine(centers, spheres, variant, component_id, deadline, reason="refined"):
        budgets.append(deadline - monotonic())
        return original_refine(centers, spheres, variant, component_id, deadline, reason)

    monkeypatch.setattr(core, "_optimize_component_spheres", observe_seed)
    monkeypatch.setattr(core, "refine_dynamic_component", observe_refine)
    params = replace(params, processing_budget_ms=1000.)
    core.generate_dynamic_spheres(current.voxel_centers, params)
    core.generate_dynamic_spheres(current.voxel_centers, replace(params, refinement_budget_ms=0.))
    assert seeds[0] == seeds[1]
    assert budgets[0] > .5 and budgets[1] <= 0.


def test_foreground_refinement_budget_shared_between_components(monkeypatch):
    params, current, _ = fixture_geometry()
    seen = []
    original = core._component_spheres

    def component(indices, variant, component_id, deadline):
        seen.append(variant.refinement_budget_ms)
        result = original(indices, variant, component_id, deadline)
        result.refinement_elapsed_ms = 4.
        return result

    monkeypatch.setattr(core, "_component_spheres", component)
    points = np.vstack((current.voxel_centers, current.voxel_centers + .5))
    core.generate_dynamic_spheres(points, replace(params, refinement_budget_ms=5.))
    assert seen == [5., 1.]


def test_search_that_finished_does_not_launch_redundant_worker_job():
    params, current, template = fixture_geometry()
    current.min_k_search_termination = "no_more_valid_merges"
    refiner = fake_refiner(template)
    refiner.busy = False
    stats = refiner.update(as_result(current), 1_100_000_000, params)
    assert not stats["refinement_submitted"] and refiner.requests.empty()


def test_background_result_reduces_a_current_rigid_cover_end_to_end():
    params, _, _ = fixture_geometry()
    params = replace(params, dynamic_merge_enable_empty_space_guard=False,
        dynamic_merge_max_radius_growth_ratio=3., dynamic_min_k_max_states=16,
        processing_budget_ms=500.)
    points = voxel_centers(np.asarray([[x, 5, 5] for x in range(4, 8)]), params)
    current = DynamicComponentResult(0, points,
        [DynamicSphere(*p, .025, .025, 0) for p in points], 1., np.empty((0, 3)), "fresh",
        min_k_search_termination="deadline")
    refiner = AsyncSphereRefiner(RefinementOptions(validation_budget_ms=100.))
    try:
        assert refiner.update(as_result(current), 1_000_000_000, params)["refinement_submitted"]
        deadline = monotonic() + 8.
        while refiner.busy and monotonic() < deadline:
            refiner._poll()
            sleep(.01)
        assert not refiner.busy and not refiner.error
        result = as_result(current)
        status = refiner.update(result, 1_100_000_000, params)
        assert status["refinement_applied_components"] == 1
        assert len(result.spheres) < len(current.spheres)
        assert np.all(_covered_mask(points, result.spheres, 0.))
    finally:
        refiner.close()


def test_partial_ipc_message_does_not_block_foreground_poll():
    params, current, template = fixture_geometry()
    entered, release, stop = Event(), Event(), Event()
    refiner = fake_refiner(template)

    class PartialTransport:
        def get(self, timeout):
            entered.set()
            release.wait(timeout=2.)  # Simulate a stalled writer mid-message.
            raise EOFError

    collector = Thread(target=_collect_replies,
        args=(PartialTransport(), refiner.replies, stop), daemon=True)
    collector.start()
    try:
        assert entered.wait(timeout=1.)
        refiner.process.is_alive = lambda: False
        status = refiner.update(as_result(current), 1_100_000_000, params)
        assert not release.is_set()
        assert status["refinement_error"] == "worker_exited"
    finally:
        stop.set()
        release.set()
        collector.join(timeout=1.)
    assert not collector.is_alive()
