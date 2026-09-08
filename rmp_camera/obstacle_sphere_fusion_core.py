"""ROS-independent static/dynamic sphere fusion and cache policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence
from time import monotonic

import numpy as np


@dataclass(frozen=True)
class FusionSphere:
    x: float
    y: float
    z: float
    raw_radius: float
    output_radius: float
    source_type: int  # 0 static, 1 dynamic, 2 semantic human
    component_id: int = -1
    component_coverage: float = 0.0
    track_id: int = -1
    age: int = 0
    confidence: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=np.float64)


@dataclass(frozen=True)
class FusionParameters:
    target_frame: str = "base_link"
    overlap_tolerance_m: float = 0.01
    dynamic_priority: bool = True
    hysteresis_sec: float = 0.5
    max_total_spheres: int = 512
    static_cache_expire_enabled: bool = False
    static_empty_confirmation_frames: int = 3
    dynamic_handover_grace_sec: float = 0.5
    keep_dynamic_until_static_overlap: bool = True
    dynamic_absolute_max_ttl_sec: float = 3.0
    human_priority: bool = True
    human_handover_grace_sec: float = 0.35
    human_absolute_max_ttl_sec: float = 0.35
    human_empty_confirmation_frames: int = 3
    coverage_guard_enabled: bool = False
    coverage_tolerance_m: float = .02
    coverage_budget_ms: float = 5.0


def field_rows(
    field_names: Sequence[str], rows: Iterable[Sequence[float]],
    required_fields: Sequence[str],
) -> list[dict[str, float]]:
    """Map PointCloud-like rows by field name, independent of column order."""
    indices = {name: index for index, name in enumerate(field_names)}
    missing = [name for name in required_fields if name not in indices]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    return [
        {name: float(row[indices[name]]) for name in required_fields}
        for row in rows
    ]


def spheres_from_field_rows(
    field_names: Sequence[str], rows: Iterable[Sequence[float]], source_type: int,
) -> list[FusionSphere]:
    rows = list(rows)
    required = ("x", "y", "z", "raw_radius", "output_radius")
    mapped = field_rows(field_names, rows, required)
    indices = {name: index for index, name in enumerate(field_names)}
    result = []
    for row, values in zip(rows, mapped):
        optional = lambda name, default: float(row[indices[name]]) if name in indices else default
        result.append(FusionSphere(
            values["x"], values["y"], values["z"], values["raw_radius"],
            values["output_radius"], source_type,
            int(optional("component_id", -1)), optional("component_coverage", 0.0),
            int(optional("track_id", -1)), int(optional("age", 0)),
            optional("confidence", 0.0)))
    return result


def spheres_overlap(a: FusionSphere, b: FusionSphere, tolerance_m: float = 0.0) -> bool:
    distance = float(np.linalg.norm(a.center - b.center))
    return distance <= a.output_radius + b.output_radius + tolerance_m


def unit_ball_samples(sample_count: int) -> np.ndarray:
    """Return deterministic, approximately uniform samples inside a unit ball."""

    count = int(sample_count)
    if count <= 0:
        raise ValueError("sample_count must be positive")
    indices = np.arange(count, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radial_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.column_stack((
        radial_xy * np.cos(indices * golden_angle),
        radial_xy * np.sin(indices * golden_angle),
        z,
    ))
    volume_radii = ((indices + 0.5) / count) ** (1.0 / 3.0)
    return directions * volume_radii[:, None]


def maximum_robot_containment_fraction(
    obstacle_center: Sequence[float],
    obstacle_radius: float,
    robot_centers: np.ndarray,
    robot_radii: np.ndarray,
    sample_count: int = 256,
) -> float:
    """Estimate the largest robot-sphere volume fraction inside an obstacle."""

    center = np.asarray(obstacle_center, dtype=np.float64)
    radius = float(obstacle_radius)
    centers = np.asarray(robot_centers, dtype=np.float64)
    radii = np.asarray(robot_radii, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("obstacle_center must be a finite three-vector")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("obstacle_radius must be finite and non-negative")
    if centers.ndim != 2 or centers.shape[1:] != (3,):
        raise ValueError("robot_centers must have shape (N, 3)")
    if radii.shape != (len(centers),):
        raise ValueError("robot_centers and robot_radii have incompatible shapes")
    if not np.isfinite(centers).all():
        raise ValueError("robot sphere centers must be finite")
    if not np.isfinite(radii).all() or np.any(radii < 0.0):
        raise ValueError("robot sphere radii must be finite and non-negative")
    if len(centers) == 0:
        return 0.0
    unit_samples = unit_ball_samples(sample_count)
    points = (
        centers[:, None, :]
        + radii[:, None, None] * unit_samples[None, :, :]
    )
    squared_distances = np.sum(
        (points - center[None, None, :]) ** 2,
        axis=2,
    )
    contained = squared_distances <= radius * radius
    fractions = np.count_nonzero(contained, axis=1) / contained.shape[1]
    return float(np.max(fractions))


def filter_spheres_containing_robot(
    spheres: Sequence[FusionSphere],
    robot_centers: np.ndarray,
    robot_radii: np.ndarray,
    coverage_threshold: float = 0.8,
    sample_count: int = 256,
) -> tuple[list[FusionSphere], list[FusionSphere]]:
    """Remove obstacles containing at least one robot sphere by threshold."""

    threshold = float(coverage_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("coverage_threshold must be in [0, 1]")
    kept = []
    removed = []
    for sphere in spheres:
        coverage = maximum_robot_containment_fraction(
            sphere.center,
            sphere.output_radius,
            robot_centers,
            robot_radii,
            sample_count,
        )
        (removed if coverage >= threshold else kept).append(sphere)
    return kept, removed


def fuse_spheres(
    static_spheres: Sequence[FusionSphere], dynamic_spheres: Sequence[FusionSphere],
    params: FusionParameters, human_spheres: Sequence[FusionSphere] = (),
    *, support=None, stats=None,
) -> list[FusionSphere]:
    """Fuse geometry paths; semantic humans suppress only overlapping copies."""
    static_items = [replace(s, source_type=0) for s in static_spheres]
    dynamic_items = [replace(s, source_type=1) for s in dynamic_spheres]
    human_items = [replace(s, source_type=2) for s in human_spheres]
    if params.coverage_guard_enabled:
        ordered = ((dynamic_items + static_items) if params.dynamic_priority
                   else (static_items + dynamic_items))
        ordered = human_items + ordered if params.human_priority else ordered + human_items
        return coverage_preserving_fusion(ordered, params, support or {}, stats)
    if params.dynamic_priority:
        static_items = [
            static for static in static_items
            if not any(spheres_overlap(static, dynamic, params.overlap_tolerance_m)
                       for dynamic in dynamic_items)
        ]
        nonhuman = dynamic_items + static_items
    else:
        dynamic_items = [
            dynamic for dynamic in dynamic_items
            if not any(spheres_overlap(dynamic, static, params.overlap_tolerance_m)
                       for static in static_items)
        ]
        nonhuman = static_items + dynamic_items
    if params.human_priority:
        nonhuman = [
            item for item in nonhuman
            if not any(spheres_overlap(item, human, params.overlap_tolerance_m)
                       for human in human_items)
        ]
        combined = human_items + nonhuman
    else:
        human_items = [
            human for human in human_items
            if not any(spheres_overlap(human, item, params.overlap_tolerance_m)
                       for item in nonhuman)
        ]
        combined = nonhuman + human_items
    return combined[:params.max_total_spheres]


def coverage_preserving_fusion(ordered, params, support, stats=None):
    """Only remove a lower-priority copy if ALL its retained support survives.

    Missing/stale/oversized support falls back to sufficient analytic whole-ball
    containment, never a sparse sampling estimate or mere surface intersection.
    Validation budget exhaustion preserves candidates. The configured output cap
    remains a hard cap, but inability to preserve coverage is explicitly reported.
    """
    start = monotonic()
    deadline = start + params.coverage_budget_ms / 1000.
    kept, removed, checked, missing, partial = [], 0, 0, 0, 0
    for sphere in ordered:
        winners = [s for s in kept if s.source_type != sphere.source_type
                   and spheres_overlap(sphere, s, params.overlap_tolerance_m)]
        redundant = False
        if winners and monotonic() < deadline:
            points = support.get(sphere)
            if (points is not None and 0 < len(points) <= 8192
                    and len(points) * len(winners) <= 200000
                    and np.isfinite(points).all()):
                checked += len(points)
                covered = np.zeros(len(points), dtype=bool)
                for winner in winners:
                    covered |= np.sum((points - winner.center) ** 2, axis=1) <= (
                        winner.raw_radius + params.coverage_tolerance_m + 1e-9) ** 2
                    if covered.all() or monotonic() >= deadline:
                        break
                redundant = bool(covered.all())
                if not redundant:
                    partial += 1
            else:
                missing += 1
                redundant = any(
                    np.linalg.norm(sphere.center - winner.center) + sphere.output_radius
                    <= winner.output_radius + 1e-9
                    and np.linalg.norm(sphere.center - winner.center) + sphere.raw_radius
                    <= winner.raw_radius + 1e-9 for winner in winners)
        if redundant:
            removed += 1
        else:
            kept.append(sphere)
    overflow = max(0, len(kept) - params.max_total_spheres)
    if stats is not None:
        stats.update(coverage_removed=removed, coverage_checked_voxels=checked,
            coverage_missing_support=missing, coverage_cap_dropped=overflow,
            coverage_kept_partial=partial,
            coverage_valid=overflow == 0, fusion_ms=(monotonic() - start) * 1000.)
    return kept[:params.max_total_spheres]


class SphereFusionCache:
    """Latest-message cache with static confirmation and dynamic handover."""

    def __init__(self, params: FusionParameters) -> None:
        self.params = params
        self.static_spheres: list[FusionSphere] = []
        self.dynamic_spheres: list[FusionSphere] = []
        self.human_spheres: list[FusionSphere] = []
        self.static_stamp: float | None = None
        self.dynamic_stamp: float | None = None
        self.human_stamp: float | None = None
        self.static_empty_count = 0
        self.human_empty_count = 0
        self.last_rejection_reason = ""

    def _accept_frame(self, frame_id: str) -> bool:
        if frame_id != self.params.target_frame:
            self.last_rejection_reason = (
                f"frame mismatch: expected {self.params.target_frame}, got {frame_id}")
            return False
        self.last_rejection_reason = ""
        return True

    def update_static(
        self, spheres: Iterable[FusionSphere], frame_id: str, timestamp_sec: float,
        *, confirmed_empty: bool = False,
    ) -> bool:
        if not self._accept_frame(frame_id):
            return False
        items = [replace(s, source_type=0) for s in spheres]
        if items:
            self.static_spheres = items
            self.static_stamp = float(timestamp_sec)
            self.static_empty_count = 0
        else:
            self.static_empty_count += 1
            if confirmed_empty or self.static_empty_count >= self.params.static_empty_confirmation_frames:
                self.static_spheres = []
                self.static_stamp = float(timestamp_sec)
        return True

    def invalidate_static_observation(self):
        # Unknown/error frames are not consecutive observations of empty space.
        self.static_empty_count = 0

    def update_dynamic(
        self, spheres: Iterable[FusionSphere], frame_id: str, timestamp_sec: float,
    ) -> bool:
        if not self._accept_frame(frame_id):
            return False
        items = [replace(s, source_type=1) for s in spheres]
        if items:
            self.dynamic_spheres = items
            self.dynamic_stamp = float(timestamp_sec)
        return True

    def update_human(
        self, spheres: Iterable[FusionSphere], frame_id: str, timestamp_sec: float,
    ) -> bool:
        if not self._accept_frame(frame_id):
            return False
        items = [replace(s, source_type=2) for s in spheres]
        if items:
            self.human_spheres = items
            self.human_stamp = float(timestamp_sec)
            self.human_empty_count = 0
        else:
            # Use a small frame-count debounce instead of starting another
            # time-based TTL. This hides sub-100 ms inference gaps while a
            # sustained empty stream still clears promptly and deterministically.
            self.human_empty_count += 1
            if self.human_empty_count >= self.params.human_empty_confirmation_frames:
                self.human_spheres = []
                self.human_stamp = float(timestamp_sec)
        return True

    def combined(self, now_sec: float, static_override=None, *, support=None, stats=None) -> list[FusionSphere]:
        now = float(now_sec)
        static = list(self.static_spheres if static_override is None else static_override)
        if self.params.static_cache_expire_enabled and self.static_stamp is not None:
            if now - self.static_stamp > self.params.hysteresis_sec:
                static = []
        dynamic = self._handover_dynamic(static, now)
        human = self._handover_human(now)
        return fuse_spheres(static, dynamic, self.params, human, support=support, stats=stats)

    def _handover_dynamic(
        self, static: Sequence[FusionSphere], now: float,
    ) -> list[FusionSphere]:
        if self.dynamic_stamp is None:
            return []
        age = now - self.dynamic_stamp
        if age > self.params.dynamic_absolute_max_ttl_sec:
            return []
        if age <= self.params.dynamic_handover_grace_sec:
            return list(self.dynamic_spheres)
        if not self.params.keep_dynamic_until_static_overlap:
            return []
        # A stale dynamic sphere remains only until a static sphere occupies its place.
        return [
            dynamic for dynamic in self.dynamic_spheres
            if not any(spheres_overlap(dynamic, stat, self.params.overlap_tolerance_m)
                       for stat in static)
        ]

    def _handover_human(self, now: float) -> list[FusionSphere]:
        if self.human_stamp is None:
            return []
        age = now - self.human_stamp
        if age > self.params.human_absolute_max_ttl_sec:
            return []
        if age <= self.params.human_handover_grace_sec:
            return list(self.human_spheres)
        return []
