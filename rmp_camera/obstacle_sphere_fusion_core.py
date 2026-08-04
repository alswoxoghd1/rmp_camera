"""ROS-independent static/dynamic sphere fusion and cache policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FusionSphere:
    x: float
    y: float
    z: float
    raw_radius: float
    output_radius: float
    source_type: int  # 0 static, 1 dynamic
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


def fuse_spheres(
    static_spheres: Sequence[FusionSphere], dynamic_spheres: Sequence[FusionSphere],
    params: FusionParameters,
) -> list[FusionSphere]:
    """Fuse using real 3D sphere overlap; dynamic wins only when configured."""
    static_items = [replace(s, source_type=0) for s in static_spheres]
    dynamic_items = [replace(s, source_type=1) for s in dynamic_spheres]
    if params.dynamic_priority:
        static_items = [
            static for static in static_items
            if not any(spheres_overlap(static, dynamic, params.overlap_tolerance_m)
                       for dynamic in dynamic_items)
        ]
        combined = dynamic_items + static_items
    else:
        dynamic_items = [
            dynamic for dynamic in dynamic_items
            if not any(spheres_overlap(dynamic, static, params.overlap_tolerance_m)
                       for static in static_items)
        ]
        combined = static_items + dynamic_items
    return combined[:params.max_total_spheres]


class SphereFusionCache:
    """Latest-message cache with static confirmation and dynamic handover."""

    def __init__(self, params: FusionParameters) -> None:
        self.params = params
        self.static_spheres: list[FusionSphere] = []
        self.dynamic_spheres: list[FusionSphere] = []
        self.static_stamp: float | None = None
        self.dynamic_stamp: float | None = None
        self.static_empty_count = 0
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
            if self.static_empty_count >= self.params.static_empty_confirmation_frames:
                self.static_spheres = []
                self.static_stamp = float(timestamp_sec)
        return True

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

    def combined(self, now_sec: float) -> list[FusionSphere]:
        now = float(now_sec)
        static = list(self.static_spheres)
        if self.params.static_cache_expire_enabled and self.static_stamp is not None:
            if now - self.static_stamp > self.params.hysteresis_sec:
                static = []
        dynamic = self._handover_dynamic(static, now)
        return fuse_spheres(static, dynamic, self.params)

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
