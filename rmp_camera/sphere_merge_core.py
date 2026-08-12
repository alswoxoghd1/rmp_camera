"""Pure NumPy geometry for conservative agglomerative sphere merging."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class PairMergeCandidate:
    """A valid pair replacement and its deterministic selection cost."""

    first_index: int
    second_index: int
    center: np.ndarray
    radius: float
    gap: float
    radius_growth: float
    volume_growth: float
    over_approximation_penalty: float

    @property
    def cost_key(self) -> tuple[float, ...]:
        return (
            self.over_approximation_penalty,
            self.volume_growth,
            self.radius_growth,
            self.gap,
            self.radius,
            float(self.center[0]),
            float(self.center[1]),
            float(self.center[2]),
            float(self.first_index),
            float(self.second_index),
        )


def minimum_enclosing_sphere_pair(
    first_center: Sequence[float],
    first_radius: float,
    second_center: Sequence[float],
    second_radius: float,
) -> tuple[np.ndarray, float]:
    """Return the exact minimum sphere containing both input spheres."""

    c1 = _center(first_center)
    c2 = _center(second_center)
    r1 = _radius(first_radius)
    r2 = _radius(second_radius)
    delta = c2 - c1
    distance = float(np.linalg.norm(delta))
    if distance + min(r1, r2) <= max(r1, r2):
        if r1 >= r2:
            return c1.copy(), r1
        return c2.copy(), r2
    if distance == 0.0:
        return c1.copy(), max(r1, r2)
    radius = 0.5 * (distance + r1 + r2)
    center = c1 + ((radius - r1) / distance) * delta
    return center, float(radius)


def sphere_pair_gap(
    first_center: Sequence[float], first_radius: float,
    second_center: Sequence[float], second_radius: float,
) -> float:
    """Surface-to-surface gap; overlapping spheres have zero gap."""

    distance = float(np.linalg.norm(_center(second_center) - _center(first_center)))
    return max(0.0, distance - _radius(first_radius) - _radius(second_radius))


def build_pair_merge_candidate(
    first_index: int,
    second_index: int,
    centers: np.ndarray,
    radii: np.ndarray,
    component_ids: np.ndarray,
    max_radius_m: float,
    max_radius_growth_ratio: float,
    max_gap_m: float,
) -> PairMergeCandidate | None:
    """Build a merge candidate when all conservative hard guards pass."""

    if component_ids[first_index] != component_ids[second_index]:
        return None
    c1, c2 = centers[first_index], centers[second_index]
    r1, r2 = float(radii[first_index]), float(radii[second_index])
    center, radius = minimum_enclosing_sphere_pair(c1, r1, c2, r2)
    gap = sphere_pair_gap(c1, r1, c2, r2)
    growth = radius / max(r1, r2, np.finfo(np.float64).eps)
    volume_sum = r1 ** 3 + r2 ** 3
    volume_growth = radius ** 3 / max(volume_sum, np.finfo(np.float64).eps)
    penalty = max(0.0, radius ** 3 - volume_sum)
    if (
        radius > max_radius_m + 1e-12
        or growth > max_radius_growth_ratio + 1e-12
        or gap > max_gap_m + 1e-12
    ):
        return None
    return PairMergeCandidate(
        min(first_index, second_index),
        max(first_index, second_index),
        center,
        radius,
        gap,
        growth,
        volume_growth,
        penalty,
    )


def select_best_merge_pair(
    centers: np.ndarray,
    radii: np.ndarray,
    component_ids: np.ndarray,
    max_radius_m: float,
    max_radius_growth_ratio: float,
    max_gap_m: float,
    validator: Callable[[PairMergeCandidate], bool] | None = None,
    deadline: float | None = None,
) -> PairMergeCandidate | None:
    """Return the lowest-cost valid pair, or ``None`` when no pair is valid."""

    centers_array = np.asarray(centers, dtype=np.float64)
    radii_array = np.asarray(radii, dtype=np.float64)
    components_array = np.asarray(component_ids, dtype=np.int64)
    count = len(radii_array)
    if centers_array.shape != (count, 3) or components_array.shape != (count,):
        raise ValueError("centers, radii, and component_ids have incompatible shapes")
    best: PairMergeCandidate | None = None
    for first_index in range(count):
        for second_index in range(first_index + 1, count):
            if deadline is not None and monotonic() >= deadline:
                return best
            candidate = build_pair_merge_candidate(
                first_index,
                second_index,
                centers_array,
                radii_array,
                components_array,
                max_radius_m,
                max_radius_growth_ratio,
                max_gap_m,
            )
            if candidate is None or (validator is not None and not validator(candidate)):
                continue
            if best is None or candidate.cost_key < best.cost_key:
                best = candidate
    return best


def validate_merge_limits(
    max_radius_m: float, max_radius_growth_ratio: float, max_gap_m: float,
) -> None:
    if not np.isfinite((
        max_radius_m, max_radius_growth_ratio, max_gap_m
    )).all():
        raise ValueError("merge limits must be finite")
    if max_radius_m <= 0.0:
        raise ValueError("merge maximum radius must be positive")
    if max_radius_growth_ratio < 1.0:
        raise ValueError("merge radius growth ratio must be at least 1")
    if max_gap_m < 0.0:
        raise ValueError("merge maximum gap must be non-negative")


def _center(value: Sequence[float]) -> np.ndarray:
    center = np.asarray(value, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("sphere center must be a finite three-vector")
    return center


def _radius(value: float) -> float:
    radius = float(value)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("sphere radius must be finite and non-negative")
    return radius
