"""ROS-independent dynamic point to obstacle-sphere algorithms.

The implementation intentionally depends only on NumPy.  Sparse voxel sets are
used for morphology and connectivity so an accidentally large workspace does
not allocate a correspondingly large dense array.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Iterable, Sequence

import numpy as np

from rmp_camera.sphere_merge_core import (
    PairMergeCandidate,
    select_best_merge_pair,
    validate_merge_limits,
)


@dataclass(frozen=True)
class DynamicSphereParameters:
    voxel_size_m: float = 0.05
    min_component_voxels: int = 6
    max_component_voxels: int = 120000
    max_components: int = 32
    max_input_points: int = 250000
    connectivity: int = 18
    dilation_voxels: int = 0
    closing_iterations: int = 1
    minimum_center_spacing_m: float = 0.08
    min_raw_radius_m: float = 0.025
    max_raw_radius_m: float = 0.45
    min_useful_adaptive_radius_m: float = 0.075
    enable_fixed_radius_fallback: bool = True
    fixed_radius_m: float = 0.10
    enable_greedy_set_cover: bool = True
    enable_single_sphere_replacement: bool = True
    single_sphere_max_radius_m: float = 0.18
    target_coverage: float = 0.95
    coverage_tolerance_m: float = 0.02
    safety_margin_m: float = 0.015
    redundancy_tolerance_m: float = 0.01
    max_spheres_per_component: int = 96
    max_iterations_per_component: int = 256
    max_total_spheres: int = 384
    processing_budget_ms: float = 35.0
    max_local_grid_voxels: int = 1200000
    dynamic_enable_agglomerative_merge: bool = True
    dynamic_merge_max_radius_m: float = 0.30
    dynamic_merge_max_radius_growth_ratio: float = 1.50
    dynamic_merge_max_gap_m: float = 0.08
    dynamic_merge_enable_empty_space_guard: bool = False
    dynamic_merge_max_empty_fraction: float = 0.70
    dynamic_merge_max_validation_voxels: int = 50000
    min_x_m: float = -3.0
    max_x_m: float = 3.0
    min_y_m: float = -3.0
    max_y_m: float = 3.0
    min_z_m: float = -0.2
    max_z_m: float = 2.5
    max_range_from_base_m: float = 5.0

    def validate(self) -> None:
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")
        if self.connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be 6, 18, or 26")
        if not 0.0 <= self.target_coverage <= 1.0:
            raise ValueError("target_coverage must be in [0, 1]")
        if self.max_input_points <= 0 or self.max_total_spheres <= 0:
            raise ValueError("point and sphere limits must be positive")
        if self.min_raw_radius_m < 0.0 or self.max_raw_radius_m < self.min_raw_radius_m:
            raise ValueError("invalid raw-radius limits")
        if self.single_sphere_max_radius_m <= 0.0:
            raise ValueError("single_sphere_max_radius_m must be positive")
        if self.single_sphere_max_radius_m < self.min_raw_radius_m:
            raise ValueError(
                "single_sphere_max_radius_m must be at least "
                "min_raw_radius_m")
        validate_merge_limits(
            self.dynamic_merge_max_radius_m,
            self.dynamic_merge_max_radius_growth_ratio,
            self.dynamic_merge_max_gap_m,
        )
        if not np.isfinite(self.dynamic_merge_max_empty_fraction):
            raise ValueError(
                "dynamic merge empty fraction must be finite")
        if not 0.0 <= self.dynamic_merge_max_empty_fraction <= 1.0:
            raise ValueError("dynamic merge empty fraction must be in [0, 1]")
        if self.dynamic_merge_max_validation_voxels <= 0:
            raise ValueError("dynamic merge validation voxel cap must be positive")
        if self.max_x_m <= self.min_x_m or self.max_y_m <= self.min_y_m:
            raise ValueError("invalid XY workspace")
        if self.max_z_m <= self.min_z_m:
            raise ValueError("invalid Z workspace")


@dataclass(frozen=True)
class DynamicSphere:
    x: float
    y: float
    z: float
    raw_radius: float
    output_radius: float
    component_id: int = -1
    component_coverage: float = 0.0
    track_id: int = -1
    age: int = 1
    confidence: float = 0.0

    @property
    def center(self) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=np.float64)


@dataclass
class DynamicComponentResult:
    component_id: int
    voxel_centers: np.ndarray
    spheres: list[DynamicSphere]
    coverage: float
    uncovered_voxels: np.ndarray
    termination_reason: str
    pre_merge_sphere_count: int = 0


@dataclass
class DynamicSphereResult:
    components: list[DynamicComponentResult] = field(default_factory=list)
    spheres: list[DynamicSphere] = field(default_factory=list)
    voxel_centers: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    uncovered_voxels: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    elapsed_ms: float = 0.0
    termination_reason: str = "empty_input"


def neighbor_offsets(connectivity: int) -> tuple[tuple[int, int, int], ...]:
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be 6, 18, or 26")
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nonzero = int(dx != 0) + int(dy != 0) + int(dz != 0)
                if connectivity == 6 and nonzero == 1:
                    offsets.append((dx, dy, dz))
                elif connectivity == 18 and nonzero <= 2:
                    offsets.append((dx, dy, dz))
                elif connectivity == 26:
                    offsets.append((dx, dy, dz))
    return tuple(offsets)


def _as_points(points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    return array[:, :3]


def filter_and_voxelize(
    points: np.ndarray | Sequence[Sequence[float]], params: DynamicSphereParameters,
) -> np.ndarray:
    """Return sorted unique integer voxel indices in the configured workspace."""
    params.validate()
    xyz = _as_points(points)
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    xyz = xyz[:params.max_input_points]
    finite = np.isfinite(xyz).all(axis=1)
    lo = np.asarray((params.min_x_m, params.min_y_m, params.min_z_m))
    hi = np.asarray((params.max_x_m, params.max_y_m, params.max_z_m))
    keep = finite & (xyz >= lo).all(axis=1) & (xyz <= hi).all(axis=1)
    if params.max_range_from_base_m > 0.0:
        keep &= np.linalg.norm(xyz[:, :2], axis=1) <= params.max_range_from_base_m
    xyz = xyz[keep]
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    indices = np.floor((xyz - lo) / params.voxel_size_m).astype(np.int64)
    indices = np.unique(indices, axis=0)
    order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
    return indices[order]


def connected_components(indices: np.ndarray, connectivity: int = 18) -> list[np.ndarray]:
    """Deterministic sparse connected components."""
    if len(indices) == 0:
        return []
    remaining = {tuple(int(v) for v in row) for row in indices}
    offsets = neighbor_offsets(connectivity)
    components: list[np.ndarray] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue = [seed]
        component = []
        cursor = 0
        while cursor < len(queue):
            voxel = queue[cursor]
            cursor += 1
            component.append(voxel)
            for offset in offsets:
                candidate = (
                    voxel[0] + offset[0], voxel[1] + offset[1], voxel[2] + offset[2])
                if candidate in remaining:
                    remaining.remove(candidate)
                    queue.append(candidate)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    components.sort(key=lambda c: (-len(c), tuple(c[0])))
    return components


def _dilate(voxels: set[tuple[int, int, int]], offsets, iterations: int):
    result = set(voxels)
    for _ in range(max(0, iterations)):
        expanded = set(result)
        for voxel in result:
            expanded.update(
                (voxel[0] + d[0], voxel[1] + d[1], voxel[2] + d[2]) for d in offsets)
        result = expanded
    return result


def _erode(voxels: set[tuple[int, int, int]], offsets, iterations: int):
    result = set(voxels)
    for _ in range(max(0, iterations)):
        result = {
            voxel for voxel in result
            if all((voxel[0] + d[0], voxel[1] + d[1], voxel[2] + d[2]) in result
                   for d in offsets)
        }
        if not result:
            break
    return result


def apply_morphology(indices: np.ndarray, params: DynamicSphereParameters) -> np.ndarray:
    if len(indices) == 0:
        return indices.copy()
    offsets = neighbor_offsets(params.connectivity)
    voxels = {tuple(int(v) for v in row) for row in indices}
    if params.closing_iterations > 0:
        closed = _dilate(voxels, offsets, params.closing_iterations)
        eroded = _erode(closed, offsets, params.closing_iterations)
        if eroded:
            voxels = eroded
    voxels = _dilate(voxels, offsets, params.dilation_voxels)
    return np.asarray(sorted(voxels), dtype=np.int64)


def voxel_centers(indices: np.ndarray, params: DynamicSphereParameters) -> np.ndarray:
    origin = np.asarray((params.min_x_m, params.min_y_m, params.min_z_m))
    return origin + (indices.astype(np.float64) + 0.5) * params.voxel_size_m


def _boundary_depths(indices: np.ndarray, connectivity: int) -> np.ndarray:
    """Return integer erosion depth (one at the boundary) for each voxel."""
    offsets = neighbor_offsets(connectivity)
    remaining = {tuple(int(v) for v in row) for row in indices}
    depths: dict[tuple[int, int, int], int] = {}
    depth = 1
    while remaining:
        boundary = {
            voxel for voxel in remaining
            if any((voxel[0] + d[0], voxel[1] + d[1], voxel[2] + d[2]) not in remaining
                   for d in offsets)
        }
        if not boundary:
            boundary = set(remaining)
        for voxel in boundary:
            depths[voxel] = depth
        remaining.difference_update(boundary)
        depth += 1
    return np.asarray([depths[tuple(int(v) for v in row)] for row in indices], dtype=np.float64)


def _covered_mask(points: np.ndarray, spheres: Sequence[DynamicSphere], tolerance: float) -> np.ndarray:
    covered = np.zeros(len(points), dtype=bool)
    for sphere in spheres:
        delta = points - sphere.center
        covered |= np.einsum("ij,ij->i", delta, delta) <= (
            sphere.raw_radius + tolerance) ** 2
    return covered


def _sphere_coverage_matrices(
    points: np.ndarray, spheres: Sequence[DynamicSphere], tolerance: float,
    redundancy_tolerance: float = 0.0, deadline: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build standard and redundancy coverage matrices from each distance once."""
    standard = np.zeros((len(spheres), len(points)), dtype=bool)
    redundant = np.zeros_like(standard)
    for index, sphere in enumerate(spheres):
        if deadline is not None and monotonic() >= deadline:
            return None, None
        delta = points - sphere.center
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        standard[index] = distance_squared <= (sphere.raw_radius + tolerance) ** 2
        redundant[index] = distance_squared <= (
            sphere.raw_radius + tolerance + redundancy_tolerance) ** 2
    return standard, redundant


def _sphere_coverage_masks(
    points: np.ndarray, spheres: Sequence[DynamicSphere], tolerance: float,
) -> np.ndarray:
    """Return one reusable boolean voxel-coverage row per sphere."""
    masks, _ = _sphere_coverage_matrices(points, spheres, tolerance)
    assert masks is not None
    return masks


def _single_sphere_candidates(
    centers: np.ndarray, spheres: Sequence[DynamicSphere],
    params: DynamicSphereParameters, component_id: int,
) -> list[DynamicSphere]:
    """Create bounded centroid, AABB-center, and largest-sphere-center candidates."""
    if len(centers) == 0:
        return []

    candidate_centers = [
        np.mean(centers, axis=0),
        0.5 * (np.min(centers, axis=0) + np.max(centers, axis=0)),
    ]
    if spheres:
        largest = min(
            enumerate(spheres),
            key=lambda item: (
                -item[1].raw_radius, item[1].x, item[1].y, item[1].z, item[0]))[1]
        candidate_centers.append(largest.center)

    # Equal centers arise frequently for symmetric components. Removing them
    # avoids duplicate matrix rows without introducing a fuzzy spatial test.
    unique_centers: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for center in candidate_centers:
        key = tuple(float(value) for value in center)
        if key not in seen:
            seen.add(key)
            unique_centers.append(np.asarray(center, dtype=np.float64))

    kth_index = max(0, int(np.ceil(params.target_coverage * len(centers))) - 1)
    radius_cap = min(params.max_raw_radius_m, params.single_sphere_max_radius_m)
    candidates = []
    for center in unique_centers:
        distances = np.linalg.norm(centers - center, axis=1)
        required_distance = float(np.sort(distances)[kth_index])
        raw_radius = max(
            params.min_raw_radius_m,
            required_distance - params.coverage_tolerance_m)
        raw_radius = min(raw_radius, radius_cap)
        candidates.append(DynamicSphere(
            float(center[0]), float(center[1]), float(center[2]), raw_radius,
            raw_radius + params.safety_margin_m, component_id))
    return candidates


def _best_single_sphere(
    candidates: Sequence[DynamicSphere], coverage_masks: np.ndarray,
    original_sphere_count: int, params: DynamicSphereParameters,
) -> DynamicSphere | None:
    """Return the smallest deterministic one-sphere target-coverage replacement."""
    if original_sphere_count <= 1 or not candidates:
        return None
    coverage = np.mean(coverage_masks, axis=1) if coverage_masks.shape[1] else np.ones(
        len(candidates), dtype=np.float64)
    valid = [
        index for index, sphere in enumerate(candidates)
        if coverage[index] + 1e-12 >= params.target_coverage
        and sphere.raw_radius <= params.single_sphere_max_radius_m + 1e-12
        and sphere.raw_radius <= params.max_raw_radius_m + 1e-12
    ]
    if not valid:
        return None
    best = min(valid, key=lambda index: (
        candidates[index].raw_radius,
        -float(coverage[index]),
        candidates[index].x,
        candidates[index].y,
        candidates[index].z,
        index,
    ))
    return candidates[best]


def _greedy_set_cover(
    spheres: Sequence[DynamicSphere], coverage_masks: np.ndarray,
    target_coverage: float, deadline: float | None = None,
) -> list[int] | None:
    """Select deterministic sphere indices that reach the requested voxel coverage."""
    if coverage_masks.shape[0] != len(spheres):
        raise ValueError("coverage matrix row count must match spheres")
    num_voxels = coverage_masks.shape[1]
    if num_voxels == 0 or target_coverage <= 0.0:
        return []

    selected: list[int] = []
    available = np.ones(len(spheres), dtype=bool)
    covered = np.zeros(num_voxels, dtype=bool)
    while float(np.mean(covered)) + 1e-12 < target_coverage:
        if deadline is not None and monotonic() >= deadline:
            return None
        gains = np.count_nonzero(coverage_masks & ~covered, axis=1)
        choices = np.flatnonzero(available & (gains > 0))
        if len(choices) == 0:
            break
        best = min(
            (int(index) for index in choices),
            key=lambda index: (
                -int(gains[index]),
                spheres[index].raw_radius,
                spheres[index].x,
                spheres[index].y,
                spheres[index].z,
                index,
            ))
        selected.append(best)
        available[best] = False
        covered |= coverage_masks[best]

    if float(np.mean(covered)) + 1e-12 < target_coverage:
        return None
    return selected


def _fixed_radius_cover(
    centers: np.ndarray, params: DynamicSphereParameters, component_id: int,
    limit: int,
) -> tuple[list[DynamicSphere], np.ndarray]:
    spheres: list[DynamicSphere] = []
    uncovered = np.ones(len(centers), dtype=bool)
    radius = min(params.max_raw_radius_m, max(params.min_raw_radius_m, params.fixed_radius_m))
    for idx in range(len(centers)):
        if not uncovered[idx] or len(spheres) >= limit:
            continue
        center = centers[idx]
        sphere = DynamicSphere(
            float(center[0]), float(center[1]), float(center[2]), radius,
            radius + params.safety_margin_m, component_id)
        spheres.append(sphere)
        delta = centers - center
        uncovered &= np.einsum("ij,ij->i", delta, delta) > (
            radius + params.coverage_tolerance_m) ** 2
        if 1.0 - float(np.count_nonzero(uncovered)) / max(1, len(centers)) >= params.target_coverage:
            break
    return spheres, uncovered


def _remove_redundant(
    centers: np.ndarray, spheres: list[DynamicSphere], params: DynamicSphereParameters,
    coverage_masks: np.ndarray | None = None,
    redundancy_masks: np.ndarray | None = None,
    deadline: float | None = None,
) -> list[DynamicSphere]:
    """Delete low-unique-contribution spheres without reducing target coverage."""
    if len(spheres) < 2 or len(centers) == 0:
        return list(spheres)
    if coverage_masks is None or redundancy_masks is None:
        coverage_masks, redundancy_masks = _sphere_coverage_matrices(
            centers, spheres, params.coverage_tolerance_m,
            params.redundancy_tolerance_m, deadline)
    if coverage_masks is None or redundancy_masks is None:
        return list(spheres)

    kept = list(range(len(spheres)))
    while len(kept) > 1:
        if deadline is not None and monotonic() >= deadline:
            break
        redundancy_counts = np.count_nonzero(redundancy_masks[kept], axis=0)
        unique_contribution = {
            index: int(np.count_nonzero(
                redundancy_masks[index] & (redundancy_counts == 1)))
            for index in kept
        }
        # Retesting after every successful deletion keeps the ordering accurate
        # while the precomputed matrices avoid repeating distance calculations.
        removal_order = sorted(kept, key=lambda index: (
            unique_contribution[index],
            spheres[index].raw_radius,
            spheres[index].x,
            spheres[index].y,
            spheres[index].z,
            index,
        ))
        removed = False
        for index in removal_order:
            trial = [candidate for candidate in kept if candidate != index]
            if not trial:
                continue
            covered = np.any(coverage_masks[trial], axis=0)
            if float(np.mean(covered)) + 1e-12 >= params.target_coverage:
                kept = trial
                removed = True
                break
        if not removed:
            break
    return [spheres[index] for index in kept]


def _optimize_component_spheres(
    centers: np.ndarray, spheres: list[DynamicSphere],
    params: DynamicSphereParameters, component_id: int, deadline: float,
) -> list[DynamicSphere]:
    """Minimize candidates using one-sphere replacement, set cover, then deletion."""
    original = list(spheres)
    if len(original) < 2 or len(centers) == 0 or monotonic() >= deadline:
        return original

    single_candidates = []
    if params.enable_single_sphere_replacement:
        single_candidates = _single_sphere_candidates(
            centers, original, params, component_id)
    candidates = original + single_candidates
    coverage_masks, redundancy_masks = _sphere_coverage_matrices(
        centers, candidates, params.coverage_tolerance_m,
        params.redundancy_tolerance_m, deadline)
    if coverage_masks is None or redundancy_masks is None:
        return original

    if single_candidates:
        best_single = _best_single_sphere(
            single_candidates, coverage_masks[len(original):],
            len(original), params)
        if best_single is not None:
            return [best_single]

    selected_indices = list(range(len(original)))
    if params.enable_greedy_set_cover:
        greedy_indices = _greedy_set_cover(
            candidates, coverage_masks, params.target_coverage, deadline)
        # Equal/larger alternatives do not further the minimization goal and
        # would unnecessarily perturb the legacy centers and radii.
        if greedy_indices is not None and len(greedy_indices) < len(original):
            selected_indices = greedy_indices

    selected_spheres = [candidates[index] for index in selected_indices]
    return _remove_redundant(
        centers, selected_spheres, params,
        coverage_masks[selected_indices], redundancy_masks[selected_indices],
        deadline)


def dynamic_merge_empty_fraction(
    centers: np.ndarray,
    merge_center: Sequence[float],
    merge_radius: float,
    params: DynamicSphereParameters,
    deadline: float | None = None,
) -> float | None:
    """Estimate the unoccupied voxel fraction inside a proposed merge sphere.

    ``None`` means that the validation deadline or voxel cap was reached, so a
    conservative caller must reject the merge.
    """

    center = np.asarray(merge_center, dtype=np.float64)
    workspace_min = np.asarray(
        (params.min_x_m, params.min_y_m, params.min_z_m), dtype=np.float64)
    lower = np.ceil(
        (center - merge_radius - workspace_min) / params.voxel_size_m - 0.5
    ).astype(np.int64)
    upper = np.floor(
        (center + merge_radius - workspace_min) / params.voxel_size_m - 0.5
    ).astype(np.int64)
    counts = np.maximum(0, upper - lower + 1)
    bounding_count = int(np.prod(counts, dtype=np.int64))
    if (
        bounding_count <= 0
        or bounding_count > params.dynamic_merge_max_validation_voxels
        or (deadline is not None and monotonic() >= deadline)
    ):
        return None
    ranges = [
        np.arange(lower[axis], upper[axis] + 1, dtype=np.int64)
        for axis in range(3)
    ]
    lattice = np.stack(
        np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape((-1, 3))
    lattice_centers = workspace_min + (
        lattice.astype(np.float64) + 0.5) * params.voxel_size_m
    inside = np.sum((lattice_centers - center) ** 2, axis=1) <= (
        merge_radius * merge_radius + 1e-12)
    validation_indices = lattice[inside]
    if len(validation_indices) == 0:
        return None
    if deadline is not None and monotonic() >= deadline:
        return None
    occupied_indices = np.rint(
        (np.asarray(centers) - workspace_min) / params.voxel_size_m - 0.5
    ).astype(np.int64)
    occupied = {
        tuple(int(value) for value in index) for index in occupied_indices
    }
    occupied_count = sum(
        tuple(int(value) for value in index) in occupied
        for index in validation_indices
    )
    return 1.0 - occupied_count / len(validation_indices)


def agglomerative_merge_dynamic_spheres(
    centers: np.ndarray,
    spheres: Sequence[DynamicSphere],
    params: DynamicSphereParameters,
    deadline: float,
) -> list[DynamicSphere]:
    """Merge same-component pairs conservatively within the shared deadline."""

    active = list(spheres)
    while len(active) >= 2 and monotonic() < deadline:
        baseline_covered = _covered_mask(
            centers, active, params.coverage_tolerance_m)

        def validator(candidate: PairMergeCandidate) -> bool:
            if monotonic() >= deadline:
                return False
            first = active[candidate.first_index]
            merged = DynamicSphere(
                float(candidate.center[0]),
                float(candidate.center[1]),
                float(candidate.center[2]),
                candidate.radius,
                candidate.radius + params.safety_margin_m,
                first.component_id,
            )
            trial = [
                sphere for index, sphere in enumerate(active)
                if index not in (candidate.first_index, candidate.second_index)
            ] + [merged]
            covered = _covered_mask(
                centers, trial, params.coverage_tolerance_m)
            if np.count_nonzero(covered) < np.count_nonzero(baseline_covered):
                return False
            if params.dynamic_merge_enable_empty_space_guard:
                empty_fraction = dynamic_merge_empty_fraction(
                    centers,
                    candidate.center,
                    candidate.radius,
                    params,
                    deadline,
                )
                if (
                    empty_fraction is None
                    or empty_fraction
                    > params.dynamic_merge_max_empty_fraction + 1e-12
                ):
                    return False
            return True

        candidate = select_best_merge_pair(
            np.asarray([sphere.center for sphere in active]),
            np.asarray([sphere.raw_radius for sphere in active]),
            np.asarray([sphere.component_id for sphere in active]),
            params.dynamic_merge_max_radius_m,
            params.dynamic_merge_max_radius_growth_ratio,
            params.dynamic_merge_max_gap_m,
            validator,
            deadline,
        )
        if candidate is None or monotonic() >= deadline:
            break
        first = active[candidate.first_index]
        merged = DynamicSphere(
            float(candidate.center[0]),
            float(candidate.center[1]),
            float(candidate.center[2]),
            candidate.radius,
            candidate.radius + params.safety_margin_m,
            first.component_id,
        )
        active = [
            sphere for index, sphere in enumerate(active)
            if index not in (candidate.first_index, candidate.second_index)
        ] + [merged]
        active.sort(key=lambda sphere: (
            sphere.component_id,
            sphere.x,
            sphere.y,
            sphere.z,
            sphere.raw_radius,
        ))
    return active


def _component_spheres(
    indices: np.ndarray, params: DynamicSphereParameters, component_id: int,
    deadline: float,
) -> DynamicComponentResult:
    centers = voxel_centers(indices, params)
    local_extent = np.ptp(indices, axis=0) + 1 if len(indices) else np.ones(3, dtype=int)
    local_volume = int(np.prod(local_extent, dtype=np.int64))
    oversized = len(indices) > params.max_component_voxels or (
        local_volume > params.max_local_grid_voxels)
    budget_exceeded = monotonic() >= deadline
    if oversized or budget_exceeded:
        reason = "oversized_component_fallback" if oversized else "processing_budget_fallback"
        if params.enable_fixed_radius_fallback:
            spheres, uncovered = _fixed_radius_cover(
                centers, params, component_id, params.max_spheres_per_component)
        else:
            spheres, uncovered = [], np.ones(len(centers), dtype=bool)
    else:
        depths = _boundary_depths(indices, params.connectivity)
        radii = np.clip(
            depths * params.voxel_size_m,
            params.min_raw_radius_m, params.max_raw_radius_m)
        order = sorted(
            range(len(centers)),
            key=lambda i: (-float(radii[i]),) + tuple(float(v) for v in centers[i]))
        spheres = []
        uncovered = np.ones(len(centers), dtype=bool)
        reason = "target_coverage"
        iterations = 0
        for idx in order:
            if monotonic() >= deadline:
                reason = "processing_budget"
                break
            if iterations >= params.max_iterations_per_component:
                reason = "iteration_limit"
                break
            if len(spheres) >= params.max_spheres_per_component:
                reason = "sphere_limit"
                break
            iterations += 1
            if not uncovered[idx]:
                continue
            radius = float(radii[idx])
            if radius < params.min_useful_adaptive_radius_m:
                continue
            center = centers[idx]
            if spheres and min(np.linalg.norm(center - s.center) for s in spheres) < (
                    params.minimum_center_spacing_m):
                continue
            spheres.append(DynamicSphere(
                float(center[0]), float(center[1]), float(center[2]), radius,
                radius + params.safety_margin_m, component_id))
            uncovered = ~_covered_mask(centers, spheres, params.coverage_tolerance_m)
            if float(np.mean(~uncovered)) + 1e-12 >= params.target_coverage:
                break
        if (not spheres or float(np.mean(~uncovered)) < params.target_coverage) and (
                params.enable_fixed_radius_fallback):
            remaining_limit = max(0, params.max_spheres_per_component - len(spheres))
            fixed, _ = _fixed_radius_cover(
                centers[uncovered], params, component_id, remaining_limit)
            spheres.extend(fixed)
            reason = "thin_component_fallback" if not oversized else reason
            uncovered = ~_covered_mask(centers, spheres, params.coverage_tolerance_m)
    if not budget_exceeded:
        spheres = _optimize_component_spheres(
            centers, spheres, params, component_id, deadline)
    pre_merge_sphere_count = len(spheres)
    if params.dynamic_enable_agglomerative_merge and len(spheres) >= 2:
        spheres = agglomerative_merge_dynamic_spheres(
            centers, spheres, params, deadline)
    uncovered = ~_covered_mask(centers, spheres, params.coverage_tolerance_m)
    coverage = float(np.mean(~uncovered)) if len(uncovered) else 1.0
    confidence = min(1.0, coverage * min(1.0, len(centers) / max(1, params.min_component_voxels * 2)))
    spheres = [replace(s, component_coverage=coverage, confidence=confidence) for s in spheres]
    return DynamicComponentResult(
        component_id, centers, spheres, coverage, centers[uncovered], reason,
        pre_merge_sphere_count)


def generate_dynamic_spheres(
    points: np.ndarray | Sequence[Sequence[float]], params: DynamicSphereParameters,
) -> DynamicSphereResult:
    """Generate bounded, deterministic obstacle spheres from dynamic XYZ points."""
    start = monotonic()
    indices = filter_and_voxelize(points, params)
    if len(indices) == 0:
        return DynamicSphereResult(elapsed_ms=(monotonic() - start) * 1000.0)
    raw_components = connected_components(indices, params.connectivity)
    raw_components = [c for c in raw_components if len(c) >= params.min_component_voxels]
    raw_components = raw_components[:params.max_components]
    if not raw_components:
        return DynamicSphereResult(
            elapsed_ms=(monotonic() - start) * 1000.0, termination_reason="noise_removed")
    budget_sec = max(0.0, params.processing_budget_ms) / 1000.0
    deadline = start + budget_sec
    component_results = []
    all_spheres: list[DynamicSphere] = []
    all_voxels = []
    all_uncovered = []
    for component_id, component in enumerate(raw_components):
        morphed = apply_morphology(component, params)
        result = _component_spheres(morphed, params, component_id, deadline)
        remaining = params.max_total_spheres - len(all_spheres)
        result.spheres = result.spheres[:max(0, remaining)]
        result.uncovered_voxels = result.voxel_centers[
            ~_covered_mask(result.voxel_centers, result.spheres, params.coverage_tolerance_m)]
        result.coverage = 1.0 - len(result.uncovered_voxels) / max(1, len(result.voxel_centers))
        result.spheres = [replace(s, component_coverage=result.coverage) for s in result.spheres]
        component_results.append(result)
        all_spheres.extend(result.spheres)
        all_voxels.append(result.voxel_centers)
        all_uncovered.append(result.uncovered_voxels)
        if len(all_spheres) >= params.max_total_spheres:
            break
    elapsed = (monotonic() - start) * 1000.0
    return DynamicSphereResult(
        components=component_results,
        spheres=all_spheres,
        voxel_centers=np.vstack(all_voxels) if all_voxels else np.empty((0, 3)),
        uncovered_voxels=np.vstack(all_uncovered) if all_uncovered else np.empty((0, 3)),
        elapsed_ms=elapsed,
        termination_reason="complete" if component_results else "noise_removed")


@dataclass
class _Track:
    track_id: int
    sphere: DynamicSphere
    first_seen: float
    last_seen: float
    missed_updates: int = 0


class DynamicSphereTracker:
    """Greedy one-to-one center-distance tracker with bounded occlusion hold."""

    def __init__(
        self, association_distance_m: float, smoothing_alpha: float,
        ttl_sec: float, max_missed_updates: int,
    ) -> None:
        self.association_distance_m = float(association_distance_m)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.ttl_sec = float(ttl_sec)
        self.max_missed_updates = int(max_missed_updates)
        self._next_id = 0
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: Iterable[DynamicSphere], timestamp_sec: float) -> list[DynamicSphere]:
        detections = list(detections)
        now = float(timestamp_sec)
        self._expire(now)
        pairs = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                distance = float(np.linalg.norm(track.sphere.center - detection.center))
                if distance <= self.association_distance_m:
                    pairs.append((distance, track_id, detection_index))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_index in sorted(pairs):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self._tracks[track_id]
            detection = detections[detection_index]
            alpha = self.smoothing_alpha
            center = alpha * detection.center + (1.0 - alpha) * track.sphere.center
            raw = alpha * detection.raw_radius + (1.0 - alpha) * track.sphere.raw_radius
            output = alpha * detection.output_radius + (1.0 - alpha) * track.sphere.output_radius
            age = track.sphere.age + 1
            track.sphere = replace(
                detection, x=float(center[0]), y=float(center[1]), z=float(center[2]),
                raw_radius=float(raw), output_radius=float(output), track_id=track_id,
                age=age, confidence=min(1.0, detection.confidence + 0.05 * age))
            track.last_seen = now
            track.missed_updates = 0
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            sphere = replace(detection, track_id=track_id, age=1)
            self._tracks[track_id] = _Track(track_id, sphere, now, now)
            matched_tracks.add(track_id)
        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track.missed_updates += 1
        self._expire(now)
        return [self._tracks[key].sphere for key in sorted(self._tracks)]

    def _expire(self, now: float) -> None:
        expired = [
            track_id for track_id, track in self._tracks.items()
            if now - track.last_seen > self.ttl_sec or
            track.missed_updates > self.max_missed_updates
        ]
        for track_id in expired:
            del self._tracks[track_id]
