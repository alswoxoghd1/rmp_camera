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
from rmp_camera.component_tracking import ComponentAssociator
from rmp_camera.local_width_sphere_cover import (
    local_width_cover, VoxelSupportGuard, visible_width_radius_cap,
)

from rmp_camera.sphere_merge_core import (
    PairMergeCandidate,
    enumerate_pair_merge_candidates,
    select_best_merge_pair,
    sphere_overlap_fraction,
    sphere_set_overlap_metrics,
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
    component_radius_scale: float = 0.0
    component_fixed_radius_scale: float = 0.0
    local_width_cover_enabled: bool = False
    local_width_ratio: float = 1.4
    local_width_budget_ms: float = 5.0
    local_width_integrated: bool = False
    observation_guard_mode: str = 'off'
    target_coverage: float = 0.95
    coverage_tolerance_m: float = 0.02
    safety_margin_m: float = 0.015
    redundancy_tolerance_m: float = 0.01
    max_spheres_per_component: int = 96
    max_iterations_per_component: int = 256
    max_total_spheres: int = 384
    processing_budget_ms: float = 35.0
    # Optional shared foreground merge/search budget. Seed/set-cover creation
    # retains its original budget; shortening that could force tiny fallback
    # spheres and make downstream tracking slower instead of faster.
    refinement_budget_ms: float | None = None
    max_local_grid_voxels: int = 1200000
    dynamic_enable_agglomerative_merge: bool = True
    dynamic_merge_max_radius_m: float = 0.30
    dynamic_merge_max_radius_growth_ratio: float = 1.50
    dynamic_merge_max_gap_m: float = 0.08
    dynamic_merge_enable_empty_space_guard: bool = False
    dynamic_merge_max_empty_fraction: float = 0.70
    dynamic_merge_max_validation_voxels: int = 50000
    dynamic_enable_min_k_search: bool = True
    dynamic_min_k_beam_width: int = 8
    dynamic_min_k_max_states: int = 128
    dynamic_max_allowed_overlap_fraction: float = 0.20
    dynamic_min_k_use_output_overlap: bool = True
    dynamic_min_k_enable_overlap_constraint: bool = True
    dynamic_min_k_prefer_lower_overlap: bool = True
    min_x_m: float = -3.0
    max_x_m: float = 3.0
    min_y_m: float = -3.0
    max_y_m: float = 3.0
    min_z_m: float = -0.2
    max_z_m: float = 2.5
    max_range_from_base_m: float = 5.0

    def validate(self) -> None:
        if self.observation_guard_mode not in ('off','audit','observed'):
            raise ValueError('invalid sphere observation guard mode')
        if (not np.isfinite(self.local_width_ratio) or self.local_width_ratio < 1.
                or not np.isfinite(self.local_width_budget_ms) or self.local_width_budget_ms < 0.):
            raise ValueError("invalid local-width ratio/budget")
        if self.refinement_budget_ms is not None and (
                not np.isfinite(self.refinement_budget_ms) or self.refinement_budget_ms < 0.):
            raise ValueError("refinement_budget_ms must be finite and non-negative")
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
        if (
            not np.isfinite(self.component_radius_scale)
            or self.component_radius_scale < 0.0
        ):
            raise ValueError("component_radius_scale must be finite and non-negative")
        if (
            not np.isfinite(self.component_fixed_radius_scale)
            or self.component_fixed_radius_scale < 0.0
        ):
            raise ValueError(
                "component_fixed_radius_scale must be finite and non-negative")
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
        if self.dynamic_min_k_beam_width <= 0:
            raise ValueError("dynamic min-k beam width must be positive")
        if self.dynamic_min_k_max_states <= 0:
            raise ValueError("dynamic min-k state cap must be positive")
        if not np.isfinite(self.dynamic_max_allowed_overlap_fraction):
            raise ValueError("dynamic overlap limit must be finite")
        if not 0.0 <= self.dynamic_max_allowed_overlap_fraction <= 1.0:
            raise ValueError("dynamic overlap limit must be in [0, 1]")
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
    # Internal provenance only; existing ROS cloud/marker schema is unchanged.
    local_width: bool = False

    @property
    def center(self) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=np.float64)


@dataclass(frozen=True)
class DynamicSphereSearchState:
    """One deterministic state in the bounded dynamic merge-state search."""

    spheres: tuple[DynamicSphere, ...]
    max_raw_overlap_fraction: float
    max_output_overlap_fraction: float
    total_raw_overlap: float
    total_output_overlap: float
    total_raw_volume: float
    total_raw_radius: float
    canonical_signature: tuple[tuple[float, ...], ...]


@dataclass
class DynamicMinimumKSearchResult:
    """Best state found within the configured bounded search budget."""

    spheres: list[DynamicSphere]
    states_explored: int
    termination_reason: str
    overlap_constraint_satisfied: bool
    state: DynamicSphereSearchState


@dataclass
class DynamicComponentResult:
    component_id: int
    voxel_centers: np.ndarray
    spheres: list[DynamicSphere]
    coverage: float
    uncovered_voxels: np.ndarray
    termination_reason: str
    pre_merge_sphere_count: int = 0
    agglomerative_sphere_count: int = 0
    min_k_sphere_count: int = 0
    final_sphere_count: int = 0
    max_raw_overlap_fraction: float = 0.0
    max_output_overlap_fraction: float = 0.0
    total_raw_overlap: float = 0.0
    total_output_overlap: float = 0.0
    min_k_search_applied: bool = False
    min_k_states_explored: int = 0
    min_k_search_termination: str = "disabled"
    overlap_constraint_satisfied: bool = True
    refinement_elapsed_ms: float = 0.0
    refinement_budget_exhausted: bool = False
    local_width_applied: bool = False
    local_width_reason: str = "disabled"
    local_width_elapsed_ms: float = 0.0
    local_width_candidates: int = 0
    local_width_regions: int = 0
    local_width_min_region_coverage: float = 1.0
    local_width_saved_spheres: int = 0


@dataclass
class DynamicSphereResult:
    components: list[DynamicComponentResult] = field(default_factory=list)
    spheres: list[DynamicSphere] = field(default_factory=list)
    voxel_centers: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    uncovered_voxels: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    robot_rejected_voxels: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    input_component_count: int = 0
    robot_rejected_component_count: int = 0
    robot_rejected_voxel_count: int = 0
    robot_rejected_component_ids: list[int] = field(default_factory=list)
    robot_component_overlap_fractions: list[float] = field(default_factory=list)
    robot_rejected_component_bounds: list[tuple[np.ndarray, np.ndarray]] = field(
        default_factory=list)
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


def voxel_robot_sphere_overlap_mask(
    indices: np.ndarray,
    params: DynamicSphereParameters,
    robot_sphere_centers: np.ndarray | Sequence[Sequence[float]],
    robot_sphere_radii: np.ndarray | Sequence[float],
    margin_m: float = 0.0,
) -> np.ndarray:
    """Return which voxel AABBs intersect at least one robot sphere."""
    voxel_indices = np.asarray(indices, dtype=np.int64)
    if voxel_indices.size == 0:
        return np.zeros(0, dtype=bool)
    if voxel_indices.ndim != 2 or voxel_indices.shape[1] != 3:
        raise ValueError("indices must have shape (N, 3)")

    centers = _as_points(robot_sphere_centers)
    radii = np.asarray(robot_sphere_radii, dtype=np.float64).reshape(-1)
    if len(centers) != len(radii):
        raise ValueError("robot sphere centers and radii must have equal length")
    if margin_m < 0.0 or not np.isfinite(margin_m):
        raise ValueError("robot sphere margin must be finite and non-negative")
    if len(centers) == 0:
        return np.zeros(len(voxel_indices), dtype=bool)
    if not np.isfinite(centers).all() or not np.isfinite(radii).all():
        raise ValueError("robot sphere geometry must be finite")
    if np.any(radii <= 0.0):
        raise ValueError("robot sphere radii must be positive")

    origin = np.asarray(
        (params.min_x_m, params.min_y_m, params.min_z_m), dtype=np.float64)
    voxel_min = origin + voxel_indices.astype(np.float64) * params.voxel_size_m
    voxel_max = voxel_min + params.voxel_size_m
    overlaps = np.zeros(len(voxel_indices), dtype=bool)
    for center, radius in zip(centers, radii):
        closest = np.clip(center, voxel_min, voxel_max)
        delta = closest - center
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        overlaps |= squared_distance <= (radius + margin_m) ** 2
        if overlaps.all():
            break
    return overlaps


def component_robot_overlap_fraction(
    indices: np.ndarray,
    params: DynamicSphereParameters,
    robot_sphere_centers: np.ndarray | Sequence[Sequence[float]],
    robot_sphere_radii: np.ndarray | Sequence[float],
    margin_m: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Return robot-intersecting voxel fraction and the per-voxel mask."""
    mask = voxel_robot_sphere_overlap_mask(
        indices,
        params,
        robot_sphere_centers,
        robot_sphere_radii,
        margin_m,
    )
    fraction = float(np.mean(mask)) if len(mask) else 0.0
    return fraction, mask


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


def _covered_mask(
    points: np.ndarray,
    spheres: Sequence[DynamicSphere],
    tolerance: float,
) -> np.ndarray:
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


def component_adaptive_radius_cap(
    centers: np.ndarray,
    params: DynamicSphereParameters,
    absolute_max_radius_m: float,
) -> float:
    """Scale the single/merge radius cap with component voxel count.

    A zero scale preserves the legacy global cap.  For a positive scale, the
    characteristic component length is ``voxel_size * cbrt(voxel_count)``.
    This keeps small hand-sized components from using a torso-sized enclosing
    sphere while still allowing dense, large components to approach the global
    radius limit.  The fixed fallback radius is retained as the lower bound so
    thin components never lose the coverage supplied by their seed spheres.
    """

    cap = min(float(absolute_max_radius_m), params.max_raw_radius_m)
    if params.component_radius_scale <= 0.0 or len(centers) == 0:
        return cap
    component_cap = (
        params.component_radius_scale
        * params.voxel_size_m
        * float(np.cbrt(len(centers)))
    )
    lower_bound = params.min_raw_radius_m
    if params.enable_fixed_radius_fallback:
        lower_bound = max(lower_bound, params.fixed_radius_m)
    return min(cap, max(lower_bound, component_cap))


def component_adaptive_fixed_radius(
    centers: np.ndarray,
    params: DynamicSphereParameters,
) -> float:
    """Return a size-aware radius for thin-component fallback spheres.

    Dynamic point clouds are usually thin surface shells, so their boundary
    depth often cannot produce a useful adaptive radius.  A single global
    fallback radius then makes wide objects use the same small spheres as a
    hand.  A positive scale grows the fallback with the characteristic
    component length while retaining ``fixed_radius_m`` as the floor and the
    existing component-aware/global radius limits as the ceiling.  Zero keeps
    the legacy fixed-radius behavior.
    """

    base_radius = min(
        params.max_raw_radius_m,
        max(params.min_raw_radius_m, params.fixed_radius_m),
    )
    if params.component_fixed_radius_scale <= 0.0 or len(centers) == 0:
        return base_radius
    scaled_radius = (
        params.component_fixed_radius_scale
        * params.voxel_size_m
        * float(np.cbrt(len(centers)))
    )
    component_cap = component_adaptive_radius_cap(
        centers, params, params.max_raw_radius_m)
    return min(component_cap, max(base_radius, scaled_radius))


class DynamicCandidateValidator:
    """One current-geometry policy for local proposal and cached reuse."""
    def __init__(self, centers, params, evidence=None):
        self.centers, self.params = centers, params
        self.evidence = evidence
        self.origin = np.asarray((params.min_x_m, params.min_y_m, params.min_z_m))
        self.maximum = np.asarray((params.max_x_m, params.max_y_m, params.max_z_m))
        self.cap = min(params.max_raw_radius_m, params.dynamic_merge_max_radius_m)
        self.legacy_cap = component_adaptive_radius_cap(centers, params, self.cap)
        self.density = (VoxelSupportGuard(centers, self.origin, params.voxel_size_m,
            params.dynamic_merge_max_empty_fraction, params.dynamic_merge_max_validation_voxels)
            if params.dynamic_merge_enable_empty_space_guard and len(centers) else None)

    def __call__(self, center, radius, deadline, *, local_width=True):
        p = self.params
        if (monotonic() >= deadline or not np.isfinite(center).all() or not np.isfinite(radius)
                or np.any(center < self.origin) or np.any(center > self.maximum)
                or not p.min_raw_radius_m <= radius <= self.cap + 1e-12
                or (p.max_range_from_base_m > 0 and np.linalg.norm(center[:2]) > p.max_range_from_base_m)
                or not len(self.centers)):
            return False
        cap = (visible_width_radius_cap(self.centers, center, radius,
                voxel_size=p.voxel_size_m, min_radius=p.min_raw_radius_m,
                max_radius=self.cap, tolerance=p.coverage_tolerance_m,
                width_ratio=p.local_width_ratio) if local_width else self.legacy_cap)
        if radius > cap + 1e-10 or not np.any(_covered_mask(self.centers,
                [DynamicSphere(*center, radius, radius)], p.coverage_tolerance_m)):
            return False
        if self.density is None:
            return True
        evaluation = self.density.evaluate(center,radius,deadline)
        if evaluation is None:
            return False
        samples,empty = evaluation
        density_ok=empty<=self.density.max_empty+1e-12
        if self.evidence is not None and p.observation_guard_mode!='off':
            ray_ok=self.evidence.validate(samples,self.density,deadline)
            if ray_ok is not None and ray_ok!=density_ok:
                key='ray_alternative_accept' if ray_ok else 'ray_alternative_reject'
                self.evidence.stats[key]=self.evidence.stats.get(key,0)+1
            if p.observation_guard_mode=='observed' and ray_ok is not None:
                return ray_ok and monotonic()<deadline
        return density_ok


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
    radius_cap = component_adaptive_radius_cap(
        centers, params, params.single_sphere_max_radius_m)
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
    limit: int, radius_m: float | None = None,
) -> tuple[list[DynamicSphere], np.ndarray]:
    spheres: list[DynamicSphere] = []
    uncovered = np.ones(len(centers), dtype=bool)
    radius = (
        component_adaptive_fixed_radius(centers, params)
        if radius_m is None
        else min(
            params.max_raw_radius_m,
            max(params.min_raw_radius_m, float(radius_m)),
        )
    )
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
        coverage = 1.0 - float(np.count_nonzero(uncovered)) / max(
            1, len(centers))
        if coverage >= params.target_coverage:
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
    *, return_masks: bool = False,
) -> list[DynamicSphere]:
    """Minimize candidates using one-sphere replacement, set cover, then deletion."""
    original = list(spheres)
    def finish(selected, candidates=None, masks=None):
        if not return_masks:
            return selected
        if masks is None:
            return selected, None
        rows = {id(s): i for i, s in enumerate(candidates)}
        return selected, masks[[rows[id(s)] for s in selected]]
    if len(original) < 2 or len(centers) == 0 or monotonic() >= deadline:
        return finish(original)

    single_candidates = []
    if params.enable_single_sphere_replacement:
        single_candidates = _single_sphere_candidates(
            centers, original, params, component_id)
    candidates = original + single_candidates
    coverage_masks, redundancy_masks = _sphere_coverage_matrices(
        centers, candidates, params.coverage_tolerance_m,
        params.redundancy_tolerance_m, deadline)
    if coverage_masks is None or redundancy_masks is None:
        return finish(original)

    if single_candidates:
        best_single = _best_single_sphere(
            single_candidates, coverage_masks[len(original):],
            len(original), params)
        if best_single is not None:
            return finish([best_single], candidates, coverage_masks)

    selected_indices = list(range(len(original)))
    if params.enable_greedy_set_cover:
        greedy_indices = _greedy_set_cover(
            candidates, coverage_masks, params.target_coverage, deadline)
        # Equal/larger alternatives do not further the minimization goal and
        # would unnecessarily perturb the legacy centers and radii.
        if greedy_indices is not None and len(greedy_indices) < len(original):
            selected_indices = greedy_indices

    selected_spheres = [candidates[index] for index in selected_indices]
    selected = _remove_redundant(
        centers, selected_spheres, params,
        coverage_masks[selected_indices], redundancy_masks[selected_indices],
        deadline)
    return finish(selected, candidates, coverage_masks)


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


def _dynamic_sphere_sort_key(
    sphere: DynamicSphere,
) -> tuple[int, float, float, float, float]:
    return (
        sphere.component_id,
        sphere.x,
        sphere.y,
        sphere.z,
        sphere.raw_radius,
    )


def build_dynamic_sphere_search_state(
    spheres: Sequence[DynamicSphere],
) -> DynamicSphereSearchState:
    """Build an order-independent state with raw and output overlap metrics."""

    ordered = tuple(sorted(spheres, key=_dynamic_sphere_sort_key))
    if ordered:
        centers = np.asarray([sphere.center for sphere in ordered])
        component_ids = np.asarray(
            [sphere.component_id for sphere in ordered], dtype=np.int64)
    else:
        centers = np.empty((0, 3), dtype=np.float64)
        component_ids = np.empty(0, dtype=np.int64)
    raw_radii = np.asarray(
        [sphere.raw_radius for sphere in ordered], dtype=np.float64)
    output_radii = np.asarray(
        [sphere.output_radius for sphere in ordered], dtype=np.float64)
    raw_metrics = sphere_set_overlap_metrics(
        centers, raw_radii, component_ids)
    output_metrics = sphere_set_overlap_metrics(
        centers, output_radii, component_ids)
    signature = tuple(
        (
            float(sphere.component_id),
            round(float(sphere.x), 9),
            round(float(sphere.y), 9),
            round(float(sphere.z), 9),
            round(float(sphere.raw_radius), 9),
        )
        for sphere in ordered
    )
    return DynamicSphereSearchState(
        spheres=ordered,
        max_raw_overlap_fraction=raw_metrics.max_overlap_fraction,
        max_output_overlap_fraction=output_metrics.max_overlap_fraction,
        total_raw_overlap=raw_metrics.total_overlap_fraction,
        total_output_overlap=output_metrics.total_overlap_fraction,
        total_raw_volume=float(
            (4.0 / 3.0) * np.pi * np.sum(raw_radii ** 3)),
        total_raw_radius=float(np.sum(raw_radii)),
        canonical_signature=signature,
    )


def _state_selected_overlap(
    state: DynamicSphereSearchState,
    params: DynamicSphereParameters,
) -> tuple[float, float]:
    if params.dynamic_min_k_use_output_overlap:
        return (
            state.max_output_overlap_fraction,
            state.total_output_overlap,
        )
    return state.max_raw_overlap_fraction, state.total_raw_overlap


def dynamic_sphere_search_state_rank_key(
    state: DynamicSphereSearchState,
    params: DynamicSphereParameters,
) -> tuple[object, ...]:
    """Rank equal-K states using the configured overlap representation."""

    if params.dynamic_min_k_use_output_overlap:
        overlap_key = (
            state.max_output_overlap_fraction,
            state.total_output_overlap,
            state.max_raw_overlap_fraction,
            state.total_raw_overlap,
        )
    else:
        overlap_key = (
            state.max_raw_overlap_fraction,
            state.total_raw_overlap,
            state.max_output_overlap_fraction,
            state.total_output_overlap,
        )
    if not params.dynamic_min_k_prefer_lower_overlap:
        overlap_key = ()
    return (
        *overlap_key,
        state.total_raw_volume,
        state.total_raw_radius,
        state.canonical_signature,
    )


def _state_overlap_constraint_satisfied(
    state: DynamicSphereSearchState,
    params: DynamicSphereParameters,
) -> bool:
    if not params.dynamic_min_k_enable_overlap_constraint:
        return True
    maximum, _ = _state_selected_overlap(state, params)
    return maximum <= params.dynamic_max_allowed_overlap_fraction + 1e-12


def _state_fallback_key(
    state: DynamicSphereSearchState,
    params: DynamicSphereParameters,
) -> tuple[object, ...]:
    maximum, total = _state_selected_overlap(state, params)
    violation = max(
        0.0, maximum - params.dynamic_max_allowed_overlap_fraction)
    return (
        violation,
        len(state.spheres),
        total,
        state.total_raw_volume,
        state.total_raw_radius,
        state.canonical_signature,
    )


def _merge_dynamic_candidate(
    spheres: Sequence[DynamicSphere],
    candidate: PairMergeCandidate,
    params: DynamicSphereParameters,
) -> list[DynamicSphere]:
    first = spheres[candidate.first_index]
    merged = DynamicSphere(
        float(candidate.center[0]),
        float(candidate.center[1]),
        float(candidate.center[2]),
        candidate.radius,
        candidate.radius + params.safety_margin_m,
        first.component_id,
        local_width=first.local_width or spheres[candidate.second_index].local_width,
    )
    active = [
        sphere for index, sphere in enumerate(spheres)
        if index not in (candidate.first_index, candidate.second_index)
    ] + [merged]
    active.sort(key=_dynamic_sphere_sort_key)
    return active


def _enumerate_valid_dynamic_merge_candidates(
    centers: np.ndarray,
    state: DynamicSphereSearchState,
    params: DynamicSphereParameters,
    deadline: float,
) -> list[PairMergeCandidate]:
    active = list(state.spheres)
    local_validator = DynamicCandidateValidator(centers, params) if any(s.local_width for s in active) else None
    legacy_cap = component_adaptive_radius_cap(centers, params, params.dynamic_merge_max_radius_m)
    baseline_covered = _covered_mask(
        centers, active, params.coverage_tolerance_m)
    baseline_count = int(np.count_nonzero(baseline_covered))

    def validator(candidate: PairMergeCandidate) -> bool:
        if monotonic() >= deadline:
            return False
        is_local = active[candidate.first_index].local_width or active[candidate.second_index].local_width
        if not is_local and candidate.radius > legacy_cap + 1e-12:
            return False
        if is_local and not local_validator(candidate.center, candidate.radius, deadline):
            return False
        trial = _merge_dynamic_candidate(active, candidate, params)
        covered = _covered_mask(
            centers, trial, params.coverage_tolerance_m)
        coverage = float(np.mean(covered)) if len(covered) else 1.0
        if (
            coverage + 1e-12 < params.target_coverage
            or np.count_nonzero(covered) < baseline_count
            or (is_local and not np.all(covered[baseline_covered]))
        ):
            return False
        if params.dynamic_merge_enable_empty_space_guard and not is_local:
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

    merge_radius_cap = local_validator.cap if local_validator is not None else legacy_cap
    return enumerate_pair_merge_candidates(
        np.asarray([sphere.center for sphere in active]),
        np.asarray([sphere.raw_radius for sphere in active]),
        np.asarray([sphere.component_id for sphere in active]),
        merge_radius_cap,
        params.dynamic_merge_max_radius_growth_ratio,
        params.dynamic_merge_max_gap_m,
        validator,
        deadline,
    )


def minimum_k_dynamic_sphere_search(
    centers: np.ndarray,
    pre_merge_spheres: Sequence[DynamicSphere],
    params: DynamicSphereParameters,
    deadline: float,
) -> DynamicMinimumKSearchResult:
    """Run a bounded deterministic minimum-K merge-state search.

    The result is the minimum K found within explored merge states, not a
    claim of a global mathematical optimum.
    """

    initial = build_dynamic_sphere_search_state(pre_merge_spheres)
    initial_satisfied = _state_overlap_constraint_satisfied(initial, params)
    if len(initial.spheres) < 2:
        return DynamicMinimumKSearchResult(
            list(initial.spheres),
            0,
            "not_needed",
            initial_satisfied,
            initial,
        )

    best_feasible = initial if initial_satisfied else None
    best_fallback = initial
    seen = {initial.canonical_signature}
    beam = [initial]
    states_explored = 0
    termination = "no_more_valid_merges"

    while beam:
        if monotonic() >= deadline:
            termination = "deadline"
            break

        child_states: dict[
            tuple[tuple[float, ...], ...],
            DynamicSphereSearchState,
        ] = {}
        stop_reason = None
        for parent in beam:
            candidates = _enumerate_valid_dynamic_merge_candidates(
                centers, parent, params, deadline)
            if monotonic() >= deadline:
                stop_reason = "deadline"
                break
            for candidate in candidates:
                if monotonic() >= deadline:
                    stop_reason = "deadline"
                    break
                child_spheres = _merge_dynamic_candidate(
                    parent.spheres, candidate, params)
                child = build_dynamic_sphere_search_state(child_spheres)
                signature = child.canonical_signature
                if signature in seen:
                    continue
                if states_explored >= params.dynamic_min_k_max_states:
                    stop_reason = "max_states"
                    break
                seen.add(signature)
                child_states[signature] = child
                states_explored += 1

                if _state_fallback_key(
                    child, params
                ) < _state_fallback_key(best_fallback, params):
                    best_fallback = child
                if _state_overlap_constraint_satisfied(child, params):
                    if (
                        best_feasible is None
                        or (
                            len(child.spheres),
                            dynamic_sphere_search_state_rank_key(child, params),
                        ) < (
                            len(best_feasible.spheres),
                            dynamic_sphere_search_state_rank_key(
                                best_feasible, params),
                        )
                    ):
                        best_feasible = child
            if stop_reason is not None:
                break

        if stop_reason is not None:
            termination = stop_reason
            break
        if not child_states:
            termination = "no_more_valid_merges"
            break

        beam = sorted(
            child_states.values(),
            key=lambda state: dynamic_sphere_search_state_rank_key(
                state, params),
        )[:params.dynamic_min_k_beam_width]
        if any(len(state.spheres) == 1 for state in beam):
            termination = "minimum_k_reached"
            break

    chosen = best_feasible if best_feasible is not None else best_fallback
    constraint_satisfied = _state_overlap_constraint_satisfied(chosen, params)
    if (
        not constraint_satisfied
        and termination not in ("deadline", "max_states")
    ):
        termination = "overlap_constraint_unmet"
    return DynamicMinimumKSearchResult(
        list(chosen.spheres),
        states_explored,
        termination,
        constraint_satisfied,
        chosen,
    )


def agglomerative_merge_dynamic_spheres(
    centers: np.ndarray,
    spheres: Sequence[DynamicSphere],
    params: DynamicSphereParameters,
    deadline: float,
) -> list[DynamicSphere]:
    """Merge same-component pairs conservatively within the shared deadline."""

    active = list(spheres)
    local_validator = DynamicCandidateValidator(centers, params) if any(s.local_width for s in active) else None
    legacy_cap = component_adaptive_radius_cap(
        centers, params, params.dynamic_merge_max_radius_m)
    merge_radius_cap = local_validator.cap if local_validator is not None else legacy_cap
    while len(active) >= 2 and monotonic() < deadline:
        baseline_covered = _covered_mask(
            centers, active, params.coverage_tolerance_m)

        def validator(candidate: PairMergeCandidate) -> bool:
            if monotonic() >= deadline:
                return False
            first = active[candidate.first_index]
            is_local = first.local_width or active[candidate.second_index].local_width
            if not is_local and candidate.radius > legacy_cap + 1e-12:
                return False
            if is_local and not local_validator(candidate.center, candidate.radius, deadline):
                return False
            merged = DynamicSphere(
                float(candidate.center[0]),
                float(candidate.center[1]),
                float(candidate.center[2]),
                candidate.radius,
                candidate.radius + params.safety_margin_m,
                first.component_id,
                local_width=is_local,
            )
            trial = [
                sphere for index, sphere in enumerate(active)
                if index not in (candidate.first_index, candidate.second_index)
            ] + [merged]
            covered = _covered_mask(
                centers, trial, params.coverage_tolerance_m)
            if (np.count_nonzero(covered) < np.count_nonzero(baseline_covered)
                    or (is_local and not np.all(covered[baseline_covered]))):
                return False
            if params.dynamic_merge_enable_empty_space_guard and not is_local:
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
            merge_radius_cap,
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
            local_width=first.local_width or active[candidate.second_index].local_width,
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
    deadline: float, *, local_width_cache=None, observation_evidence=None,
) -> DynamicComponentResult:
    centers = voxel_centers(indices, params)
    component_fixed_radius = component_adaptive_fixed_radius(centers, params)
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
        near_existing_center = np.zeros(len(centers), dtype=bool)
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
            if near_existing_center[idx]:
                continue
            spheres.append(DynamicSphere(
                float(center[0]), float(center[1]), float(center[2]), radius,
                radius + params.safety_margin_m, component_id))
            # Accumulate only the newly added sphere. Rechecking all prior
            # spheres for every seed gave the same union at quadratic cost.
            uncovered &= ~_covered_mask(centers, spheres[-1:], params.coverage_tolerance_m)
            near_existing_center |= np.linalg.norm(centers - center, axis=1) < (
                params.minimum_center_spacing_m)
            if float(np.mean(~uncovered)) + 1e-12 >= params.target_coverage:
                break
        if (not spheres or float(np.mean(~uncovered)) < params.target_coverage) and (
                params.enable_fixed_radius_fallback):
            remaining_limit = max(0, params.max_spheres_per_component - len(spheres))
            fixed, _ = _fixed_radius_cover(
                centers[uncovered], params, component_id, remaining_limit,
                radius_m=component_fixed_radius)
            spheres.extend(fixed)
            reason = "thin_component_fallback" if not oversized else reason
            uncovered = ~_covered_mask(centers, spheres, params.coverage_tolerance_m)
    integrate_local = params.local_width_cover_enabled and params.local_width_integrated
    coverage_masks = None
    if not budget_exceeded:
        optimized = _optimize_component_spheres(
            centers, spheres, params, component_id, deadline,
            **({'return_masks': True} if integrate_local else {}))
        if integrate_local:
            spheres, coverage_masks = optimized
        else:
            spheres = optimized
    local_result = None
    if integrate_local:
        covered = (np.any(coverage_masks, axis=0) if coverage_masks is not None
                   else _covered_mask(centers, spheres, params.coverage_tolerance_m))
        local_result = DynamicComponentResult(component_id, centers, spheres,
            float(np.mean(covered)), centers[~covered], reason)
        _apply_local_width_cover(local_result, params, coverage_masks=coverage_masks,
                                attempt_cache=local_width_cache, observation_evidence=observation_evidence)
        spheres = local_result.spheres
    refinement_deadline = deadline
    if params.refinement_budget_ms is not None:
        refinement_deadline = min(deadline, monotonic() + params.refinement_budget_ms / 1000.)
    result = refine_dynamic_component(
        centers, spheres, params, component_id, refinement_deadline, reason)
    if local_result is not None:
        for name, value in vars(local_result).items():
            if name.startswith('local_width_'):
                setattr(result, name, value)
    elif params.local_width_cover_enabled:
        _apply_local_width_cover(result, params)
    return result


def _apply_local_width_cover(result, params, *, coverage_masks=None, attempt_cache=None, observation_evidence=None):
    """Reconstruct from local voxel widths, not enclosing old inflated balls."""
    centers, spheres = result.voxel_centers, result.spheres
    if not 2 <= len(spheres) <= 64 or len(centers) > 8192 or params.local_width_budget_ms <= 0:
        result.local_width_reason = "not_needed_or_limit"
        return
    radius_cap = min(params.max_raw_radius_m, params.dynamic_merge_max_radius_m)
    if radius_cap < params.min_raw_radius_m:
        result.local_width_reason = "radius_limit"
        return
    start = monotonic()
    origin = np.asarray((params.min_x_m, params.min_y_m, params.min_z_m))
    descriptor = None
    if attempt_cache is not None:
        descriptor = attempt_cache.descriptor(centers, params.voxel_size_m, origin, len(spheres))
        if attempt_cache.matches(descriptor, start):
            result.local_width_reason = 'recent_no_gain'
            result.local_width_elapsed_ms = (monotonic()-start)*1000.
            return
    maximum = np.asarray((params.max_x_m, params.max_y_m, params.max_z_m))
    guard = (VoxelSupportGuard(centers, origin, params.voxel_size_m,
                params.dynamic_merge_max_empty_fraction, params.dynamic_merge_max_validation_voxels)
             if params.dynamic_merge_enable_empty_space_guard and not params.local_width_integrated else None)

    def valid(center, radius, deadline):
        # Match the existing center crop; sphere volume may cross the ROI just
        # as legacy boundary spheres do. No new outside-ROI center is allowed.
        return (np.all(center >= origin) and np.all(center <= maximum)
                and (params.max_range_from_base_m <= 0
                     or np.linalg.norm(center[:2]) <= params.max_range_from_base_m)
                and (guard is None or guard(center, radius, deadline)))

    if params.local_width_integrated:
        valid = DynamicCandidateValidator(centers, params, observation_evidence)

    proposal = local_width_cover(centers, [s.center for s in spheres],
        [s.raw_radius for s in spheres], voxel_size=params.voxel_size_m,
        min_radius=params.min_raw_radius_m,
        # Local width replaces the whole-component cbrt(N) heuristic here.
        # Absolute raw/merge limits stay unchanged; narrow local parts still
        # have the stricter width-derived cap inside local_width_cover().
        max_radius=radius_cap,
        tolerance=params.coverage_tolerance_m, target_coverage=params.target_coverage,
        width_ratio=params.local_width_ratio,
        budget_ms=max(0., params.local_width_budget_ms - (monotonic() - start) * 1000.),
        validator=valid, old_coverage_masks=coverage_masks)
    result.local_width_reason = proposal.reason
    result.local_width_candidates = proposal.candidates
    result.local_width_regions = proposal.regions
    result.local_width_min_region_coverage = proposal.min_region_coverage
    if proposal.applied:
        trial = [spheres[index] if index < len(spheres) else DynamicSphere(
            *center, float(radius), float(radius) + params.safety_margin_m,
            result.component_id, local_width=True) for index, center, radius in zip(
                proposal.selected, proposal.centers, proposal.radii)]
        state = build_dynamic_sphere_search_state(trial)
        required = _covered_mask(centers, spheres, params.coverage_tolerance_m)
        covered = _covered_mask(centers, trial, params.coverage_tolerance_m)
        # Don't trade a feasible overlap configuration for an infeasible one.
        # Existing infeasible sets may improve K without increasing total overlap.
        baseline = build_dynamic_sphere_search_state(spheres)
        if not np.all(covered[required]):
            result.local_width_reason = "coverage_rejected"
        elif ((_state_overlap_constraint_satisfied(baseline, params)
             and not _state_overlap_constraint_satisfied(state, params))
                or (_state_selected_overlap(state, params)[1]
                    > _state_selected_overlap(baseline, params)[1] + 1e-12)):
            result.local_width_reason = "overlap_rejected"
        else:
            result.local_width_applied = True
            result.local_width_saved_spheres = len(spheres) - len(trial)
            result.coverage = float(np.mean(covered))
            result.uncovered_voxels = centers[~covered]
            confidence = max((s.confidence for s in spheres), default=0.)
            result.spheres = [replace(s, component_coverage=result.coverage, confidence=confidence) for s in trial]
            result.final_sphere_count = len(trial)
            result.max_raw_overlap_fraction = state.max_raw_overlap_fraction
            result.max_output_overlap_fraction = state.max_output_overlap_fraction
            result.total_raw_overlap = state.total_raw_overlap
            result.total_output_overlap = state.total_output_overlap
            result.overlap_constraint_satisfied = _state_overlap_constraint_satisfied(state, params)
    result.local_width_elapsed_ms = (monotonic() - start) * 1000.
    if attempt_cache is not None:
        attempt_cache.remember(descriptor, result.local_width_reason, monotonic())


def refine_dynamic_component(
    centers: np.ndarray, spheres: Sequence[DynamicSphere],
    params: DynamicSphereParameters, component_id: int, deadline: float,
    reason: str = "refined",
) -> DynamicComponentResult:
    """Refine an existing cover without repeating voxelization or seed creation.

    The synchronous path and the optional background worker use exactly the
    same merge/search selection, limits and coverage checks.
    """
    refinement_start = monotonic()
    pre_merge_spheres = list(spheres)
    pre_merge_sphere_count = len(pre_merge_spheres)
    agglomerative_spheres = list(pre_merge_spheres)
    if (
        params.dynamic_enable_agglomerative_merge
        and len(pre_merge_spheres) >= 2
    ):
        agglomerative_spheres = agglomerative_merge_dynamic_spheres(
            centers, pre_merge_spheres, params, deadline)
    spheres = agglomerative_spheres
    agglomerative_sphere_count = len(agglomerative_spheres)
    min_k_search_applied = (
        params.dynamic_enable_min_k_search
        and len(pre_merge_spheres) >= 2
    )
    min_k_states_explored = 0
    min_k_termination = (
        "not_needed"
        if params.dynamic_enable_min_k_search
        else "disabled"
    )
    min_k_sphere_count = pre_merge_sphere_count

    if params.dynamic_enable_min_k_search:
        search_result = minimum_k_dynamic_sphere_search(
            centers, pre_merge_spheres, params, deadline)
        min_k_states_explored = search_result.states_explored
        min_k_termination = search_result.termination_reason
        min_k_sphere_count = len(search_result.spheres)
        baseline_state = build_dynamic_sphere_search_state(
            agglomerative_spheres)
        search_state = search_result.state
        candidate_states = (baseline_state, search_state)
        feasible_states = [
            state for state in candidate_states
            if _state_overlap_constraint_satisfied(state, params)
        ]
        if feasible_states:
            selected_state = min(
                feasible_states,
                key=lambda state: (
                    len(state.spheres),
                    dynamic_sphere_search_state_rank_key(state, params),
                ),
            )
        else:
            selected_state = min(
                candidate_states,
                key=lambda state: _state_fallback_key(state, params),
            )
        spheres = list(selected_state.spheres)
        if (
            selected_state.canonical_signature
            == baseline_state.canonical_signature
            and selected_state.canonical_signature
            != search_state.canonical_signature
            and min_k_termination not in ("deadline", "max_states")
        ):
            min_k_termination = "fallback_agglomerative"

    final_state = (selected_state if params.dynamic_enable_min_k_search
                   else build_dynamic_sphere_search_state(spheres))
    overlap_constraint_satisfied = _state_overlap_constraint_satisfied(
        final_state, params)
    if (
        params.dynamic_enable_min_k_search
        and not overlap_constraint_satisfied
        and min_k_termination not in ("deadline", "max_states")
    ):
        min_k_termination = "overlap_constraint_unmet"

    uncovered = ~_covered_mask(centers, spheres, params.coverage_tolerance_m)
    coverage = float(np.mean(~uncovered)) if len(uncovered) else 1.0
    confidence = min(
        1.0,
        coverage * min(
            1.0, len(centers) / max(1, params.min_component_voxels * 2)),
    )
    spheres = [replace(s, component_coverage=coverage, confidence=confidence) for s in spheres]
    return DynamicComponentResult(
        component_id=component_id,
        voxel_centers=centers,
        spheres=spheres,
        coverage=coverage,
        uncovered_voxels=centers[uncovered],
        termination_reason=reason,
        pre_merge_sphere_count=pre_merge_sphere_count,
        agglomerative_sphere_count=agglomerative_sphere_count,
        min_k_sphere_count=min_k_sphere_count,
        final_sphere_count=len(spheres),
        max_raw_overlap_fraction=final_state.max_raw_overlap_fraction,
        max_output_overlap_fraction=final_state.max_output_overlap_fraction,
        total_raw_overlap=final_state.total_raw_overlap,
        total_output_overlap=final_state.total_output_overlap,
        min_k_search_applied=min_k_search_applied,
        min_k_states_explored=min_k_states_explored,
        min_k_search_termination=min_k_termination,
        overlap_constraint_satisfied=overlap_constraint_satisfied,
        refinement_elapsed_ms=(monotonic() - refinement_start) * 1000.,
        refinement_budget_exhausted=monotonic() >= deadline,
    )


def generate_dynamic_spheres(
    points: np.ndarray | Sequence[Sequence[float]],
    params: DynamicSphereParameters,
    *,
    robot_sphere_centers: np.ndarray | Sequence[Sequence[float]] | None = None,
    robot_sphere_radii: np.ndarray | Sequence[float] | None = None,
    robot_component_overlap_threshold: float = 1.0,
    robot_sphere_margin_m: float = 0.0,
    local_width_cache=None,
    observation_evidence=None,
) -> DynamicSphereResult:
    """Generate bounded, deterministic obstacle spheres from dynamic XYZ points."""
    start = monotonic()
    if not 0.0 <= robot_component_overlap_threshold <= 1.0:
        raise ValueError("robot component overlap threshold must be in [0, 1]")
    use_robot_filter = (
        robot_sphere_centers is not None or robot_sphere_radii is not None)
    if use_robot_filter and (
        robot_sphere_centers is None or robot_sphere_radii is None
    ):
        raise ValueError("robot sphere centers and radii must be supplied together")
    indices = filter_and_voxelize(points, params)
    if len(indices) == 0:
        if local_width_cache is not None:
            local_width_cache.clear()
        return DynamicSphereResult(elapsed_ms=(monotonic() - start) * 1000.0)
    raw_components = connected_components(indices, params.connectivity)
    raw_components = [c for c in raw_components if len(c) >= params.min_component_voxels]
    raw_components = raw_components[:params.max_components]
    input_component_count = len(raw_components)
    if not raw_components:
        if local_width_cache is not None:
            local_width_cache.clear()
        return DynamicSphereResult(
            elapsed_ms=(monotonic() - start) * 1000.0, termination_reason="noise_removed")
    budget_sec = max(0.0, params.processing_budget_ms) / 1000.0
    deadline = start + budget_sec
    component_results = []
    all_spheres: list[DynamicSphere] = []
    all_voxels = []
    all_uncovered = []
    robot_rejected_voxels = []
    robot_rejected_component_ids = []
    robot_component_overlap_fractions = []
    robot_rejected_component_bounds = []
    remaining_refinement_ms = params.refinement_budget_ms
    remaining_local_width_ms = params.local_width_budget_ms
    for component_id, component in enumerate(raw_components):
        morphed = apply_morphology(component, params)
        if use_robot_filter:
            overlap_fraction, _ = component_robot_overlap_fraction(
                morphed,
                params,
                robot_sphere_centers,
                robot_sphere_radii,
                robot_sphere_margin_m,
            )
            robot_component_overlap_fractions.append(overlap_fraction)
            if overlap_fraction + 1e-12 >= robot_component_overlap_threshold:
                robot_rejected_component_ids.append(component_id)
                rejected_centers = voxel_centers(morphed, params)
                robot_rejected_voxels.append(rejected_centers)
                half_voxel = 0.5 * params.voxel_size_m
                robot_rejected_component_bounds.append((
                    np.min(rejected_centers, axis=0) - half_voxel,
                    np.max(rejected_centers, axis=0) + half_voxel,
                ))
                continue
        component_params = (replace(params, refinement_budget_ms=remaining_refinement_ms)
                            if remaining_refinement_ms is not None else params)
        component_params = replace(component_params, local_width_budget_ms=remaining_local_width_ms)
        result = _component_spheres(morphed, component_params, component_id, deadline,
            **({'local_width_cache': local_width_cache} if local_width_cache is not None else {}),
            **({'observation_evidence': observation_evidence} if observation_evidence is not None else {}))
        remaining_local_width_ms = max(0., remaining_local_width_ms - result.local_width_elapsed_ms)
        if remaining_refinement_ms is not None:
            remaining_refinement_ms = max(0., remaining_refinement_ms - result.refinement_elapsed_ms)
        remaining = params.max_total_spheres - len(all_spheres)
        if len(result.spheres) > max(0, remaining):
            # Component results already contain exact coverage/overlap. Only
            # invalidate that work when the global cap actually truncates it.
            result.spheres = result.spheres[:max(0, remaining)]
            result.uncovered_voxels = result.voxel_centers[
                ~_covered_mask(result.voxel_centers, result.spheres, params.coverage_tolerance_m)]
            result.coverage = 1.0 - len(result.uncovered_voxels) / max(1, len(result.voxel_centers))
            result.spheres = [replace(s, component_coverage=result.coverage) for s in result.spheres]
            final_state = build_dynamic_sphere_search_state(result.spheres)
            result.final_sphere_count = len(result.spheres)
            result.max_raw_overlap_fraction = final_state.max_raw_overlap_fraction
            result.max_output_overlap_fraction = final_state.max_output_overlap_fraction
            result.total_raw_overlap = final_state.total_raw_overlap
            result.total_output_overlap = final_state.total_output_overlap
            result.overlap_constraint_satisfied = (
                _state_overlap_constraint_satisfied(final_state, params))
        component_results.append(result)
        all_spheres.extend(result.spheres)
        all_voxels.append(result.voxel_centers)
        all_uncovered.append(result.uncovered_voxels)
        if len(all_spheres) >= params.max_total_spheres:
            break
    elapsed = (monotonic() - start) * 1000.0
    rejected_voxels = (
        np.vstack(robot_rejected_voxels)
        if robot_rejected_voxels else np.empty((0, 3)))
    if component_results:
        termination_reason = "complete"
    elif robot_rejected_component_ids:
        termination_reason = "robot_components_removed"
    else:
        termination_reason = "noise_removed"
    if not component_results and local_width_cache is not None:
        local_width_cache.clear()
    return DynamicSphereResult(
        components=component_results,
        spheres=all_spheres,
        voxel_centers=np.vstack(all_voxels) if all_voxels else np.empty((0, 3)),
        uncovered_voxels=np.vstack(all_uncovered) if all_uncovered else np.empty((0, 3)),
        robot_rejected_voxels=rejected_voxels,
        input_component_count=input_component_count,
        robot_rejected_component_count=len(robot_rejected_component_ids),
        robot_rejected_voxel_count=len(rejected_voxels),
        robot_rejected_component_ids=robot_rejected_component_ids,
        robot_component_overlap_fractions=robot_component_overlap_fractions,
        robot_rejected_component_bounds=robot_rejected_component_bounds,
        elapsed_ms=elapsed,
        termination_reason=termination_reason)


def prune_overlapping_tracked_spheres(
    voxel_groups: Sequence[np.ndarray],
    spheres: Sequence[DynamicSphere],
    target_coverage: float,
    coverage_tolerance_m: float,
    max_overlap_fraction: float,
    max_spheres: int = 64,
    max_removals: int = 16,
    max_matrix_elements: int = 2000000,
    *, fresh_track_ids: set[int] | None = None,
) -> tuple[list[DynamicSphere], set[int]]:
    """Remove redundant tracked spheres while preserving every component.

    Output-radius overlap selects candidates, while raw-radius voxel coverage
    guards each current component.  Hard sphere, removal, and matrix caps keep
    this post-tracking pass bounded at the node's high update rate.
    """

    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be in [0, 1]")
    if coverage_tolerance_m < 0.0:
        raise ValueError("coverage_tolerance_m must be non-negative")
    if not 0.0 <= max_overlap_fraction <= 1.0:
        raise ValueError("tracked overlap fraction must be in [0, 1]")
    if max_spheres <= 0 or max_removals < 0 or max_matrix_elements <= 0:
        raise ValueError("tracked pruning limits are invalid")

    active_spheres = list(spheres)
    groups = [_as_points(group) for group in voxel_groups if len(group)]
    total_points = sum(len(group) for group in groups)
    if (
        len(active_spheres) < 2
        or len(active_spheres) > max_spheres
        or max_removals == 0
        or total_points == 0
        or len(active_spheres) * total_points > max_matrix_elements
    ):
        return active_spheres, set()

    points = np.vstack(groups)
    coverage_masks = _sphere_coverage_masks(
        points, active_spheres, coverage_tolerance_m)
    group_slices = []
    start = 0
    for group in groups:
        group_slices.append(slice(start, start + len(group)))
        start += len(group)
    baseline_covered = np.any(coverage_masks, axis=0)
    required_coverage = [
        min(target_coverage, float(np.mean(baseline_covered[group_slice])))
        for group_slice in group_slices
    ]

    active = list(range(len(active_spheres)))
    removed_track_ids: set[int] = set()
    removed_count = 0
    while len(active) > 1 and removed_count < max_removals:
        coverage_count = np.count_nonzero(coverage_masks[active], axis=0)
        unique_contribution = {
            index: int(np.count_nonzero(
                coverage_masks[index] & (coverage_count == 1)))
            for index in active
        }
        overlapping = set()
        for first_position, first_index in enumerate(active):
            first = active_spheres[first_index]
            for second_index in active[first_position + 1:]:
                second = active_spheres[second_index]
                overlap = sphere_overlap_fraction(
                    first.center,
                    first.output_radius,
                    second.center,
                    second.output_radius,
                )
                if overlap > max_overlap_fraction + 1e-12:
                    overlapping.add(first_index)
                    overlapping.add(second_index)
        if not overlapping:
            break

        removal_order = sorted(overlapping, key=lambda index: (
            unique_contribution[index],
            int(active_spheres[index].track_id in fresh_track_ids) if fresh_track_ids is not None else 0,
            active_spheres[index].confidence,
            active_spheres[index].age,
            -active_spheres[index].output_radius,
            active_spheres[index].track_id,
            index,
        ))
        removed = False
        for index in removal_order:
            trial = [candidate for candidate in active if candidate != index]
            covered = np.any(coverage_masks[trial], axis=0)
            if all(
                float(np.mean(covered[group_slice])) + 1e-12 >= required
                for group_slice, required in zip(
                    group_slices, required_coverage)
            ):
                active = trial
                track_id = active_spheres[index].track_id
                if track_id >= 0:
                    removed_track_ids.add(track_id)
                removed_count += 1
                removed = True
                break
        if not removed:
            break

    return [active_spheres[index] for index in active], removed_track_ids


@dataclass
class _Track:
    track_id: int
    sphere: DynamicSphere
    first_seen: float
    last_seen: float
    missed_updates: int = 0
    object_id: int = -1


class DynamicSphereTracker:
    """Greedy one-to-one center-distance tracker with bounded occlusion hold."""

    def __init__(
        self, association_distance_m: float, smoothing_alpha: float,
        ttl_sec: float, max_missed_updates: int,
        radius_smoothing_alpha: float | None = None,
        component_tracking_enabled: bool = False,
        max_radius_ratio: float = 2.0,
        component_distance_m: float = .30,
        voxel_size_m: float = .05,
    ) -> None:
        self.association_distance_m = float(association_distance_m)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.radius_smoothing_alpha = float(np.clip(
            self.smoothing_alpha
            if radius_smoothing_alpha is None
            else radius_smoothing_alpha,
            0.0,
            1.0,
        ))
        self.ttl_sec = float(ttl_sec)
        self.max_missed_updates = int(max_missed_updates)
        self._next_id = 0
        self._tracks: dict[int, _Track] = {}
        if not np.isfinite(max_radius_ratio) or max_radius_ratio < 1.:
            raise ValueError("tracking radius ratio must be finite and >= 1")
        self.component_tracking_enabled = component_tracking_enabled
        self.max_radius_ratio = max_radius_ratio
        self.component_associator = ComponentAssociator(
            distance=component_distance_m, voxel_size=voxel_size_m, ttl=self.ttl_sec)
        self.last_observation_time = None

    def update(
        self,
        detections: Iterable[DynamicSphere],
        timestamp_sec: float,
        *, components=None,
    ) -> list[DynamicSphere]:
        detections = list(detections)
        now = float(timestamp_sec)
        if (self.component_tracking_enabled and self.last_observation_time is not None
                and now < self.last_observation_time):
            self.clear()
        self.last_observation_time = now
        self._expire(now)
        use_components = self.component_tracking_enabled and components is not None
        object_ids, translations = (self.component_associator.update(components, now)
                                    if use_components else ({}, {}))
        pairs = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                expected = track.sphere.center
                radius_cost = 0.
                if use_components:
                    object_id = object_ids.get(detection.component_id, -1)
                    if object_id < 0 or track.object_id != object_id:
                        continue
                    ratio = max(track.sphere.raw_radius, detection.raw_radius) / max(
                        1e-6, min(track.sphere.raw_radius, detection.raw_radius))
                    if ratio > self.max_radius_ratio:
                        continue
                    # Use measured object translation for association only;
                    # published centers still come from current observations.
                    expected = expected + translations[detection.component_id]
                    radius_cost = .03 * abs(np.log(ratio))
                distance = float(np.linalg.norm(expected - detection.center))
                if distance <= self.association_distance_m:
                    pairs.append((distance + radius_cost, track_id, detection_index))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_index in sorted(pairs):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self._tracks[track_id]
            detection = detections[detection_index]
            center_alpha = self.smoothing_alpha
            radius_alpha = self.radius_smoothing_alpha
            center = (
                center_alpha * detection.center
                + (1.0 - center_alpha) * track.sphere.center
            )
            raw = (
                radius_alpha * detection.raw_radius
                + (1.0 - radius_alpha) * track.sphere.raw_radius
            )
            output = (
                radius_alpha * detection.output_radius
                + (1.0 - radius_alpha) * track.sphere.output_radius
            )
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
            self._tracks[track_id] = _Track(track_id, sphere, now, now,
                object_id=object_ids.get(detection.component_id, -1))
            matched_tracks.add(track_id)
        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track.missed_updates += 1
        self._expire(now)
        return [self._tracks[key].sphere for key in sorted(self._tracks)]

    def clear(self) -> None:
        """Drop all retained tracks while preserving monotonically increasing IDs."""
        self._tracks.clear()
        self.component_associator.clear()
        self.last_observation_time = None

    def fresh_track_ids(self, timestamp_sec):
        return {key for key, track in self._tracks.items() if track.last_seen == timestamp_sec}

    def snapshot(self, timestamp_sec: float) -> list[DynamicSphere]:
        """Return live tracks without counting a missing detection update.

        The sphere node runs from a fixed-rate timer that can be faster than
        the upstream dynamic PointCloud publisher.  A timer tick with no new
        PointCloud is therefore not evidence that an obstacle disappeared.
        TTL expiry still applies so tracks cannot survive indefinitely when
        the upstream publisher stops.
        """

        self._expire(float(timestamp_sec))
        return [self._tracks[key].sphere for key in sorted(self._tracks)]

    def remove_track_ids(self, track_ids: Iterable[int]) -> None:
        """Drop tracks removed by a downstream geometry post-pass."""

        for track_id in track_ids:
            self._tracks.pop(int(track_id), None)

    def remove_tracks_in_bounds(
        self,
        bounds: Iterable[tuple[np.ndarray, np.ndarray]],
        padding_m: float = 0.0,
    ) -> None:
        """Drop tracks centered in rejected component bounds."""
        boxes = [
            (
                np.asarray(lower, dtype=np.float64) - padding_m,
                np.asarray(upper, dtype=np.float64) + padding_m,
            )
            for lower, upper in bounds
        ]
        rejected = [
            track_id
            for track_id, track in self._tracks.items()
            if any(
                np.all(track.sphere.center >= lower)
                and np.all(track.sphere.center <= upper)
                for lower, upper in boxes
            )
        ]
        for track_id in rejected:
            del self._tracks[track_id]

    def _expire(self, now: float) -> None:
        expired = [
            track_id for track_id, track in self._tracks.items()
            if now - track.last_seen > self.ttl_sec or
            track.missed_updates > self.max_missed_updates
        ]
        for track_id in expired:
            del self._tracks[track_id]
