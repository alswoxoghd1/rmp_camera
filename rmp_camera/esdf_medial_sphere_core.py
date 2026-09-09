"""Pure NumPy medial-sphere extraction from a dense signed ESDF grid.

Array indices are ordered ``(x, y, z)``. Distances, origins, voxel sizes,
spacings, tolerances, and radii are expressed in metres.
"""

from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import List, Sequence, Tuple

import numpy as np

from rmp_camera.sphere_merge_core import (
    PairMergeCandidate,
    enumerate_pair_merge_candidates,
    select_best_merge_pair,
    sphere_overlap_fraction,
    validate_merge_limits,
)


from rmp_camera.local_width_sphere_cover import local_width_cover, VoxelSupportGuard

Index3 = Tuple[int, int, int]


@dataclass
class Sphere:
    """One obstacle sphere expressed in the ESDF target frame, in metres."""

    center: np.ndarray
    raw_radius: float
    output_radius: float
    component_id: int
    source_index: Index3
    is_merged: bool = False


@dataclass
class ComponentResult:
    """Sphere approximation and raw-radius coverage for one inside component."""

    component_id: int
    voxel_indices: np.ndarray
    spheres: List[Sphere]
    coverage: float
    termination_reason: str
    uncovered_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )
    pre_merge_sphere_count: int = 0
    agglomerative_sphere_count: int = 0
    min_k_sphere_count: int = 0
    final_sphere_count: int = 0
    min_k_search_applied: bool = False
    min_k_states_explored: int = 0
    min_k_search_termination: str = "disabled"
    post_overlap_sphere_count: int = 0
    post_overlap_removed_count: int = 0
    coarse_candidate_count: int = 0
    coarse_selected_count: int = 0
    local_width_saved_spheres: int = 0
    local_width_elapsed_ms: float = 0.0
    local_width_reason: str = "disabled"


@dataclass(frozen=True)
class StaticSphereSearchState:
    """One order-independent state in the bounded static merge search."""

    spheres: tuple[Sphere, ...]
    total_raw_volume: float
    total_raw_radius: float
    canonical_signature: tuple[tuple[object, ...], ...]


@dataclass
class StaticMinimumKSearchResult:
    """Best valid static sphere state found within configured search bounds."""

    spheres: List[Sphere]
    states_explored: int
    termination_reason: str
    state: StaticSphereSearchState


@dataclass
class GenerationResult:
    """Complete deterministic result for one dense ESDF grid."""

    inside_mask: np.ndarray
    component_labels: np.ndarray
    components: List[ComponentResult]
    spheres: List[Sphere]
    removed_small_components: int
    total_limit_applied: bool
    coverage_lost_component_ids: List[int]
    input_component_count: int = 0
    robot_rejected_component_count: int = 0
    robot_rejected_voxel_count: int = 0
    robot_rejected_component_ids: List[int] = field(default_factory=list)
    robot_component_overlap_fractions: List[float] = field(default_factory=list)
    robot_rejected_voxel_indices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.int64)
    )


def make_neighbor_offsets_18() -> Tuple[Index3, ...]:
    """Return the 18 face/edge neighbor index offsets in deterministic order."""

    return tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0) and abs(dx) + abs(dy) + abs(dz) <= 2
    )


NEIGHBOR_OFFSETS_18 = make_neighbor_offsets_18()


def extract_inside_mask(
    esdf_grid: np.ndarray,
    unobserved_distance_value: float,
    inside_epsilon_m: float,
) -> np.ndarray:
    """Return observed voxels whose signed ESDF is below ``-epsilon``.

    ``esdf_grid`` is a dense ``(x, y, z)`` scalar array in metres. The
    unobserved sentinel is excluded explicitly; ``inside_epsilon_m`` should be
    a small non-negative noise threshold. A value that is too large erodes thin
    obstacles, while zero may retain noisy sign flips at the surface.
    """

    values = np.asarray(esdf_grid)
    if values.ndim != 3:
        raise ValueError("esdf_grid must be a three-dimensional array")
    if inside_epsilon_m < 0.0:
        raise ValueError("inside_epsilon_m must be non-negative")
    sentinel_tolerance = max(1e-9, abs(float(unobserved_distance_value)) * 1e-12)
    observed = np.isfinite(values) & (
        np.abs(values - float(unobserved_distance_value)) > sentinel_tolerance
    )
    return observed & (values < -float(inside_epsilon_m))


def connected_components_18(inside_mask: np.ndarray) -> List[np.ndarray]:
    """Flood-fill true ``(x, y, z)`` voxels using 18-connectivity.

    Returns a deterministic list of ``(N, 3)`` integer index arrays. Only true
    voxels are visited, and each component's indices are lexicographically
    ordered by the traversal seeded from ``numpy.argwhere``.
    """

    mask = np.asarray(inside_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("inside_mask must be a three-dimensional array")
    visited = np.zeros(mask.shape, dtype=bool)
    shape = mask.shape
    components: List[np.ndarray] = []

    for seed_array in np.argwhere(mask):
        seed = tuple(int(value) for value in seed_array)
        if visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        indices = []
        while queue:
            current = queue.popleft()
            indices.append(current)
            for offset in NEIGHBOR_OFFSETS_18:
                neighbor = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if (
                    0 <= neighbor[0] < shape[0]
                    and 0 <= neighbor[1] < shape[1]
                    and 0 <= neighbor[2] < shape[2]
                    and mask[neighbor]
                    and not visited[neighbor]
                ):
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(np.asarray(indices, dtype=np.int64))
    return components


def static_voxel_robot_sphere_overlap_mask(
    voxel_indices: np.ndarray,
    origin_m: Sequence[float],
    voxel_size_m: float,
    robot_sphere_centers: np.ndarray | Sequence[Sequence[float]],
    robot_sphere_radii: np.ndarray | Sequence[float],
    margin_m: float = 0.0,
) -> np.ndarray:
    """Return which static voxel AABBs intersect a robot sphere."""

    indices = _as_index_array(voxel_indices)
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if margin_m < 0.0 or not np.isfinite(margin_m):
        raise ValueError("robot sphere margin must be finite and non-negative")
    if len(indices) == 0:
        return np.zeros(0, dtype=bool)

    centers = np.asarray(robot_sphere_centers, dtype=np.float64)
    if centers.size == 0:
        centers = np.empty((0, 3), dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("robot sphere centers must have shape (N, 3)")
    radii = np.asarray(robot_sphere_radii, dtype=np.float64).reshape(-1)
    if len(centers) != len(radii):
        raise ValueError("robot sphere centers and radii must have equal length")
    if not np.isfinite(centers).all() or not np.isfinite(radii).all():
        raise ValueError("robot sphere geometry must be finite")
    if np.any(radii <= 0.0):
        raise ValueError("robot sphere radii must be positive")
    if len(centers) == 0:
        return np.zeros(len(indices), dtype=bool)

    origin = _as_origin(origin_m)
    voxel_min = origin + indices.astype(np.float64) * voxel_size_m
    voxel_max = voxel_min + voxel_size_m
    overlaps = np.zeros(len(indices), dtype=bool)
    for center, radius in zip(centers, radii):
        closest = np.clip(center, voxel_min, voxel_max)
        delta = closest - center
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        overlaps |= squared_distance <= (radius + margin_m) ** 2
        if overlaps.all():
            break
    return overlaps


def static_component_robot_overlap_fraction(
    voxel_indices: np.ndarray,
    origin_m: Sequence[float],
    voxel_size_m: float,
    robot_sphere_centers: np.ndarray | Sequence[Sequence[float]],
    robot_sphere_radii: np.ndarray | Sequence[float],
    margin_m: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """Return robot-intersecting static voxel fraction and its mask."""

    mask = static_voxel_robot_sphere_overlap_mask(
        voxel_indices,
        origin_m,
        voxel_size_m,
        robot_sphere_centers,
        robot_sphere_radii,
        margin_m,
    )
    return (float(np.mean(mask)) if len(mask) else 0.0), mask


def previous_support_is_observed(points, observed_mask, origin_m, voxel_size_m):
    """Do not turn map loss/out-of-bounds old geometry into a valid empty."""
    if not np.any(observed_mask):
        return False
    points = np.asarray(points).reshape(-1, 3)
    if not len(points):
        return True
    indices = np.floor((points - origin_m) / voxel_size_m).astype(np.int64)
    in_bounds = np.all((indices >= 0) & (indices < observed_mask.shape), axis=1)
    return bool(in_bounds.all() and observed_mask[tuple(indices.T)].all())


def has_static_obstacle_component(
    values, origin_m, voxel_size_m, *, unobserved_distance_value=-1000.0,
    inside_epsilon_m=0.005, min_component_voxels=30,
    robot_sphere_centers=None, robot_sphere_radii=None,
    robot_component_overlap_threshold=0.02, robot_sphere_margin_m=0.0,
):
    """Conservative empty-only check using the full solver's admission rules.

    False guarantees that there is no component on which the sphere solver
    would run. True does not guarantee a sphere (e.g. minimum-radius limits).
    This deliberately does not use coverage, pruning, merging or Minimum-K.
    Observation validity must be checked separately by the caller.
    """
    if min_component_voxels < 1:
        raise ValueError("min_component_voxels must be positive")
    use_robot = robot_sphere_centers is not None or robot_sphere_radii is not None
    if use_robot and (robot_sphere_centers is None or robot_sphere_radii is None):
        raise ValueError("robot centers and radii must be supplied together")
    inside = extract_inside_mask(values, unobserved_distance_value, inside_epsilon_m)
    for indices in connected_components_18(inside):
        if len(indices) < min_component_voxels:
            continue
        if use_robot:
            fraction, _ = static_component_robot_overlap_fraction(
                indices, origin_m, voxel_size_m, robot_sphere_centers,
                robot_sphere_radii, robot_sphere_margin_m)
            if fraction + 1e-12 >= robot_component_overlap_threshold:
                continue
        return True
    return False


def find_local_minimum_candidates(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
) -> np.ndarray:
    """Find component voxels no greater than any valid 18-neighbor.

    Inputs and returned candidates use integer ``(x, y, z)`` indices. Only
    neighbors belonging to the same component participate in the comparison.
    """

    values = np.asarray(esdf_grid)
    indices = _as_index_array(component_indices)
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.int64)
    component_set = {tuple(int(v) for v in index) for index in indices}
    candidates = []
    for index_array in indices:
        index = tuple(int(v) for v in index_array)
        value = float(values[index])
        is_minimum = True
        for offset in NEIGHBOR_OFFSETS_18:
            neighbor = (
                index[0] + offset[0],
                index[1] + offset[1],
                index[2] + offset[2],
            )
            if neighbor in component_set and value > float(values[neighbor]):
                is_minimum = False
                break
        if is_minimum:
            candidates.append(index)
    return np.asarray(candidates, dtype=np.int64).reshape((-1, 3))


def group_or_reduce_plateaus(
    esdf_grid: np.ndarray,
    candidate_indices: np.ndarray,
    voxel_size_m: float,
    plateau_epsilon_m: float,
    minimum_center_spacing_m: float,
) -> np.ndarray:
    """Reduce adjacent equal-depth minima while retaining long-plateau extent.

    Candidate voxels connected by 18-neighbors and differing by at most
    ``plateau_epsilon_m`` are one plateau. Within each plateau, representatives
    are greedily retained in deepest-ESDF then lexicographic order whenever
    they are at least ``minimum_center_spacing_m`` from existing
    representatives. This permits multiple centres along a long plateau rather
    than collapsing the whole plateau to one mean point.
    """

    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if plateau_epsilon_m < 0.0 or minimum_center_spacing_m < 0.0:
        raise ValueError("plateau epsilon and center spacing must be non-negative")
    values = np.asarray(esdf_grid)
    candidates = _as_index_array(candidate_indices)
    if len(candidates) == 0:
        return candidates.copy()

    candidate_set = {tuple(int(v) for v in index) for index in candidates}
    unvisited = set(candidate_set)
    plateaus: List[List[Index3]] = []
    for seed in sorted(candidate_set):
        if seed not in unvisited:
            continue
        unvisited.remove(seed)
        queue = deque([seed])
        plateau = []
        while queue:
            current = queue.popleft()
            plateau.append(current)
            current_value = float(values[current])
            for offset in NEIGHBOR_OFFSETS_18:
                neighbor = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if (
                    neighbor in unvisited
                    and abs(current_value - float(values[neighbor]))
                    <= plateau_epsilon_m
                ):
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
        plateaus.append(plateau)

    # Apply spacing globally as well as across each plateau boundary. Adjacent
    # plateaus can be separated solely by a tiny value step, so per-group
    # spacing alone can still emit practically duplicate centres.
    ordered_candidates = sorted(
        (index for plateau in plateaus for index in plateau),
        key=lambda idx: (float(values[idx]), idx),
    )
    representatives: List[Index3] = []
    for candidate in ordered_candidates:
        if representatives:
            distances = np.linalg.norm(
                (
                    np.asarray(representatives, dtype=np.float64)
                    - np.asarray(candidate)
                )
                * float(voxel_size_m),
                axis=1,
            )
            if np.any(distances + 1e-12 < minimum_center_spacing_m):
                continue
        representatives.append(candidate)
    return np.asarray(representatives, dtype=np.int64).reshape((-1, 3))


def create_initial_spheres(
    esdf_grid: np.ndarray,
    candidate_indices: np.ndarray,
    origin_m: Sequence[float],
    voxel_size_m: float,
    min_raw_sphere_radius_m: float,
    safety_margin_m: float,
    component_id: int,
    max_raw_sphere_radius_m: float = float("inf"),
) -> List[Sphere]:
    """Create bounded spheres at voxel centres from the negative ESDF.

    Coordinates and all scalar distances are metres. Candidates below the
    configurable minimum raw radius are discarded; a high minimum can remove
    thin obstacle parts. Deep candidates are clipped to the maximum raw
    radius, and ``output_radius`` alone receives the safety margin.
    """

    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    if (
        min_raw_sphere_radius_m < 0.0
        or max_raw_sphere_radius_m < min_raw_sphere_radius_m
        or safety_margin_m < 0.0
    ):
        raise ValueError("sphere radius threshold and safety margin must be non-negative")
    values = np.asarray(esdf_grid)
    origin = _as_origin(origin_m)
    spheres = []
    for index_array in _as_index_array(candidate_indices):
        source_index = tuple(int(v) for v in index_array)
        raw_radius = min(
            -float(values[source_index]), float(max_raw_sphere_radius_m))
        if not np.isfinite(raw_radius) or raw_radius < min_raw_sphere_radius_m:
            continue
        center = origin + (index_array.astype(np.float64) + 0.5) * voxel_size_m
        spheres.append(
            Sphere(
                center=center,
                raw_radius=raw_radius,
                output_radius=raw_radius + safety_margin_m,
                component_id=int(component_id),
                source_index=source_index,
            )
        )
    return spheres


def calculate_component_coverage(
    component_indices: np.ndarray,
    spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    coverage_tolerance_m: float,
    chunk_size: int = 16384,
) -> Tuple[float, np.ndarray]:
    """Return raw-sphere coverage and a per-voxel covered mask.

    Coverage tests component voxel centres in metres against
    ``raw_radius + coverage_tolerance_m``. Safety margins are intentionally
    ignored. Voxels are processed in chunks to bound temporary memory.
    """

    indices = _as_index_array(component_indices)
    if len(indices) == 0:
        return 1.0, np.empty(0, dtype=bool)
    if voxel_size_m <= 0.0 or coverage_tolerance_m < 0.0 or chunk_size <= 0:
        raise ValueError("voxel size/chunk size must be positive and tolerance non-negative")
    if not spheres:
        return 0.0, np.zeros(len(indices), dtype=bool)

    origin = _as_origin(origin_m)
    centers = np.asarray([sphere.center for sphere in spheres], dtype=np.float64)
    radii_squared = np.square(
        np.asarray([sphere.raw_radius for sphere in spheres], dtype=np.float64)
        + coverage_tolerance_m
    )
    covered = np.zeros(len(indices), dtype=bool)
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        points = origin + (indices[start:stop].astype(np.float64) + 0.5) * voxel_size_m
        distances_squared = np.sum(
            np.square(points[:, None, :] - centers[None, :, :]), axis=2
        )
        covered[start:stop] = np.any(
            distances_squared <= radii_squared[None, :] + 1e-15, axis=1
        )
    return float(np.count_nonzero(covered)) / float(len(indices)), covered


def find_uncovered_voxels(
    component_indices: np.ndarray,
    spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    coverage_tolerance_m: float,
) -> np.ndarray:
    """Return ``(x, y, z)`` component indices outside all raw spheres."""

    indices = _as_index_array(component_indices)
    _, covered = calculate_component_coverage(
        indices, spheres, origin_m, voxel_size_m, coverage_tolerance_m
    )
    return indices[~covered]


def add_spheres_until_coverage(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    initial_spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    target_coverage: float,
    coverage_tolerance_m: float,
    minimum_center_spacing_m: float,
    min_raw_sphere_radius_m: float,
    safety_margin_m: float,
    max_spheres_per_component: int,
    max_iterations_per_component: int,
    component_id: int,
    max_raw_sphere_radius_m: float = float("inf"),
) -> Tuple[List[Sphere], float, str, np.ndarray]:
    """Add deepest uncovered valid voxels until coverage or a hard limit.

    The returned coverage uses raw radii. Selection is deterministic by ESDF
    value then voxel index. Spacing, sphere count, and iteration caps guarantee
    termination; too-large spacing or radius thresholds can prevent the target.
    """

    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError("target_coverage must be in [0, 1]")
    if max_spheres_per_component < 0 or max_iterations_per_component < 0:
        raise ValueError("sphere and iteration limits must be non-negative")
    if max_raw_sphere_radius_m < min_raw_sphere_radius_m:
        raise ValueError("maximum raw radius must be at least the minimum")
    values = np.asarray(esdf_grid)
    indices = _as_index_array(component_indices)
    origin = _as_origin(origin_m)
    spheres = list(initial_spheres)
    iterations = 0

    while True:
        coverage, covered = calculate_component_coverage(
            indices, spheres, origin, voxel_size_m, coverage_tolerance_m
        )
        uncovered = indices[~covered]
        if coverage + 1e-12 >= target_coverage:
            return spheres, coverage, "target_coverage", uncovered
        if len(uncovered) == 0:
            return spheres, coverage, "no_uncovered_voxels", uncovered
        if len(spheres) >= max_spheres_per_component:
            return spheres, coverage, "max_spheres_per_component", uncovered
        if iterations >= max_iterations_per_component:
            return spheres, coverage, "max_iterations_per_component", uncovered

        ordered = sorted(
            (tuple(int(v) for v in index) for index in uncovered),
            key=lambda idx: (float(values[idx]), idx),
        )
        selected = None
        for candidate in ordered:
            raw_radius = min(
                -float(values[candidate]), float(max_raw_sphere_radius_m))
            if not np.isfinite(raw_radius) or raw_radius < min_raw_sphere_radius_m:
                continue
            center = origin + (np.asarray(candidate, dtype=np.float64) + 0.5) * voxel_size_m
            if spheres:
                center_distances = np.linalg.norm(
                    np.asarray([sphere.center for sphere in spheres]) - center, axis=1
                )
                if np.any(center_distances + 1e-12 < minimum_center_spacing_m):
                    continue
            selected = Sphere(
                center=center,
                raw_radius=raw_radius,
                output_radius=raw_radius + safety_margin_m,
                component_id=int(component_id),
                source_index=candidate,
            )
            break
        if selected is None:
            return spheres, coverage, "no_valid_candidate", uncovered
        spheres.append(selected)
        iterations += 1


def remove_redundant_spheres(
    component_indices: np.ndarray,
    spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    target_coverage: float,
    coverage_tolerance_m: float,
    redundancy_tolerance_m: float,
) -> Tuple[List[Sphere], float]:
    """Remove contained spheres only when target raw coverage remains.

    Larger radii are considered first and ties use the source index, making the
    operation deterministic. A large redundancy tolerance removes more
    candidates, but every removal is guarded by a complete coverage check.
    """

    if redundancy_tolerance_m < 0.0:
        raise ValueError("redundancy_tolerance_m must be non-negative")
    active = [True] * len(spheres)
    order = sorted(
        range(len(spheres)),
        key=lambda i: (-spheres[i].raw_radius, spheres[i].source_index, i),
    )
    for order_position, container_index in enumerate(order):
        if not active[container_index]:
            continue
        container = spheres[container_index]
        for candidate_index in order[order_position + 1:]:
            if not active[candidate_index]:
                continue
            candidate = spheres[candidate_index]
            contained = (
                float(np.linalg.norm(container.center - candidate.center))
                + candidate.raw_radius
                <= container.raw_radius + redundancy_tolerance_m + 1e-12
            )
            if not contained:
                continue
            trial = [
                sphere
                for index, sphere in enumerate(spheres)
                if active[index] and index != candidate_index
            ]
            coverage, _ = calculate_component_coverage(
                component_indices,
                trial,
                origin_m,
                voxel_size_m,
                coverage_tolerance_m,
            )
            if coverage + 1e-12 >= target_coverage:
                active[candidate_index] = False
    result = [sphere for index, sphere in enumerate(spheres) if active[index]]
    coverage, _ = calculate_component_coverage(
        component_indices, result, origin_m, voxel_size_m, coverage_tolerance_m
    )
    return result, coverage


def make_surface_shell_mask(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    inside_epsilon_m: float,
    surface_shell_thickness_m: float,
) -> np.ndarray:
    """Return component-order voxels lying in the configured inner shell."""

    if inside_epsilon_m < 0.0 or surface_shell_thickness_m < 0.0:
        raise ValueError("inside epsilon and shell thickness must be non-negative")
    indices = _as_index_array(component_indices)
    if len(indices) == 0 or surface_shell_thickness_m == 0.0:
        return np.zeros(len(indices), dtype=bool)
    values = np.asarray(esdf_grid, dtype=np.float64)
    distances = values[tuple(indices.T)]
    return (
        np.isfinite(distances)
        & (distances >= -surface_shell_thickness_m)
        & (distances < -inside_epsilon_m)
    )


def sphere_coverage_masks(
    component_indices: np.ndarray,
    spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    coverage_tolerance_m: float,
    max_matrix_elements: int = 20000000,
) -> np.ndarray | None:
    """Build one reusable component-voxel coverage row per candidate sphere."""

    indices = _as_index_array(component_indices)
    if voxel_size_m <= 0.0 or coverage_tolerance_m < 0.0:
        raise ValueError("voxel size must be positive and tolerance non-negative")
    if max_matrix_elements <= 0:
        raise ValueError("max_matrix_elements must be positive")
    if len(indices) * len(spheres) > max_matrix_elements:
        return None
    points = _as_origin(origin_m) + (
        indices.astype(np.float64) + 0.5
    ) * voxel_size_m
    masks = np.zeros((len(spheres), len(indices)), dtype=bool)
    for sphere_index, sphere in enumerate(spheres):
        delta = points - np.asarray(sphere.center, dtype=np.float64)
        distance_squared = np.einsum("ij,ij->i", delta, delta)
        masks[sphere_index] = distance_squared <= (
            sphere.raw_radius + coverage_tolerance_m
        ) ** 2 + 1e-15
    return masks


def calculate_mask_coverage(
    covered_mask: np.ndarray,
    evaluation_mask: np.ndarray | None = None,
) -> float:
    """Calculate coverage over all voxels or a same-length evaluation subset."""

    covered = np.asarray(covered_mask, dtype=bool).reshape(-1)
    if evaluation_mask is None:
        return 1.0 if len(covered) == 0 else float(np.mean(covered))
    evaluation = np.asarray(evaluation_mask, dtype=bool).reshape(-1)
    if len(evaluation) != len(covered):
        raise ValueError("covered and evaluation masks must have equal length")
    denominator = int(np.count_nonzero(evaluation))
    if denominator == 0:
        return 1.0
    return float(np.count_nonzero(covered & evaluation)) / float(denominator)


def select_best_single_sphere(
    spheres: Sequence[Sphere],
    coverage_masks: np.ndarray,
    shell_mask: np.ndarray,
    required_volume_coverage: float,
    required_shell_coverage: float,
    shell_guard_enabled: bool,
    valid_candidate_mask: np.ndarray | None = None,
) -> int | None:
    """Select the smallest deterministic candidate satisfying both guards."""

    if len(spheres) < 2:
        return None
    valid = (
        np.ones(len(spheres), dtype=bool)
        if valid_candidate_mask is None
        else np.asarray(valid_candidate_mask, dtype=bool)
    )
    eligible = []
    for index, sphere in enumerate(spheres):
        volume_coverage = calculate_mask_coverage(coverage_masks[index])
        shell_coverage = calculate_mask_coverage(coverage_masks[index], shell_mask)
        if (
            valid[index]
            and volume_coverage + 1e-12 >= required_volume_coverage
            and (
                not shell_guard_enabled
                or shell_coverage + 1e-12 >= required_shell_coverage
            )
        ):
            eligible.append((
                sphere.raw_radius,
                -shell_coverage,
                -volume_coverage,
                sphere.source_index,
                index,
            ))
    return None if not eligible else min(eligible)[-1]


def greedy_set_cover_spheres(
    spheres: Sequence[Sphere],
    coverage_masks: np.ndarray,
    shell_mask: np.ndarray,
    required_volume_coverage: float,
    required_shell_coverage: float,
    shell_guard_enabled: bool,
    valid_candidate_mask: np.ndarray | None = None,
) -> List[int] | None:
    """Select deterministic candidate indices using shell-aware marginal gain."""

    valid = (
        np.ones(len(spheres), dtype=bool)
        if valid_candidate_mask is None
        else np.asarray(valid_candidate_mask, dtype=bool)
    )
    available = valid.copy()
    covered = np.zeros(coverage_masks.shape[1], dtype=bool)
    selected: List[int] = []
    while True:
        volume_coverage = calculate_mask_coverage(covered)
        shell_coverage = calculate_mask_coverage(covered, shell_mask)
        volume_met = volume_coverage + 1e-12 >= required_volume_coverage
        shell_met = (
            not shell_guard_enabled
            or shell_coverage + 1e-12 >= required_shell_coverage
        )
        if volume_met and shell_met:
            return selected

        uncovered = ~covered
        volume_gains = np.count_nonzero(coverage_masks & uncovered, axis=1)
        shell_gains = np.count_nonzero(
            coverage_masks & uncovered & shell_mask[None, :], axis=1
        )
        choices = np.flatnonzero(available & (volume_gains > 0))
        if len(choices) == 0:
            return None
        shell_is_priority = shell_guard_enabled and not shell_met
        best = min(
            (int(index) for index in choices),
            key=lambda index: (
                -int(shell_gains[index])
                if shell_is_priority
                else -int(volume_gains[index]),
                -int(volume_gains[index])
                if shell_is_priority
                else -int(shell_gains[index]),
                -spheres[index].raw_radius,
                spheres[index].source_index,
                index,
            ),
        )
        selected.append(best)
        available[best] = False
        covered |= coverage_masks[best]


def general_coverage_pruning(
    spheres: Sequence[Sphere],
    coverage_masks: np.ndarray,
    shell_mask: np.ndarray,
    selected_indices: Sequence[int],
    required_volume_coverage: float,
    required_shell_coverage: float,
    shell_guard_enabled: bool,
) -> List[int]:
    """Remove low-unique-contribution spheres while both guards remain met."""

    active = list(selected_indices)
    while len(active) > 1:
        coverage_count = np.count_nonzero(coverage_masks[active], axis=0)
        unique_volume = {
            index: int(np.count_nonzero(
                coverage_masks[index] & (coverage_count == 1)))
            for index in active
        }
        unique_shell = {
            index: int(np.count_nonzero(
                coverage_masks[index] & shell_mask & (coverage_count == 1)))
            for index in active
        }
        removal_order = sorted(active, key=lambda index: (
            unique_shell[index],
            unique_volume[index],
            spheres[index].raw_radius,
            spheres[index].source_index,
            index,
        ))
        removed = False
        for index in removal_order:
            trial = [candidate for candidate in active if candidate != index]
            trial_covered = np.any(coverage_masks[trial], axis=0)
            volume_coverage = calculate_mask_coverage(trial_covered)
            shell_coverage = calculate_mask_coverage(trial_covered, shell_mask)
            if (
                volume_coverage + 1e-12 >= required_volume_coverage
                and (
                    not shell_guard_enabled
                    or shell_coverage + 1e-12 >= required_shell_coverage
                )
            ):
                active = trial
                removed = True
                break
        if not removed:
            break
    return active


def prune_overlapping_static_spheres(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    inside_epsilon_m: float,
    target_coverage: float,
    coverage_tolerance_m: float,
    enable_surface_shell_guard: bool,
    surface_shell_thickness_m: float,
    target_shell_coverage: float,
    shell_coverage_loss_tolerance: float,
    max_optimization_matrix_elements: int,
    max_overlap_fraction: float,
    max_spheres: int,
    max_removals: int,
) -> tuple[List[Sphere], int]:
    """Remove highly overlapping final spheres without losing coverage.

    This bounded pass runs after merging.  It reuses one sphere-by-voxel
    matrix, preserves both volume and surface-shell coverage, and only
    considers spheres participating in a pair above ``max_overlap_fraction``.
    The pass is skipped for unexpectedly large sets instead of adding an
    unbounded quadratic workload.
    """

    if not 0.0 <= max_overlap_fraction <= 1.0:
        raise ValueError("post-merge overlap fraction must be in [0, 1]")
    if max_spheres <= 0 or max_removals < 0:
        raise ValueError(
            "post-merge sphere cap must be positive and removal cap non-negative")
    active_spheres = list(spheres)
    if (
        len(active_spheres) < 2
        or len(active_spheres) > max_spheres
        or max_removals == 0
    ):
        return active_spheres, 0

    indices = _as_index_array(component_indices)
    coverage_masks = sphere_coverage_masks(
        indices,
        active_spheres,
        origin_m,
        voxel_size_m,
        coverage_tolerance_m,
        max_optimization_matrix_elements,
    )
    if coverage_masks is None:
        return active_spheres, 0

    shell_mask = make_surface_shell_mask(
        esdf_grid,
        indices,
        inside_epsilon_m,
        surface_shell_thickness_m,
    )
    shell_guard_enabled = (
        enable_surface_shell_guard
        and surface_shell_thickness_m > 0.0
        and bool(np.any(shell_mask))
    )
    baseline_covered = np.any(coverage_masks, axis=0)
    baseline_volume = calculate_mask_coverage(baseline_covered)
    baseline_shell = calculate_mask_coverage(baseline_covered, shell_mask)
    required_volume = min(float(target_coverage), baseline_volume)
    required_shell = max(
        0.0,
        min(float(target_shell_coverage), baseline_shell)
        - float(shell_coverage_loss_tolerance),
    )

    active = list(range(len(active_spheres)))
    removed_count = 0
    while len(active) > 1 and removed_count < max_removals:
        coverage_count = np.count_nonzero(coverage_masks[active], axis=0)
        unique_volume = {
            index: int(np.count_nonzero(
                coverage_masks[index] & (coverage_count == 1)))
            for index in active
        }
        unique_shell = {
            index: int(np.count_nonzero(
                coverage_masks[index] & shell_mask & (coverage_count == 1)))
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
            unique_shell[index],
            unique_volume[index],
            -active_spheres[index].output_radius,
            active_spheres[index].source_index,
            index,
        ))
        removed = False
        for index in removal_order:
            trial = [candidate for candidate in active if candidate != index]
            covered = np.any(coverage_masks[trial], axis=0)
            if (
                calculate_mask_coverage(covered) + 1e-12 >= required_volume
                and (
                    not shell_guard_enabled
                    or calculate_mask_coverage(covered, shell_mask) + 1e-12
                    >= required_shell
                )
            ):
                active = trial
                removed_count += 1
                removed = True
                break
        if not removed:
            break

    return [active_spheres[index] for index in active], removed_count


def _sphere_is_esdf_valid(
    sphere: Sphere,
    esdf_grid: np.ndarray,
    component_index_set: set[Index3],
    origin_m: np.ndarray,
    voxel_size_m: float,
    inside_epsilon_m: float,
    component_id: int,
) -> bool:
    source_index = tuple(int(value) for value in sphere.source_index)
    if source_index not in component_index_set or sphere.component_id != component_id:
        return False
    distance = float(esdf_grid[source_index])
    expected_center = origin_m + (
        np.asarray(source_index, dtype=np.float64) + 0.5
    ) * voxel_size_m
    return (
        np.isfinite(distance)
        and distance < -inside_epsilon_m
        and sphere.raw_radius <= -distance + 1e-12
        and np.array_equal(np.asarray(sphere.center), expected_center)
    )


def optimize_component_spheres(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    candidate_spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    inside_epsilon_m: float,
    target_coverage: float,
    coverage_tolerance_m: float,
    enable_single_sphere_replacement: bool = True,
    enable_greedy_set_cover: bool = True,
    enable_general_coverage_pruning: bool = True,
    enable_surface_shell_guard: bool = True,
    surface_shell_thickness_m: float = 0.10,
    target_shell_coverage: float = 0.98,
    shell_coverage_loss_tolerance: float = 0.005,
    max_optimization_matrix_elements: int = 20000000,
    component_id: int = 0,
    externally_valid_candidate_mask: np.ndarray | Sequence[bool] | None = None,
) -> Tuple[List[Sphere], float, float, np.ndarray, bool]:
    """Optimize one ESDF-valid pool and report volume/shell coverage."""

    _validate_optimization_parameters(
        surface_shell_thickness_m,
        target_shell_coverage,
        shell_coverage_loss_tolerance,
        max_optimization_matrix_elements,
    )
    indices = _as_index_array(component_indices)
    origin = _as_origin(origin_m)
    candidates = list(candidate_spheres)
    shell_mask = make_surface_shell_mask(
        esdf_grid, indices, inside_epsilon_m, surface_shell_thickness_m
    )
    shell_guard_enabled = (
        enable_surface_shell_guard
        and surface_shell_thickness_m > 0.0
        and bool(np.any(shell_mask))
    )
    coverage_masks = sphere_coverage_masks(
        indices,
        candidates,
        origin,
        voxel_size_m,
        coverage_tolerance_m,
        max_optimization_matrix_elements,
    )
    if coverage_masks is None:
        baseline_volume, covered = calculate_component_coverage(
            indices, candidates, origin, voxel_size_m, coverage_tolerance_m
        )
        baseline_shell = calculate_mask_coverage(covered, shell_mask)
        return candidates, baseline_volume, baseline_shell, covered, False

    baseline_covered = (
        np.any(coverage_masks, axis=0)
        if len(candidates)
        else np.zeros(len(indices), dtype=bool)
    )
    baseline_volume = calculate_mask_coverage(baseline_covered)
    baseline_shell = calculate_mask_coverage(baseline_covered, shell_mask)
    required_shell_coverage = max(
        0.0,
        min(target_shell_coverage, baseline_shell)
        - shell_coverage_loss_tolerance,
    )
    component_index_set = {
        tuple(int(value) for value in index) for index in indices
    }
    valid_candidates = np.asarray([
        _sphere_is_esdf_valid(
            sphere,
            esdf_grid,
            component_index_set,
            origin,
            voxel_size_m,
            inside_epsilon_m,
            component_id,
        )
        for sphere in candidates
    ], dtype=bool)
    if externally_valid_candidate_mask is not None:
        externally_valid = np.asarray(
            externally_valid_candidate_mask, dtype=bool).reshape(-1)
        if len(externally_valid) != len(candidates):
            raise ValueError(
                "external candidate validity must match the candidate count")
        valid_candidates |= externally_valid
    if baseline_volume + 1e-12 < target_coverage or not np.all(valid_candidates):
        return (
            candidates,
            baseline_volume,
            baseline_shell,
            baseline_covered,
            False,
        )

    selected = list(range(len(candidates)))
    if enable_single_sphere_replacement:
        single = select_best_single_sphere(
            candidates,
            coverage_masks,
            shell_mask,
            target_coverage,
            required_shell_coverage,
            shell_guard_enabled,
            valid_candidates,
        )
        if single is not None:
            selected = [single]
    if len(selected) != 1 and enable_greedy_set_cover:
        greedy = greedy_set_cover_spheres(
            candidates,
            coverage_masks,
            shell_mask,
            target_coverage,
            required_shell_coverage,
            shell_guard_enabled,
            valid_candidates,
        )
        if greedy is not None:
            selected = greedy
    if enable_general_coverage_pruning:
        selected = general_coverage_pruning(
            candidates,
            coverage_masks,
            shell_mask,
            selected,
            target_coverage,
            required_shell_coverage,
            shell_guard_enabled,
        )

    final_covered = (
        np.any(coverage_masks[selected], axis=0)
        if selected
        else np.zeros(len(indices), dtype=bool)
    )
    final_volume = calculate_mask_coverage(final_covered)
    final_shell = calculate_mask_coverage(final_covered, shell_mask)
    if (
        final_volume + 1e-12 < target_coverage
        or (
            shell_guard_enabled
            and final_shell + 1e-12 < required_shell_coverage
        )
    ):
        return (
            candidates,
            baseline_volume,
            baseline_shell,
            baseline_covered,
            False,
        )
    return (
        [candidates[index] for index in selected],
        final_volume,
        final_shell,
        final_covered,
        True,
    )


def fibonacci_sphere_surface_points(
    center_m: Sequence[float], radius_m: float, sample_count: int,
) -> np.ndarray:
    """Return deterministic, near-uniform samples on a sphere surface."""

    if sample_count <= 0:
        raise ValueError("merge surface sample count must be positive")
    center = _as_origin(center_m)
    sample = np.arange(sample_count, dtype=np.float64) + 0.5
    y = 1.0 - 2.0 * sample / sample_count
    radial = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    angle = np.pi * (3.0 - np.sqrt(5.0)) * sample
    directions = np.column_stack(
        (radial * np.cos(angle), y, radial * np.sin(angle)))
    return center + float(radius_m) * directions


def _trilinear_observed_esdf_samples(
    esdf_grid: np.ndarray,
    points_m: np.ndarray,
    origin_m: np.ndarray,
    voxel_size_m: float,
    unobserved_distance_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample centre-valued ESDF data and mark outside/unobserved samples."""

    values = np.asarray(esdf_grid, dtype=np.float64)
    points = np.asarray(points_m, dtype=np.float64)
    observed = np.zeros(len(points), dtype=bool)
    samples = np.full(len(points), np.nan, dtype=np.float64)
    coordinates = (points - origin_m) / voxel_size_m - 0.5
    upper = np.asarray(values.shape, dtype=np.float64) - 1.0
    sentinel_tolerance = max(
        1e-9, abs(float(unobserved_distance_value)) * 1e-12)

    for sample_index, coordinate in enumerate(coordinates):
        if np.any(coordinate < -1e-12) or np.any(coordinate > upper + 1e-12):
            continue
        coordinate = np.clip(coordinate, 0.0, upper)
        lower = np.floor(coordinate).astype(np.int64)
        higher = np.minimum(lower + 1, np.asarray(values.shape) - 1)
        fractions = coordinate - lower
        axes = []
        for axis in range(3):
            if lower[axis] == higher[axis]:
                axes.append(((int(lower[axis]), 1.0),))
            else:
                axes.append((
                    (int(lower[axis]), float(1.0 - fractions[axis])),
                    (int(higher[axis]), float(fractions[axis])),
                ))
        weighted_value = 0.0
        valid = True
        for x_index, x_weight in axes[0]:
            for y_index, y_weight in axes[1]:
                for z_index, z_weight in axes[2]:
                    weight = x_weight * y_weight * z_weight
                    if weight <= 0.0:
                        continue
                    corner = float(values[x_index, y_index, z_index])
                    if (
                        not np.isfinite(corner)
                        or abs(corner - unobserved_distance_value)
                        <= sentinel_tolerance
                    ):
                        valid = False
                        break
                    weighted_value += weight * corner
                if not valid:
                    break
            if not valid:
                break
        if valid:
            observed[sample_index] = True
            samples[sample_index] = weighted_value
    return observed, samples


def merged_sphere_passes_esdf_guard(
    esdf_grid: np.ndarray,
    center_m: Sequence[float],
    radius_m: float,
    origin_m: Sequence[float],
    voxel_size_m: float,
    unobserved_distance_value: float,
    max_free_space_distance_m: float,
    surface_sample_count: int,
    min_observed_surface_fraction: float,
) -> bool:
    """Reject a merge whose sampled surface enters too much known free space."""

    points = fibonacci_sphere_surface_points(
        center_m, radius_m, surface_sample_count)
    observed, samples = _trilinear_observed_esdf_samples(
        esdf_grid,
        points,
        _as_origin(origin_m),
        voxel_size_m,
        unobserved_distance_value,
    )
    if float(np.mean(observed)) + 1e-12 < min_observed_surface_fraction:
        return False
    return bool(
        np.any(observed)
        and np.max(samples[observed]) <= max_free_space_distance_m + 1e-12
    )


def create_component_coarse_sphere_candidates(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    seed_spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    safety_margin_m: float,
    unobserved_distance_value: float,
    component_id: int,
    min_component_voxels: int,
    radius_scale: float,
    max_radius_m: float,
    max_empty_fraction: float,
    max_free_space_distance_m: float,
    surface_sample_count: int,
    min_observed_surface_fraction: float,
) -> List[Sphere]:
    """Generate guarded, size-aware candidates for large static components.

    A camera often observes a wide object's surface as a thin negative-ESDF
    shell.  Its medial radii therefore remain close to one voxel even when the
    object's visible extent is large.  This pass enlarges existing valid seed
    centres according to component size, then rejects candidates that contain
    too little component support or extend too far into observed free space.
    The returned candidates are intended for the existing shell-aware greedy
    cover, not for unconditional publication.
    """

    indices = _as_index_array(component_indices)
    if (
        len(indices) < int(min_component_voxels)
        or not seed_spheres
        or radius_scale <= 0.0
    ):
        return []
    if voxel_size_m <= 0.0 or safety_margin_m < 0.0:
        raise ValueError("voxel size must be positive and margin non-negative")
    if min_component_voxels <= 0:
        raise ValueError("coarse component voxel threshold must be positive")
    if max_radius_m <= 0.0 or max_free_space_distance_m < 0.0:
        raise ValueError("coarse radius must be positive and free-space limit non-negative")
    if not 0.0 <= max_empty_fraction <= 1.0:
        raise ValueError("coarse empty fraction must be in [0, 1]")

    radius = min(
        float(max_radius_m),
        float(radius_scale)
        * float(voxel_size_m)
        * float(np.cbrt(len(indices))),
    )
    largest_seed_radius = max(float(sphere.raw_radius) for sphere in seed_spheres)
    if radius <= largest_seed_radius + 1e-12:
        return []

    origin = _as_origin(origin_m)
    points = origin + (indices.astype(np.float64) + 0.5) * voxel_size_m
    # Approximate how many voxel centres a solid sphere of this radius can
    # contain.  This deliberately conservative density check prevents a large
    # candidate from spanning unrelated sparse fragments.
    sphere_voxel_capacity = max(
        1.0,
        (4.0 / 3.0) * np.pi * (radius / voxel_size_m) ** 3,
    )
    candidates: List[Sphere] = []
    for seed in sorted(
        seed_spheres,
        key=lambda sphere: (
            sphere.source_index,
            -sphere.raw_radius,
        ),
    ):
        delta = points - np.asarray(seed.center, dtype=np.float64)
        supported_voxels = int(np.count_nonzero(
            np.einsum("ij,ij->i", delta, delta) <= radius ** 2 + 1e-15))
        empty_fraction = max(
            0.0,
            1.0 - float(supported_voxels) / sphere_voxel_capacity,
        )
        if empty_fraction > max_empty_fraction + 1e-12:
            continue
        candidate = Sphere(
            center=np.asarray(seed.center, dtype=np.float64).copy(),
            raw_radius=radius,
            output_radius=radius + safety_margin_m,
            component_id=int(component_id),
            source_index=seed.source_index,
            is_merged=True,
        )
        if not merged_sphere_passes_esdf_guard(
            esdf_grid,
            candidate.center,
            candidate.raw_radius,
            origin,
            voxel_size_m,
            unobserved_distance_value,
            max_free_space_distance_m,
            surface_sample_count,
            min_observed_surface_fraction,
        ):
            continue
        candidates.append(candidate)
    return candidates


def agglomerative_merge_static_spheres(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    coverage_tolerance_m: float,
    safety_margin_m: float,
    unobserved_distance_value: float = -1000.0,
    merge_max_radius_m: float = 0.35,
    merge_max_radius_growth_ratio: float = 1.45,
    merge_max_gap_m: float = 0.05,
    merge_enable_esdf_guard: bool = True,
    merge_max_free_space_distance_m: float = 0.08,
    merge_surface_sample_count: int = 64,
    merge_min_observed_surface_fraction: float = 0.70,
) -> List[Sphere]:
    """Repeatedly replace the least-cost valid pair with its containing sphere."""

    _validate_static_merge_parameters(
        merge_max_radius_m, merge_max_radius_growth_ratio, merge_max_gap_m,
        merge_max_free_space_distance_m, merge_surface_sample_count,
        merge_min_observed_surface_fraction)
    active = list(spheres)
    indices = _as_index_array(component_indices)
    origin = _as_origin(origin_m)

    while len(active) >= 2:
        _, baseline_covered = calculate_component_coverage(
            indices, active, origin, voxel_size_m, coverage_tolerance_m)

        def validator(candidate: PairMergeCandidate) -> bool:
            first = active[candidate.first_index]
            second = active[candidate.second_index]
            merged = Sphere(
                center=candidate.center,
                raw_radius=candidate.radius,
                output_radius=candidate.radius + safety_margin_m,
                component_id=first.component_id,
                source_index=min(first.source_index, second.source_index),
                is_merged=True,
            )
            trial = [
                sphere for index, sphere in enumerate(active)
                if index not in (candidate.first_index, candidate.second_index)
            ] + [merged]
            _, covered = calculate_component_coverage(
                indices, trial, origin, voxel_size_m, coverage_tolerance_m)
            if np.count_nonzero(covered) < np.count_nonzero(baseline_covered):
                return False
            return (
                not merge_enable_esdf_guard
                or merged_sphere_passes_esdf_guard(
                    esdf_grid,
                    merged.center,
                    merged.raw_radius,
                    origin,
                    voxel_size_m,
                    unobserved_distance_value,
                    merge_max_free_space_distance_m,
                    merge_surface_sample_count,
                    merge_min_observed_surface_fraction,
                )
            )

        candidate = select_best_merge_pair(
            np.asarray([sphere.center for sphere in active]),
            np.asarray([sphere.raw_radius for sphere in active]),
            np.asarray([sphere.component_id for sphere in active]),
            merge_max_radius_m,
            merge_max_radius_growth_ratio,
            merge_max_gap_m,
            validator,
        )
        if candidate is None:
            break
        first = active[candidate.first_index]
        second = active[candidate.second_index]
        merged = Sphere(
            candidate.center,
            candidate.radius,
            candidate.radius + safety_margin_m,
            first.component_id,
            min(first.source_index, second.source_index),
            True,
        )
        active = [
            sphere for index, sphere in enumerate(active)
            if index not in (candidate.first_index, candidate.second_index)
        ] + [merged]
        active.sort(key=lambda sphere: (
            sphere.component_id,
            float(sphere.center[0]),
            float(sphere.center[1]),
            float(sphere.center[2]),
            sphere.raw_radius,
            sphere.source_index,
        ))
    return active


def _static_sphere_sort_key(
    sphere: Sphere,
) -> tuple[object, ...]:
    return (
        sphere.component_id,
        float(sphere.center[0]),
        float(sphere.center[1]),
        float(sphere.center[2]),
        sphere.raw_radius,
        sphere.source_index,
    )


def build_static_sphere_search_state(
    spheres: Sequence[Sphere],
) -> StaticSphereSearchState:
    """Build a deterministic, order-independent static merge state."""

    ordered = tuple(sorted(spheres, key=_static_sphere_sort_key))
    radii = np.asarray(
        [sphere.raw_radius for sphere in ordered], dtype=np.float64)
    signature = tuple(
        (
            sphere.component_id,
            round(float(sphere.center[0]), 9),
            round(float(sphere.center[1]), 9),
            round(float(sphere.center[2]), 9),
            round(float(sphere.raw_radius), 9),
            tuple(int(value) for value in sphere.source_index),
        )
        for sphere in ordered
    )
    return StaticSphereSearchState(
        spheres=ordered,
        total_raw_volume=float(
            (4.0 / 3.0) * np.pi * np.sum(radii ** 3)),
        total_raw_radius=float(np.sum(radii)),
        canonical_signature=signature,
    )


def static_sphere_search_state_rank_key(
    state: StaticSphereSearchState,
) -> tuple[object, ...]:
    """Prefer less over-approximation when two valid states have equal K."""

    return (
        state.total_raw_volume,
        state.total_raw_radius,
        state.canonical_signature,
    )


def _merge_static_candidate(
    spheres: Sequence[Sphere],
    candidate: PairMergeCandidate,
    safety_margin_m: float,
) -> List[Sphere]:
    first = spheres[candidate.first_index]
    second = spheres[candidate.second_index]
    merged = Sphere(
        center=np.asarray(candidate.center, dtype=np.float64),
        raw_radius=float(candidate.radius),
        output_radius=float(candidate.radius + safety_margin_m),
        component_id=first.component_id,
        source_index=min(first.source_index, second.source_index),
        is_merged=True,
    )
    active = [
        sphere for index, sphere in enumerate(spheres)
        if index not in (candidate.first_index, candidate.second_index)
    ] + [merged]
    active.sort(key=_static_sphere_sort_key)
    return active


def _enumerate_valid_static_merge_candidates(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    state: StaticSphereSearchState,
    origin_m: Sequence[float],
    voxel_size_m: float,
    coverage_tolerance_m: float,
    safety_margin_m: float,
    unobserved_distance_value: float,
    merge_max_radius_m: float,
    merge_max_radius_growth_ratio: float,
    merge_max_gap_m: float,
    merge_enable_esdf_guard: bool,
    merge_max_free_space_distance_m: float,
    merge_surface_sample_count: int,
    merge_min_observed_surface_fraction: float,
    deadline: float,
) -> List[PairMergeCandidate]:
    active = list(state.spheres)
    _, baseline_covered = calculate_component_coverage(
        component_indices,
        active,
        origin_m,
        voxel_size_m,
        coverage_tolerance_m,
    )
    baseline_count = int(np.count_nonzero(baseline_covered))

    def validator(candidate: PairMergeCandidate) -> bool:
        if monotonic() >= deadline:
            return False
        trial = _merge_static_candidate(active, candidate, safety_margin_m)
        _, covered = calculate_component_coverage(
            component_indices,
            trial,
            origin_m,
            voxel_size_m,
            coverage_tolerance_m,
        )
        if int(np.count_nonzero(covered)) < baseline_count:
            return False
        if not merge_enable_esdf_guard:
            return True
        merged = next(sphere for sphere in trial if sphere.is_merged and (
            np.array_equal(sphere.center, candidate.center)
            and abs(sphere.raw_radius - candidate.radius) <= 1e-12
        ))
        return merged_sphere_passes_esdf_guard(
            esdf_grid,
            merged.center,
            merged.raw_radius,
            origin_m,
            voxel_size_m,
            unobserved_distance_value,
            merge_max_free_space_distance_m,
            merge_surface_sample_count,
            merge_min_observed_surface_fraction,
        )

    return enumerate_pair_merge_candidates(
        np.asarray([sphere.center for sphere in active], dtype=np.float64),
        np.asarray([sphere.raw_radius for sphere in active], dtype=np.float64),
        np.asarray([sphere.component_id for sphere in active], dtype=np.int64),
        merge_max_radius_m,
        merge_max_radius_growth_ratio,
        merge_max_gap_m,
        validator,
        deadline,
    )


def minimum_k_static_sphere_search(
    esdf_grid: np.ndarray,
    component_indices: np.ndarray,
    pre_merge_spheres: Sequence[Sphere],
    origin_m: Sequence[float],
    voxel_size_m: float,
    coverage_tolerance_m: float,
    safety_margin_m: float,
    unobserved_distance_value: float,
    merge_max_radius_m: float,
    merge_max_radius_growth_ratio: float,
    merge_max_gap_m: float,
    merge_enable_esdf_guard: bool,
    merge_max_free_space_distance_m: float,
    merge_surface_sample_count: int,
    merge_min_observed_surface_fraction: float,
    beam_width: int,
    max_states: int,
    deadline: float,
) -> StaticMinimumKSearchResult:
    """Explore valid merge orders and return the minimum K found in bounds."""

    initial = build_static_sphere_search_state(pre_merge_spheres)
    if len(initial.spheres) < 2:
        return StaticMinimumKSearchResult(
            list(initial.spheres), 0, "not_needed", initial)

    best = initial
    seen = {initial.canonical_signature}
    beam = [initial]
    states_explored = 0
    termination = "no_more_valid_merges"

    while beam:
        if monotonic() >= deadline:
            termination = "deadline"
            break
        child_states: dict[
            tuple[tuple[object, ...], ...], StaticSphereSearchState
        ] = {}
        stop_reason = None
        for parent in beam:
            candidates = _enumerate_valid_static_merge_candidates(
                esdf_grid,
                component_indices,
                parent,
                origin_m,
                voxel_size_m,
                coverage_tolerance_m,
                safety_margin_m,
                unobserved_distance_value,
                merge_max_radius_m,
                merge_max_radius_growth_ratio,
                merge_max_gap_m,
                merge_enable_esdf_guard,
                merge_max_free_space_distance_m,
                merge_surface_sample_count,
                merge_min_observed_surface_fraction,
                deadline,
            )
            if monotonic() >= deadline:
                stop_reason = "deadline"
                break
            for candidate in candidates:
                if states_explored >= max_states:
                    stop_reason = "max_states"
                    break
                child = build_static_sphere_search_state(
                    _merge_static_candidate(
                        parent.spheres, candidate, safety_margin_m))
                if child.canonical_signature in seen:
                    continue
                seen.add(child.canonical_signature)
                child_states[child.canonical_signature] = child
                states_explored += 1
                if (
                    len(child.spheres),
                    static_sphere_search_state_rank_key(child),
                ) < (
                    len(best.spheres),
                    static_sphere_search_state_rank_key(best),
                ):
                    best = child
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
            key=lambda state: (
                len(state.spheres),
                static_sphere_search_state_rank_key(state),
            ),
        )[:beam_width]
        if len(best.spheres) == 1:
            termination = "minimum_k_reached"
            break

    return StaticMinimumKSearchResult(
        list(best.spheres), states_explored, termination, best)


def _validate_static_merge_parameters(
    merge_max_radius_m: float,
    merge_max_radius_growth_ratio: float,
    merge_max_gap_m: float,
    merge_max_free_space_distance_m: float,
    merge_surface_sample_count: int,
    merge_min_observed_surface_fraction: float,
) -> None:
    validate_merge_limits(
        merge_max_radius_m, merge_max_radius_growth_ratio, merge_max_gap_m)
    if not np.isfinite((
        merge_max_free_space_distance_m,
        merge_min_observed_surface_fraction,
    )).all():
        raise ValueError("static merge guard parameters must be finite")
    if merge_max_free_space_distance_m < 0.0:
        raise ValueError("merge free-space distance must be non-negative")
    if merge_surface_sample_count <= 0:
        raise ValueError("merge surface sample count must be positive")
    if not 0.0 <= merge_min_observed_surface_fraction <= 1.0:
        raise ValueError("merge observed surface fraction must be in [0, 1]")


def _validate_optimization_parameters(
    surface_shell_thickness_m: float,
    target_shell_coverage: float,
    shell_coverage_loss_tolerance: float,
    max_optimization_matrix_elements: int,
) -> None:
    if surface_shell_thickness_m < 0.0:
        raise ValueError("surface_shell_thickness_m must be non-negative")
    if not 0.0 <= target_shell_coverage <= 1.0:
        raise ValueError("target_shell_coverage must be in [0, 1]")
    if not 0.0 <= shell_coverage_loss_tolerance <= 1.0:
        raise ValueError("shell_coverage_loss_tolerance must be in [0, 1]")
    if max_optimization_matrix_elements <= 0:
        raise ValueError("max_optimization_matrix_elements must be positive")


def _validate_static_min_k_parameters(
    min_k_beam_width: int,
    min_k_max_states: int,
    min_k_processing_budget_ms: float,
) -> None:
    if min_k_beam_width <= 0:
        raise ValueError("minimum-K beam width must be positive")
    if min_k_max_states <= 0:
        raise ValueError("minimum-K state cap must be positive")
    if (
        not np.isfinite(min_k_processing_budget_ms)
        or min_k_processing_budget_ms <= 0.0
    ):
        raise ValueError("minimum-K processing budget must be finite and positive")


def generate_medial_spheres(
    esdf_grid: np.ndarray,
    origin_m: Sequence[float],
    voxel_size_m: float,
    unobserved_distance_value: float = -1000.0,
    inside_epsilon_m: float = 0.005,
    target_coverage: float = 0.95,
    coverage_tolerance_m: float = 0.01,
    plateau_epsilon_m: float = 0.001,
    minimum_center_spacing_m: float = 0.05,
    min_component_voxels: int = 8,
    min_raw_sphere_radius_m: float = 0.01,
    max_raw_sphere_radius_m: float = float("inf"),
    safety_margin_m: float = 0.02,
    redundancy_tolerance_m: float = 0.001,
    max_spheres_per_component: int = 128,
    max_iterations_per_component: int = 256,
    max_total_spheres: int = 256,
    enable_single_sphere_replacement: bool = True,
    enable_greedy_set_cover: bool = True,
    enable_general_coverage_pruning: bool = True,
    enable_surface_shell_guard: bool = True,
    surface_shell_thickness_m: float = 0.10,
    target_shell_coverage: float = 0.98,
    shell_coverage_loss_tolerance: float = 0.005,
    max_optimization_matrix_elements: int = 20000000,
    enable_component_coarse_cover: bool = False,
    component_coarse_min_voxels: int = 80,
    component_coarse_radius_scale: float = 0.45,
    component_coarse_max_radius_m: float = 0.32,
    component_coarse_max_empty_fraction: float = 0.75,
    component_coarse_max_free_space_distance_m: float = 0.15,
    enable_agglomerative_merge: bool = True,
    merge_max_radius_m: float = 0.35,
    merge_max_radius_growth_ratio: float = 1.45,
    merge_max_gap_m: float = 0.05,
    merge_enable_esdf_guard: bool = True,
    merge_max_free_space_distance_m: float = 0.08,
    merge_surface_sample_count: int = 64,
    merge_min_observed_surface_fraction: float = 0.70,
    enable_min_k_search: bool = False,
    min_k_beam_width: int = 8,
    min_k_max_states: int = 128,
    min_k_processing_budget_ms: float = 200.0,
    enable_post_merge_overlap_pruning: bool = False,
    post_merge_max_overlap_fraction: float = 0.20,
    post_merge_pruning_max_spheres: int = 64,
    post_merge_pruning_max_removals: int = 32,
    robot_sphere_centers: np.ndarray | Sequence[Sequence[float]] | None = None,
    robot_sphere_radii: np.ndarray | Sequence[float] | None = None,
    robot_component_overlap_threshold: float = 1.0,
    robot_sphere_margin_m: float = 0.0,
    enable_local_width_cover: bool = False,
    local_width_ratio: float = 1.4,
    local_width_budget_ms: float = 15.0,
) -> GenerationResult:
    """Run the complete component-wise signed-ESDF sphere algorithm.

    The dense grid is ``(x, y, z)`` and all distances are metres. Coverage is
    measured per component using raw radii before applying ``safety_margin_m``.
    Small components are removed before sphere generation. The global cap uses
    per-component preservation and greedy marginal coverage instead of list
    truncation.
    """

    values = np.asarray(esdf_grid, dtype=np.float64)
    if (not np.isfinite(local_width_ratio) or local_width_ratio < 1.
            or not np.isfinite(local_width_budget_ms) or local_width_budget_ms < 0.):
        raise ValueError("invalid static local-width ratio/budget")
    remaining_local_width_ms = local_width_budget_ms
    origin = _as_origin(origin_m)
    if values.ndim != 3 or voxel_size_m <= 0.0:
        raise ValueError("esdf_grid must be 3D and voxel_size_m must be positive")
    if min_component_voxels < 1 or max_total_spheres < 0:
        raise ValueError("min_component_voxels must be positive and total cap non-negative")
    if (
        min_raw_sphere_radius_m < 0.0
        or max_raw_sphere_radius_m < min_raw_sphere_radius_m
    ):
        raise ValueError("invalid static raw-radius limits")
    if not 0.0 <= robot_component_overlap_threshold <= 1.0:
        raise ValueError("robot component overlap threshold must be in [0, 1]")
    if component_coarse_min_voxels <= 0:
        raise ValueError("coarse component voxel threshold must be positive")
    if (
        not np.isfinite(component_coarse_radius_scale)
        or component_coarse_radius_scale < 0.0
    ):
        raise ValueError("coarse radius scale must be finite and non-negative")
    if (
        not np.isfinite(component_coarse_max_radius_m)
        or component_coarse_max_radius_m <= 0.0
    ):
        raise ValueError("coarse maximum radius must be finite and positive")
    if not 0.0 <= component_coarse_max_empty_fraction <= 1.0:
        raise ValueError("coarse empty fraction must be in [0, 1]")
    if (
        not np.isfinite(component_coarse_max_free_space_distance_m)
        or component_coarse_max_free_space_distance_m < 0.0
    ):
        raise ValueError(
            "coarse free-space distance must be finite and non-negative")
    use_robot_filter = (
        robot_sphere_centers is not None or robot_sphere_radii is not None)
    if use_robot_filter and (
        robot_sphere_centers is None or robot_sphere_radii is None
    ):
        raise ValueError("robot sphere centers and radii must be supplied together")
    _validate_optimization_parameters(
        surface_shell_thickness_m,
        target_shell_coverage,
        shell_coverage_loss_tolerance,
        max_optimization_matrix_elements,
    )
    _validate_static_merge_parameters(
        merge_max_radius_m,
        merge_max_radius_growth_ratio,
        merge_max_gap_m,
        merge_max_free_space_distance_m,
        merge_surface_sample_count,
        merge_min_observed_surface_fraction,
    )
    if merge_max_radius_m > max_raw_sphere_radius_m:
        raise ValueError(
            "merge_max_radius_m cannot exceed max_raw_sphere_radius_m")
    _validate_static_min_k_parameters(
        min_k_beam_width,
        min_k_max_states,
        min_k_processing_budget_ms,
    )
    if not 0.0 <= post_merge_max_overlap_fraction <= 1.0:
        raise ValueError("post-merge overlap fraction must be in [0, 1]")
    if post_merge_pruning_max_spheres <= 0:
        raise ValueError("post-merge pruning sphere cap must be positive")
    if post_merge_pruning_max_removals < 0:
        raise ValueError("post-merge pruning removal cap must be non-negative")

    inside_mask = extract_inside_mask(
        values, unobserved_distance_value, inside_epsilon_m
    )
    raw_components = connected_components_18(inside_mask)
    kept_components = [
        indices for indices in raw_components if len(indices) >= min_component_voxels
    ]
    removed_small_components = len(raw_components) - len(kept_components)
    component_labels = np.full(values.shape, -1, dtype=np.int32)
    filtered_inside_mask = inside_mask.copy()
    results: List[ComponentResult] = []
    robot_rejected_component_ids: List[int] = []
    robot_component_overlap_fractions: List[float] = []
    robot_rejected_voxel_indices: List[np.ndarray] = []

    for component_id, component_indices in enumerate(kept_components):
        component_labels[tuple(component_indices.T)] = component_id
        if use_robot_filter:
            overlap_fraction, _ = static_component_robot_overlap_fraction(
                component_indices,
                origin,
                voxel_size_m,
                robot_sphere_centers,
                robot_sphere_radii,
                robot_sphere_margin_m,
            )
            robot_component_overlap_fractions.append(overlap_fraction)
            if overlap_fraction + 1e-12 >= robot_component_overlap_threshold:
                robot_rejected_component_ids.append(component_id)
                robot_rejected_voxel_indices.append(component_indices)
                filtered_inside_mask[tuple(component_indices.T)] = False
                continue
        minima = find_local_minimum_candidates(values, component_indices)
        representatives = group_or_reduce_plateaus(
            values,
            minima,
            voxel_size_m,
            plateau_epsilon_m,
            minimum_center_spacing_m,
        )
        initial_spheres = create_initial_spheres(
            values,
            representatives,
            origin,
            voxel_size_m,
            min_raw_sphere_radius_m,
            safety_margin_m,
            component_id,
            max_raw_sphere_radius_m,
        )
        spheres, coverage, reason, uncovered = add_spheres_until_coverage(
            values,
            component_indices,
            initial_spheres,
            origin,
            voxel_size_m,
            target_coverage,
            coverage_tolerance_m,
            minimum_center_spacing_m,
            min_raw_sphere_radius_m,
            safety_margin_m,
            max_spheres_per_component,
            max_iterations_per_component,
            component_id,
            max_raw_sphere_radius_m,
        )
        base_candidate_pool = list(spheres)
        optimization_enabled = (
            enable_single_sphere_replacement
            or enable_greedy_set_cover
            or enable_general_coverage_pruning
        )
        coarse_candidates: List[Sphere] = []
        if (
            enable_component_coarse_cover
            and enable_greedy_set_cover
            and optimization_enabled
        ):
            coarse_candidates = create_component_coarse_sphere_candidates(
                values,
                component_indices,
                base_candidate_pool,
                origin,
                voxel_size_m,
                safety_margin_m,
                unobserved_distance_value,
                component_id,
                component_coarse_min_voxels,
                component_coarse_radius_scale,
                min(component_coarse_max_radius_m, max_raw_sphere_radius_m),
                component_coarse_max_empty_fraction,
                component_coarse_max_free_space_distance_m,
                merge_surface_sample_count,
                merge_min_observed_surface_fraction,
            )
        candidate_pool = base_candidate_pool + coarse_candidates
        if (
            len(candidate_pool) * len(component_indices)
            > max_optimization_matrix_elements
        ):
            coarse_candidates = []
            candidate_pool = base_candidate_pool
        matrix_too_large = (
            len(candidate_pool) * len(component_indices)
            > max_optimization_matrix_elements
        )
        if not optimization_enabled or matrix_too_large:
            spheres, coverage = remove_redundant_spheres(
                component_indices,
                candidate_pool,
                origin,
                voxel_size_m,
                target_coverage,
                coverage_tolerance_m,
                redundancy_tolerance_m,
            )
            uncovered = find_uncovered_voxels(
                component_indices,
                spheres,
                origin,
                voxel_size_m,
                coverage_tolerance_m,
            )
        else:
            optimized_spheres, coverage, _, covered, optimization_applied = (
                optimize_component_spheres(
                values,
                component_indices,
                candidate_pool,
                origin,
                voxel_size_m,
                inside_epsilon_m,
                target_coverage,
                coverage_tolerance_m,
                enable_single_sphere_replacement,
                enable_greedy_set_cover,
                enable_general_coverage_pruning,
                enable_surface_shell_guard,
                surface_shell_thickness_m,
                target_shell_coverage,
                shell_coverage_loss_tolerance,
                max_optimization_matrix_elements,
                component_id,
                externally_valid_candidate_mask=(
                    [False] * len(base_candidate_pool)
                    + [True] * len(coarse_candidates)
                ),
            ))
            if coarse_candidates and not optimization_applied:
                spheres, coverage = remove_redundant_spheres(
                    component_indices,
                    base_candidate_pool,
                    origin,
                    voxel_size_m,
                    target_coverage,
                    coverage_tolerance_m,
                    redundancy_tolerance_m,
                )
                uncovered = find_uncovered_voxels(
                    component_indices,
                    spheres,
                    origin,
                    voxel_size_m,
                    coverage_tolerance_m,
                )
            else:
                spheres = optimized_spheres
                uncovered = component_indices[~covered]
        coarse_selected_count = int(sum(sphere.is_merged for sphere in spheres))
        pre_merge_spheres = list(spheres)
        pre_merge_sphere_count = len(pre_merge_spheres)
        agglomerative_spheres = list(pre_merge_spheres)
        if enable_agglomerative_merge and len(pre_merge_spheres) >= 2:
            agglomerative_spheres = agglomerative_merge_static_spheres(
                values,
                component_indices,
                pre_merge_spheres,
                origin,
                voxel_size_m,
                coverage_tolerance_m,
                safety_margin_m,
                unobserved_distance_value,
                merge_max_radius_m,
                merge_max_radius_growth_ratio,
                merge_max_gap_m,
                merge_enable_esdf_guard,
                merge_max_free_space_distance_m,
                merge_surface_sample_count,
                merge_min_observed_surface_fraction,
            )
        spheres = agglomerative_spheres
        agglomerative_sphere_count = len(agglomerative_spheres)
        min_k_search_applied = enable_min_k_search and len(pre_merge_spheres) >= 2
        min_k_states_explored = 0
        min_k_search_termination = (
            "not_needed" if enable_min_k_search else "disabled")
        min_k_sphere_count = pre_merge_sphere_count
        if min_k_search_applied:
            search_result = minimum_k_static_sphere_search(
                values,
                component_indices,
                pre_merge_spheres,
                origin,
                voxel_size_m,
                coverage_tolerance_m,
                safety_margin_m,
                unobserved_distance_value,
                merge_max_radius_m,
                merge_max_radius_growth_ratio,
                merge_max_gap_m,
                merge_enable_esdf_guard,
                merge_max_free_space_distance_m,
                merge_surface_sample_count,
                merge_min_observed_surface_fraction,
                min_k_beam_width,
                min_k_max_states,
                monotonic() + min_k_processing_budget_ms / 1000.0,
            )
            min_k_states_explored = search_result.states_explored
            min_k_search_termination = search_result.termination_reason
            min_k_sphere_count = len(search_result.spheres)
            baseline_state = build_static_sphere_search_state(
                agglomerative_spheres)
            selected_state = min(
                (baseline_state, search_result.state),
                key=lambda state: (
                    len(state.spheres),
                    static_sphere_search_state_rank_key(state),
                ),
            )
            spheres = list(selected_state.spheres)
            if (
                selected_state.canonical_signature
                == baseline_state.canonical_signature
                and selected_state.canonical_signature
                != search_result.state.canonical_signature
                and min_k_search_termination not in ("deadline", "max_states")
            ):
                min_k_search_termination = "fallback_agglomerative"
        post_overlap_removed_count = 0
        if enable_post_merge_overlap_pruning:
            spheres, post_overlap_removed_count = (
                prune_overlapping_static_spheres(
                    values,
                    component_indices,
                    spheres,
                    origin,
                    voxel_size_m,
                    inside_epsilon_m,
                    target_coverage,
                    coverage_tolerance_m,
                    enable_surface_shell_guard,
                    surface_shell_thickness_m,
                    target_shell_coverage,
                    shell_coverage_loss_tolerance,
                    max_optimization_matrix_elements,
                    post_merge_max_overlap_fraction,
                    post_merge_pruning_max_spheres,
                    post_merge_pruning_max_removals,
                )
            )
        local_saved, local_elapsed = 0, 0.
        local_reason = "not_needed_or_limit" if enable_local_width_cover else "disabled"
        local_radius_cap = min(max_raw_sphere_radius_m, component_coarse_max_radius_m, merge_max_radius_m)
        if (enable_local_width_cover and 2 <= len(spheres) <= 64
                and len(component_indices) <= 8192 and remaining_local_width_ms > 0.
                and local_radius_cap >= min_raw_sphere_radius_m):
            local_start = monotonic()
            points = origin + (component_indices.astype(float) + .5) * voxel_size_m
            density = VoxelSupportGuard(points, origin, voxel_size_m,
                component_coarse_max_empty_fraction)

            def valid_local(center, radius, deadline):
                if not density(center, radius, deadline):
                    return False
                return merged_sphere_passes_esdf_guard(values, center, radius,
                    origin, voxel_size_m, unobserved_distance_value,
                    min(component_coarse_max_free_space_distance_m, merge_max_free_space_distance_m),
                    merge_surface_sample_count, merge_min_observed_surface_fraction)

            local = local_width_cover(points, [s.center for s in spheres],
                [s.raw_radius for s in spheres], voxel_size=voxel_size_m,
                min_radius=min_raw_sphere_radius_m,
                max_radius=local_radius_cap,
                tolerance=coverage_tolerance_m, target_coverage=target_coverage,
                width_ratio=local_width_ratio,
                budget_ms=max(0., remaining_local_width_ms - (monotonic() - local_start) * 1000.),
                validator=valid_local)
            local_reason = local.reason
            if local.applied:
                trial = [spheres[index] if index < len(spheres) else Sphere(
                    center.copy(), float(radius), float(radius) + safety_margin_m,
                    component_id, tuple(component_indices[np.argmin(np.sum((points - center)**2, axis=1))]),
                    is_merged=True) for index, center, radius in zip(local.selected, local.centers, local.radii)]
                # Compare actual pairwise output-volume overlap, not just K.
                def total_overlap(items):
                    return sum(sphere_overlap_fraction(a.center, a.output_radius, b.center, b.output_radius)
                               for i, a in enumerate(items) for b in items[i + 1:])
                # Recheck with the source's precise coverage convention
                # (including its 1e-15 squared-distance roundoff allowance).
                _, old_support = calculate_component_coverage(component_indices, spheres,
                    origin, voxel_size_m, coverage_tolerance_m)
                _, new_support = calculate_component_coverage(component_indices, trial,
                    origin, voxel_size_m, coverage_tolerance_m)
                if not np.all(new_support[old_support]):
                    local_reason = "coverage_rejected"
                elif total_overlap(trial) <= total_overlap(spheres) + 1e-12:
                    local_saved = len(spheres) - len(trial)
                    spheres = trial
                else:
                    local_reason = "overlap_rejected"
            local_elapsed = (monotonic() - local_start) * 1000.
            remaining_local_width_ms = max(0., remaining_local_width_ms - local_elapsed)
        coverage, covered = calculate_component_coverage(
            component_indices,
            spheres,
            origin,
            voxel_size_m,
            coverage_tolerance_m,
        )
        uncovered = component_indices[~covered]
        results.append(
            ComponentResult(
                component_id=component_id,
                voxel_indices=component_indices,
                spheres=spheres,
                coverage=coverage,
                termination_reason=reason,
                uncovered_indices=uncovered,
                pre_merge_sphere_count=pre_merge_sphere_count,
                agglomerative_sphere_count=agglomerative_sphere_count,
                min_k_sphere_count=min_k_sphere_count,
                final_sphere_count=len(spheres),
                min_k_search_applied=min_k_search_applied,
                min_k_states_explored=min_k_states_explored,
                min_k_search_termination=min_k_search_termination,
                post_overlap_sphere_count=len(spheres),
                post_overlap_removed_count=post_overlap_removed_count,
                coarse_candidate_count=len(coarse_candidates),
                coarse_selected_count=coarse_selected_count,
                local_width_saved_spheres=local_saved,
                local_width_elapsed_ms=local_elapsed,
                local_width_reason=local_reason,
            )
        )

    total_limit_applied = sum(len(result.spheres) for result in results) > max_total_spheres
    coverage_lost_component_ids: List[int] = []
    if total_limit_applied:
        _apply_total_sphere_limit(
            results,
            max_total_spheres,
            origin,
            voxel_size_m,
            coverage_tolerance_m,
            target_coverage,
        )
        coverage_lost_component_ids = [
            result.component_id
            for result in results
            if result.coverage + 1e-12 < target_coverage
        ]
    spheres = [sphere for result in results for sphere in result.spheres]
    rejected_indices = (
        np.vstack(robot_rejected_voxel_indices)
        if robot_rejected_voxel_indices
        else np.empty((0, 3), dtype=np.int64)
    )
    return GenerationResult(
        inside_mask=filtered_inside_mask,
        component_labels=component_labels,
        components=results,
        spheres=spheres,
        removed_small_components=removed_small_components,
        total_limit_applied=total_limit_applied,
        coverage_lost_component_ids=coverage_lost_component_ids,
        input_component_count=len(kept_components),
        robot_rejected_component_count=len(robot_rejected_component_ids),
        robot_rejected_voxel_count=len(rejected_indices),
        robot_rejected_component_ids=robot_rejected_component_ids,
        robot_component_overlap_fractions=robot_component_overlap_fractions,
        robot_rejected_voxel_indices=rejected_indices,
    )


def _apply_total_sphere_limit(
    results: List[ComponentResult],
    max_total_spheres: int,
    origin_m: np.ndarray,
    voxel_size_m: float,
    coverage_tolerance_m: float,
    target_coverage: float,
) -> None:
    """Mutate component results using deterministic marginal-coverage ranking."""

    nonempty = [result for result in results if result.spheres]
    selected = {result.component_id: [] for result in results}
    component_order = sorted(
        nonempty, key=lambda result: (-len(result.voxel_indices), result.component_id)
    )

    # Preserve one useful sphere per component whenever the global cap permits.
    for result in component_order[:max_total_spheres]:
        best = max(
            result.spheres,
            key=lambda sphere: (
                _coverage_with(
                    result.voxel_indices,
                    [sphere],
                    origin_m,
                    voxel_size_m,
                    coverage_tolerance_m,
                ),
                sphere.raw_radius,
                tuple(-value for value in sphere.source_index),
            ),
        )
        selected[result.component_id].append(best)

    slots = max_total_spheres - sum(len(items) for items in selected.values())
    while slots > 0:
        best_choice = None
        for result in results:
            chosen = selected[result.component_id]
            base_coverage = _coverage_with(
                result.voxel_indices,
                chosen,
                origin_m,
                voxel_size_m,
                coverage_tolerance_m,
            )
            for sphere in result.spheres:
                if any(id(existing) == id(sphere) for existing in chosen):
                    continue
                new_coverage = _coverage_with(
                    result.voxel_indices,
                    chosen + [sphere],
                    origin_m,
                    voxel_size_m,
                    coverage_tolerance_m,
                )
                score = (
                    new_coverage - base_coverage,
                    sphere.raw_radius,
                    len(result.voxel_indices),
                    -result.component_id,
                    tuple(-value for value in sphere.source_index),
                )
                if best_choice is None or score > best_choice[0]:
                    best_choice = (score, result.component_id, sphere)
        if best_choice is None:
            break
        selected[best_choice[1]].append(best_choice[2])
        slots -= 1

    for result in results:
        chosen_ids = {id(sphere) for sphere in selected[result.component_id]}
        result.spheres = [
            sphere for sphere in result.spheres if id(sphere) in chosen_ids
        ]
        result.coverage, covered = calculate_component_coverage(
            result.voxel_indices,
            result.spheres,
            origin_m,
            voxel_size_m,
            coverage_tolerance_m,
        )
        result.uncovered_indices = result.voxel_indices[~covered]
        result.final_sphere_count = len(result.spheres)
        if result.coverage + 1e-12 < target_coverage:
            result.termination_reason = "max_total_spheres"


def _coverage_with(
    component_indices: np.ndarray,
    spheres: Sequence[Sphere],
    origin_m: np.ndarray,
    voxel_size_m: float,
    coverage_tolerance_m: float,
) -> float:
    return calculate_component_coverage(
        component_indices,
        spheres,
        origin_m,
        voxel_size_m,
        coverage_tolerance_m,
    )[0]


def _as_index_array(indices: np.ndarray) -> np.ndarray:
    result = np.asarray(indices, dtype=np.int64)
    if result.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("voxel indices must have shape (N, 3)")
    return result


def _as_origin(origin_m: Sequence[float]) -> np.ndarray:
    result = np.asarray(origin_m, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("origin_m must contain three finite values")
    return result
