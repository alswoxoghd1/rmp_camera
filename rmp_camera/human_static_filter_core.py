"""Bounded human voxel history and conservative depth free-space evidence."""

from dataclasses import dataclass, replace
from itertools import product

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class HumanStaticParameters:
    voxel_size_m: float = 0.05
    match_distance_m: float = 0.08
    surface_back_band_m: float = 0.15
    active_hold_s: float = 0.35
    history_s: float = 5.0
    depth_max_age_s: float = 0.25
    free_clearance_m: float = 0.03
    max_history_voxels: int = 20000
    occlusion_hold_enabled: bool = False
    occlusion_max_depth_m: float = 0.45
    occlusion_match_distance_m: float = 0.05
    occlusion_max_sync_delta_s: float = 0.10
    occlusion_min_valid_fraction: float = 1.0
    occlusion_min_human_fraction: float = 0.60
    ownership_hold_s: float = 2.0

    def __post_init__(self):
        distances = (self.voxel_size_m, self.match_distance_m,
                     self.active_hold_s, self.history_s, self.depth_max_age_s)
        if not all(np.isfinite(x) and x > 0 for x in distances):
            raise ValueError("human static distances and durations must be positive")
        if (self.history_s < self.active_hold_s
                or not np.isfinite(self.free_clearance_m)
                or self.free_clearance_m < 0
                or not np.isfinite(self.surface_back_band_m)
                or not 0 <= self.surface_back_band_m <= 0.30
                or self.max_history_voxels < 1):
            raise ValueError("invalid human static history/band/clearance limits")
        if (not isinstance(self.occlusion_hold_enabled, (bool, np.bool_))
                or not np.isfinite(self.occlusion_max_depth_m)
                or not 0 < self.occlusion_max_depth_m <= 0.5
                or not np.isfinite(self.occlusion_match_distance_m)
                or not 0 < self.occlusion_match_distance_m <= 0.1
                or not np.isfinite(self.occlusion_max_sync_delta_s)
                or not 0 < self.occlusion_max_sync_delta_s <= self.depth_max_age_s
                or not np.isfinite(self.occlusion_min_valid_fraction)
                or not 0.5 <= self.occlusion_min_valid_fraction <= 1.0
                or not np.isfinite(self.occlusion_min_human_fraction)
                or not 0.5 <= self.occlusion_min_human_fraction <= 1.0
                or not np.isfinite(self.ownership_hold_s)
                or not 0 <= self.ownership_hold_s <= 5.0):
            raise ValueError("invalid human static occlusion evidence limits")


def human_occludes_voxels(points, depth_m, intrinsics, camera_from_target,
                         human_points, params, recent_ownership=None):
    """Confirm old HUMAN support, not arbitrary space behind a person.

    The caller must first limit candidates to the bounded human history. Recent
    measured human labels may persist briefly (ownership_hold_s) unless a
    nonhuman surface is actually reobserved in the voxel. This does not assert
    that hidden space is free. Older labels require current human occlusion:
    a voxel's entire projected box, every VALID pixel must either show free
    space beyond the whole voxel or a currently measured human surface. A
    configured small fraction of depth holes is allowed only with majority
    measured human support; holes themselves are never free-space evidence.
    One unmatched observed obstacle preserves the voxel. Human matches
    use unexpanded measured surface points (never the historical back band).
    This is semantic ownership evidence, NOT an ESDF free-space observation.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    human = np.asarray(human_points, dtype=float).reshape(-1, 3)
    owned = np.zeros(len(points), dtype=bool)
    recent = (np.zeros(len(points), dtype=bool) if recent_ownership is None
              else np.asarray(recent_ownership, dtype=bool))
    if not len(points) or (not len(human) and not recent.any()):
        return owned
    transform = np.asarray(camera_from_target)
    human_camera = human @ transform[:3, :3].T + transform[:3, 3]
    human_camera = human_camera[np.all(np.isfinite(human_camera), axis=1)
                               & (human_camera[:, 2] > 0)]
    tree = cKDTree(human_camera) if len(human_camera) else None
    offsets = np.asarray(list(product((-0.5, 0.5), repeat=3))) * params.voxel_size_m
    corners = (points[:, None, :] + offsets) @ transform[:3, :3].T + transform[:3, 3]
    z = corners[:, :, 2]
    lower = corners.min(axis=1)
    upper = corners.max(axis=1)
    front = np.all(np.isfinite(corners), axis=(1, 2)) & np.all(z > 0, axis=1)
    fx, fy, cx, cy = intrinsics
    u = corners[:, :, 0] / np.maximum(z, 1e-9) * fx + cx
    v = corners[:, :, 1] / np.maximum(z, 1e-9) * fy + cy
    h, w = depth_m.shape
    # Overlapping projected voxel boxes share pixels. Query each occupied pixel
    # once, in a single batch, rather than doing a KD-tree call per voxel.
    patches = []
    pixel_tests = 0
    for i in np.flatnonzero(front):
        x0, x1 = int(np.floor(u[i].min())), int(np.ceil(u[i].max()))
        y0, y1 = int(np.floor(v[i].min())), int(np.ceil(v[i].max()))
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            continue
        pixel_tests += (x1 - x0 + 1) * (y1 - y0 + 1)
        if pixel_tests > 131072:
            # Bounded scratch/work for very near or very large components.
            # Unprocessed candidates stay static; never drop them on a budget.
            break
        patch = depth_m[y0:y1+1, x0:x1+1]
        valid = np.isfinite(patch) & (patch > 0)
        if np.mean(valid) < params.occlusion_min_valid_fraction:
            continue
        rear = float(z[i].max())
        if recent[i]:
            # Semantic labels persist briefly through motion/occlusion. Missing
            # mask support is not a new "static object" observation. However,
            # ANY actually observed nonhuman surface in the old voxel (plus
            # clearance) vetoes ownership immediately. All-invalid/mostly
            # missing depth was rejected above, and map distances stay intact.
            possible = (valid & (patch >= lower[i, 2] - params.free_clearance_m)
                        & (patch <= upper[i, 2] + params.free_clearance_m))
            if not possible.any():
                owned[i] = True
                continue
            yy, xx = np.nonzero(possible)
            zz = patch[possible]
            measured = np.column_stack(((xx+x0-cx)*zz/fx, (yy+y0-cy)*zz/fy, zz))
            in_voxel = np.all((measured >= lower[i] - params.free_clearance_m)
                              & (measured <= upper[i] + params.free_clearance_m), axis=1)
            if not in_voxel.any():
                owned[i] = True
                continue
            if tree is None:
                continue
            patches.append((i, (yy[in_voxel]+y0)*w + xx[in_voxel]+x0))
            continue
        occupied = valid & (patch <= rear + params.free_clearance_m)
        if (not occupied.any()
                or (not valid.all() and np.mean(occupied) < params.occlusion_min_human_fraction)
                or np.any(rear - patch[occupied] > params.occlusion_max_depth_m)):
            continue
        yy, xx = np.nonzero(occupied)
        patches.append((i, (yy + y0) * w + xx + x0))
    if not patches or tree is None:
        return owned
    pixels, inverse = np.unique(np.concatenate([p for _, p in patches]), return_inverse=True)
    yy, xx = np.divmod(pixels, w)
    zz = depth_m[yy, xx]
    observed = np.column_stack(((xx - cx) * zz / fx, (yy - cy) * zz / fy, zz))
    matched = tree.query(observed, distance_upper_bound=params.occlusion_match_distance_m)[0]
    matched = np.isfinite(matched)[inverse]
    cursor = 0
    for i, pixels in patches:
        owned[i] = matched[cursor:cursor + len(pixels)].all()
        cursor += len(pixels)
    return owned


def depth_proves_voxels_free(points, depth_m, intrinsics, camera_from_target,
                            voxel_size_m, clearance_m):
    """Require valid background depth across each voxel's entire image box.

    Unknown, out-of-view, invalid and occluded voxels are never free evidence.
    Checking the full projected box also protects thin foreground obstacles
    between the voxel centre and its projected corners.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    free = np.zeros(len(points), dtype=bool)
    if not len(points):
        return free
    transform = np.asarray(camera_from_target)
    offsets = np.asarray(list(product((-0.5, 0.5), repeat=3))) * voxel_size_m
    corners = points[:, None, :] + offsets
    camera = corners @ transform[:3, :3].T + transform[:3, 3]
    z = camera[:, :, 2]
    front = np.all(np.isfinite(camera), axis=(1, 2)) & np.all(z > 0.0, axis=1)
    fx, fy, cx, cy = intrinsics
    safe_z = np.maximum(z, 1e-9)
    u = camera[:, :, 0] / safe_z * fx + cx
    v = camera[:, :, 1] / safe_z * fy + cy
    h, w = depth_m.shape
    for i in np.flatnonzero(front):
        x0, x1 = int(np.floor(u[i].min())), int(np.ceil(u[i].max()))
        y0, y1 = int(np.floor(v[i].min())), int(np.ceil(v[i].max()))
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            continue
        patch = depth_m[y0:y1 + 1, x0:x1 + 1]
        free[i] = (np.all(np.isfinite(patch))
                   and np.min(patch) > np.max(z[i]) + clearance_m)
    return free


class HumanVoxelHistory:
    def __init__(self, params=HumanStaticParameters()):
        self.params = params
        self.voxels = {}
        self.last_stamp = None
        self.version = 0
        self.surface_points = np.empty((0, 3), dtype=float)
        self.last_occluded_count = 0

    def add(self, points, stamp, camera_origin=None):
        # Late out-of-order masks do not rewind history. ROS clock jumps are
        # handled explicitly by the adapter's jump callback.
        if self.last_stamp is not None and stamp < self.last_stamp:
            return
        self.last_stamp = stamp
        p = self.params
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        self.surface_points = points.copy()
        if len(points) and camera_origin is not None and p.surface_back_band_m:
            direction = points - camera_origin
            direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-9)
            offsets = np.arange(0.0, p.surface_back_band_m + 1e-9, p.voxel_size_m)
            points = (points[:, None, :] + offsets[None, :, None]
                      * direction[:, None, :]).reshape(-1, 3)
        keys = np.unique(np.floor(points / p.voxel_size_m).astype(np.int64), axis=0)
        self.voxels.update((tuple(key), stamp) for key in keys)
        self.voxels = {key: seen for key, seen in self.voxels.items()
                       if stamp - seen <= p.history_s}
        if len(self.voxels) > p.max_history_voxels:
            self.voxels = dict(sorted(self.voxels.items(), key=lambda x: x[1])[
                -p.max_history_voxels:])
        self.version += 1

    def clear(self):
        self.voxels.clear()
        self.last_stamp = None
        self.version += 1
        self.surface_points = np.empty((0, 3), dtype=float)
        self.last_occluded_count = 0

    def classify(self, points, now, depth=None, intrinsics=None,
                 camera_from_target=None, depth_stamp=None):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        active = np.zeros(len(points), dtype=bool)
        cleared = active.copy()
        self.last_occluded_count = 0
        if not len(points) or not self.voxels:
            return active, cleared
        keys = np.asarray(list(self.voxels), dtype=float)
        seen = np.asarray(list(self.voxels.values()))
        age = now - seen
        valid = (age >= -0.05) & (age <= self.params.history_s)
        centers = (keys + 0.5) * self.params.voxel_size_m
        current = valid & (age <= self.params.active_hold_s)
        if np.any(current):
            active = cKDTree(centers[current]).query(points)[0] <= self.params.match_distance_m
        candidates = np.zeros(len(points), dtype=bool)
        recent_ownership = np.zeros(len(points), dtype=bool)
        if np.any(valid):
            candidates = cKDTree(centers[valid]).query(points)[0] <= self.params.match_distance_m
        if self.params.occlusion_hold_enabled:
            recent = valid & (age <= self.params.ownership_hold_s)
            if recent.any():
                recent_ownership = cKDTree(centers[recent]).query(points)[0] <= self.params.match_distance_m
        if (depth is not None and depth_stamp is not None
                and -0.05 <= now - depth_stamp <= self.params.depth_max_age_s):
            selected = np.flatnonzero(candidates & ~active)
            cleared[selected] = depth_proves_voxels_free(
                points[selected], depth, intrinsics, camera_from_target,
                self.params.voxel_size_m, self.params.free_clearance_m)
            if (self.params.occlusion_hold_enabled and self.last_stamp is not None
                    and -0.05 <= now - self.last_stamp <= self.params.depth_max_age_s
                    and abs(depth_stamp - self.last_stamp)
                    <= self.params.occlusion_max_sync_delta_s):
                selected = np.flatnonzero(candidates & ~active & ~cleared)
                owned = human_occludes_voxels(points[selected], depth, intrinsics,
                    camera_from_target, self.surface_points, self.params,
                    recent_ownership[selected])
                active[selected[owned]] = True
                self.last_occluded_count = int(owned.sum())
        return active, cleared


def fully_excluded_spheres(sphere_indices, excluded, sphere_count):
    """No supporting voxels or a single retained voxel prevents deletion."""
    indices = np.asarray(sphere_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= sphere_count):
        raise ValueError("support sphere index out of range")
    total = np.bincount(indices, minlength=sphere_count)
    removed = np.bincount(indices[np.asarray(excluded, dtype=bool)], minlength=sphere_count)
    return (total > 0) & (total == removed)


def refit_mixed_static_spheres(spheres, points, sphere_indices, excluded,
                             min_radius_m=0.04, coverage_tolerance_m=0.02):
    """Tighten mixed spheres around ALL retained support, without adding balls.

    This is a bounded four-centre fit, not another Minimum-K solve. A replacement
    must stay inside its original output ball, preserve the original margin and
    cover every retained voxel centre at the existing coverage tolerance. It
    must also cover fewer excluded centres. Otherwise retain the original ball;
    the next ESDF solve can split/rebuild the component with fresh observations.
    Missing support never authorizes deletion. Fully excluded balls are handled
    separately by the caller. No semantic label is interpreted as free space.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    indices = np.asarray(sphere_indices, dtype=np.int64)
    excluded = np.asarray(excluded, dtype=bool)
    if (indices.shape != (len(points),) or excluded.shape != indices.shape
            or not np.isfinite(points).all()
            or np.any(indices < 0) or np.any(indices >= len(spheres))
            or not np.isfinite(min_radius_m) or min_radius_m <= 0
            or not np.isfinite(coverage_tolerance_m) or coverage_tolerance_m < 0):
        raise ValueError("invalid static refit support or radius/tolerance")
    result = list(spheres)
    stats = dict(mixed=0, refitted=0, excluded_support_avoided=0,
                 refit_volume_before_m3=0.0, refit_volume_after_m3=0.0)
    if not excluded.any():
        return result, stats
    for index, sphere in enumerate(spheres):
        selected = indices == index
        retained = points[selected & ~excluded]
        removed = points[selected & excluded]
        if not len(retained) or not len(removed):
            continue
        stats['mixed'] += 1
        original_center = sphere.center
        margin = sphere.output_radius - sphere.raw_radius
        if (margin < 0 or not np.isfinite(original_center).all()
                or not np.isfinite((sphere.raw_radius, sphere.output_radius)).all()
                or sphere.raw_radius <= 0):
            continue
        first = retained[np.argmax(np.linalg.norm(retained - retained[0], axis=1))]
        second = retained[np.argmax(np.linalg.norm(retained - first, axis=1))]
        candidates = (original_center, retained.mean(axis=0),
                      (retained.min(axis=0) + retained.max(axis=0)) * .5,
                      (first + second) * .5)
        best = None
        for center in candidates:
            radius = max(min_radius_m, float(np.linalg.norm(
                retained - center, axis=1).max()) - coverage_tolerance_m)
            output_radius = radius + margin
            # No newly occupied output volume and no larger radius, even if
            # retained points lie near the parent's coverage-tolerance fringe.
            shift = float(np.linalg.norm(center - original_center))
            if shift + output_radius > sphere.output_radius + 1e-9:
                continue
            remaining = int(np.count_nonzero(np.linalg.norm(
                removed - center, axis=1) <= radius + coverage_tolerance_m + 1e-9))
            if remaining >= len(removed):
                continue
            rank = (remaining, radius, shift)
            if best is None or rank < best[0]:
                best = (rank, center, radius, output_radius)
        if best is None:
            continue
        rank, center, radius, output_radius = best
        result[index] = replace(sphere, x=float(center[0]), y=float(center[1]),
            z=float(center[2]), raw_radius=radius, output_radius=output_radius)
        stats['refitted'] += 1
        stats['excluded_support_avoided'] += len(removed) - rank[0]
        stats['refit_volume_before_m3'] += 4 * np.pi / 3 * sphere.output_radius ** 3
        stats['refit_volume_after_m3'] += 4 * np.pi / 3 * output_radius ** 3
    return result, stats


def esdf_proves_voxels_free(points, values, origin, voxel_size, sentinel,
                           clearance=0.03):
    """Require observed positive ESDF beyond the whole voxel's half diagonal."""
    points = np.asarray(points).reshape(-1, 3)
    indices = np.floor((points - origin) / voxel_size).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(values.shape)), axis=1)
    free = np.zeros(len(points), dtype=bool)
    selected = np.flatnonzero(valid)
    distances = values[tuple(indices[selected].T)]
    free[selected] = (np.isfinite(distances)
        & ~np.isclose(distances, sentinel, rtol=0, atol=1e-9)
        & (distances > np.sqrt(3) * voxel_size * .5 + clearance))
    return free
