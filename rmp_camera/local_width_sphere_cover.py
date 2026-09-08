"""Bounded, surface-aware local-width sphere reconstruction (NumPy only).

PCA partitions are geometric regions, not semantic body parts. The second
principal extent measures visible width rather than mistaking a thin depth
surface for the thickness of the whole object. No unseen volume is inferred.
New geometry is transactional: keep the old cover on timeout, failed guards,
or no count improvement. Never sacrifice an old supported voxel for lower K.
"""
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable

import numpy as np


@dataclass
class LocalWidthResult:
    centers: np.ndarray
    radii: np.ndarray
    applied: bool = False
    reason: str = "not_needed"
    candidates: int = 0
    regions: int = 0
    min_region_coverage: float = 1.0
    elapsed_ms: float = 0.0
    # Selection contains old indices first, then generated candidate indices.
    selected: list[int] = field(default_factory=list)


def coverage_matrix(points, centers, radii, tolerance):
    # Row-wise allocation avoids an N*K*3 intermediate.
    masks = np.empty((len(centers), len(points)), dtype=bool)
    for i, (center, radius) in enumerate(zip(centers, radii)):
        delta = points - center
        masks[i] = np.einsum("ij,ij->i", delta, delta) <= (radius + tolerance) ** 2
    return masks


def visible_width_radius_cap(points, center, radius, *, voxel_size,
                             min_radius, max_radius, tolerance, width_ratio):
    """Current local support width; shared by proposal AND template reuse.

    The input is the entire current robot-filtered component, not a historical
    component voxel count. No current support means no reusable local sphere.
    """
    delta = np.asarray(points) - center
    nearby = delta[np.einsum('ij,ij->i', delta, delta) <= (radius + tolerance + 1e-10) ** 2]
    if not len(nearby):
        return 0.
    if len(nearby) < 3:
        return min(max_radius, max(min_radius, .5 * width_ratio * voxel_size))
    nearby = nearby - np.mean(nearby, axis=0)
    _, axes = np.linalg.eigh(nearby.T @ nearby / len(nearby))
    width = float(np.sort(np.ptp(nearby @ axes, axis=0) + voxel_size)[-2])
    return min(max_radius, max(min_radius, .5 * width_ratio * width))


def local_width_cover(points, old_centers, old_radii, *, voxel_size,
                      min_radius, max_radius, tolerance, target_coverage,
                      width_ratio=1.4, budget_ms=5.0, max_points=8192,
                      max_candidates=64, max_matrix_elements=500000,
                      validator: Callable[[np.ndarray, float, float], bool] | None = None,
                      old_coverage_masks=None):
    """Propose fewer spheres, preserving old support AND regional coverage.

    ``validator(center, radius, deadline)`` supplies source-specific occupancy,
    ESDF and workspace checks; False/timeout never removes legacy geometry.
    Width ratio is radius / visible half-width. Long narrow groups are split
    along their longest principal direction, producing variable-radius chains.
    Regional coverage cannot fall below min(target, previous region coverage);
    this pass doesn't require an incomplete old cover to recover unseen points.
    All caps are per call. Deadlines are soft (one bounded NumPy op may finish).
    """
    started = monotonic()
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    old_centers = np.asarray(old_centers, dtype=float).reshape(-1, 3)
    old_radii = np.asarray(old_radii, dtype=float).reshape(-1)
    out = LocalWidthResult(old_centers, old_radii, selected=list(range(len(old_radii))))

    def finish(reason):
        out.reason = reason
        out.elapsed_ms = (monotonic() - started) * 1000.
        return out

    if (not np.isfinite([voxel_size, min_radius, max_radius, tolerance,
                         target_coverage, width_ratio, budget_ms]).all()
            or voxel_size <= 0 or min_radius < 0 or max_radius < min_radius
            or tolerance < 0 or not 0 <= target_coverage <= 1
            or width_ratio < 1 or budget_ms < 0
            or max_points < 1 or max_candidates < 1 or max_matrix_elements < 1):
        raise ValueError("invalid local-width cover limits")
    if (len(old_centers) != len(old_radii) or not np.isfinite(points).all()
            or not np.isfinite(old_centers).all() or not np.isfinite(old_radii).all()
            or np.any(old_radii < 0)):
        raise ValueError("invalid local-width cover geometry")
    if not len(points) or len(old_radii) < 2:
        return finish("not_needed")
    if (len(points) > max_points
            or (len(old_radii) + max_candidates) * len(points) > max_matrix_elements):
        return finish("size_limit")
    if budget_ms == 0:
        return finish("deadline")
    deadline = started + budget_ms / 1000.
    # Reserve time for coverage proof; don't spend the whole budget on proposals.
    proposal_deadline = started + .65 * budget_ms / 1000.
    candidate_centers, candidate_radii = [], []
    pending = [np.arange(len(points))]
    regions = []
    seen = set()
    nodes = 0
    while pending and nodes < max_candidates and monotonic() < proposal_deadline:
        indices = pending.pop(0)
        local = points[indices]
        mean = np.mean(local, axis=0)
        centered = local - mean
        _, axes = np.linalg.eigh(centered.T @ centered / max(1, len(local)))
        projected = centered @ axes
        lo, hi = np.min(projected, axis=0), np.max(projected, axis=0)
        extents = hi - lo + voxel_size
        width = float(np.sort(extents)[-2])
        radius_cap = min(max_radius, max(min_radius, .5 * width_ratio * width))
        midpoint = mean + axes @ (.5 * (lo + hi))
        # A partial depth surface can have a centroid in unobserved space.
        # Include a real support anchor rather than inventing volume there.
        anchor = local[np.argmin(np.sum(centered ** 2, axis=1))]
        for center in (midpoint, mean, anchor):
            if len(candidate_radii) >= max_candidates or monotonic() >= proposal_deadline:
                break
            distances = np.linalg.norm(local - center, axis=1)
            radius = min(radius_cap, max(min_radius, float(np.max(distances)) - tolerance + 1e-9))
            # A mixed torso+arm partition must not lend its torso width to a
            # sphere centered on the distal arm. Re-estimate width at the
            # actual candidate, with bounded local PCA tightening passes.
            for _ in range(2):
                nearby = local[distances <= radius + tolerance + 1e-10]
                if len(nearby) < 3:
                    break
                delta = nearby - np.mean(nearby, axis=0)
                _, local_axes = np.linalg.eigh(delta.T @ delta / len(nearby))
                local_width = float(np.sort(np.ptp(delta @ local_axes, axis=0) + voxel_size)[-2])
                radius = min(radius, max(min_radius, .5 * width_ratio * local_width))
            # A wide but thin observed shell may fail the volumetric guard at
            # its largest geometric radius. Try smaller supported scales; do
            # not simply reject the entire region or relax the empty guard.
            for scale in (1., .8, .6, .4):
                bounded_radius = max(min_radius, radius * scale)
                key = tuple(np.round(np.r_[center, bounded_radius], 8))
                if key in seen or monotonic() >= proposal_deadline:
                    continue
                seen.add(key)
                if validator is not None and not validator(center, bounded_radius, proposal_deadline):
                    continue
                candidate_centers.append(center)
                candidate_radii.append(bounded_radius)
                break
        nodes += 1
        # Keep subdividing large/elongated regions even when a central sphere
        # was valid: trunk coverage must not hide a missed hand at the boundary.
        if len(local) <= 2 or float(np.max(extents)) <= max(2 * voxel_size, 2 * radius_cap):
            regions.append(indices)
            continue
        axis = int(np.argmax(extents))
        order = np.argsort(projected[:, axis], kind="stable")
        middle = len(indices) // 2
        pending.extend((indices[order[:middle]], indices[order[middle:]]))
    regions.extend(pending)  # Unvisited points still participate in the proof.
    out.regions = len(regions)
    out.candidates = len(candidate_radii)
    if monotonic() >= deadline:
        return finish("deadline")
    if not candidate_radii:
        return finish("no_valid_candidates")
    centers = np.vstack((old_centers, candidate_centers))
    radii = np.r_[old_radii, candidate_radii]
    if old_coverage_masks is None:
        masks = coverage_matrix(points, centers, radii, tolerance)
    else:
        previous_masks = np.asarray(old_coverage_masks, dtype=bool)
        if previous_masks.shape != (len(old_radii), len(points)):
            raise ValueError('cached coverage does not match current candidates/voxels')
        masks = np.vstack((previous_masks, coverage_matrix(points,
            candidate_centers, candidate_radii, tolerance)))
    required = np.any(masks[:len(old_radii)], axis=0)
    regional_required = np.asarray([min(int(np.ceil(target_coverage * len(r) - 1e-10)),
                                       np.count_nonzero(required[r])) for r in regions])
    covered = np.zeros(len(points), dtype=bool)
    selected = []
    available = np.ones(len(radii), dtype=bool)
    # A strictly smaller cover is the only admissible replacement.
    while len(selected) < len(old_radii):
        if monotonic() >= deadline:
            return finish("deadline")
        regional_counts = np.asarray([np.count_nonzero(covered[r]) for r in regions])
        if np.all(covered[required]) and np.all(regional_counts >= regional_required):
            break
        if len(selected) >= len(old_radii) - 1:
            return finish("no_count_improvement")
        # Preserve exact old support first, while giving each deficient region
        # equal weight so a small hand isn't dominated by torso voxel count.
        weight = (required & ~covered).astype(float)
        for region, count, needed in zip(regions, regional_counts, regional_required):
            if count < needed:
                weight[region] += (~covered[region]) * (len(points) / (len(regions) * len(region)))
        gains = masks @ weight
        gains[~available] = -1.
        choices = np.flatnonzero(gains > 0)
        if not len(choices):
            return finish("insufficient_support")
        best = min(choices, key=lambda i: (-gains[i], radii[i], int(i)))
        selected.append(int(best))
        available[best] = False
        covered |= masks[best]
    if monotonic() >= deadline:
        return finish("deadline")
    out.min_region_coverage = min(float(np.mean(covered[r])) for r in regions)
    if (not np.all(covered[required]) or any(np.count_nonzero(covered[r]) < needed
                                         for r, needed in zip(regions, regional_required))):
        return finish("coverage_rejected")
    out.centers, out.radii = centers[selected], radii[selected]
    out.selected = selected
    out.applied = True
    return finish("applied")


class VoxelSupportGuard:
    """Reusable exact lattice occupancy guard for candidate spheres.

    Unlike rebuilding Python tuple sets for every candidate, sorted integer
    voxel keys amortize setup across the bounded proposal pool.
    """
    def __init__(self, points, origin, voxel_size, max_empty_fraction, max_voxels=50000):
        self.origin = np.asarray(origin, dtype=float)
        self.voxel_size = float(voxel_size)
        self.max_empty = float(max_empty_fraction)
        self.max_voxels = int(max_voxels)
        indices = np.rint((np.asarray(points) - self.origin) / voxel_size - .5).astype(np.int64)
        self.lower = indices.min(axis=0)
        self.shape = indices.max(axis=0) - self.lower + 1
        self.keys = np.unique(self._keys(indices))

    def _keys(self, indices):
        local = indices - self.lower
        return (local[:, 0] * self.shape[1] + local[:, 1]) * self.shape[2] + local[:, 2]

    def __call__(self, center, radius, deadline):
        result = self.evaluate(center, radius, deadline)
        return result is not None and result[1] <= self.max_empty + 1e-12

    def support_at_points(self, xyz):
        lattice = np.floor((xyz-self.origin)/self.voxel_size).astype(np.int64)
        bounded = np.all((lattice >= self.lower) & (lattice < self.lower+self.shape), axis=1)
        result = np.zeros(len(xyz),dtype=bool)
        keys = self._keys(lattice[bounded])
        positions = np.searchsorted(self.keys,keys)
        valid = positions < len(self.keys)
        result[np.flatnonzero(bounded)[valid]] = self.keys[positions[valid]] == keys[valid]
        return result

    def evaluate(self, center, radius, deadline):
        if monotonic() >= deadline:
            return None
        low = np.ceil((center - radius - self.origin) / self.voxel_size - .5).astype(np.int64)
        high = np.floor((center + radius - self.origin) / self.voxel_size - .5).astype(np.int64)
        counts = high - low + 1
        if np.any(counts <= 0) or np.prod(counts) > self.max_voxels:
            return None
        lattice = np.indices(tuple(counts)).reshape(3, -1).T + low
        xyz = self.origin + (lattice + .5) * self.voxel_size
        inside = np.einsum("ij,ij->i", xyz - center, xyz - center) <= radius ** 2 + 1e-12
        lattice = lattice[inside]
        if not len(lattice):
            return None
        # Reject out-of-box indices BEFORE flattening to avoid key aliasing.
        bounded = np.all((lattice >= self.lower) & (lattice < self.lower + self.shape), axis=1)
        keys = self._keys(lattice[bounded])
        positions = np.searchsorted(self.keys, keys)
        valid = positions < len(self.keys)
        support = np.count_nonzero(self.keys[positions[valid]] == keys[valid])
        return None if monotonic() >= deadline else (xyz[inside], 1.-support/len(lattice))


class LocalWidthAttemptCache:
    """Short negative-result memoization, NEVER cached obstacle publication.

    On nearly unchanged geometry, skip only an extra optimization that just
    failed to reduce K. The caller still constructs a fresh current cover.
    Major shape changes, splits, count changes and expiry immediately retry.
    """
    def __init__(self, hold_s=.25, min_similarity=.9, capacity=16):
        self.hold_s, self.min_similarity, self.capacity = hold_s, min_similarity, capacity
        self.entries = []

    def clear(self):
        self.entries.clear()

    @staticmethod
    def descriptor(points, voxel_size, origin, count):
        indices = np.rint((points - origin) / voxel_size - .5).astype(np.int64)
        lower = indices.min(axis=0)
        return (count, lower * voxel_size, frozenset(map(tuple, indices-lower)))

    def matches(self, descriptor, now):
        self.entries = [e for e in self.entries if 0 <= now-e[0] < self.hold_s]
        count, position, keys = descriptor
        for _, old_count, old_position, old_keys in self.entries:
            if (count != old_count or np.linalg.norm(position-old_position) > .3
                    or min(len(keys),len(old_keys))/max(len(keys),len(old_keys)) < self.min_similarity):
                continue
            if len(keys & old_keys)/max(len(keys),len(old_keys)) >= self.min_similarity:
                return True
        return False

    def remember(self, descriptor, reason, now):
        if reason not in ('no_count_improvement','no_valid_candidates','overlap_rejected'):
            return
        self.entries.append((now,*descriptor))
        self.entries = self.entries[-self.capacity:]
