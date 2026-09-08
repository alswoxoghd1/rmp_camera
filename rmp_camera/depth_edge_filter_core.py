"""Single-frame rejection of weakly supported intermediate depth measurements.

No depth is synthesized and no frame history, robot pose or semantic mask is
used. A foreground silhouette alone is not grounds for rejection: a candidate
must lie between supported nearer AND farther measurements, with too few
neighbors supporting its own depth. This is a heuristic, not a validity proof.
"""

from dataclasses import dataclass
import math
from numbers import Integral

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthEdgeFilterConfig:
    neighborhood_radius_px: int = 3
    min_depth_jump_m: float = 0.10
    support_tolerance_m: float = 0.04
    min_support_pixels: int = 12
    min_side_pixels: int = 2

    def __post_init__(self):
        for name in ('neighborhood_radius_px', 'min_support_pixels', 'min_side_pixels'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f'{name} must be an integer')
        if not 1 <= self.neighborhood_radius_px <= 8:
            raise ValueError('neighborhood_radius_px must be in [1, 8]')
        neighbors = (2 * self.neighborhood_radius_px + 1) ** 2 - 1
        if not 1 <= self.min_support_pixels <= neighbors:
            raise ValueError('min_support_pixels exceeds the neighborhood')
        if not 1 <= self.min_side_pixels <= neighbors // 2:
            raise ValueError('min_side_pixels exceeds the neighborhood')
        if not (math.isfinite(self.min_depth_jump_m) and self.min_depth_jump_m > 0):
            raise ValueError('min_depth_jump_m must be finite and positive')
        if not (math.isfinite(self.support_tolerance_m)
                and 0 < self.support_tolerance_m < self.min_depth_jump_m):
            raise ValueError('support_tolerance_m must be positive and below min_depth_jump_m')


@dataclass
class DepthEdgeFilterResult:
    rejected: np.ndarray
    valid_pixels: int
    candidate_pixels: int


def depth_edge_rejection_mask(depth_m, config=DepthEdgeFilterConfig()):
    """Return a rejection mask without mutating input; zeros/NaNs are unknown.

    Counts exclude the center pixel. Only candidate pixels gather a full
    neighborhood, keeping the expensive part off uniform foreground surfaces.
    The outer radius-wide image border is retained rather than inventing
    neighbors. All decisions use the ORIGINAL frame, never an iterative mask.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('depth_m must be a two-dimensional optical-Z image')
    valid = np.isfinite(depth) & (depth > 0)
    rejected = np.zeros(depth.shape, dtype=bool)
    valid_count = int(np.count_nonzero(valid))
    r = config.neighborhood_radius_px
    if min(depth.shape) <= 2 * r or not valid_count:
        return DepthEdgeFilterResult(rejected, valid_count, 0)
    kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
    nearest = cv2.erode(np.where(valid, depth, np.inf), kernel)
    farthest = cv2.dilate(np.where(valid, depth, -np.inf), kernel)
    candidates = valid & (depth >= nearest + config.min_depth_jump_m)
    candidates &= depth <= farthest - config.min_depth_jump_m
    candidates[:r] = candidates[-r:] = False
    candidates[:, :r] = candidates[:, -r:] = False
    v, u = np.nonzero(candidates)
    count = len(v)
    if not count:
        return DepthEdgeFilterResult(rejected, valid_count, 0)
    center = depth[v, u]
    indices = v * depth.shape[1] + u
    flat = depth.ravel()
    valid_flat = valid.ravel()
    support = np.zeros(count, dtype=np.uint16)
    near = np.zeros(count, dtype=np.uint16)
    far = np.zeros(count, dtype=np.uint16)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == dy == 0:
                continue
            neighbor_indices = indices + dy * depth.shape[1] + dx
            neighbor = flat[neighbor_indices]
            ok = valid_flat[neighbor_indices]
            delta = neighbor - center
            support += ok & (np.abs(delta) <= config.support_tolerance_m)
            near += ok & (delta <= -config.min_depth_jump_m)
            far += ok & (delta >= config.min_depth_jump_m)
    remove = ((support < config.min_support_pixels)
              & (near >= config.min_side_pixels) & (far >= config.min_side_pixels))
    rejected[v[remove], u[remove]] = True
    return DepthEdgeFilterResult(rejected, valid_count, count)
