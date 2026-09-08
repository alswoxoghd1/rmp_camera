"""Geometry helpers for projecting a semantic person mask onto depth."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def filter_small_mask_components(
    mask: np.ndarray,
    *,
    mask_threshold: int = 1,
    min_component_pixels: int = 0,
) -> np.ndarray:
    """Remove isolated semantic-mask regions smaller than a pixel threshold."""

    person_mask = np.asarray(mask)
    if person_mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    minimum = int(min_component_pixels)
    if minimum < 0:
        raise ValueError("min_component_pixels must be non-negative")
    if minimum <= 1:
        return person_mask.copy()
    binary = (person_mask >= int(mask_threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    keep_label = np.zeros(count, dtype=bool)
    if count > 1:
        keep_label[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    return np.where(keep_label[labels], person_mask, 0).astype(
        person_mask.dtype, copy=False)


def foreground_depth_keep_mask(
    projected_u: np.ndarray,
    projected_v: np.ndarray,
    depth_in_mask_frame_m: np.ndarray,
    person_mask: np.ndarray,
    *,
    mask_threshold: int = 1,
    neighborhood_radius_pixels: int = 0,
    max_local_depth_jump_m: float = 0.0,
    max_component_depth_span_m: float = 0.0,
) -> np.ndarray:
    """Reject background depth leaking through a foreground semantic mask.

    Depth and RGB have different optical centers. Near a person silhouette,
    several depth samples can therefore land on mask pixels while belonging
    to a farther wall or table. A local inverse-depth z-buffer removes samples
    behind nearby foreground. The optional component bound also rejects a
    distant depth layer that fills a wider mask hole after temporal skew.
    """

    u = np.asarray(projected_u, dtype=np.int64).reshape(-1)
    v = np.asarray(projected_v, dtype=np.int64).reshape(-1)
    z = np.asarray(depth_in_mask_frame_m, dtype=np.float64).reshape(-1)
    mask = np.asarray(person_mask)
    if mask.ndim != 2:
        raise ValueError("person_mask must be two-dimensional")
    if len(u) != len(v) or len(u) != len(z):
        raise ValueError("projected coordinates and depth must have equal length")
    radius = int(neighborhood_radius_pixels)
    local_jump = float(max_local_depth_jump_m)
    component_span = float(max_component_depth_span_m)
    if radius < 0:
        raise ValueError("neighborhood_radius_pixels must be non-negative")
    if local_jump < 0.0 or component_span < 0.0:
        raise ValueError("foreground depth tolerances must be non-negative")
    if len(z) == 0:
        return np.empty(0, dtype=bool)
    if (
        np.any(u < 0) or np.any(u >= mask.shape[1])
        or np.any(v < 0) or np.any(v >= mask.shape[0])
        or np.any(~np.isfinite(z)) or np.any(z <= 0.0)
    ):
        raise ValueError("projected foreground samples must be finite and in bounds")

    keep = np.ones(len(z), dtype=bool)
    if local_jump > 0.0:
        # The greatest inverse depth is the closest metric depth. maximum.at
        # also provides a deterministic z-buffer where source pixels collide.
        inverse_depth = np.zeros(mask.shape, dtype=np.float32)
        np.maximum.at(
            inverse_depth, (v, u), (1.0 / z).astype(np.float32, copy=False))
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        local_inverse_depth = cv2.dilate(inverse_depth, kernel)
        nearest = 1.0 / np.maximum(local_inverse_depth[v, u], 1e-12)
        keep &= z <= nearest + local_jump

    if component_span > 0.0:
        binary = (mask >= int(mask_threshold)).astype(np.uint8)
        _, labels = cv2.connectedComponents(binary, connectivity=8)
        sample_labels = labels[v, u]
        for label in np.unique(sample_labels):
            if label == 0:
                keep[sample_labels == label] = False
                continue
            members = sample_labels == label
            # A low percentile is more robust than one noisy minimum pixel,
            # while remaining anchored to the visible foreground surface.
            near_depth = float(np.percentile(z[members], 10.0))
            keep[members] &= z[members] <= near_depth + component_span
    return keep


def project_masked_depth(
    depth_m: np.ndarray,
    depth_intrinsics: Sequence[float],
    mask: np.ndarray,
    mask_intrinsics: Sequence[float],
    transform_mask_from_depth: np.ndarray,
    transform_target_from_depth: np.ndarray,
    *,
    stride: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 4.0,
    mask_threshold: int = 1,
    foreground_neighborhood_pixels: int = 0,
    foreground_max_local_depth_jump_m: float = 0.0,
    foreground_max_component_depth_span_m: float = 0.0,
) -> np.ndarray:
    """Back-project depth pixels whose projection falls inside ``mask``.

    Depth and RGB do not need to share a resolution or optical frame.  Both
    transforms use homogeneous 4x4 matrices and the returned points are in the
    target frame.
    """

    depth = np.asarray(depth_m)
    person_mask = np.asarray(mask)
    if depth.ndim != 2 or person_mask.ndim != 2:
        raise ValueError("depth and mask must be two-dimensional")
    if len(depth_intrinsics) != 4 or len(mask_intrinsics) != 4:
        raise ValueError("camera intrinsics must be (fx, fy, cx, cy)")
    transform_mask_from_depth = np.asarray(
        transform_mask_from_depth, dtype=np.float64)
    transform_target_from_depth = np.asarray(
        transform_target_from_depth, dtype=np.float64)
    if transform_mask_from_depth.shape != (4, 4):
        raise ValueError("transform_mask_from_depth must be 4x4")
    if transform_target_from_depth.shape != (4, 4):
        raise ValueError("transform_target_from_depth must be 4x4")
    pixel_stride = max(1, int(stride))
    fx_d, fy_d, cx_d, cy_d = (float(value) for value in depth_intrinsics)
    fx_m, fy_m, cx_m, cy_m = (float(value) for value in mask_intrinsics)
    if min(fx_d, fy_d, fx_m, fy_m) <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    sampled = depth[::pixel_stride, ::pixel_stride].astype(
        np.float64, copy=False)
    valid = (
        np.isfinite(sampled)
        & (sampled > float(min_depth_m))
        & (sampled < float(max_depth_m))
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    sampled_height, sampled_width = sampled.shape
    image_u = np.arange(sampled_width, dtype=np.float64) * pixel_stride
    image_v = np.arange(sampled_height, dtype=np.float64) * pixel_stride
    grid_u, grid_v = np.meshgrid(image_u, image_v)
    z_depth = sampled[valid]
    points_depth = np.column_stack((
        (grid_u[valid] - cx_d) * z_depth / fx_d,
        (grid_v[valid] - cy_d) * z_depth / fy_d,
        z_depth,
    ))

    homogeneous = np.column_stack((
        points_depth, np.ones(len(points_depth), dtype=np.float64)))
    points_mask = (
        transform_mask_from_depth @ homogeneous.T).T[:, :3]
    in_front = points_mask[:, 2] > 1e-6
    if not np.any(in_front):
        return np.empty((0, 3), dtype=np.float64)

    mask_u = np.zeros(len(points_mask), dtype=np.int64)
    mask_v = np.zeros(len(points_mask), dtype=np.int64)
    mask_u[in_front] = np.rint(
        fx_m * points_mask[in_front, 0] / points_mask[in_front, 2] + cx_m
    ).astype(np.int64)
    mask_v[in_front] = np.rint(
        fy_m * points_mask[in_front, 1] / points_mask[in_front, 2] + cy_m
    ).astype(np.int64)
    inside_image = (
        in_front
        & (mask_u >= 0)
        & (mask_u < person_mask.shape[1])
        & (mask_v >= 0)
        & (mask_v < person_mask.shape[0])
    )
    selected = np.zeros(len(points_depth), dtype=bool)
    valid_indices = np.flatnonzero(inside_image)
    selected[valid_indices] = (
        person_mask[mask_v[valid_indices], mask_u[valid_indices]]
        >= int(mask_threshold)
    )
    if not np.any(selected):
        return np.empty((0, 3), dtype=np.float64)

    selected_indices = np.flatnonzero(selected)
    foreground_keep = foreground_depth_keep_mask(
        mask_u[selected_indices],
        mask_v[selected_indices],
        points_mask[selected_indices, 2],
        person_mask,
        mask_threshold=mask_threshold,
        neighborhood_radius_pixels=foreground_neighborhood_pixels,
        max_local_depth_jump_m=foreground_max_local_depth_jump_m,
        max_component_depth_span_m=foreground_max_component_depth_span_m,
    )
    selected[selected_indices[~foreground_keep]] = False
    if not np.any(selected):
        return np.empty((0, 3), dtype=np.float64)

    return (
        transform_target_from_depth @ homogeneous[selected].T
    ).T[:, :3]


def crop_points(
    points: np.ndarray,
    minimum_xyz: Sequence[float],
    maximum_xyz: Sequence[float],
    max_range_xy_m: float = 0.0,
) -> np.ndarray:
    """Crop target-frame points to the configured robot workspace."""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    minimum = np.asarray(minimum_xyz, dtype=np.float64)
    maximum = np.asarray(maximum_xyz, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("crop limits must be three-vectors")
    keep = np.all((values >= minimum) & (values <= maximum), axis=1)
    if max_range_xy_m > 0.0:
        keep &= np.sum(values[:, :2] ** 2, axis=1) <= max_range_xy_m ** 2
    return values[keep]
