"""Pure helpers for depth-aware robot self filtering."""

from array import array

import numpy as np


def as_ros_image_data(image):
    """
    Return contiguous bytes through the ROS ``uint8[]`` fast path.

    Assigning a Python ``bytes`` object to a generated ROS 2 ``uint8[]``
    field makes the Python message setter validate every byte twice.  An
    ``array('B')`` is accepted directly and avoids millions of Python-level
    checks for each depth frame.
    """
    contiguous = np.ascontiguousarray(image)
    return array("B", contiguous.tobytes(order="C"))


def predict_sphere_surface_depth(
    image_shape,
    centers,
    radii,
    fx,
    fy,
    cx,
    cy,
    min_depth_m=0.05,
    max_depth_m=5.0,
):
    """
    Render the nearest optical-Z intersection of robot spheres per pixel.

    Depth images report optical-axis Z rather than Euclidean ray length.  Rays
    are therefore parameterized as ``[(u-cx)/fx * z, (v-cy)/fy * z, z]`` so
    the quadratic roots can be compared directly with the camera depth image.
    Pixels with no robot intersection are returned as ``np.inf``.
    """
    height, width = (int(image_shape[0]), int(image_shape[1]))
    predicted = np.full((height, width), np.inf, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if height <= 0 or width <= 0 or len(centers) == 0:
        return predicted
    if len(centers) != len(radii):
        raise ValueError("centers and radii must contain the same number of spheres")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    min_depth_m = max(float(min_depth_m), 1e-6)
    max_depth_m = float(max_depth_m)

    for center, radius in zip(centers, radii):
        if not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0.0:
            continue
        sphere_x, sphere_y, sphere_z = center
        if sphere_z + radius < min_depth_m or sphere_z - radius > max_depth_m:
            continue

        # Project a conservative axis-aligned bounding box for this sphere.
        # Testing the exact ray/sphere intersection inside the ROI removes any
        # false positives introduced by this coarse projection.
        z_near = max(sphere_z - radius, min_depth_m)
        z_far = max(sphere_z + radius, min_depth_m)
        projected_u = [
            fx * x_value / z_value + cx
            for x_value in (sphere_x - radius, sphere_x + radius)
            for z_value in (z_near, z_far)
        ]
        projected_v = [
            fy * y_value / z_value + cy
            for y_value in (sphere_y - radius, sphere_y + radius)
            for z_value in (z_near, z_far)
        ]
        u_min = max(0, int(np.floor(min(projected_u))) - 1)
        u_max = min(width, int(np.ceil(max(projected_u))) + 2)
        v_min = max(0, int(np.floor(min(projected_v))) - 1)
        v_max = min(height, int(np.ceil(max(projected_v))) + 2)
        if u_min >= u_max or v_min >= v_max:
            continue

        ray_x = (np.arange(u_min, u_max, dtype=np.float64) - cx) / fx
        ray_y = (np.arange(v_min, v_max, dtype=np.float64) - cy) / fy
        ray_x = ray_x[None, :]
        ray_y = ray_y[:, None]
        quadratic_a = ray_x * ray_x + ray_y * ray_y + 1.0
        ray_dot_center = (
            ray_x * sphere_x + ray_y * sphere_y + sphere_z
        )
        sphere_term = float(np.dot(center, center) - radius * radius)
        discriminant = ray_dot_center * ray_dot_center - quadratic_a * sphere_term
        intersects = discriminant >= 0.0
        if not np.any(intersects):
            continue

        sqrt_discriminant = np.sqrt(np.maximum(discriminant, 0.0))
        front_z = (ray_dot_center - sqrt_discriminant) / quadratic_a
        back_z = (ray_dot_center + sqrt_discriminant) / quadratic_a
        intersection_z = np.where(front_z >= min_depth_m, front_z, back_z)
        valid = (
            intersects
            & (intersection_z >= min_depth_m)
            & (intersection_z <= max_depth_m)
        )
        roi = predicted[v_min:v_max, u_min:u_max]
        candidate = np.where(valid, intersection_z, np.inf).astype(np.float32)
        np.minimum(roi, candidate, out=roi)

    return predicted


def surface_depth_removal_mask(
    measured_depth_m,
    predicted_robot_depth_m,
    front_tolerance_m,
    back_tolerance_m,
    min_depth_m=0.05,
    max_depth_m=5.0,
    mask_shadow_behind_robot=False,
):
    """
    Classify measured depth that belongs to the predicted robot surface.

    Measurements clearly in front of the predicted surface are preserved.  A
    configurable asymmetric band accounts for calibration and timing errors.
    Measurements behind the predicted robot can optionally be invalidated as
    physically occluded shadow data.
    """
    measured = np.asarray(measured_depth_m)
    predicted = np.asarray(predicted_robot_depth_m)
    if measured.shape != predicted.shape:
        raise ValueError("measured and predicted depth images must have equal shape")

    valid_measured = (
        np.isfinite(measured)
        & (measured >= float(min_depth_m))
        & (measured <= float(max_depth_m))
    )
    has_robot_surface = np.isfinite(predicted)
    delta = measured - predicted
    remove = (
        valid_measured
        & has_robot_surface
        & (delta >= -max(0.0, float(front_tolerance_m)))
        & (delta <= max(0.0, float(back_tolerance_m)))
    )
    if mask_shadow_behind_robot:
        remove |= (
            valid_measured
            & has_robot_surface
            & (delta > max(0.0, float(back_tolerance_m)))
        )
    return remove
