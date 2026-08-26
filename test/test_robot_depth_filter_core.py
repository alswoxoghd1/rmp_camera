import numpy as np

from rmp_camera.robot_depth_filter_core import (
    as_ros_image_data,
    predict_sphere_surface_depth,
    surface_depth_removal_mask,
)


def render(centers, radii, shape=(9, 9)):
    return predict_sphere_surface_depth(
        shape,
        np.asarray(centers, dtype=float),
        np.asarray(radii, dtype=float),
        fx=100.0,
        fy=100.0,
        cx=4.0,
        cy=4.0,
        min_depth_m=0.05,
        max_depth_m=5.0,
    )


def test_center_pixel_is_nearest_sphere_surface_depth():
    predicted = render([[0.0, 0.0, 1.0]], [0.1])
    assert np.isclose(predicted[4, 4], 0.9)


def test_pixels_outside_projected_sphere_have_no_robot_depth():
    predicted = render([[0.0, 0.0, 1.0]], [0.02])
    assert np.isfinite(predicted[4, 4])
    assert np.isinf(predicted[0, 0])


def test_nearest_surface_wins_for_overlapping_spheres():
    predicted = render(
        [[0.0, 0.0, 1.2], [0.0, 0.0, 0.8]],
        [0.1, 0.1],
    )
    assert np.isclose(predicted[4, 4], 0.7)


def test_foreground_obstacle_is_preserved():
    predicted = np.asarray([[1.0]], dtype=np.float32)
    measured = np.asarray([[0.8]], dtype=np.float32)
    remove = surface_depth_removal_mask(measured, predicted, 0.02, 0.03)
    assert not remove[0, 0]


def test_measurement_matching_robot_surface_is_removed():
    predicted = np.asarray([[1.0, 1.0]], dtype=np.float32)
    measured = np.asarray([[0.985, 1.025]], dtype=np.float32)
    remove = surface_depth_removal_mask(measured, predicted, 0.02, 0.03)
    assert remove.tolist() == [[True, True]]


def test_measurement_behind_robot_uses_explicit_shadow_policy():
    predicted = np.asarray([[1.0]], dtype=np.float32)
    measured = np.asarray([[1.2]], dtype=np.float32)
    keep_shadow = surface_depth_removal_mask(
        measured, predicted, 0.02, 0.03, mask_shadow_behind_robot=False)
    remove_shadow = surface_depth_removal_mask(
        measured, predicted, 0.02, 0.03, mask_shadow_behind_robot=True)
    assert not keep_shadow[0, 0]
    assert remove_shadow[0, 0]


def test_invalid_measurement_and_pixels_without_robot_are_preserved():
    measured = np.asarray([[0.0, 1.0]], dtype=np.float32)
    predicted = np.asarray([[1.0, np.inf]], dtype=np.float32)
    remove = surface_depth_removal_mask(measured, predicted, 0.02, 0.03)
    assert not np.any(remove)


def test_mismatched_center_and_radius_counts_are_rejected():
    try:
        render([[0.0, 0.0, 1.0]], [])
    except ValueError as exc:
        assert "same number" in str(exc)
    else:
        raise AssertionError("expected mismatched sphere arrays to fail")


def test_ros_image_data_uses_uint8_array_fast_path():
    image = np.asarray([[1, 256], [1024, 65535]], dtype=np.uint16)
    data = as_ros_image_data(image)
    assert data.typecode == "B"
    assert data.tobytes() == image.tobytes()


def test_ros_image_data_makes_noncontiguous_input_contiguous():
    image = np.arange(24, dtype=np.uint16).reshape(4, 6)[:, ::2]
    data = as_ros_image_data(image)
    assert data.tobytes() == np.ascontiguousarray(image).tobytes()
