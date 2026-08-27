import math

import numpy as np

from rmp_camera.robot_sphere_marker_correction_core import (
    correction_in_marker_frame,
    matrix_to_quaternion,
    rpy_correction_matrix,
    transform_pose_matrix,
)


def test_translation_in_correction_frame_is_converted_to_marker_frame():
    marker_to_correction = np.eye(4)
    marker_to_correction[:3, :3] = rpy_correction_matrix(
        [0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2.0])[:3, :3]
    correction = rpy_correction_matrix(
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    in_marker = correction_in_marker_frame(marker_to_correction, correction)

    assert np.allclose(in_marker[:3, 3], [0.0, -1.0, 0.0], atol=1e-12)


def test_correction_rotates_pose_about_correction_frame_origin():
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 0.0, 0.0]
    correction = rpy_correction_matrix(
        [0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2.0])

    corrected = transform_pose_matrix(pose, correction)

    assert np.allclose(corrected[:3, 3], [0.0, 1.0, 0.0], atol=1e-12)


def test_matrix_to_quaternion_round_trips_rpy_rotation():
    expected = rpy_correction_matrix(
        [0.0, 0.0, 0.0], [0.3, -0.2, 0.7])[:3, :3]
    x, y, z, w = matrix_to_quaternion(expected)
    reconstructed = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

    assert np.allclose(reconstructed, expected, atol=1e-12)
