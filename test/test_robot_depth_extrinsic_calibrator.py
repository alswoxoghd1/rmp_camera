import struct

import numpy as np

from rmp_camera.robot_depth_extrinsic_calibrator import (
    candidate_base_from_camera,
    nearest_index,
    read_binary_stl,
    sample_triangle_surface,
)


def test_nearest_index_clamps_and_uses_earlier_tie():
    values = [100, 200, 300]
    assert nearest_index(values, 1) == 0
    assert nearest_index(values, 250) == 1
    assert nearest_index(values, 999) == 2


def test_sample_triangle_surface_stays_in_triangle_plane():
    triangle = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    points = sample_triangle_surface(triangle, 100, seed=7)
    assert points.shape == (100, 3)
    assert np.allclose(points[:, 2], 0.0)
    assert np.all(points[:, :2] >= 0.0)
    assert np.all(points[:, 0] + points[:, 1] <= 1.0 + 1e-12)


def test_read_binary_stl_decodes_vertices(tmp_path):
    path = tmp_path / "one_triangle.stl"
    header = b"test".ljust(80, b"\0")
    triangle = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    )
    path.write_bytes(header + struct.pack("<I", 1) + triangle)
    vertices = read_binary_stl(path)
    assert vertices.shape == (1, 3, 3)
    assert np.allclose(vertices[0], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_camera_correction_translation_changes_candidate_with_inverse_sign():
    initial = np.eye(4)
    candidate = candidate_base_from_camera(
        initial, np.asarray([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    )
    assert np.allclose(candidate[:3, 3], [-0.1, 0.2, -0.3])
