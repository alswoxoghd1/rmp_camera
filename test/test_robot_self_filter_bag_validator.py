from types import SimpleNamespace

import numpy as np

from rmp_camera.robot_self_filter_bag_validator import (
    SphereSpec,
    depth_image_to_meters,
    nearest_state_index,
    resolve_transform,
    sphere_centers_in_camera,
    volume_overlap_count,
)


def translation(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = [x, y, z]
    return matrix


def test_nearest_state_index_uses_closest_timestamp_and_earlier_tie():
    stamps = [100, 200, 300]
    assert nearest_state_index(stamps, 10) == 0
    assert nearest_state_index(stamps, 151) == 1
    assert nearest_state_index(stamps, 250) == 1
    assert nearest_state_index(stamps, 999) == 2


def test_resolve_transform_composes_parent_from_child_chain():
    tree = {
        "camera": ("base", translation(1.0, 0.0, 0.0)),
        "optical": ("camera", translation(0.0, 2.0, 0.0)),
    }
    base_from_optical = resolve_transform(tree, "optical", "base")
    assert np.allclose(base_from_optical[:3, 3], [1.0, 2.0, 0.0])


def test_sphere_centers_are_transformed_into_camera_frame():
    tree = {
        "camera": ("base", translation(1.0, 0.0, 0.0)),
        "link": ("base", translation(2.0, 0.0, 0.0)),
    }
    spheres = [SphereSpec("link", np.asarray([0.5, 0.0, 0.0]), 0.1)]
    centers = sphere_centers_in_camera(tree, spheres, "base", "camera")
    assert np.allclose(centers, [[1.5, 0.0, 0.0]])


def test_depth_image_decoder_handles_padded_16uc1_rows():
    values = np.asarray([[1000, 2000, 99], [3000, 4000, 99]], dtype="<u2")
    msg = SimpleNamespace(
        encoding="16UC1",
        is_bigendian=False,
        width=2,
        height=2,
        step=6,
        data=values.tobytes(),
    )
    depth_m = depth_image_to_meters(msg)
    assert np.allclose(depth_m, [[1.0, 2.0], [3.0, 4.0]])


def test_volume_overlap_counts_each_point_at_most_once():
    points = np.asarray([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]])
    centers = np.asarray([[0.0, 0.0, 1.0], [0.02, 0.0, 1.0]])
    assert volume_overlap_count(points, centers, np.asarray([0.1, 0.1])) == 1
