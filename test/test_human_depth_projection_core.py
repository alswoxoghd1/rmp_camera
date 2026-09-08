import numpy as np

from rmp_camera.human_depth_projection_core import (
    crop_points,
    filter_small_mask_components,
    foreground_depth_keep_mask,
    project_masked_depth,
)


def test_filters_small_semantic_mask_components():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0, 0] = 1
    mask[3:5, 3:5] = 1
    filtered = filter_small_mask_components(
        mask, mask_threshold=1, min_component_pixels=3)
    assert filtered[0, 0] == 0
    assert np.count_nonzero(filtered) == 4


def test_foreground_filter_rejects_depth_behind_nearby_person_surface():
    keep = foreground_depth_keep_mask(
        np.asarray((1, 2)),
        np.asarray((0, 0)),
        np.asarray((1.0, 2.0)),
        np.ones((1, 4), dtype=np.uint8),
        neighborhood_radius_pixels=1,
        max_local_depth_jump_m=0.2,
    )
    assert keep.tolist() == [True, False]


def test_foreground_filter_rejects_distant_layer_in_one_mask_component():
    keep = foreground_depth_keep_mask(
        np.asarray((0, 3)),
        np.asarray((0, 0)),
        np.asarray((1.0, 2.0)),
        np.ones((1, 4), dtype=np.uint8),
        max_component_depth_span_m=0.5,
    )
    assert keep.tolist() == [True, False]


def test_projects_only_depth_pixels_inside_person_mask():
    depth = np.ones((3, 3), dtype=np.float32)
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 2] = 1
    points = project_masked_depth(
        depth,
        (1.0, 1.0, 1.0, 1.0),
        mask,
        (1.0, 1.0, 1.0, 1.0),
        np.eye(4),
        np.eye(4),
        stride=1,
        min_depth_m=0.1,
        max_depth_m=2.0,
    )
    assert points.shape == (1, 3)
    assert np.allclose(points[0], (1.0, 0.0, 1.0))


def test_supports_different_depth_and_mask_intrinsics():
    depth = np.asarray(((1.0, 1.0),), dtype=np.float32)
    mask = np.zeros((1, 4), dtype=np.uint8)
    mask[0, 2] = 255
    points = project_masked_depth(
        depth,
        (1.0, 1.0, 0.0, 0.0),
        mask,
        (2.0, 1.0, 0.0, 0.0),
        np.eye(4),
        np.eye(4),
        stride=1,
        min_depth_m=0.1,
        max_depth_m=2.0,
    )
    assert points.shape == (1, 3)
    assert np.allclose(points[0], (1.0, 0.0, 1.0))


def test_applies_depth_to_mask_and_target_transforms():
    depth = np.ones((1, 1), dtype=np.float32)
    mask = np.zeros((1, 2), dtype=np.uint8)
    mask[0, 1] = 1
    mask_from_depth = np.eye(4)
    mask_from_depth[0, 3] = 1.0
    target_from_depth = np.eye(4)
    target_from_depth[2, 3] = 2.0
    points = project_masked_depth(
        depth,
        (1.0, 1.0, 0.0, 0.0),
        mask,
        (1.0, 1.0, 0.0, 0.0),
        mask_from_depth,
        target_from_depth,
        min_depth_m=0.1,
        max_depth_m=2.0,
    )
    assert np.allclose(points, ((0.0, 0.0, 3.0),))


def test_empty_mask_returns_empty_points():
    points = project_masked_depth(
        np.ones((2, 2), dtype=np.float32),
        (1.0, 1.0, 0.0, 0.0),
        np.zeros((2, 2), dtype=np.uint8),
        (1.0, 1.0, 0.0, 0.0),
        np.eye(4),
        np.eye(4),
        min_depth_m=0.1,
        max_depth_m=2.0,
    )
    assert points.shape == (0, 3)


def test_projection_z_buffer_keeps_nearest_depth_for_colliding_mask_pixel():
    depth = np.asarray(((1.0, 2.0),), dtype=np.float32)
    points = project_masked_depth(
        depth,
        (100.0, 1.0, 0.0, 0.0),
        np.ones((1, 1), dtype=np.uint8),
        (1.0, 1.0, 0.0, 0.0),
        np.eye(4),
        np.eye(4),
        stride=1,
        min_depth_m=0.1,
        max_depth_m=3.0,
        foreground_neighborhood_pixels=0,
        foreground_max_local_depth_jump_m=0.2,
    )
    assert points.shape == (1, 3)
    assert np.allclose(points[0], (0.0, 0.0, 1.0))


def test_crop_points_applies_box_and_radial_limits():
    points = np.asarray((
        (0.0, 0.0, 0.5),
        (2.0, 0.0, 0.5),
        (0.5, 0.5, 3.0),
        (0.8, 0.8, 0.5),
    ))
    result = crop_points(
        points, (-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), 1.0)
    assert np.allclose(result, ((0.0, 0.0, 0.5),))
