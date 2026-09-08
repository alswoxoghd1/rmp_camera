from dataclasses import replace

import numpy as np
import pytest

from rmp_camera.depth_edge_filter_core import DepthEdgeFilterConfig, depth_edge_rejection_mask


def transition_image():
    depth = np.ones((37, 45), dtype=np.float32)
    depth[:, 23:] = 2.3
    depth[:, 22] = 1.8
    return depth


def test_rejects_unsupported_intermediate_band_not_real_surfaces():
    depth = transition_image()
    original = depth.copy()
    result = depth_edge_rejection_mask(depth)
    expected = np.zeros(depth.shape, dtype=bool)
    expected[3:-3, 22] = True
    np.testing.assert_array_equal(result.rejected, expected)
    np.testing.assert_array_equal(depth, original)
    assert result.valid_pixels == depth.size


@pytest.mark.parametrize('width', [1, 2, 5])
def test_preserves_thin_foreground_against_background(width):
    depth = np.full((37, 45), 2.3, np.float32)
    depth[:, 22:22 + width] = 1.4
    assert not depth_edge_rejection_mask(depth).rejected.any()


def test_preserves_supported_third_surface():
    depth = transition_image()
    depth[:, 20:25] = 1.8
    assert not depth_edge_rejection_mask(depth).rejected.any()


@pytest.mark.parametrize('invalid', [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_is_unknown_not_far_or_near_support(invalid):
    depth = transition_image()
    depth[:, 23:] = invalid
    result = depth_edge_rejection_mask(depth)
    assert not result.rejected.any()
    assert result.valid_pixels == 37 * 23


def test_needs_multiple_support_points_on_both_sides():
    depth = np.zeros((17, 17), np.float32)
    depth[8, 8] = 1.8
    depth[7, 8] = 1.4
    depth[9, 8] = 2.3
    assert not depth_edge_rejection_mask(depth).rejected[8, 8]
    depth[7, 9] = 1.4
    depth[9, 9] = 2.3
    assert depth_edge_rejection_mask(depth).rejected[8, 8]


def test_no_temporal_state_and_no_cascading_deletion():
    original = transition_image()
    before = depth_edge_rejection_mask(original).rejected
    depth_edge_rejection_mask(np.full_like(original, 1.5))
    after = depth_edge_rejection_mask(original).rejected
    np.testing.assert_array_equal(before, after)


def test_strided_readonly_array_and_smooth_sloped_surface():
    depth = np.tile(np.linspace(1, 1.9, 100, dtype=np.float32), (50, 1))[:, ::2]
    depth.flags.writeable = False
    assert not depth_edge_rejection_mask(depth).rejected.any()


@pytest.mark.parametrize('shape', [(0, 0), (1, 20), (20, 1), (3, 3)])
def test_small_or_empty_images_keep_shape(shape):
    result = depth_edge_rejection_mask(np.ones(shape, np.float32))
    assert result.rejected.shape == shape
    assert not result.rejected.any()


@pytest.mark.parametrize('changes', [
    {'neighborhood_radius_px': 0}, {'neighborhood_radius_px': 9},
    {'neighborhood_radius_px': 3.0}, {'min_support_pixels': True},
    {'min_support_pixels': 0}, {'min_support_pixels': 49},
    {'min_side_pixels': 0}, {'min_side_pixels': 25},
    {'min_depth_jump_m': 0}, {'min_depth_jump_m': np.inf},
    {'support_tolerance_m': np.nan}, {'support_tolerance_m': 0},
    {'support_tolerance_m': .10},
])
def test_rejects_invalid_configuration(changes):
    with pytest.raises(ValueError):
        replace(DepthEdgeFilterConfig(), **changes)


def test_rejects_non_image_input():
    with pytest.raises(ValueError):
        depth_edge_rejection_mask(np.ones((4, 3, 2)))


def test_support_threshold_is_inclusive_and_does_not_include_center():
    # Intermediate column has six equal-depth neighbors in a 7x7 window.
    depth = transition_image()
    assert not depth_edge_rejection_mask(depth, DepthEdgeFilterConfig(
        min_support_pixels=6)).rejected.any()
    assert depth_edge_rejection_mask(depth, DepthEdgeFilterConfig(
        min_support_pixels=7)).rejected[18, 22]
