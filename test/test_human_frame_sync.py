import math

import pytest

from rmp_camera.human_frame_sync import select_latest_depth_mask_pair as select_pair


def test_latest_mask_wins_over_older_perfect_sync():
    assert select_pair([1.0, 1.1], [1.0, 1.09], .05)[:2] == (1, 1)


def test_nearest_depth_for_latest_mask_not_merely_newest_depth():
    assert select_pair([1.0, 1.03], [1.005], .05)[:2] == (0, 0)


def test_newer_depth_wins_equal_delta_tie():
    assert select_pair([1.0, 1.5], [1.25], .3)[:2] == (1, 0)


def test_watermarks_advance_both_streams_and_disallow_duplicates():
    assert select_pair([1.0, 1.1], [1.0, 1.1], .05, 1.1, 1.0) is None
    assert select_pair([1.0, 1.1], [1.0, 1.1], .05, 1.0, 1.1) is None
    assert select_pair([1.0, 1.1], [1.0, 1.1], .05, 1.0, 1.0)[:2] == (1, 1)


def test_no_compatible_pair_does_not_relax_sync_threshold():
    assert select_pair([1.0], [1.2], .05) is None


def test_unordered_queues_still_select_by_source_stamp():
    assert select_pair([1.1, 1.0], [1.09, 1.0], .05)[:2] == (0, 0)


@pytest.mark.parametrize('depth,mask', [([], [1.0]), ([1.0], []),
    ([math.nan, math.inf], [1.0]), ([1.0], [math.nan, -math.inf])])
def test_empty_and_nonfinite_stamps_do_not_form_pairs(depth, mask):
    assert select_pair(depth, mask, .05) is None


@pytest.mark.parametrize('delta', [-1, math.nan, math.inf])
def test_invalid_sync_tolerance(delta):
    with pytest.raises(ValueError):
        select_pair([1], [1], delta)
