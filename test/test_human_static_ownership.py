"""Old person positions must not become static merely because a person occludes them."""
from dataclasses import replace
import numpy as np
import pytest

from rmp_camera.human_static_filter_core import (
    HumanStaticParameters, HumanVoxelHistory, human_occludes_voxels,
)

K = (100., 100., 50., 50.)
OLD = np.array([[0., 0., 1.3]])
CURRENT = np.array([[0., 0., 1.]])
DEPTH = np.full((101, 101), 1.)
PARAMS = HumanStaticParameters(occlusion_hold_enabled=True, ownership_hold_s=0.)


def history():
    h = HumanVoxelHistory(PARAMS)
    h.add(OLD, 1., np.zeros(3))
    h.add(CURRENT, 2., np.zeros(3))
    return h


def classify(h, depth=DEPTH, now=2., depth_stamp=2.):
    return h.classify(OLD, now, depth, K, np.eye(4), depth_stamp)


def test_old_body_is_owned_when_current_person_occludes_it_not_marked_free():
    h = history()
    active, free = classify(h)
    assert active.tolist() == [True]
    assert not free.any()
    assert h.last_occluded_count == 1
    # Feature off is the exact old policy, even with the same history/depth.
    h.params = replace(PARAMS, occlusion_hold_enabled=False)
    assert not np.any(classify(h))
    assert h.last_occluded_count == 0


def test_background_that_was_never_human_is_not_removed():
    h = HumanVoxelHistory(PARAMS)
    h.add(CURRENT, 2., np.zeros(3))
    assert not np.any(classify(h))
    # No unbounded shadow: distant historic objects also remain.
    h.add([[0, 0, 1.8]], 2.)
    h.add(CURRENT, 3., np.zeros(3))
    active, free = h.classify([[0, 0, 1.8]], 3., DEPTH, K, np.eye(4), 3.)
    assert not (active | free).any()


@pytest.mark.parametrize('value', [0., np.nan, np.inf, .7, 1.30])
def test_one_unknown_or_nonhuman_pixel_preserves_the_voxel(value):
    depth = DEPTH.copy()
    depth[49, 51] = value  # Interior pixel, not just the centre/corners.
    assert not np.any(classify(history(), depth))


def test_background_free_and_current_person_can_jointly_explain_old_body():
    depth = DEPTH.copy()
    depth[49, 51] = 2.0
    active, free = classify(history(), depth)
    assert active[0] and not free[0]
    # Person gone: use the existing full-voxel free-space proof, not ownership.
    active, free = classify(history(), np.full_like(depth, 2.))
    assert not active[0] and free[0]


def test_current_human_must_match_measured_surface_not_extruded_back_band():
    assert not np.any(classify(history(), np.full_like(DEPTH, 1.15)))


def test_stale_skewed_missing_and_empty_semantics_do_not_authorize_ownership():
    assert not np.any(classify(history(), now=2.4, depth_stamp=2.4))
    assert not np.any(classify(history(), depth_stamp=1.85))
    assert not np.any(classify(history(), depth=None))
    h = history()
    h.add([], 2.01)
    assert not np.any(classify(h, now=2.01, depth_stamp=2.01))
    h = history()
    h.clear()
    assert not len(h.surface_points)
    assert not np.any(classify(h))


def test_late_packet_does_not_rewind_current_human_surface():
    h = history()
    h.add([[.9, .9, 1.]], 1.5)
    np.testing.assert_array_equal(h.surface_points, CURRENT)
    assert classify(h)[0][0]


def test_historical_ownership_is_bounded_and_not_refreshed_by_inference():
    h = history()
    h.add(CURRENT, 7., np.zeros(3))
    assert not np.any(classify(h, now=7., depth_stamp=7.))


def test_arbitrary_camera_transform_and_duplicate_queries():
    tf = np.eye(4)
    tf[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    tf[:3, 3] = [.2, .3, .4]
    to_target = lambda p: (p - tf[:3, 3]) @ tf[:3, :3]
    pts = to_target(np.vstack([OLD, OLD, [10, 0, 1.3], [0, 0, -1]]))
    result = human_occludes_voxels(pts, DEPTH, K, tf, to_target(CURRENT), PARAMS)
    assert result.tolist() == [True, True, False, False]


@pytest.mark.parametrize('kw', [dict(occlusion_max_depth_m=.6),
    dict(occlusion_match_distance_m=.2), dict(occlusion_max_sync_delta_s=.3),
    dict(occlusion_max_depth_m=float('nan')), dict(occlusion_hold_enabled='false')])
def test_invalid_ownership_parameters_rejected(kw):
    with pytest.raises(ValueError):
        HumanStaticParameters(**kw)


def test_recent_semantic_label_survives_motion_but_reoccupation_vetoes():
    h = history()
    h.params = replace(PARAMS, ownership_hold_s=2., occlusion_min_valid_fraction=.75)
    # The old body position does not turn static merely because the mask moved.
    h.add([], 2.01)
    active, free = classify(h, now=2.01, depth_stamp=2.01)
    assert active[0] and not free[0]
    depth = DEPTH.copy()
    depth[49, 51] = 1.3  # A newly observed thin nonhuman object IN the old voxel.
    assert not np.any(classify(h, depth, now=2.01, depth_stamp=2.01))
    # After the label grace period, only the stricter current-human proof applies.
    h.add([], 3.1)
    assert not np.any(classify(h, now=3.1, depth_stamp=3.1))


def test_recent_label_cannot_hide_unlabelled_background_or_entirely_missing_depth():
    h = history()
    h.params = replace(PARAMS, ownership_hold_s=2., occlusion_min_valid_fraction=.75)
    assert not np.any(classify(h, np.zeros_like(DEPTH)))
    active, free = h.classify([[.5, 0, 1.3]], 2., DEPTH, K, np.eye(4), 2.)
    assert not (active | free).any()


def test_small_depth_holes_require_valid_majority_and_preserve_observed_obstacle():
    h = history()
    h.params = replace(PARAMS, occlusion_min_valid_fraction=.75)
    depth = DEPTH.copy()
    depth[49, 51] = 0.
    active, free = classify(h, depth)
    assert active[0] and not free[0]  # Ownership, never a free-space conclusion.
    depth[51, 49] = 1.3
    assert not np.any(classify(h, depth))
    depth[48:53, 48:53] = 0.
    assert not np.any(classify(h, depth))


def test_holes_cannot_be_cleared_by_mostly_free_depth_and_one_human_pixel():
    h = history()
    h.params = replace(PARAMS, occlusion_min_valid_fraction=.75)
    depth = np.full_like(DEPTH, 2.)
    depth[49, 51] = 0.
    depth[50, 50] = 1.
    assert not np.any(classify(h, depth))


def test_ownership_reduces_static_without_recolouring_or_changing_human_geometry():
    from types import SimpleNamespace
    from rmp_camera.obstacle_sphere_fusion_core import (
        FusionSphere, FusionParameters, SphereFusionCache,
    )
    from rmp_camera.obstacle_sphere_fusion_node import ObstacleSphereFusionNode
    old_body = FusionSphere(0, 0, 1.3, .07, .075, 0)
    ordinary = FusionSphere(.4, 0, 1.3, .07, .075, 0)
    human = FusionSphere(0, 0, 1., .05, .055, 2)
    cache = SphereFusionCache(FusionParameters())
    cache.update_static([old_body, ordinary], 'base_link', 2.)
    cache.update_human([human], 'base_link', 2.)
    h = history()
    h.params = replace(PARAMS, ownership_hold_s=2.)
    def excluded(points, now):
        active, free = h.classify(points, now, DEPTH, K, np.eye(4), 2.)
        return active | free
    fake = SimpleNamespace(cache=cache, static_generation_stamp=100,
        static_support={100:(np.array([[0,0,1.3],[.4,0,1.3]]), np.array([0,1]))},
        static_free_support={}, static_esdf_cleared={},
        human_filter=SimpleNamespace(enabled=True, excluded=excluded, last_stats=(1,0)),
        last_human_filter_log=float('inf'))
    assert len(cache.combined(2.)) == 3  # No sphere intersection before ownership.
    retained = ObstacleSphereFusionNode._human_filtered_static(fake, 2.)
    assert retained == [ordinary]
    assert cache.combined(2., static_override=retained) == [human, ordinary]
    assert cache.static_spheres == [old_body, ordinary]  # Source cache/map unchanged.


def test_footprint_budget_preserves_unprocessed_candidates():
    points = np.repeat(OLD, 6000, axis=0)
    owned = human_occludes_voxels(points, DEPTH, K, np.eye(4), CURRENT, PARAMS)
    assert owned[0] and not owned[-1]
