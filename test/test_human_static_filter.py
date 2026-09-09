from types import SimpleNamespace

import numpy as np

from rmp_camera.human_static_filter_core import (
    HumanStaticParameters, HumanVoxelHistory, depth_proves_voxels_free,
    fully_excluded_spheres,
    esdf_proves_voxels_free,
    refit_mixed_static_spheres,
)
from rmp_camera.obstacle_sphere_fusion_core import FusionParameters, FusionSphere, SphereFusionCache
from rmp_camera.obstacle_sphere_fusion_node import ObstacleSphereFusionNode
from rmp_camera.human_static_filter import HumanStaticFilter


K = (100.0, 100.0, 50.0, 50.0)
POINT = np.asarray([[0.0, 0.0, 1.0]])


def test_enabled_ros_adapter_initializes_and_resets_history():
    import rclpy
    from rclpy.context import Context
    from rclpy.node import Node
    from rclpy.parameter import Parameter

    context = Context()
    rclpy.init(context=context)
    node = Node('test_human_static_filter', context=context, parameter_overrides=[
        Parameter('human_static_filter_enabled', value=True)])
    try:
        node.target_frame = 'base_link'
        adapter = HumanStaticFilter(node)
        assert adapter.enabled
        adapter.history.add(POINT, 10.0)
        adapter._jump(None)
        assert not adapter.history.voxels
        assert adapter.depth is None
        assert adapter.transform is None
    finally:
        node.destroy_node()
        rclpy.shutdown(context=context)


def test_free_evidence_requires_valid_background_across_whole_voxel():
    depth = np.full((101, 101), 2.0)
    check = lambda: depth_proves_voxels_free(POINT, depth, K, np.eye(4), 0.05, 0.03)[0]
    assert check()
    # A thin foreground obstacle between centre/corner samples must protect it.
    depth[49, 51] = 0.8
    assert not check()
    depth[49, 51] = 0.0
    assert not check()
    depth[49, 51] = np.nan
    assert not check()
    depth[:] = 1.04  # Centre is free, but not the voxel rear + clearance.
    assert not check()


def test_out_of_view_and_behind_camera_are_not_free():
    points = np.asarray([[0, 0, -1], [2, 0, 1], [0.5, 0, 1]])
    assert not depth_proves_voxels_free(
        points, np.full((101, 101), 3.0), K, np.eye(4), .05, .03).any()


def test_departed_person_needs_depth_evidence_and_neighbour_is_preserved():
    history = HumanVoxelHistory()
    history.add(POINT, 1.0)
    queries = np.vstack([POINT, [0.2, 0, 1]])
    active, free = history.classify(queries, 1.1)
    assert active.tolist() == [True, False]
    assert not free.any()
    # Expired semantic detection alone does not erase anything.
    active, free = history.classify(queries, 1.6)
    assert not (active | free).any()
    active, free = history.classify(queries, 1.6, np.full((101, 101), 2.0),
                                    K, np.eye(4), 1.6)
    assert free.tolist() == [True, False]
    # Newly occupied human location is kept, as is an occluded location.
    for measured in (1.0, 0.5, 0.0):
        _, free = history.classify(queries, 1.6, np.full((101, 101), measured),
                                  K, np.eye(4), 1.6)
        assert not free.any()


def test_stale_depth_and_expired_history_cannot_clear():
    history = HumanVoxelHistory(HumanStaticParameters(history_s=1.0))
    history.add(POINT, 1.0)
    assert not history.classify(POINT, 1.7, np.full((101, 101), 2.0),
                                K, np.eye(4), 1.0)[1].any()
    active, free = history.classify(POINT, 3.0, np.full((101, 101), 2.0),
                                    K, np.eye(4), 3.0)
    assert not (active | free).any()


def test_history_bound_and_clock_reset():
    history = HumanVoxelHistory(HumanStaticParameters(max_history_voxels=4))
    history.add(np.arange(30).reshape(-1, 3) * .1, 10.0)
    assert len(history.voxels) == 4
    history.add(POINT, 9.0)  # Ignore late masks, do not overwrite newer evidence.
    assert history.last_stamp == 10.0
    history.clear()
    history.add(POINT, 1.0)
    assert history.classify(POINT, 1.0)[0].all()


def test_back_band_is_bounded_along_view_ray():
    history = HumanVoxelHistory(HumanStaticParameters(match_distance_m=.05))
    history.add(POINT, 1.0, np.zeros(3))
    active, _ = history.classify([[0, 0, 1.13], [.2, 0, 1.13], [0, 0, 1.4]], 1.0)
    assert active.tolist() == [True, False, False]


def test_only_all_excluded_support_can_remove_a_sphere():
    assert fully_excluded_spheres([0, 0, 1, 1], [True, True, True, False], 3).tolist() == [True, False, False]


def test_fusion_fast_clear_uses_matching_support_and_keeps_mixed_geometry():
    spheres = [FusionSphere(x, 0, 1, .1, .1, 0) for x in (0, .3)]
    cache = SphereFusionCache(FusionParameters(static_empty_confirmation_frames=2))
    cache.update_static(spheres, 'base_link', 1.0)
    cache.update_static([], 'base_link', 1.1)  # Normal static clear still waits.
    fake = SimpleNamespace(
        cache=cache, static_generation_stamp=100,
        static_free_support={}, static_esdf_cleared={},
        static_support={100: (np.asarray([[0, 0, 1], [.3, 0, 1], [.35, 0, 1]]), np.array([0, 1, 1]))},
        human_filter=SimpleNamespace(enabled=True,
            excluded=lambda points, now: np.asarray([True, True, False]), last_stats=(1, 1)),
        last_human_filter_log=float('inf'),
    )
    retained = ObstacleSphereFusionNode._human_filtered_static(fake, 1.1)
    assert retained == [spheres[1]]
    assert cache.combined(1.1, static_override=retained) == [spheres[1]]
    # Missing/mismatched evidence protects every sphere.
    fake.static_generation_stamp = 200
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 1.1) == spheres


def test_esdf_free_requires_observed_distance_for_whole_voxel():
    values = np.array([.2, -.1, -1000.0, np.nan, .04]).reshape(5, 1, 1)
    points = np.array([[.025 + .05 * i, .025, .025] for i in range(6)])
    assert esdf_proves_voxels_free(points, values, np.zeros(3), .05, -1000).tolist() == [
        True, False, False, False, False, False]


def test_new_esdf_empty_evidence_clears_old_generation_without_reappearance():
    spheres = [FusionSphere(x, 0, 1, .1, .1, 0) for x in (0, .3)]
    cache = SphereFusionCache(FusionParameters(static_empty_confirmation_frames=2))
    cache.update_static(spheres, 'base_link', 1.0)
    cache.update_static([], 'base_link', 2.0)
    points = np.asarray([[0, 0, 1], [.3, 0, 1], [.35, 0, 1]])
    ids = np.array([0, 1, 1])
    fake = SimpleNamespace(cache=cache, static_generation_stamp=100,
        static_support={100: (points, ids)}, static_esdf_cleared={},
        static_free_support={100: (2.0, points, ids, np.array([True, True, False]))},
        human_filter=SimpleNamespace(enabled=True,
            excluded=lambda points, now: np.zeros(len(points), dtype=bool), last_stats=(0, 0)),
        last_human_filter_log=float('inf'))
    # Stale evidence cannot trigger removal.
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 2.5) == spheres
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 2.1) == [spheres[1]]
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 2.5) == [spheres[1]]
    # New generation can represent newly occupied space at the same position.
    fake.static_generation_stamp = 200
    fake.static_support[200] = (points, ids)
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 2.6) == spheres


def test_human_majority_exclusion_retires_static_for_the_generation():
    spheres = [FusionSphere(0, 0, 1, .2, .2, 0)]
    points = np.asarray([[x, 0, 1] for x in (-.1, -.05, 0, .05, .1)])
    excluded = np.asarray([True, True, True, True, False])
    fake = SimpleNamespace(cache=SimpleNamespace(static_spheres=spheres),
        static_generation_stamp=100, static_support={100: (points, np.zeros(5, dtype=int))},
        static_free_support={}, static_esdf_cleared={}, static_human_suppressed={},
        human_filter=SimpleNamespace(enabled=True, excluded=lambda _p, _n: excluded,
            last_stats=(4, 0)), human_static_exclusion_ratio=.70,
        human_static_refit_enabled=False, fusion_coverage_guard_enabled=False,
        last_human_filter_log=float('inf'))
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 1.0) == []
    fake.human_filter.excluded = lambda _p, _n: np.zeros(5, dtype=bool)
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 2.0) == []
    fake.static_generation_stamp = 200
    fake.static_support[200] = (points, np.zeros(5, dtype=int))
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 2.0) == spheres


def test_esdf_evidence_wire_format_and_generation_pairing():
    from builtin_interfaces.msg import Time
    from std_msgs.msg import Header
    from rmp_camera.esdf_medial_sphere_node import EsdfMedialSphereNode
    from sensor_msgs_py import point_cloud2

    published = []
    rows = np.array([[.025, .025, .025, 0], [.075, .025, .025, 1]])
    sender = SimpleNamespace(support_history=[(Time(sec=1), rows)],
        last_support_evidence_stamp=None,
        unobserved_distance_value=-1000.0,
        human_filter=SimpleNamespace(params=HumanStaticParameters()),
        free_support_pub=SimpleNamespace(publish=published.append),
        _header=lambda stamp: Header(stamp=stamp, frame_id='base_link'))
    grid = SimpleNamespace(values=np.array([.2, -1000.0]).reshape(2, 1, 1),
        origin_m=np.zeros(3), voxel_size_m=.05)
    EsdfMedialSphereNode.publish_esdf_free_support(sender, Time(sec=2), grid)
    message = published[-1]
    assert message.header.stamp.sec == 1
    wire_rows = point_cloud2.read_points_list(message)
    assert [row.esdf_free for row in wire_rows] == [1, 0]
    assert all(row.evidence_stamp == 2.0 for row in wire_rows)
    receiver = SimpleNamespace(target_frame='base_link', static_free_support={},
        _stamp_key=ObstacleSphereFusionNode._stamp_key)
    ObstacleSphereFusionNode._free_support_callback(receiver, message)
    evidence = receiver.static_free_support[1_000_000_000]
    assert evidence[0] == 2.0
    assert evidence[2].tolist() == [0, 1]
    assert evidence[3].tolist() == [True, False]
    # Replaying /clock backwards must not reuse a previous pass's support.
    EsdfMedialSphereNode.publish_esdf_free_support(sender, Time(sec=0), grid)
    assert not sender.support_history


def test_fusion_clock_reset_does_not_reuse_old_generation_tombstones():
    fake = SimpleNamespace(static_support={100: object()},
        static_free_support={100: object()}, static_esdf_cleared={100: {0}},
        static_generation_stamp=100)
    ObstacleSphereFusionNode._reset_human_static_cache(fake)
    assert not fake.static_support
    assert not fake.static_free_support
    assert not fake.static_esdf_cleared
    assert fake.static_generation_stamp is None


def test_mixed_refit_covers_all_retained_support_without_growing_or_adding():
    original = FusionSphere(0, 0, 1, .30, .305, 0, component_id=7)
    retained = np.array([[.14, -.02, 1], [.18, .02, 1], [.18, 0, 1.02]])
    human = np.array([[-.18, 0, 1], [-.12, 0, 1]])
    points = np.vstack((retained, human))
    output, stats = refit_mixed_static_spheres(
        [original], points, np.zeros(5, dtype=int), [False] * 3 + [True] * 2)
    assert len(output) == 1 and stats['refitted'] == 1
    new = output[0]
    assert new.raw_radius >= .04
    assert new.component_id == original.component_id
    assert np.all(np.linalg.norm(retained - new.center, axis=1) <= new.raw_radius + .02 + 1e-9)
    assert np.linalg.norm(new.center - original.center) + new.output_radius <= original.output_radius + 1e-9
    assert np.isclose(new.output_radius - new.raw_radius, .005)
    assert stats['excluded_support_avoided'] == 2
    assert original.raw_radius == .30  # Cached source geometry is immutable.


def test_refit_cannot_erase_unlabelled_or_occluded_support():
    original = FusionSphere(0, 0, 1, .30, .305, 0)
    # Retained points on opposite edges prevent a smaller enclosing ball.
    points = np.array([[-.30, 0, 1], [.30, 0, 1], [0, 0, 1]])
    output, stats = refit_mixed_static_spheres([original], points, [0, 0, 0],
                                             [False, False, True])
    assert output == [original] and stats['refitted'] == 0
    assert stats['mixed'] == 1


def test_refit_missing_pure_human_and_ordinary_support_are_not_deleted_by_fitting():
    spheres = [FusionSphere(x, 0, 1, .1, .105, 0) for x in (0, .5, 1)]
    # First all human, second no human, third has no support message rows.
    result, stats = refit_mixed_static_spheres(spheres,
        [[0, 0, 1], [.5, 0, 1]], [0, 1], [True, False])
    assert result == spheres and stats['mixed'] == 0


def test_refit_rejects_malformed_support_and_invalid_parameters():
    import pytest
    sphere = FusionSphere(0, 0, 1, .1, .1, 0)
    for points, ids, mask in (([[np.nan, 0, 1]], [0], [True]),
                             (POINT, [1], [True]), (POINT, [-1], [True]),
                             (POINT, [0], [])):
        with pytest.raises(ValueError):
            refit_mixed_static_spheres([sphere], points, ids, mask)
    for options in ({'min_radius_m': -1}, {'coverage_tolerance_m': float('nan')}):
        with pytest.raises(ValueError):
            refit_mixed_static_spheres([sphere], POINT, [0], [True], **options)


def test_fusion_refit_cache_changes_with_mask_and_does_not_mutate_source():
    sphere = FusionSphere(0, 0, 1, .30, .305, 0)
    cache = SphereFusionCache(FusionParameters())
    cache.update_static([sphere], 'base_link', 1.)
    points = np.array([[-.15, 0, 1], [.15, 0, 1], [.18, 0, 1]])
    mask = np.array([True, False, False])
    fake = SimpleNamespace(cache=cache, static_generation_stamp=100,
        static_support={100: (points, np.zeros(3, dtype=int))},
        static_free_support={}, static_esdf_cleared={},
        human_filter=SimpleNamespace(enabled=True,
            excluded=lambda points, now: mask, last_stats=(1, 0)),
        human_static_refit_enabled=True, human_static_refit_min_radius_m=.04,
        human_static_refit_coverage_tolerance_m=.02,
        static_refit_key=None, static_refit_result=None,
        last_human_filter_log=float('inf'))
    first = ObstacleSphereFusionNode._human_filtered_static(fake, 1.1)
    stored = fake.static_refit_result
    assert first[0].raw_radius < sphere.raw_radius
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 1.2) == first
    assert fake.static_refit_result is stored
    assert cache.static_spheres == [sphere]
    mask[:] = False
    assert ObstacleSphereFusionNode._human_filtered_static(fake, 1.3) == [sphere]


def test_mixed_person_object_keeps_object_in_final_fusion_after_refit():
    from rmp_camera.obstacle_sphere_fusion_core import fuse_spheres
    original = FusionSphere(0, 0, 1, .30, .305, 0)
    human = FusionSphere(-.15, 0, 1, .08, .08, 2)
    points = [[-.15, 0, 1], [.14, 0, 1], [.18, .02, 1]]
    refitted, _ = refit_mixed_static_spheres([original], points, [0, 0, 0],
                                           [True, False, False])
    fused = fuse_spheres(refitted, [], FusionParameters(), [human])
    assert len(fused) == 2 and {s.source_type for s in fused} == {0, 2}
    assert np.all(np.linalg.norm(np.array(points[1:]) - refitted[0].center,
                                axis=1) <= refitted[0].raw_radius + .02 + 1e-9)


def test_refit_randomized_coverage_containment_and_count_invariants():
    rng = np.random.default_rng(24)
    original = FusionSphere(0, 0, 1, .3, .305, 0)
    for _ in range(100):
        points = rng.uniform(-.16, .16, (50, 3)) + original.center
        excluded = points[:, 0] < 0
        output, _ = refit_mixed_static_spheres([original], points,
            np.zeros(len(points), dtype=int), excluded)
        assert len(output) == 1
        new = output[0]
        assert new.raw_radius >= .04
        assert np.all(np.linalg.norm(points[~excluded] - new.center,
                                     axis=1) <= new.raw_radius + .02 + 1e-9)
        assert np.linalg.norm(new.center - original.center) + new.output_radius <= original.output_radius + 1e-9
