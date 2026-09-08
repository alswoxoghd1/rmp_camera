from dataclasses import replace
from types import SimpleNamespace
import struct
import time

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2

from rmp_camera.pointcloud_numpy import LatestCloudBuffer, read_numeric_cloud
from rmp_camera.dynamic_obstacle_sphere_core import (
    DynamicSphere, DynamicSphereTracker, DynamicComponentResult,
    prune_overlapping_tracked_spheres,
)
from rmp_camera.dynamic_obstacle_sphere_node import DynamicObstacleSphereNode
from rmp_camera.obstacle_sphere_fusion_core import FusionSphere, FusionParameters, fuse_spheres
from rmp_camera.obstacle_sphere_fusion_node import ObstacleSphereFusionNode
from rmp_camera.sphere_support import current_support, support_cloud


def cloud(stamp=1., big=False):
    # Organized rows with padding, out-of-order field layout and unaligned x.
    data = bytearray(64)
    for index, xyz in enumerate(((1., 2., 3.), (4., 5., 6.))):
        base = index * 32
        for offset, value in zip((4, 12, 0), xyz):
            struct.pack_into(('>' if big else '<') + 'f', data, base + offset, value)
    fields = [PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
              for name, offset in (('z', 0), ('x', 4), ('y', 12))]
    return PointCloud2(header=Header(frame_id='base_link', stamp=Time(
        sec=int(stamp), nanosec=round((stamp - int(stamp)) * 1e9))),
        height=2, width=1, point_step=20, row_step=32, fields=fields,
        is_bigendian=big, data=bytes(data))


@pytest.mark.parametrize('big', [True, False])
def test_numpy_cloud_honors_field_order_endian_and_row_padding(big):
    message = cloud(big=big)
    original = bytes(message.data)
    np.testing.assert_array_equal(read_numeric_cloud(message), [[1, 2, 3], [4, 5, 6]])
    assert bytes(message.data) == original


@pytest.mark.parametrize('bad', ['payload', 'stride', 'field', 'count'])
def test_malformed_cloud_is_not_an_empty_observation(bad):
    message = cloud()
    if bad == 'payload':
        message.data = bytes(8)
    elif bad == 'stride':
        message.row_step = 1
    elif bad == 'field':
        message.fields[0].offset = 20
    else:
        message.fields[0].count = 2
    with pytest.raises(ValueError):
        read_numeric_cloud(message)


def test_latest_buffer_uses_newest_synchronized_not_newest_unsynchronized():
    buffer = LatestCloudBuffer()
    for stamp in (1., 1.05, 1.10):
        buffer.add(cloud(stamp), stamp, int(stamp * 1e9))
    selected = buffer.take(1_120_000_000, 1.12, lambda m: m.header.stamp.nanosec <= 50_000_000)
    assert selected[2] == 1_050_000_000
    assert len(buffer.frames) == 1
    assert buffer.take(1_150_000_000, 1.15, lambda _: True)[2] == 1_100_000_000


def test_buffer_bounds_age_order_and_clock_rewind():
    buffer = LatestCloudBuffer(capacity=2)
    for stamp in (1., 1.01, 1.02):
        assert buffer.add(cloud(stamp), stamp, int(stamp * 1e9))[0]
    assert len(buffer.frames) == 2 and buffer.superseded == 1
    assert not buffer.add(cloud(1.), 1.03, 1_030_000_000)[0]
    assert buffer.take(1_500_000_000, 1.5, lambda _: True) is None
    assert buffer.expired == 2
    accepted, rewound = buffer.add(cloud(.1), 2., 100_000_000)
    assert accepted and rewound
    assert buffer.take(100_000_000, 2., lambda _: True)[2] == 100_000_000


def test_node_selects_before_decoding_and_keeps_robot_sync(monkeypatch):
    import rmp_camera.dynamic_obstacle_sphere_node as adapter
    buffer = LatestCloudBuffer()
    now_wall = time.monotonic()
    buffer.add(cloud(1.), now_wall, 1_000_000_000)
    buffer.add(cloud(1.1), now_wall, 1_100_000_000)
    decoded = []
    original = read_numeric_cloud
    monkeypatch.setattr(adapter, 'read_numeric_cloud', lambda m: (decoded.append(m), original(m))[1])
    fake = SimpleNamespace(input_buffer=buffer, target_frame='base_link',
        input_wait_started=None, input_wait_reported=False,
        robot_component_filter_enabled=True, robot_component_filter_fail_closed=True,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_110_000_000)),
        _stamp_to_ns=DynamicObstacleSphereNode._stamp_to_ns,
        _select_robot_markers=lambda stamp: None if stamp.nanosec else 'valid',
        _robot_sphere_geometry=lambda marker, stamp: (np.zeros((35, 3)), np.ones(35)) if marker else None)
    assert DynamicObstacleSphereNode._prepare_latest_input(fake)
    assert len(decoded) == 1 and decoded[0].header.stamp.nanosec == 0
    assert fake.selected_robot_geometry[0] == 1_000_000_000


def test_continuously_new_unsynchronized_frames_do_not_reset_wait_deadline():
    calls = []
    now_wall = time.monotonic()
    buffer = LatestCloudBuffer(capacity=2)
    # All old frames have already been superseded; only fresh ones remain.
    buffer.add(cloud(1.), now_wall, 1_000_000_000)
    now = SimpleNamespace(nanoseconds=1_010_000_000, to_msg=lambda: Time(sec=1, nanosec=10_000_000))
    fake = SimpleNamespace(input_buffer=buffer, target_frame='base_link',
        input_wait_started=now_wall - .3, input_wait_reported=False,
        robot_component_filter_enabled=True, robot_component_filter_fail_closed=True,
        robot_marker_sync_wait_timeout_s=.25, get_clock=lambda: SimpleNamespace(now=lambda: now),
        _select_robot_markers=lambda stamp: None, _robot_sphere_geometry=lambda m, stamp: None,
        tracker=SimpleNamespace(clear=lambda: calls.append('clear')), refiner=None,
        _publish_sphere_cloud=lambda *a: None, _publish_markers=lambda *a: None,
        _publish_result_status=lambda *a: calls.append(a[3]))
    assert not DynamicObstacleSphereNode._prepare_latest_input(fake)
    assert calls == ['clear', 'input_expired_or_unsynchronized']
    assert not DynamicObstacleSphereNode._prepare_latest_input(fake)
    assert len(calls) == 2  # No endless new-frame-reset wait and no repeated clearing.


def component(index, center, size=.1):
    points = np.asarray([[x, y, z] for x in (-size, size) for y in (-size, size)
                         for z in (-size, size)]) + center
    return DynamicComponentResult(index, points, [], 1., np.empty((0, 3)), 'test')


def ball(x, radius=.08, component_id=0):
    return DynamicSphere(x, 0., 1., radius, radius, component_id)


def test_component_tracking_survives_frame_local_id_changes_and_measured_translation():
    tracker = DynamicSphereTracker(.1, 1., .3, 2, component_tracking_enabled=True)
    first = tracker.update([ball(0.)], 1., components=[component(0, (0, 0, 1))])
    second = tracker.update([ball(.2, component_id=7)], 1.1,
        components=[component(7, (.2, 0, 1))])
    assert len(second) == 1 and second[0].track_id == first[0].track_id
    assert second[0].x == .2


def test_large_torso_track_cannot_shrink_into_hand_track():
    tracker = DynamicSphereTracker(.3, .8, .3, 2, component_tracking_enabled=True)
    first = tracker.update([ball(0., .25)], 1., components=[component(0, (0, 0, 1))])
    second = tracker.update([ball(.02, .04)], 1.1, components=[component(0, (.02, 0, 1))])
    hand = next(s for s in second if s.track_id != first[0].track_id)
    assert hand.raw_radius == .04
    # The explicitly postponed unsupported-track deletion policy is NOT added.
    assert any(s.track_id == first[0].track_id for s in second)


def test_small_new_component_does_not_steal_old_large_object_id():
    tracker = DynamicSphereTracker(.3, 1., .3, 2, component_tracking_enabled=True)
    first = tracker.update([ball(0.)], 1., components=[component(0, (0, 0, 1), .3)])
    second = tracker.update([ball(.02)], 1.1, components=[component(0, (.02, 0, 1), .025)])
    assert len(second) == 2 and any(s.track_id != first[0].track_id for s in second)
    tracker.clear()
    assert not tracker.component_associator.shapes


def test_overlap_pruning_keeps_fresh_hand_instead_of_old_torso_when_coverage_equal():
    old = replace(ball(0., .25), track_id=1, age=20, confidence=1.)
    hand = replace(ball(0., .04), track_id=2, age=1, confidence=.6)
    points = np.asarray([[0., 0., 1.], [.01, 0., 1.]])
    kept, removed = prune_overlapping_tracked_spheres([points], [old, hand], .9, .02, .2,
        fresh_track_ids={2})
    assert kept == [hand] and removed == {1}
    # Unsupported, spatially separate old tracks are not immediately deleted.
    separate = replace(old, x=1.)
    kept, removed = prune_overlapping_tracked_spheres([points], [separate, hand], .9, .02, .2,
        fresh_track_ids={2})
    assert kept == [separate, hand] and not removed


def fusion_ball(x, radius, source):
    return FusionSphere(x, 0., 1., radius, radius, source)


def test_partial_human_overlap_does_not_erase_adjacent_box_without_support():
    box, human = fusion_ball(0., .2, 0), fusion_ball(.34, .15, 2)
    assert fuse_spheres([box], [], FusionParameters(), [human]) == [human]
    result = fuse_spheres([box], [], FusionParameters(coverage_guard_enabled=True), [human])
    assert result == [human, box]


def test_fusion_requires_every_box_support_voxel_not_just_high_percentage():
    box, human = fusion_ball(0., .2, 0), fusion_ball(.1, .2, 2)
    points = np.asarray([[.1, 0, 1]] * 99 + [[-.19, 0, 1]])
    result = fuse_spheres([box], [], FusionParameters(coverage_guard_enabled=True), [human],
                         support={box: points})
    assert box in result


def test_actual_duplicate_support_can_be_covered_by_union_of_human_spheres():
    box = fusion_ball(0., .3, 0)
    humans = [fusion_ball(-.2, .1, 2), fusion_ball(.2, .1, 2)]
    support = {box: np.asarray([[-.2, 0, 1], [.2, 0, 1]])}
    stats = {}
    result = fuse_spheres([box], [], FusionParameters(coverage_guard_enabled=True), humans,
                         support=support, stats=stats)
    assert result == humans and stats['coverage_removed'] == 1


def test_fusion_uses_surviving_winners_and_reports_cap_failure():
    box, dynamic, human = fusion_ball(0, .2, 0), fusion_ball(.15, .2, 1), fusion_ball(.3, .2, 2)
    support = {box: np.asarray([[0., 0., 1.]]), dynamic: np.asarray([[.3, 0., 1.]])}
    params = FusionParameters(coverage_guard_enabled=True)
    assert fuse_spheres([box], [dynamic], params, [human], support=support) == [human, box]
    stats = {}
    fuse_spheres([box], [dynamic], replace(params, max_total_spheres=1), [human], support=support, stats=stats)
    assert stats['coverage_cap_dropped'] == 1 and not stats['coverage_valid']


def test_missing_or_oversize_support_uses_sufficient_containment_not_sampling():
    small, large = fusion_ball(0., .05, 0), fusion_ball(0., .1, 2)
    params = FusionParameters(coverage_guard_enabled=True)
    assert fuse_spheres([small], [], params, [large]) == [large]
    box = fusion_ball(.15, .2, 0)
    assert box in fuse_spheres([box], [], params, [large],
        support={box: np.zeros((8193, 3))})


def test_fusion_validation_timeout_keeps_geometry():
    box, human = fusion_ball(0., .1, 0), fusion_ball(0., .2, 2)
    params = FusionParameters(coverage_guard_enabled=True, coverage_budget_ms=0.)
    assert fuse_spheres([box], [], params, [human]) == [human, box]


def test_support_packet_preserves_pairing_evidence_and_ignores_stale_generation():
    sphere = replace(ball(0.), track_id=12)
    comp = component(0, (0, 0, 1), .02)
    groups = current_support([sphere], [comp], .02, {12})
    packet = support_cloud(Header(frame_id='base_link', stamp=Time(sec=2)), [sphere], groups, 1.95)
    rows = read_numeric_cloud(packet, ('x', 'y', 'z', 'sphere_index', 'evidence_stamp'))
    assert len(rows) == 8 and np.all(rows[:, 3] == 0) and np.all(rows[:, 4] == 1.95)
    fused = fusion_ball(0., .08, 1)
    fake = SimpleNamespace(source_support={1: {}, 2: {}}, source_generation={1: 2_000_000_000},
        cache=SimpleNamespace(dynamic_spheres=[fused], human_spheres=[]),
        current_fusion_support={}, target_frame='base_link',
        _stamp_key=ObstacleSphereFusionNode._stamp_key)
    ObstacleSphereFusionNode._source_support_callback(fake, packet, 1)
    ObstacleSphereFusionNode._paired_source_support(fake, 2.)
    assert fused in fake.current_fusion_support
    fake.current_fusion_support = {}
    ObstacleSphereFusionNode._paired_source_support(fake, 2.3)
    assert not fake.current_fusion_support
    fake.source_generation[1] += 1
    ObstacleSphereFusionNode._paired_source_support(fake, 2.)
    assert not fake.current_fusion_support


def test_static_support_after_human_refit_preserves_only_retained_nonhuman_voxels():
    sphere = fusion_ball(0., .2, 0)
    points = np.asarray([[.1, 0., 1.], [-.15, 0., 1.]])
    fake = SimpleNamespace(cache=SimpleNamespace(static_spheres=[sphere]),
        static_support={1: (points, np.asarray([0, 0]))}, static_generation_stamp=1,
        human_filter=SimpleNamespace(enabled=True, excluded=lambda pts, now: pts[:, 0] > 0,
                                     last_stats=(1, 0)),
        static_free_support={}, static_esdf_cleared={}, static_refit_key=None,
        human_static_refit_enabled=True, human_static_refit_min_radius_m=.04,
        human_static_refit_coverage_tolerance_m=.02,
        last_human_filter_log=time.monotonic(), fusion_coverage_guard_enabled=True,
        current_fusion_support={})
    kept = ObstacleSphereFusionNode._human_filtered_static(fake, 1.)
    assert len(kept) == 1 and kept[0].raw_radius < sphere.raw_radius
    np.testing.assert_allclose(fake.current_fusion_support[kept[0]], [[-.15, 0, 1]])


def test_clock_rewind_cannot_reuse_previous_object_identity():
    tracker = DynamicSphereTracker(.3, 1., .3, 2, component_tracking_enabled=True)
    comp = component(0, (0, 0, 1))
    first = tracker.update([ball(0.)], 10., components=[comp])
    second = tracker.update([ball(0.)], 1., components=[comp])
    assert len(second) == 1 and second[0].track_id != first[0].track_id


@pytest.mark.parametrize('radius', [float('nan'), -1.])
def test_invalid_fusion_geometry_is_rejected_not_converted_to_empty(radius):
    fields = [PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
              for index, name in enumerate(('x', 'y', 'z', 'raw_radius', 'output_radius'))]
    message = point_cloud2.create_cloud(Header(frame_id='base_link'), fields, [(0., 0., 1., radius, .1)])
    with pytest.raises(ValueError):
        ObstacleSphereFusionNode._parse_cloud(SimpleNamespace(), message, 1)
