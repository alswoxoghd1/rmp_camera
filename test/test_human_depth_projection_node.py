from collections import deque
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from rmp_camera.human_depth_projection_node import HumanDepthProjectionNode


def message(stamp):
    sec = int(stamp)
    return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
        sec=sec, nanosec=round((stamp-sec)*1e9))))


def make_node(latest=True):
    clock = SimpleNamespace(ns=10_200_000_000)
    clock.now = lambda: SimpleNamespace(nanoseconds=clock.ns)
    node = SimpleNamespace(prefer_latest_frames=latest, max_frame_age_s=.25,
        max_sync_delta_s=.05, max_rate_hz=0.,
        depth_frames=deque(maxlen=12), mask_frames=deque(maxlen=2 if latest else 12),
        depth_camera_info=object(), mask_camera_info=object(),
        last_process_stamp_sec=None, last_mask_stamp_sec=None, last_pair_clock_ns=None,
        projected=[], get_clock=lambda: clock)
    node._stamp_sec = HumanDepthProjectionNode._stamp_sec
    node._pop_index = HumanDepthProjectionNode._pop_index
    node._project_pair = lambda d, m, delta: node.projected.append(
        (node._stamp_sec(d), node._stamp_sec(m), delta))
    for name in ('_check_pair_clock', '_process_nearest_pair', '_legacy_nearest_pair',
                 '_depth_callback', '_mask_callback'):
        setattr(node, name, MethodType(getattr(HumanDepthProjectionNode, name), node))
    return node, clock


def test_newest_pair_is_used_and_older_pairs_cannot_reappear():
    node, _ = make_node()
    node.depth_frames.extend([message(10.), message(10.1)])
    node.mask_frames.extend([message(10.), message(10.09)])
    node._process_nearest_pair()
    assert node.projected[0][:2] == (10.1, 10.09)
    node._process_nearest_pair()
    assert len(node.projected) == 1
    assert not node.depth_frames and not node.mask_frames


def test_late_mask_uses_retained_depth_history():
    node, _ = make_node()
    node._depth_callback(message(10.10))
    node._depth_callback(message(10.18))
    node._mask_callback(message(10.11))
    assert node.projected[0][:2] == (10.1, 10.11)


def test_no_synthetic_output_on_missing_or_stale_input():
    node, _ = make_node()
    node._depth_callback(message(9.))
    node._mask_callback(message(9.))
    assert node.projected == []
    node._depth_callback(message(10.1))
    assert node.projected == []


def test_new_mask_does_not_reuse_consumed_depth():
    node, _ = make_node()
    node._depth_callback(message(10.1))
    node._mask_callback(message(10.11))
    node._mask_callback(message(10.12))
    assert len(node.projected) == 1
    node._depth_callback(message(10.14))
    assert len(node.projected) == 2
    assert node.projected[-1][:2] == (10.14, 10.12)


def test_old_dds_packet_does_not_reset_current_epoch_watermarks():
    node, _ = make_node()
    node._depth_callback(message(10.1))
    node._mask_callback(message(10.11))
    node._depth_callback(message(9.))
    node._mask_callback(message(9.))
    assert len(node.projected) == 1
    assert node.last_process_stamp_sec == 10.1


def test_actual_clock_rewind_resets_both_queues_and_watermarks():
    node, clock = make_node()
    node._depth_callback(message(10.1))
    node._mask_callback(message(10.11))
    clock.ns = 1_100_000_000
    node._depth_callback(message(1.02))
    assert node.last_process_stamp_sec is None
    assert node.last_mask_stamp_sec is None
    node._mask_callback(message(1.03))
    assert node.projected[-1][:2] == (1.02, 1.03)


def test_old_epoch_packets_cannot_publish_future_pose_after_clock_rewind():
    node, clock = make_node()
    node._depth_callback(message(10.1))
    node._mask_callback(message(10.11))
    clock.ns = 1_100_000_000
    node._check_pair_clock()
    node._depth_callback(message(10.1))
    node._mask_callback(message(10.11))
    assert len(node.projected) == 1
    assert node.last_process_stamp_sec is None


def test_legacy_toggle_preserves_old_nearest_pair_choice():
    node, _ = make_node(latest=False)
    node.depth_frames.extend([message(10.), message(10.1)])
    node.mask_frames.extend([message(10.), message(10.09)])
    node._process_nearest_pair()
    assert node.projected[0][:2] == (10., 10.)


def test_rate_limit_remains_source_time_based():
    node, _ = make_node()
    node.max_rate_hz = 10.
    node._depth_callback(message(10.1))
    node._mask_callback(message(10.1))
    node._depth_callback(message(10.15))
    node._mask_callback(message(10.15))
    assert len(node.projected) == 1
