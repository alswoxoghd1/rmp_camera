from types import SimpleNamespace
import time

import numpy as np
from builtin_interfaces.msg import Time

from rmp_camera.dynamic_obstacle_sphere_core import DynamicSphereParameters
from rmp_camera.dynamic_obstacle_sphere_node import DynamicObstacleSphereNode
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


class _FakeNow:
    nanoseconds = 1_000_000_000

    @staticmethod
    def to_msg():
        return SimpleNamespace(sec=1, nanosec=0)


class _FakeClock:
    @staticmethod
    def now():
        return _FakeNow()


def test_empty_timer_tick_does_not_recheck_old_point_marker_timestamp():
    def unexpected_marker_lookup(_stamp):
        raise AssertionError("empty timer tick must not select robot markers")

    fake_node = SimpleNamespace(
        ever_received=True,
        have_message=False,
        latest_points=np.asarray(((1.0, 2.0, 3.0),)),
        latest_stamp=SimpleNamespace(sec=10, nanosec=0),
        latest_points_received_monotonic=time.monotonic(),
        pending_points_frame=None,
        latest_robot_marker_delta_s=2.0,
        robot_component_filter_enabled=True,
        robot_component_filter_fail_closed=True,
        robot_component_overlap_threshold=0.10,
        robot_sphere_filter_margin_m=0.0,
        dynamic_tracking_enabled=False,
        publish_debug_clouds=False,
        _select_robot_markers=unexpected_marker_lookup,
        _core_parameters=lambda: DynamicSphereParameters(),
        get_clock=lambda: _FakeClock(),
        _publish_sphere_cloud=lambda _stamp, _spheres: None,
        _publish_markers=lambda _stamp, _spheres: None,
        _info_throttled=lambda _message: None,
    )
    # An empty scheduler tick is not an empty observation: don't invalidate
    # the negative optimizer cache between slower incoming camera frames.
    from rmp_camera.local_width_sphere_cover import LocalWidthAttemptCache
    fake_node.dynamic_local_width_integrated = True
    fake_node.local_width_attempt_cache = LocalWidthAttemptCache()
    descriptor = fake_node.local_width_attempt_cache.descriptor(
        np.asarray([[.025, .025, .025]]), .05, np.zeros(3), 2)
    fake_node.local_width_attempt_cache.remember(descriptor, 'no_valid_candidates', time.monotonic())
    fake_node.local_width_last_stamp_ns = 10_000_000_000

    DynamicObstacleSphereNode._tick(fake_node)

    assert fake_node.latest_robot_marker_delta_s is None
    assert not fake_node.have_message
    assert fake_node.local_width_attempt_cache.entries
    assert fake_node.local_width_last_stamp_ns == 10_000_000_000


def test_empty_timer_tick_uses_tracker_snapshot_not_detection_update():
    calls = {"snapshot": 0}

    class Tracker:
        @staticmethod
        def snapshot(timestamp_sec):
            assert timestamp_sec == 1.0
            calls["snapshot"] += 1
            return []

        @staticmethod
        def update(_spheres, _timestamp_sec):
            raise AssertionError("timer-only tick must not count as an update")

        @staticmethod
        def remove_tracks_in_bounds(_bounds, _padding):
            raise AssertionError("timer-only tick has no component bounds")

    fake_node = SimpleNamespace(
        ever_received=True,
        have_message=False,
        latest_points=np.asarray(((1.0, 2.0, 3.0),)),
        latest_stamp=SimpleNamespace(sec=10, nanosec=0),
        latest_points_received_monotonic=time.monotonic(),
        pending_points_frame=None,
        latest_robot_marker_delta_s=2.0,
        robot_component_filter_enabled=True,
        robot_component_overlap_threshold=0.10,
        robot_sphere_filter_margin_m=0.0,
        dynamic_tracking_enabled=True,
        dynamic_post_tracking_overlap_pruning_enabled=False,
        dynamic_association_distance_m=0.12,
        publish_debug_clouds=False,
        tracker=Tracker(),
        _core_parameters=lambda: DynamicSphereParameters(),
        get_clock=lambda: _FakeClock(),
        _publish_sphere_cloud=lambda _stamp, _spheres: None,
        _publish_markers=lambda _stamp, _spheres: None,
        _info_throttled=lambda _message: None,
    )

    DynamicObstacleSphereNode._tick(fake_node)

    assert calls["snapshot"] == 1


def test_steady_watchdog_clears_tracks_when_input_stops():
    state = {
        "tracker_cleared": False,
        "sphere_publications": [],
        "marker_publications": [],
    }

    class Tracker:
        @staticmethod
        def clear():
            state["tracker_cleared"] = True

    fake_node = SimpleNamespace(
        dynamic_input_stale_timeout_s=0.30,
        ever_received=True,
        input_stale_cleared=False,
        latest_points_received_monotonic=time.monotonic() - 0.50,
        input_watchdog_last_ros_time_ns=_FakeNow.nanoseconds,
        input_watchdog_last_ros_advance_monotonic=time.monotonic() - 0.50,
        have_message=True,
        pending_points_frame=(np.ones((1, 3)), Time()),
        latest_points=np.ones((1, 3)),
        tracker=Tracker(),
        publish_debug_clouds=False,
        get_clock=lambda: _FakeClock(),
        _publish_sphere_cloud=lambda _stamp, spheres: (
            state["sphere_publications"].append(spheres)),
        _publish_markers=lambda _stamp, spheres: (
            state["marker_publications"].append(spheres)),
        _info_throttled=lambda _message: None,
    )

    DynamicObstacleSphereNode._input_watchdog_tick(fake_node)

    assert state["tracker_cleared"]
    assert state["sphere_publications"] == [[]]
    assert state["marker_publications"] == [[]]
    assert fake_node.input_stale_cleared
    assert not fake_node.have_message
    assert fake_node.pending_points_frame is None
    assert fake_node.latest_points.shape == (0, 3)


def test_steady_watchdog_does_not_clear_fresh_input():
    fake_node = SimpleNamespace(
        dynamic_input_stale_timeout_s=0.30,
        ever_received=True,
        input_stale_cleared=False,
        latest_points_received_monotonic=time.monotonic(),
        input_watchdog_last_ros_time_ns=_FakeNow.nanoseconds,
        input_watchdog_last_ros_advance_monotonic=time.monotonic() - 0.50,
        get_clock=lambda: _FakeClock(),
        tracker=SimpleNamespace(clear=lambda: (_ for _ in ()).throw(
            AssertionError("fresh input must not clear tracker"))),
    )

    DynamicObstacleSphereNode._input_watchdog_tick(fake_node)

    assert not fake_node.input_stale_cleared


def test_steady_watchdog_does_not_clear_while_ros_time_advances():
    class AdvancingNow(_FakeNow):
        nanoseconds = 2_000_000_000

    class AdvancingClock:
        @staticmethod
        def now():
            return AdvancingNow()

    fake_node = SimpleNamespace(
        dynamic_input_stale_timeout_s=0.30,
        ever_received=True,
        input_stale_cleared=False,
        latest_points_received_monotonic=time.monotonic() - 0.50,
        input_watchdog_last_ros_time_ns=_FakeNow.nanoseconds,
        input_watchdog_last_ros_advance_monotonic=time.monotonic() - 0.50,
        get_clock=lambda: AdvancingClock(),
        tracker=SimpleNamespace(clear=lambda: (_ for _ in ()).throw(
            AssertionError("advancing ROS time must use normal TTL expiry"))),
    )

    DynamicObstacleSphereNode._input_watchdog_tick(fake_node)

    assert not fake_node.input_stale_cleared
    assert fake_node.input_watchdog_last_ros_time_ns == AdvancingNow.nanoseconds


def test_frame_waits_for_delayed_robot_marker_before_fail_closed_drop():
    def unavailable_markers(_stamp):
        return None

    class Tracker:
        @staticmethod
        def clear():
            raise AssertionError("tracker must not clear during marker wait")

    fake_node = SimpleNamespace(
        ever_received=True,
        have_message=True,
        latest_points=np.asarray(((1.0, 2.0, 3.0),)),
        latest_stamp=SimpleNamespace(sec=10, nanosec=0),
        latest_points_received_monotonic=time.monotonic(),
        pending_points_frame=None,
        latest_robot_marker_delta_s=0.20,
        robot_component_filter_enabled=True,
        robot_component_filter_fail_closed=True,
        robot_marker_sync_wait_timeout_s=0.25,
        _select_robot_markers=unavailable_markers,
        _robot_sphere_geometry=lambda _markers, _stamp: None,
        tracker=Tracker(),
    )

    DynamicObstacleSphereNode._tick(fake_node)

    assert fake_node.pending_points_frame is not None
    assert not fake_node.have_message


def test_frame_drops_fail_closed_after_robot_marker_wait_timeout():
    state = {"cleared": False, "sphere_publications": []}

    class Tracker:
        @staticmethod
        def clear():
            state["cleared"] = True

    fake_node = SimpleNamespace(
        ever_received=True,
        have_message=True,
        latest_points=np.asarray(((1.0, 2.0, 3.0),)),
        latest_stamp=SimpleNamespace(sec=10, nanosec=0),
        latest_points_received_monotonic=time.monotonic() - 0.30,
        pending_points_frame=None,
        latest_robot_marker_delta_s=0.20,
        robot_component_filter_enabled=True,
        robot_component_filter_fail_closed=True,
        robot_marker_sync_wait_timeout_s=0.25,
        publish_debug_clouds=False,
        _select_robot_markers=lambda _stamp: None,
        _robot_sphere_geometry=lambda _markers, _stamp: None,
        _warn_throttled=lambda _message: None,
        get_clock=lambda: _FakeClock(),
        _publish_sphere_cloud=lambda _stamp, spheres: (
            state["sphere_publications"].append(spheres)),
        _publish_markers=lambda _stamp, _spheres: None,
        tracker=Tracker(),
    )

    DynamicObstacleSphereNode._tick(fake_node)

    assert fake_node.pending_points_frame is None
    assert state["cleared"]
    assert state["sphere_publications"] == [[]]


def test_markers_clear_previous_process_and_have_finite_lifetime():
    published = []
    fake_node = SimpleNamespace(
        target_frame="base_link",
        clear_markers_on_next_publish=True,
        previous_marker_ids=set(),
        marker_lifetime_s=0.5,
        marker_pub=SimpleNamespace(publish=published.append),
        _header=lambda stamp: Header(stamp=stamp, frame_id="base_link"),
    )
    sphere = SimpleNamespace(
        track_id=7,
        x=0.1,
        y=0.2,
        z=0.3,
        output_radius=0.15,
    )
    stamp = Time(sec=1, nanosec=0)

    DynamicObstacleSphereNode._publish_markers(fake_node, stamp, [sphere])

    first_markers = published[-1].markers
    assert first_markers[0].action == Marker.DELETEALL
    assert first_markers[1].action == Marker.ADD
    assert first_markers[1].id == 7
    assert first_markers[1].lifetime.sec == 0
    assert first_markers[1].lifetime.nanosec == 500_000_000

    DynamicObstacleSphereNode._publish_markers(fake_node, stamp, [])

    second_markers = published[-1].markers
    assert len(second_markers) == 1
    assert second_markers[0].action == Marker.DELETE
    assert second_markers[0].id == 7
