from types import SimpleNamespace
import time

import numpy as np

from rmp_camera.dynamic_obstacle_sphere_core import DynamicSphereParameters
from rmp_camera.dynamic_obstacle_sphere_node import DynamicObstacleSphereNode


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

    DynamicObstacleSphereNode._tick(fake_node)

    assert fake_node.latest_robot_marker_delta_s is None
    assert not fake_node.have_message


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
