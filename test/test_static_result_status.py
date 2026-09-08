import pytest
import numpy as np
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticStatus

from rmp_camera.static_result_status import (
    StaticResultGate, make_result_status, read_result_status,
)
from rmp_camera.obstacle_sphere_fusion_core import FusionParameters, FusionSphere, SphereFusionCache
from rmp_camera.esdf_medial_sphere_core import (
    has_static_obstacle_component, previous_support_is_observed, generate_medial_spheres,
)


def test_status_roundtrip_and_rejects_inconsistent_message():
    for state in ('valid', 'empty', 'unknown'):
        msg = make_result_status(Time(sec=3), 'base_link', state, 'test')
        assert read_result_status(msg) == state
    msg.status[0].level = DiagnosticStatus.OK
    with pytest.raises(ValueError):
        read_result_status(msg)


@pytest.mark.parametrize('first,second', [('cloud', 'status'), ('status', 'cloud')])
def test_gate_matches_exact_generation_in_either_order(first, second):
    gate = StaticResultGate()
    payload = {'cloud': object(), 'status': 'empty'}
    assert gate.add(100, first, payload[first], 0.0) is None
    assert gate.add(200, second, payload[second], 0.1) is None
    assert gate.add(100, second, payload[second], 0.2) == (payload['cloud'], 'empty')
    assert gate.add(100, first, payload[first], 0.3) is None  # no duplicate confirmations


def test_error_missing_metadata_timeout_and_clock_reset():
    gate = StaticResultGate(timeout_s=.2, capacity=2)
    assert gate.add(100, 'cloud', [], 0.0) is None
    assert gate.add(100, 'status', 'unknown', .1) is None
    assert gate.add(100, 'status', 'empty', .1) is None
    assert gate.add(200, 'cloud', [], .2) is None
    assert gate.add(200, 'status', 'empty', .5) is None  # original cloud expired
    for key in range(300, 310):
        gate.add(key, 'cloud', [], .5)
    assert len(gate.pending) == 2
    gate.reset()
    assert gate.last_completed is None
    assert not gate.pending
    assert gate.add(50, 'cloud', [], .6) is None
    assert gate.add(50, 'status', 'empty', .7) == ([], 'empty')


def test_valid_empty_clears_immediately_but_unknown_preserves_static_obstacle():
    cache = SphereFusionCache(FusionParameters(static_empty_confirmation_frames=2))
    obstacle = FusionSphere(0., .2, 1., .2, .2, 0)
    cache.update_static([obstacle], 'base_link', 1.0)
    cache.update_static([], 'base_link', 2.0)
    assert cache.static_spheres == [obstacle]
    cache.invalidate_static_observation()
    cache.update_static([], 'base_link', 3.0)
    assert cache.static_spheres == [obstacle]  # interruption breaks consecutive empty evidence
    cache.update_static([], 'base_link', 4.0, confirmed_empty=True)
    assert not cache.static_spheres
    cache.update_static([obstacle], 'base_link', 5.0)
    cache.invalidate_static_observation()
    assert cache.static_spheres == [obstacle]  # real nonhuman obstacle stays through failures


def test_previous_support_unknown_or_out_of_bounds_blocks_empty_confirmation():
    observed = np.ones((3, 3, 3), dtype=bool)
    point = np.array([[.075, .075, .075]])
    assert previous_support_is_observed(point, observed, np.zeros(3), .05)
    observed[1, 1, 1] = False
    assert not previous_support_is_observed(point, observed, np.zeros(3), .05)
    assert not previous_support_is_observed([[1, 1, 1]], observed, np.zeros(3), .05)
    assert not previous_support_is_observed([], np.zeros_like(observed), np.zeros(3), .05)


def test_empty_checks_do_not_increase_full_optimization_rate_or_skip_legacy_ticks():
    from types import SimpleNamespace
    from rmp_camera.esdf_medial_sphere_node import EsdfMedialSphereNode
    fake = SimpleNamespace(update_rate_hz=1., empty_check_rate_hz=5., last_full_solve_stamp=10.)
    assert not EsdfMedialSphereNode._full_solve_due(fake, 10.2)
    assert not EsdfMedialSphereNode._full_solve_due(fake, 10.8)
    assert EsdfMedialSphereNode._full_solve_due(fake, 11.0)
    assert EsdfMedialSphereNode._full_solve_due(fake, 1.0)  # new bag pass
    fake.empty_check_rate_hz = 0.
    assert EsdfMedialSphereNode._full_solve_due(fake, 10.999)  # normal timer jitter is not a skipped solve


def test_fast_empty_check_keeps_real_component_and_uses_existing_size_threshold():
    grid = np.full((12, 12, 12), .2)
    grid[3:6, 3:6, 3:6] = -.1
    assert has_static_obstacle_component(grid, np.zeros(3), .05, min_component_voxels=27)
    assert not has_static_obstacle_component(grid, np.zeros(3), .05, min_component_voxels=28)
    # This is the same admission rule as the full solver, not a looser shortcut.
    assert not generate_medial_spheres(grid, np.zeros(3), .05, min_component_voxels=28).spheres


def test_fast_empty_robot_only_does_not_erase_separate_ordinary_obstacle():
    grid = np.full((14, 14, 14), .2)
    grid[2:5, 2:5, 2:5] = -.1
    params = dict(min_component_voxels=20, robot_sphere_centers=[[.175, .175, .175]],
                  robot_sphere_radii=[.2], robot_component_overlap_threshold=.02)
    assert not has_static_obstacle_component(grid, np.zeros(3), .05, **params)
    assert not generate_medial_spheres(grid, np.zeros(3), .05, **params).spheres
    grid[10:13, 10:13, 10:13] = -.1
    assert has_static_obstacle_component(grid, np.zeros(3), .05, **params)


def test_fusion_adapter_unknown_holds_valid_empty_clears_and_new_obstacle_returns(monkeypatch):
    import time
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header
    from diagnostic_msgs.msg import DiagnosticArray
    from rmp_camera.obstacle_sphere_fusion_node import ObstacleSphereFusionNode

    # Never inject synthetic collision geometry into the user's live ROS graph.
    monkeypatch.setenv('ROS_DOMAIN_ID', '79')
    rclpy.init()
    fusion = ObstacleSphereFusionNode()
    source = Node('static_validity_test_source')
    executor = SingleThreadedExecutor()
    executor.add_node(fusion)
    executor.add_node(source)
    clouds = []
    source.create_subscription(PointCloud2, fusion.combined_sphere_cloud_topic,
                               lambda m: clouds.append(m.width * m.height), 10)
    cloud_pub = source.create_publisher(PointCloud2, fusion.static_sphere_cloud_topic, 5)
    status_pub = source.create_publisher(DiagnosticArray, fusion.static_result_status_topic, 5)
    fields = [PointField(name=name, offset=i * 4, datatype=PointField.FLOAT32, count=1)
              for i, name in enumerate(('x', 'y', 'z', 'raw_radius', 'output_radius'))]

    def spin_until(predicate, timeout=2.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=.02)
            if predicate():
                return
        assert predicate()

    def send(rows, state):
        stamp = source.get_clock().now().to_msg()
        cloud_pub.publish(point_cloud2.create_cloud(Header(stamp=stamp, frame_id='base_link'), fields, rows))
        if state is not None:
            status_pub.publish(make_result_status(stamp, 'base_link', state))

    try:
        spin_until(lambda: cloud_pub.get_subscription_count() > 0 and status_pub.get_subscription_count() > 0)
        obstacle = [(0., .3, 1., .15, .15)]
        send(obstacle, 'valid')
        spin_until(lambda: bool(clouds) and clouds[-1] == 1)
        send([], 'unknown')
        end = time.monotonic() + .15
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=.02)
        assert len(fusion.cache.static_spheres) == 1
        assert clouds[-1] == 1
        send([], None)  # Unpaired empty cloud also cannot delete the obstacle.
        end = time.monotonic() + .15
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=.02)
        assert len(fusion.cache.static_spheres) == 1
        send([], 'empty')
        spin_until(lambda: clouds[-1] == 0)
        assert not fusion.cache.static_spheres
        send(obstacle, 'valid')
        spin_until(lambda: clouds[-1] == 1)
    finally:
        executor.shutdown()
        source.destroy_node()
        fusion.destroy_node()
        rclpy.shutdown()
