"""ROS 2 adapter from native nvblox dynamic points to tracked spheres."""

from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.dynamic_obstacle_sphere_core import (
    DynamicSphereParameters,
    DynamicSphereTracker,
    generate_dynamic_spheres,
)


def _rotation_matrix(x, y, z, w):
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


class DynamicObstacleSphereNode(Node):
    def __init__(self):
        super().__init__("dynamic_obstacle_sphere_node")
        self._declare_parameters()
        self._read_parameters()
        self.tf_buffer = Buffer()
        self._core_parameters().validate()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_points = np.empty((0, 3), dtype=np.float64)
        self.latest_stamp = self.get_clock().now().to_msg()
        self.latest_frame = self.target_frame
        self.have_message = False
        self.ever_received = False
        self.previous_marker_ids: set[int] = set()
        self.last_warn_time = 0.0
        self.last_info_time = 0.0
        self.tracker = DynamicSphereTracker(
            self.dynamic_association_distance_m,
            self.dynamic_smoothing_alpha,
            self.dynamic_sphere_ttl_sec,
            self.dynamic_max_missed_updates,
        )
        self.subscription = self.create_subscription(
            PointCloud2, self.input_dynamic_points_topic, self._points_callback,
            qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 2)
        self.sphere_pub = self.create_publisher(PointCloud2, self.sphere_cloud_topic, 2)
        self.voxel_pub = self.create_publisher(PointCloud2, self.dynamic_voxel_cloud_topic, 2)
        self.uncovered_pub = self.create_publisher(
            PointCloud2, self.uncovered_voxel_cloud_topic, 2)
        period = 1.0 if self.dynamic_update_rate_hz <= 0.0 else max(
            0.01, 1.0 / self.dynamic_update_rate_hz)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"Dynamic sphere adapter started: input={self.input_dynamic_points_topic}, "
            f"frame={self.target_frame}, rate={self.dynamic_update_rate_hz:.2f} Hz, "
            f"merge={self.dynamic_enable_agglomerative_merge}, "
            f"merge_radius<={self.dynamic_merge_max_radius_m:.3f} m, "
            f"merge_growth<={self.dynamic_merge_max_radius_growth_ratio:.3f}, "
            f"merge_gap<={self.dynamic_merge_max_gap_m:.3f} m, "
            f"empty_guard={self.dynamic_merge_enable_empty_space_guard}")

    def _declare_parameters(self):
        defaults = {
            "input_dynamic_points_topic": "/nvblox_node/dynamic_points",
            "target_frame": "base_link",
            "marker_topic": "/rmp_camera/dynamic_obstacle_sphere_markers",
            "sphere_cloud_topic": "/rmp_camera/dynamic_obstacle_sphere_cloud",
            "dynamic_voxel_cloud_topic": "/rmp_camera/dynamic_obstacle_voxels",
            "uncovered_voxel_cloud_topic": "/rmp_camera/dynamic_uncovered_voxels",
            "min_x_m": -3.0, "max_x_m": 3.0,
            "min_y_m": -3.0, "max_y_m": 3.0,
            "min_z_m": -0.2, "max_z_m": 2.5,
            "max_range_from_base_m": 5.0,
            "dynamic_voxel_size_m": 0.05,
            "dynamic_min_component_voxels": 6,
            "dynamic_max_component_voxels": 120000,
            "dynamic_max_components": 32,
            "dynamic_max_input_points": 250000,
            "dynamic_connectivity": 18,
            "dynamic_dilation_voxels": 0,
            "dynamic_closing_iterations": 1,
            "dynamic_minimum_center_spacing_m": 0.08,
            "dynamic_min_raw_radius_m": 0.025,
            "dynamic_max_raw_radius_m": 0.45,
            "dynamic_min_useful_adaptive_radius_m": 0.075,
            "dynamic_enable_fixed_radius_fallback": True,
            "dynamic_fixed_radius_m": 0.10,
            "dynamic_enable_greedy_set_cover": True,
            "dynamic_enable_single_sphere_replacement": True,
            "dynamic_single_sphere_max_radius_m": 0.18,
            "dynamic_target_coverage": 0.95,
            "dynamic_coverage_tolerance_m": 0.02,
            "dynamic_safety_margin_m": 0.015,
            "dynamic_redundancy_tolerance_m": 0.01,
            "dynamic_max_spheres_per_component": 96,
            "dynamic_max_iterations_per_component": 256,
            "dynamic_max_total_spheres": 384,
            "dynamic_processing_budget_ms": 35.0,
            "dynamic_max_local_grid_voxels": 1200000,
            "dynamic_enable_agglomerative_merge": True,
            "dynamic_merge_max_radius_m": 0.30,
            "dynamic_merge_max_radius_growth_ratio": 1.50,
            "dynamic_merge_max_gap_m": 0.08,
            "dynamic_merge_enable_empty_space_guard": False,
            "dynamic_merge_max_empty_fraction": 0.70,
            "dynamic_merge_max_validation_voxels": 50000,
            "dynamic_tracking_enabled": True,
            "dynamic_association_distance_m": 0.20,
            "dynamic_smoothing_alpha": 0.55,
            "dynamic_sphere_ttl_sec": 0.8,
            "dynamic_max_missed_updates": 8,
            "dynamic_update_rate_hz": 10.0,
            "publish_debug_clouds": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        string_names = (
            "input_dynamic_points_topic", "target_frame", "marker_topic",
            "sphere_cloud_topic", "dynamic_voxel_cloud_topic",
            "uncovered_voxel_cloud_topic")
        for name in string_names:
            setattr(self, name, str(self.get_parameter(name).value))
        float_names = (
            "min_x_m", "max_x_m", "min_y_m", "max_y_m", "min_z_m", "max_z_m",
            "max_range_from_base_m", "dynamic_voxel_size_m",
            "dynamic_minimum_center_spacing_m", "dynamic_min_raw_radius_m",
            "dynamic_max_raw_radius_m", "dynamic_min_useful_adaptive_radius_m",
            "dynamic_fixed_radius_m", "dynamic_target_coverage",
            "dynamic_single_sphere_max_radius_m",
            "dynamic_coverage_tolerance_m", "dynamic_safety_margin_m",
            "dynamic_redundancy_tolerance_m", "dynamic_processing_budget_ms",
            "dynamic_association_distance_m", "dynamic_smoothing_alpha",
            "dynamic_sphere_ttl_sec", "dynamic_update_rate_hz",
            "dynamic_merge_max_radius_m", "dynamic_merge_max_radius_growth_ratio",
            "dynamic_merge_max_gap_m", "dynamic_merge_max_empty_fraction")
        for name in float_names:
            setattr(self, name, float(self.get_parameter(name).value))
        int_names = (
            "dynamic_min_component_voxels", "dynamic_max_component_voxels",
            "dynamic_max_components", "dynamic_max_input_points", "dynamic_connectivity",
            "dynamic_dilation_voxels", "dynamic_closing_iterations",
            "dynamic_max_spheres_per_component", "dynamic_max_iterations_per_component",
            "dynamic_max_total_spheres", "dynamic_max_local_grid_voxels",
            "dynamic_max_missed_updates",
            "dynamic_merge_max_validation_voxels")
        for name in int_names:
            setattr(self, name, int(self.get_parameter(name).value))
        self.dynamic_enable_fixed_radius_fallback = bool(
            self.get_parameter("dynamic_enable_fixed_radius_fallback").value)
        self.dynamic_enable_greedy_set_cover = bool(
            self.get_parameter("dynamic_enable_greedy_set_cover").value)
        self.dynamic_enable_single_sphere_replacement = bool(
            self.get_parameter("dynamic_enable_single_sphere_replacement").value)
        self.dynamic_enable_agglomerative_merge = bool(
            self.get_parameter("dynamic_enable_agglomerative_merge").value)
        self.dynamic_merge_enable_empty_space_guard = bool(
            self.get_parameter("dynamic_merge_enable_empty_space_guard").value)
        self.dynamic_tracking_enabled = bool(
            self.get_parameter("dynamic_tracking_enabled").value)
        self.publish_debug_clouds = bool(self.get_parameter("publish_debug_clouds").value)

    def _core_parameters(self):
        return DynamicSphereParameters(
            voxel_size_m=self.dynamic_voxel_size_m,
            min_component_voxels=self.dynamic_min_component_voxels,
            max_component_voxels=self.dynamic_max_component_voxels,
            max_components=self.dynamic_max_components,
            max_input_points=self.dynamic_max_input_points,
            connectivity=self.dynamic_connectivity,
            dilation_voxels=self.dynamic_dilation_voxels,
            closing_iterations=self.dynamic_closing_iterations,
            minimum_center_spacing_m=self.dynamic_minimum_center_spacing_m,
            min_raw_radius_m=self.dynamic_min_raw_radius_m,
            max_raw_radius_m=self.dynamic_max_raw_radius_m,
            min_useful_adaptive_radius_m=self.dynamic_min_useful_adaptive_radius_m,
            enable_fixed_radius_fallback=self.dynamic_enable_fixed_radius_fallback,
            fixed_radius_m=self.dynamic_fixed_radius_m,
            enable_greedy_set_cover=self.dynamic_enable_greedy_set_cover,
            enable_single_sphere_replacement=self.dynamic_enable_single_sphere_replacement,
            single_sphere_max_radius_m=self.dynamic_single_sphere_max_radius_m,
            target_coverage=self.dynamic_target_coverage,
            coverage_tolerance_m=self.dynamic_coverage_tolerance_m,
            safety_margin_m=self.dynamic_safety_margin_m,
            redundancy_tolerance_m=self.dynamic_redundancy_tolerance_m,
            max_spheres_per_component=self.dynamic_max_spheres_per_component,
            max_iterations_per_component=self.dynamic_max_iterations_per_component,
            max_total_spheres=self.dynamic_max_total_spheres,
            processing_budget_ms=self.dynamic_processing_budget_ms,
            max_local_grid_voxels=self.dynamic_max_local_grid_voxels,
            dynamic_enable_agglomerative_merge=self.dynamic_enable_agglomerative_merge,
            dynamic_merge_max_radius_m=self.dynamic_merge_max_radius_m,
            dynamic_merge_max_radius_growth_ratio=self.dynamic_merge_max_radius_growth_ratio,
            dynamic_merge_max_gap_m=self.dynamic_merge_max_gap_m,
            dynamic_merge_enable_empty_space_guard=self.dynamic_merge_enable_empty_space_guard,
            dynamic_merge_max_empty_fraction=self.dynamic_merge_max_empty_fraction,
            dynamic_merge_max_validation_voxels=self.dynamic_merge_max_validation_voxels,
            min_x_m=self.min_x_m, max_x_m=self.max_x_m,
            min_y_m=self.min_y_m, max_y_m=self.max_y_m,
            min_z_m=self.min_z_m, max_z_m=self.max_z_m,
            max_range_from_base_m=self.max_range_from_base_m)

    def _points_callback(self, message):
        names = {field.name for field in message.fields}
        if not {"x", "y", "z"}.issubset(names):
            self._warn_throttled("dynamic PointCloud2 is missing x/y/z fields")
            return
        rows = point_cloud2.read_points_list(
            message, field_names=("x", "y", "z"), skip_nans=True)
        points = np.asarray([(row.x, row.y, row.z) for row in rows], dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float64)
        elif message.header.frame_id != self.target_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, message.header.frame_id, message.header.stamp,
                    timeout=Duration(seconds=0.05))
                q = transform.transform.rotation
                t = transform.transform.translation
                points = points @ _rotation_matrix(q.x, q.y, q.z, q.w).T
                points += np.asarray((t.x, t.y, t.z), dtype=np.float64)
            except TransformException as exc:
                self._warn_throttled(
                    f"dropping dynamic points: cannot transform "
                    f"{message.header.frame_id}->{self.target_frame}: {exc}")
                return
        self.latest_points = points
        self.latest_stamp = message.header.stamp
        self.latest_frame = self.target_frame
        self.have_message = True
        self.ever_received = True

    def _tick(self):
        if not self.ever_received:
            return
        points = self.latest_points if self.have_message else np.empty((0, 3), dtype=np.float64)
        self.have_message = False
        result = generate_dynamic_spheres(points, self._core_parameters())
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        spheres = result.spheres
        if self.dynamic_tracking_enabled:
            spheres = self.tracker.update(spheres, now_sec)
        stamp = now.to_msg()
        self._publish_sphere_cloud(stamp, spheres)
        self._publish_markers(stamp, spheres)
        if self.publish_debug_clouds:
            self._publish_xyz_cloud(self.voxel_pub, stamp, result.voxel_centers)
            self._publish_xyz_cloud(self.uncovered_pub, stamp, result.uncovered_voxels)
        coverage = 1.0 - len(result.uncovered_voxels) / max(
            1, len(result.voxel_centers))
        self._info_throttled(
            f"dynamic points={len(points)}, voxels={len(result.voxel_centers)}, "
            f"pre_merge_spheres={sum(c.pre_merge_sphere_count for c in result.components)}, "
            f"generated_spheres={len(result.spheres)}, "
            f"tracked_spheres={len(spheres)}, "
            f"coverage={coverage:.3f}, uncovered={len(result.uncovered_voxels)}, "
            f"core={result.elapsed_ms:.1f} ms")

    def _publish_sphere_cloud(self, stamp, spheres):
        rows = [(
            s.x, s.y, s.z, s.raw_radius, s.output_radius, s.component_id,
            s.component_coverage, s.track_id, s.age, s.confidence) for s in spheres]
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="raw_radius", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="output_radius", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="component_id", offset=20, datatype=PointField.INT32, count=1),
            PointField(name="component_coverage", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="track_id", offset=28, datatype=PointField.INT32, count=1),
            PointField(name="age", offset=32, datatype=PointField.INT32, count=1),
            PointField(name="confidence", offset=36, datatype=PointField.FLOAT32, count=1),
        ]
        self.sphere_pub.publish(point_cloud2.create_cloud(self._header(stamp), fields, rows))

    def _publish_markers(self, stamp, spheres):
        markers = []
        current = set()
        for index, sphere in enumerate(spheres):
            marker_id = sphere.track_id if sphere.track_id >= 0 else index
            current.add(marker_id)
            marker = Marker()
            marker.header = self._header(stamp)
            marker.ns = "dynamic_obstacle_spheres"
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=sphere.x, y=sphere.y, z=sphere.z)
            marker.pose.orientation.w = 1.0
            diameter = 2.0 * sphere.output_radius
            marker.scale.x = marker.scale.y = marker.scale.z = diameter
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.12, 0.04, 0.70
            markers.append(marker)
        for marker_id in sorted(self.previous_marker_ids - current):
            marker = Marker()
            marker.header = self._header(stamp)
            marker.ns = "dynamic_obstacle_spheres"
            marker.id = marker_id
            marker.action = Marker.DELETE
            markers.append(marker)
        self.previous_marker_ids = current
        self.marker_pub.publish(MarkerArray(markers=markers))

    def _publish_xyz_cloud(self, publisher, stamp, points):
        publisher.publish(point_cloud2.create_cloud_xyz32(
            self._header(stamp), np.asarray(points, dtype=np.float32).tolist()))

    def _header(self, stamp):
        return Header(stamp=stamp, frame_id=self.target_frame)

    def _info_throttled(self, message):
        now = time.monotonic()
        if now - self.last_info_time >= 2.0:
            self.last_info_time = now
            self.get_logger().info(message)

    def _warn_throttled(self, message):
        now = time.monotonic()
        if now - self.last_warn_time >= 2.0:
            self.last_warn_time = now
            self.get_logger().warn(message)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleSphereNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
