"""ROS 2 latest-cache fusion adapter for static and dynamic sphere clouds."""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.obstacle_sphere_fusion_core import (
    FusionParameters,
    SphereFusionCache,
    spheres_from_field_rows,
)


class ObstacleSphereFusionNode(Node):
    def __init__(self):
        super().__init__("obstacle_sphere_fusion_node")
        self._declare_parameters()
        self._read_parameters()
        self.cache = SphereFusionCache(self._core_parameters())
        self.previous_marker_keys: set[tuple[str, int]] = set()
        self.last_warn_time = 0.0
        self.static_sub = self.create_subscription(
            PointCloud2, self.static_sphere_cloud_topic, self._static_callback, 5)
        self.dynamic_sub = self.create_subscription(
            PointCloud2, self.dynamic_sphere_cloud_topic, self._dynamic_callback, 5)
        self.cloud_pub = self.create_publisher(PointCloud2, self.combined_sphere_cloud_topic, 2)
        self.marker_pub = self.create_publisher(MarkerArray, self.combined_marker_topic, 2)
        period = 1.0 if self.fusion_rate_hz <= 0.0 else max(0.01, 1.0 / self.fusion_rate_hz)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"Sphere fusion started: static={self.static_sphere_cloud_topic}, "
            f"dynamic={self.dynamic_sphere_cloud_topic}, frame={self.target_frame}")

    def _declare_parameters(self):
        defaults = {
            "static_sphere_cloud_topic": "/rmp_camera/esdf_medial_sphere_cloud",
            "dynamic_sphere_cloud_topic": "/rmp_camera/dynamic_obstacle_sphere_cloud",
            "combined_sphere_cloud_topic": "/rmp_camera/combined_obstacle_sphere_cloud",
            "combined_marker_topic": "/rmp_camera/combined_obstacle_sphere_markers",
            "target_frame": "base_link",
            "fusion_rate_hz": 10.0,
            "fusion_overlap_tolerance_m": 0.01,
            "fusion_dynamic_priority": True,
            "fusion_hysteresis_sec": 0.5,
            "fusion_max_total_spheres": 512,
            "static_cache_expire_enabled": False,
            "static_empty_confirmation_frames": 3,
            "dynamic_handover_grace_sec": 0.5,
            "keep_dynamic_until_static_overlap": True,
            "dynamic_absolute_max_ttl_sec": 3.0,
            "static_marker_color": [0.15, 0.55, 1.0],
            "dynamic_marker_color": [1.0, 0.12, 0.04],
            "marker_alpha": 0.65,
            "marker_lifetime_s": 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        strings = (
            "static_sphere_cloud_topic", "dynamic_sphere_cloud_topic",
            "combined_sphere_cloud_topic", "combined_marker_topic", "target_frame")
        for name in strings:
            setattr(self, name, str(self.get_parameter(name).value))
        floats = (
            "fusion_rate_hz", "fusion_overlap_tolerance_m", "fusion_hysteresis_sec",
            "dynamic_handover_grace_sec", "dynamic_absolute_max_ttl_sec",
            "marker_alpha", "marker_lifetime_s")
        for name in floats:
            setattr(self, name, float(self.get_parameter(name).value))
        integers = ("fusion_max_total_spheres", "static_empty_confirmation_frames")
        for name in integers:
            setattr(self, name, int(self.get_parameter(name).value))
        booleans = (
            "fusion_dynamic_priority", "static_cache_expire_enabled",
            "keep_dynamic_until_static_overlap")
        for name in booleans:
            setattr(self, name, bool(self.get_parameter(name).value))
        self.static_marker_color = tuple(float(v) for v in self.get_parameter(
            "static_marker_color").value)
        self.dynamic_marker_color = tuple(float(v) for v in self.get_parameter(
            "dynamic_marker_color").value)
        if len(self.static_marker_color) != 3 or len(self.dynamic_marker_color) != 3:
            raise ValueError("marker colors must contain exactly three RGB values")

    def _core_parameters(self):
        return FusionParameters(
            target_frame=self.target_frame,
            overlap_tolerance_m=self.fusion_overlap_tolerance_m,
            dynamic_priority=self.fusion_dynamic_priority,
            hysteresis_sec=self.fusion_hysteresis_sec,
            max_total_spheres=self.fusion_max_total_spheres,
            static_cache_expire_enabled=self.static_cache_expire_enabled,
            static_empty_confirmation_frames=self.static_empty_confirmation_frames,
            dynamic_handover_grace_sec=self.dynamic_handover_grace_sec,
            keep_dynamic_until_static_overlap=self.keep_dynamic_until_static_overlap,
            dynamic_absolute_max_ttl_sec=self.dynamic_absolute_max_ttl_sec)

    @staticmethod
    def _stamp_sec(message):
        return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9

    def _parse_cloud(self, message, source_type):
        names = [field.name for field in message.fields]
        required = {"x", "y", "z", "raw_radius", "output_radius"}
        if not required.issubset(names):
            raise ValueError(f"sphere cloud missing fields: {sorted(required - set(names))}")
        rows = point_cloud2.read_points_list(message, field_names=names, skip_nans=True)
        positional = [[getattr(row, name) for name in names] for row in rows]
        return spheres_from_field_rows(names, positional, source_type)

    def _static_callback(self, message):
        try:
            spheres = self._parse_cloud(message, 0)
            accepted = self.cache.update_static(
                spheres, message.header.frame_id, self._stamp_sec(message))
            if not accepted:
                self._warn_throttled(self.cache.last_rejection_reason)
        except (ValueError, TypeError) as exc:
            self._warn_throttled(f"rejecting static sphere cloud: {exc}")

    def _dynamic_callback(self, message):
        try:
            spheres = self._parse_cloud(message, 1)
            accepted = self.cache.update_dynamic(
                spheres, message.header.frame_id, self._stamp_sec(message))
            if not accepted:
                self._warn_throttled(self.cache.last_rejection_reason)
        except (ValueError, TypeError) as exc:
            self._warn_throttled(f"rejecting dynamic sphere cloud: {exc}")

    def _tick(self):
        now = self.get_clock().now()
        spheres = self.cache.combined(now.nanoseconds * 1e-9)
        stamp = now.to_msg()
        self._publish_cloud(stamp, spheres)
        self._publish_markers(stamp, spheres)

    def _publish_cloud(self, stamp, spheres):
        rows = [(
            s.x, s.y, s.z, s.raw_radius, s.output_radius, s.source_type,
            s.component_id, s.component_coverage, s.track_id, s.age,
            s.confidence) for s in spheres]
        specs = [
            ("x", PointField.FLOAT32), ("y", PointField.FLOAT32),
            ("z", PointField.FLOAT32), ("raw_radius", PointField.FLOAT32),
            ("output_radius", PointField.FLOAT32), ("source_type", PointField.INT32),
            ("component_id", PointField.INT32),
            ("component_coverage", PointField.FLOAT32),
            ("track_id", PointField.INT32), ("age", PointField.INT32),
            ("confidence", PointField.FLOAT32)]
        fields = [
            PointField(name=name, offset=index * 4, datatype=datatype, count=1)
            for index, (name, datatype) in enumerate(specs)]
        self.cloud_pub.publish(point_cloud2.create_cloud(
            Header(stamp=stamp, frame_id=self.target_frame), fields, rows))

    def _publish_markers(self, stamp, spheres):
        markers = []
        current = set()
        for index, sphere in enumerate(spheres):
            marker_id = index
            namespace = "combined_static" if sphere.source_type == 0 else "combined_dynamic"
            current.add((namespace, marker_id))
            marker = Marker()
            marker.header = Header(stamp=stamp, frame_id=self.target_frame)
            marker.ns = namespace
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=sphere.x, y=sphere.y, z=sphere.z)
            marker.pose.orientation.w = 1.0
            diameter = 2.0 * sphere.output_radius
            marker.scale.x = marker.scale.y = marker.scale.z = diameter
            color = self.static_marker_color if sphere.source_type == 0 else self.dynamic_marker_color
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = self.marker_alpha
            if self.marker_lifetime_s > 0.0:
                marker.lifetime = Duration(seconds=self.marker_lifetime_s).to_msg()
            markers.append(marker)
        for namespace, marker_id in sorted(self.previous_marker_keys - current):
            marker = Marker()
            marker.header = Header(stamp=stamp, frame_id=self.target_frame)
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            markers.append(marker)
        self.previous_marker_keys = current
        self.marker_pub.publish(MarkerArray(markers=markers))

    def _warn_throttled(self, message):
        now = time.monotonic()
        if now - self.last_warn_time >= 2.0:
            self.last_warn_time = now
            self.get_logger().warn(message)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSphereFusionNode()
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
