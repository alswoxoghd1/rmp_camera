import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros
from visualization_msgs.msg import MarkerArray

from rmp_camera.vision_geometry import (
    marker_spheres,
    transform_points,
    transform_to_matrix,
    voxel_downsample,
)


class RobotPointCloudFilterNode(Node):
    def __init__(self):
        super().__init__("robot_pointcloud_filter_node")

        self.declare_parameter(
            "input_cloud_topic",
            "/nvblox_node/back_projected_depth/camera0_depth_optical_frame",
        )
        self.declare_parameter("robot_sphere_marker_topic", "/rmp_camera/robot_collision_sphere_markers")
        self.declare_parameter("output_cloud_topic", "/rmp_camera/robot_free_pointcloud")
        self.declare_parameter("removed_cloud_topic", "/rmp_camera/robot_pointcloud_removed_points")
        self.declare_parameter("publish_removed_cloud", True)
        self.declare_parameter("filter_robot_volume", True)
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("robot_margin_m", 0.05)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 4.0)
        self.declare_parameter("voxel_size_m", 0.015)
        self.declare_parameter("max_input_points", 120000)
        self.declare_parameter("max_publish_points", 80000)
        self.declare_parameter("max_rate_hz", 10.0)
        self.declare_parameter("tf_timeout_s", 0.2)
        self.declare_parameter("passthrough_on_missing_robot", True)
        self.declare_parameter("use_time_synchronized_robot_markers", True)
        self.declare_parameter("robot_marker_buffer_duration_s", 1.0)
        self.declare_parameter("robot_marker_max_stamp_delta_s", 0.15)
        self.declare_parameter("fallback_to_latest_marker_on_time_miss", True)

        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.robot_sphere_marker_topic = self.get_parameter("robot_sphere_marker_topic").value
        self.output_cloud_topic = self.get_parameter("output_cloud_topic").value
        self.removed_cloud_topic = self.get_parameter("removed_cloud_topic").value
        self.publish_removed_cloud = bool(self.get_parameter("publish_removed_cloud").value)
        self.filter_robot_volume = bool(self.get_parameter("filter_robot_volume").value)
        self.target_frame = self.get_parameter("target_frame").value
        self.robot_margin_m = float(self.get_parameter("robot_margin_m").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        self.max_input_points = int(self.get_parameter("max_input_points").value)
        self.max_publish_points = int(self.get_parameter("max_publish_points").value)
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self.tf_timeout = Duration(seconds=float(self.get_parameter("tf_timeout_s").value))
        self.passthrough_on_missing_robot = bool(
            self.get_parameter("passthrough_on_missing_robot").value
        )
        self.use_time_synchronized_robot_markers = bool(
            self.get_parameter("use_time_synchronized_robot_markers").value
        )
        self.robot_marker_buffer_duration_ns = int(
            max(0.0, float(self.get_parameter("robot_marker_buffer_duration_s").value)) * 1e9
        )
        self.robot_marker_max_stamp_delta_ns = int(
            max(0.0, float(self.get_parameter("robot_marker_max_stamp_delta_s").value)) * 1e9
        )
        self.fallback_to_latest_marker_on_time_miss = bool(
            self.get_parameter("fallback_to_latest_marker_on_time_miss").value
        )

        self.latest_markers = None
        self.marker_buffer = []
        self.last_process_time = 0.0
        self.last_log_time = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        marker_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2)
        self.marker_sub = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self.marker_callback,
            marker_qos,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.input_cloud_topic,
            self.cloud_callback,
            cloud_qos,
        )
        self.cloud_pub = self.create_publisher(PointCloud2, self.output_cloud_topic, 2)
        self.removed_cloud_pub = self.create_publisher(PointCloud2, self.removed_cloud_topic, 2)

        self.get_logger().info(
            "Robot point cloud filter started: "
            f"{self.input_cloud_topic} -> {self.output_cloud_topic}, "
            f"volume_filter={'on' if self.filter_robot_volume else 'off'}, "
            f"target_frame={self.target_frame}, robot_margin={self.robot_margin_m:.3f} m, "
            f"removed_cloud={self.removed_cloud_topic if self.publish_removed_cloud else 'off'}"
        )

    def marker_callback(self, msg):
        self.latest_markers = msg
        stamp_ns = self.marker_stamp_ns(msg)
        if stamp_ns == 0:
            stamp_ns = self.stamp_to_ns(self.get_clock().now().to_msg())
        self.marker_buffer.append((stamp_ns, msg))
        self.prune_marker_buffer(stamp_ns)

    def cloud_callback(self, msg):
        now = time.monotonic()
        if self.max_rate_hz > 0.0 and now - self.last_process_time < 1.0 / self.max_rate_hz:
            return
        self.last_process_time = now

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=self.tf_timeout,
            )
        except Exception as exc:
            self.log_throttled(
                f"Waiting for TF {self.target_frame} <- {msg.header.frame_id}: {exc}",
                warn=True,
            )
            return

        points = self.read_xyz_points(msg)
        raw_count = len(points)
        if raw_count == 0:
            return
        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        depth_mask = (points[:, 2] > self.min_depth_m) & (points[:, 2] < self.max_depth_m)
        points = points[depth_mask]
        if self.max_input_points > 0 and len(points) > self.max_input_points:
            step = int(np.ceil(len(points) / self.max_input_points))
            points = points[::step]

        target_from_cloud = transform_to_matrix(transform.transform)
        points_base = transform_points(points, target_from_cloud)

        keep_mask = np.ones(len(points_base), dtype=bool)
        if self.filter_robot_volume:
            robot_markers = self.select_robot_markers(msg.header.stamp)
            if robot_markers is None:
                if not self.passthrough_on_missing_robot:
                    self.log_throttled("Waiting for robot collision sphere markers.", warn=True)
                    return
                self.log_throttled(
                    "Robot collision sphere markers are unavailable; publishing unfiltered cloud.",
                    warn=True,
                )
            else:
                centers, radii, _ = marker_spheres(robot_markers)
                if len(centers) == 0:
                    if not self.passthrough_on_missing_robot:
                        self.log_throttled("Robot marker array has no sphere markers.", warn=True)
                        return
                    self.log_throttled(
                        "Robot marker array has no sphere markers; publishing unfiltered cloud.",
                        warn=True,
                    )
                else:
                    expanded_radii = radii + self.robot_margin_m
                    for center, radius in zip(centers, expanded_radii):
                        inside = np.sum((points_base - center) ** 2, axis=1) < radius * radius
                        keep_mask &= ~inside

        filtered = points_base[keep_mask]
        removed = points_base[~keep_mask]
        filtered = voxel_downsample(filtered, self.voxel_size_m)
        removed = voxel_downsample(removed, self.voxel_size_m)
        if self.max_publish_points > 0 and len(filtered) > self.max_publish_points:
            step = int(np.ceil(len(filtered) / self.max_publish_points))
            filtered = filtered[::step]
        if self.max_publish_points > 0 and len(removed) > self.max_publish_points:
            step = int(np.ceil(len(removed) / self.max_publish_points))
            removed = removed[::step]

        out = point_cloud2.create_cloud_xyz32(msg.header, filtered.astype(np.float32))
        out.header.frame_id = self.target_frame
        out.header.stamp = self.get_clock().now().to_msg()
        self.cloud_pub.publish(out)

        removed_count = len(points_base) - int(np.count_nonzero(keep_mask))
        if self.publish_removed_cloud:
            removed_out = point_cloud2.create_cloud_xyz32(msg.header, removed.astype(np.float32))
            removed_out.header.frame_id = self.target_frame
            removed_out.header.stamp = out.header.stamp
            self.removed_cloud_pub.publish(removed_out)

        self.log_throttled(
            "Robot-free pointcloud: "
            f"raw={raw_count}, valid={len(points_base)}, removed_robot={removed_count}, "
            f"published={len(filtered)}"
        )

    @staticmethod
    def read_xyz_points(msg):
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        points = np.asarray(points)
        if points.dtype.fields:
            points = np.vstack([points["x"], points["y"], points["z"]]).T
        if points.ndim == 1 and points.size:
            points = points.reshape(-1, 3)
        return points.astype(np.float64, copy=False)

    def select_robot_markers(self, stamp):
        if not self.use_time_synchronized_robot_markers:
            return self.latest_markers
        if not self.marker_buffer:
            return self.latest_markers

        target_ns = self.stamp_to_ns(stamp)
        if target_ns == 0:
            return self.latest_markers

        best_stamp_ns, best_markers = min(
            self.marker_buffer,
            key=lambda item: abs(item[0] - target_ns),
        )
        delta_ns = abs(best_stamp_ns - target_ns)
        if self.robot_marker_max_stamp_delta_ns > 0 and delta_ns > self.robot_marker_max_stamp_delta_ns:
            delta_s = delta_ns / 1e9
            self.log_throttled(
                "No close time-synchronized robot markers for pointcloud filter: "
                f"delta={delta_s:.3f}s. "
                f"{'Using latest marker fallback.' if self.fallback_to_latest_marker_on_time_miss else 'Skipping frame.'}",
                warn=True,
            )
            if self.fallback_to_latest_marker_on_time_miss:
                return self.latest_markers
            return None
        return best_markers

    def prune_marker_buffer(self, latest_stamp_ns):
        if self.robot_marker_buffer_duration_ns <= 0:
            self.marker_buffer = self.marker_buffer[-1:]
            return
        cutoff = latest_stamp_ns - self.robot_marker_buffer_duration_ns
        self.marker_buffer = [item for item in self.marker_buffer if item[0] >= cutoff]

    @classmethod
    def marker_stamp_ns(cls, marker_array):
        for marker in marker_array.markers:
            stamp_ns = cls.stamp_to_ns(marker.header.stamp)
            if stamp_ns != 0:
                return stamp_ns
        return 0

    @staticmethod
    def stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def log_throttled(self, message, warn=False):
        now = time.monotonic()
        if now - self.last_log_time < 2.0:
            return
        self.last_log_time = now
        if warn:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    rclpy.init(args=args)
    node = RobotPointCloudFilterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
