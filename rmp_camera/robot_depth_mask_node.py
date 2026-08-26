import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros
from visualization_msgs.msg import MarkerArray

from rmp_camera.robot_depth_filter_core import (
    as_ros_image_data,
    predict_sphere_surface_depth,
    surface_depth_removal_mask,
)
from rmp_camera.vision_geometry import marker_spheres, transform_to_matrix


class RobotDepthMaskNode(Node):
    def __init__(self):
        super().__init__("robot_depth_mask_node")

        self.declare_parameter("input_depth_topic", "/camera0/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera0/camera/depth/camera_info")
        self.declare_parameter(
            "robot_sphere_marker_topic",
            "/rmp_camera/robot_collision_sphere_markers",
        )
        self.declare_parameter(
            "output_depth_topic", "/rmp_camera/robot_masked_depth/image_rect_raw")
        self.declare_parameter(
            "predicted_depth_topic", "/rmp_camera/robot_predicted_depth/image_rect_raw")
        self.declare_parameter(
            "removed_points_topic", "/rmp_camera/robot_depth_mask_removed_points")
        self.declare_parameter("publish_removed_points", False)
        self.declare_parameter("publish_predicted_depth", False)
        self.declare_parameter("filter_mode", "volume")
        self.declare_parameter("robot_margin_m", 0.05)
        self.declare_parameter("surface_sphere_padding_m", 0.0)
        self.declare_parameter("surface_front_tolerance_m", 0.02)
        self.declare_parameter("surface_back_tolerance_m", 0.03)
        self.declare_parameter("mask_shadow_behind_robot", False)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)
        self.declare_parameter("max_rate_hz", 0.0)
        self.declare_parameter("statistics_period_s", 5.0)
        self.declare_parameter("tf_timeout_s", 0.05)
        self.declare_parameter("passthrough_on_missing_robot", True)
        self.declare_parameter("use_time_synchronized_robot_markers", True)
        self.declare_parameter("robot_marker_buffer_duration_s", 1.0)
        self.declare_parameter("robot_marker_max_stamp_delta_s", 0.15)
        self.declare_parameter("fallback_to_latest_marker_on_time_miss", True)

        self.input_depth_topic = self.get_parameter("input_depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.robot_sphere_marker_topic = self.get_parameter("robot_sphere_marker_topic").value
        self.output_depth_topic = self.get_parameter("output_depth_topic").value
        self.predicted_depth_topic = self.get_parameter("predicted_depth_topic").value
        self.removed_points_topic = self.get_parameter("removed_points_topic").value
        self.publish_removed_points = bool(self.get_parameter("publish_removed_points").value)
        self.publish_predicted_depth = bool(
            self.get_parameter("publish_predicted_depth").value)
        self.filter_mode = str(self.get_parameter("filter_mode").value).strip().lower()
        if self.filter_mode not in ("volume", "surface_depth"):
            raise ValueError(
                "filter_mode must be either 'volume' or 'surface_depth'")
        self.robot_margin_m = float(self.get_parameter("robot_margin_m").value)
        self.surface_sphere_padding_m = max(
            0.0,
            float(self.get_parameter("surface_sphere_padding_m").value),
        )
        self.surface_front_tolerance_m = float(
            self.get_parameter("surface_front_tolerance_m").value)
        self.surface_back_tolerance_m = float(
            self.get_parameter("surface_back_tolerance_m").value)
        self.mask_shadow_behind_robot = bool(
            self.get_parameter("mask_shadow_behind_robot").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self.statistics_period_s = max(
            0.0, float(self.get_parameter("statistics_period_s").value))
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

        self.latest_camera_info = None
        self.latest_robot_markers = None
        self.robot_marker_buffer = []
        self.last_process_time = 0.0
        self.last_log_time = 0.0
        self.statistics_started_at = time.monotonic()
        self.statistics_active = False
        self.statistics_received = 0
        self.statistics_processed = 0
        self.statistics_rate_limited = 0
        self.statistics_passthrough = 0
        self.statistics_missing_drops = 0
        self.statistics_processing_time_s = 0.0
        self.statistics_max_processing_time_s = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        marker_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )
        self.marker_sub = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self.robot_marker_callback,
            marker_qos,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.input_depth_topic,
            self.depth_callback,
            sensor_qos,
        )
        self.depth_pub = self.create_publisher(Image, self.output_depth_topic, sensor_qos)
        self.predicted_depth_pub = self.create_publisher(
            Image, self.predicted_depth_topic, sensor_qos)
        self.removed_points_pub = self.create_publisher(
            PointCloud2,
            self.removed_points_topic,
            2,
        )

        self.get_logger().info(
            "Robot depth mask started: "
            f"{self.input_depth_topic} -> {self.output_depth_topic}, "
            f"mode={self.filter_mode}, camera_info={self.camera_info_topic}, "
            f"robot_margin={self.robot_margin_m:.3f} m, "
            f"surface_band=[-{self.surface_front_tolerance_m:.3f}, "
            f"+{self.surface_back_tolerance_m:.3f}] m, "
            f"predicted_depth={'on' if self.publish_predicted_depth else 'off'}, "
            f"removed_points={self.removed_points_topic if self.publish_removed_points else 'off'}"
        )

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def robot_marker_callback(self, msg):
        self.latest_robot_markers = msg
        stamp_ns = self.marker_stamp_ns(msg)
        if stamp_ns == 0:
            stamp_ns = self.stamp_to_ns(self.get_clock().now().to_msg())
        self.robot_marker_buffer.append((stamp_ns, msg))
        self.prune_robot_marker_buffer(stamp_ns)

    def depth_callback(self, msg):
        callback_started_at = time.perf_counter()
        now = time.monotonic()
        if not self.statistics_active:
            self.statistics_started_at = now
            self.statistics_active = True
        self.statistics_received += 1
        if self.max_rate_hz > 0.0 and now - self.last_process_time < 1.0 / self.max_rate_hz:
            self.statistics_rate_limited += 1
            self.maybe_log_statistics(now)
            return
        self.last_process_time = now

        if self.latest_camera_info is None:
            self.log_throttled("Waiting for camera_info before depth masking.", warn=True)
            self.handle_unavailable_depth(msg)
            return

        robot_markers = self.select_robot_markers(msg.header.stamp)
        if robot_markers is None or not robot_markers.markers:
            self.log_throttled(
                "Waiting for robot collision sphere markers; "
                + self.unavailable_depth_action(),
                warn=True,
            )
            self.handle_unavailable_depth(msg)
            return

        marker_frame = self.first_marker_frame(robot_markers)
        if not marker_frame:
            self.log_throttled("Robot marker array has no valid marker frame.", warn=True)
            self.handle_unavailable_depth(msg)
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id,
                marker_frame,
                rclpy.time.Time(),
                timeout=self.tf_timeout,
            )
        except Exception as exc:
            self.log_throttled(
                f"Waiting for TF {msg.header.frame_id} <- {marker_frame}: {exc}",
                warn=True,
            )
            self.handle_unavailable_depth(msg)
            return

        parsed = self.depth_image_to_array(msg)
        if parsed is None:
            self.handle_unavailable_depth(msg)
            return
        depth_array, full_array, depth_scale_m, invalid_value = parsed

        fx, fy, cx, cy = self.camera_intrinsics(self.latest_camera_info)
        if fx <= 0.0 or fy <= 0.0:
            self.log_throttled(
                "Invalid camera intrinsics; " + self.unavailable_depth_action(),
                warn=True,
            )
            self.handle_unavailable_depth(msg)
            return

        centers_base, radii, _ = marker_spheres(robot_markers)
        if len(centers_base) == 0:
            self.log_throttled(
                "Robot marker array has no sphere markers; "
                + self.unavailable_depth_action(),
                warn=True,
            )
            self.handle_unavailable_depth(msg)
            return

        marker_to_depth = transform_to_matrix(transform.transform)
        centers_depth = (
            marker_to_depth
            @ np.c_[centers_base, np.ones(len(centers_base), dtype=np.float64)].T
        ).T[:, :3]
        predicted_depth = None
        if self.filter_mode == "surface_depth":
            predicted_depth = predict_sphere_surface_depth(
                depth_array.shape,
                centers_depth,
                radii + self.surface_sphere_padding_m,
                fx,
                fy,
                cx,
                cy,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
            measured_depth_m = depth_array.astype(np.float32) * depth_scale_m
            removed_mask = surface_depth_removal_mask(
                measured_depth_m,
                predicted_depth,
                self.surface_front_tolerance_m,
                self.surface_back_tolerance_m,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
                mask_shadow_behind_robot=self.mask_shadow_behind_robot,
            )
            masked_depth = depth_array.copy()
            masked_depth[removed_mask] = invalid_value
        else:
            expanded_radii = radii + self.robot_margin_m
            masked_depth, removed_mask = self.mask_depth(
                depth_array,
                centers_depth,
                expanded_radii,
                fx,
                fy,
                cx,
                cy,
                depth_scale_m,
                invalid_value,
            )
        removed_pixels = int(np.count_nonzero(removed_mask))
        full_array[:, : msg.width] = masked_depth

        output = Image()
        output.header = msg.header
        output.height = msg.height
        output.width = msg.width
        output.encoding = msg.encoding
        output.is_bigendian = msg.is_bigendian
        output.step = msg.step
        output.data = as_ros_image_data(full_array)
        self.depth_pub.publish(output)
        if self.publish_predicted_depth and predicted_depth is not None:
            self.publish_robot_predicted_depth(msg.header, predicted_depth)
        if self.publish_removed_points:
            self.publish_removed_pointcloud(
                msg.header,
                depth_array,
                removed_mask,
                fx,
                fy,
                cx,
                cy,
                depth_scale_m,
            )

        processing_time_s = time.perf_counter() - callback_started_at
        self.statistics_processed += 1
        self.statistics_processing_time_s += processing_time_s
        self.statistics_max_processing_time_s = max(
            self.statistics_max_processing_time_s, processing_time_s)
        self.maybe_log_statistics()

        self.log_throttled(
            "Robot-masked depth: "
            f"spheres={len(centers_depth)}, removed_pixels={removed_pixels}, "
            f"frame={msg.header.frame_id}"
        )

    def handle_unavailable_depth(self, msg):
        if self.passthrough_on_missing_robot:
            self.depth_pub.publish(msg)
            self.statistics_passthrough += 1
        else:
            self.statistics_missing_drops += 1
        self.maybe_log_statistics()

    def unavailable_depth_action(self):
        if self.passthrough_on_missing_robot:
            return "publishing unmasked depth."
        return "dropping depth frame."

    def depth_image_to_array(self, msg):
        dtype, scale_m, invalid_value = self.encoding_info(msg.encoding, msg.is_bigendian)
        if dtype is None:
            self.log_throttled(f"Unsupported depth encoding: {msg.encoding}", warn=True)
            return None

        item_size = np.dtype(dtype).itemsize
        if msg.step < msg.width * item_size:
            self.log_throttled(
                f"Invalid image step={msg.step} for width={msg.width}, encoding={msg.encoding}",
                warn=True,
            )
            return None

        row_items = msg.step // item_size
        expected_items = row_items * msg.height
        full_array = np.frombuffer(msg.data, dtype=dtype, count=expected_items).copy()
        full_array = full_array.reshape(msg.height, row_items)
        return full_array[:, : msg.width], full_array, scale_m, invalid_value

    @staticmethod
    def encoding_info(encoding, is_bigendian):
        endian = ">" if is_bigendian else "<"
        if encoding in ("16UC1", "mono16"):
            return np.dtype(endian + "u2"), 0.001, 0
        if encoding == "32FC1":
            return np.dtype(endian + "f4"), 1.0, 0.0
        return None, None, None

    @staticmethod
    def camera_intrinsics(camera_info):
        if camera_info.p[0] != 0.0 and camera_info.p[5] != 0.0:
            return (
                float(camera_info.p[0]),
                float(camera_info.p[5]),
                float(camera_info.p[2]),
                float(camera_info.p[6]),
            )
        return (
            float(camera_info.k[0]),
            float(camera_info.k[4]),
            float(camera_info.k[2]),
            float(camera_info.k[5]),
        )

    @staticmethod
    def first_marker_frame(marker_array):
        for marker in marker_array.markers:
            if marker.header.frame_id:
                return marker.header.frame_id
        return ""

    def select_robot_markers(self, stamp):
        if not self.use_time_synchronized_robot_markers:
            return self.latest_robot_markers
        if not self.robot_marker_buffer:
            return self.latest_robot_markers

        target_ns = self.stamp_to_ns(stamp)
        if target_ns == 0:
            return self.latest_robot_markers

        best_stamp_ns, best_markers = min(
            self.robot_marker_buffer,
            key=lambda item: abs(item[0] - target_ns),
        )
        delta_ns = abs(best_stamp_ns - target_ns)
        if (
            self.robot_marker_max_stamp_delta_ns > 0
            and delta_ns > self.robot_marker_max_stamp_delta_ns
        ):
            delta_s = delta_ns / 1e9
            action = (
                "Using latest marker fallback."
                if self.fallback_to_latest_marker_on_time_miss
                else "Skipping frame."
            )
            self.log_throttled(
                "No close time-synchronized robot markers for depth mask: "
                f"delta={delta_s:.3f}s. "
                + action,
                warn=True,
            )
            if self.fallback_to_latest_marker_on_time_miss:
                return self.latest_robot_markers
            return None
        return best_markers

    def prune_robot_marker_buffer(self, latest_stamp_ns):
        if self.robot_marker_buffer_duration_ns <= 0:
            self.robot_marker_buffer = self.robot_marker_buffer[-1:]
            return
        cutoff = latest_stamp_ns - self.robot_marker_buffer_duration_ns
        self.robot_marker_buffer = [
            item for item in self.robot_marker_buffer if item[0] >= cutoff
        ]

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

    def mask_depth(
        self,
        depth_array,
        centers,
        radii,
        fx,
        fy,
        cx,
        cy,
        depth_scale_m,
        invalid_value,
    ):
        masked = depth_array.copy()
        height, width = masked.shape
        removed_mask_total = np.zeros((height, width), dtype=bool)

        for center, radius in zip(centers, radii):
            x, y, z = center
            if not np.isfinite(center).all() or not np.isfinite(radius):
                continue
            if radius <= 0.0 or z + radius < self.min_depth_m:
                continue

            z_for_projection = max(z - radius, self.min_depth_m)
            pixel_radius_x = int(np.ceil(fx * radius / z_for_projection)) + 2
            pixel_radius_y = int(np.ceil(fy * radius / z_for_projection)) + 2
            if z <= 1e-6:
                u_center = int(round(cx))
                v_center = int(round(cy))
            else:
                u_center = int(round(fx * x / z + cx))
                v_center = int(round(fy * y / z + cy))

            u_min = max(0, u_center - pixel_radius_x)
            u_max = min(width, u_center + pixel_radius_x + 1)
            v_min = max(0, v_center - pixel_radius_y)
            v_max = min(height, v_center + pixel_radius_y + 1)
            if u_min >= u_max or v_min >= v_max:
                continue

            depth_roi_raw = masked[v_min:v_max, u_min:u_max]
            depth_m = depth_roi_raw.astype(np.float32) * depth_scale_m
            valid = (depth_m > self.min_depth_m) & (depth_m < self.max_depth_m)
            if not np.any(valid):
                continue

            us = np.arange(u_min, u_max, dtype=np.float32)[None, :]
            vs = np.arange(v_min, v_max, dtype=np.float32)[:, None]
            xs = (us - cx) * depth_m / fx
            ys = (vs - cy) * depth_m / fy
            inside = (
                valid
                & ((xs - x) ** 2 + (ys - y) ** 2 + (depth_m - z) ** 2 <= radius * radius)
            )
            if not np.any(inside):
                continue

            depth_roi_raw[inside] = invalid_value
            removed_mask_total[v_min:v_max, u_min:u_max] |= inside

        return masked, removed_mask_total

    def publish_robot_predicted_depth(self, header, predicted_depth):
        """Publish the rendered robot front surface as a 32FC1 depth image."""
        visualization = np.asarray(predicted_depth, dtype=np.float32).copy()
        visualization[~np.isfinite(visualization)] = 0.0
        output = Image()
        output.header = header
        output.height = int(visualization.shape[0])
        output.width = int(visualization.shape[1])
        output.encoding = "32FC1"
        output.is_bigendian = False
        output.step = output.width * np.dtype(np.float32).itemsize
        output.data = as_ros_image_data(visualization)
        self.predicted_depth_pub.publish(output)

    def publish_removed_pointcloud(
        self,
        header,
        depth_array,
        removed_mask,
        fx,
        fy,
        cx,
        cy,
        depth_scale_m,
    ):
        if removed_mask is None or not np.any(removed_mask):
            cloud = point_cloud2.create_cloud_xyz32(header, np.empty((0, 3), dtype=np.float32))
            self.removed_points_pub.publish(cloud)
            return

        depth_m = depth_array.astype(np.float32) * depth_scale_m
        valid = removed_mask & (depth_m > self.min_depth_m) & (depth_m < self.max_depth_m)
        if not np.any(valid):
            cloud = point_cloud2.create_cloud_xyz32(header, np.empty((0, 3), dtype=np.float32))
            self.removed_points_pub.publish(cloud)
            return

        vs, us = np.nonzero(valid)
        z = depth_m[valid]
        x = (us.astype(np.float32) - cx) * z / fx
        y = (vs.astype(np.float32) - cy) * z / fy
        points = np.stack((x, y, z), axis=1)
        cloud = point_cloud2.create_cloud_xyz32(header, points.astype(np.float32, copy=False))
        self.removed_points_pub.publish(cloud)

    def log_throttled(self, message, warn=False):
        now = time.monotonic()
        if now - self.last_log_time < 2.0:
            return
        self.last_log_time = now
        if warn:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def maybe_log_statistics(self, now=None):
        """Periodically report rates, drops, and successful callback latency."""
        if self.statistics_period_s <= 0.0:
            return
        now = time.monotonic() if now is None else now
        elapsed_s = now - self.statistics_started_at
        if elapsed_s < self.statistics_period_s:
            return
        processed = self.statistics_processed
        published = processed + self.statistics_passthrough
        average_ms = (
            self.statistics_processing_time_s * 1000.0 / processed
            if processed > 0 else 0.0
        )
        self.get_logger().info(
            "Robot depth mask stats: "
            f"input={self.statistics_received / elapsed_s:.2f} Hz, "
            f"output={published / elapsed_s:.2f} Hz, "
            f"filtered={processed}, "
            f"rate_limited={self.statistics_rate_limited}, "
            f"passthrough={self.statistics_passthrough}, "
            f"missing_drops={self.statistics_missing_drops}, "
            f"processing_avg={average_ms:.2f} ms, "
            f"processing_max={self.statistics_max_processing_time_s * 1000.0:.2f} ms"
        )
        self.statistics_started_at = now
        self.statistics_received = 0
        self.statistics_processed = 0
        self.statistics_rate_limited = 0
        self.statistics_passthrough = 0
        self.statistics_missing_drops = 0
        self.statistics_processing_time_s = 0.0
        self.statistics_max_processing_time_s = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = RobotDepthMaskNode()
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
