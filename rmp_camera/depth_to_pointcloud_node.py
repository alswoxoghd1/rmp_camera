import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2


class DepthToPointCloudNode(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud_node")

        self.declare_parameter("input_depth_topic", "/rmp_camera/robot_masked_depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera0/camera/depth/camera_info")
        self.declare_parameter("output_cloud_topic", "/rmp_camera/robot_masked_depth/points")
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 4.0)
        self.declare_parameter("stride", 3)
        self.declare_parameter("max_points", 50000)
        self.declare_parameter("max_rate_hz", 10.0)

        self.input_depth_topic = self.get_parameter("input_depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.output_cloud_topic = self.get_parameter("output_cloud_topic").value
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.stride = max(1, int(self.get_parameter("stride").value))
        self.max_points = int(self.get_parameter("max_points").value)
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)

        self.latest_camera_info = None
        self.last_process_time = 0.0
        self.last_log_time = 0.0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.input_depth_topic,
            self.depth_callback,
            sensor_qos,
        )
        self.cloud_pub = self.create_publisher(PointCloud2, self.output_cloud_topic, 2)

        self.get_logger().info(
            "Depth to pointcloud started: "
            f"{self.input_depth_topic} -> {self.output_cloud_topic}, "
            f"camera_info={self.camera_info_topic}, stride={self.stride}, "
            f"depth_range=[{self.min_depth_m:.2f}, {self.max_depth_m:.2f}] m"
        )

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg

    def depth_callback(self, msg):
        now = time.monotonic()
        if self.max_rate_hz > 0.0 and now - self.last_process_time < 1.0 / self.max_rate_hz:
            return
        self.last_process_time = now

        if self.latest_camera_info is None:
            self.log_throttled("Waiting for camera_info before pointcloud projection.", warn=True)
            return

        parsed = self.depth_image_to_array(msg)
        if parsed is None:
            return
        depth_array, depth_scale_m = parsed

        fx, fy, cx, cy = self.camera_intrinsics(self.latest_camera_info)
        if fx <= 0.0 or fy <= 0.0:
            self.log_throttled("Invalid camera intrinsics; cannot project depth.", warn=True)
            return

        points = self.project_depth(depth_array, depth_scale_m, fx, fy, cx, cy)
        if self.max_points > 0 and len(points) > self.max_points:
            step = int(np.ceil(len(points) / self.max_points))
            points = points[::step]

        header = msg.header
        if not header.frame_id and self.latest_camera_info.header.frame_id:
            header.frame_id = self.latest_camera_info.header.frame_id
        cloud = point_cloud2.create_cloud_xyz32(header, points.astype(np.float32, copy=False))
        self.cloud_pub.publish(cloud)

        self.log_throttled(
            "Depth pointcloud: "
            f"frame={header.frame_id}, valid_points={len(points)}, stride={self.stride}"
        )

    def project_depth(self, depth_array, depth_scale_m, fx, fy, cx, cy):
        depth = depth_array[:: self.stride, :: self.stride].astype(np.float32) * depth_scale_m
        valid = np.isfinite(depth) & (depth > self.min_depth_m) & (depth < self.max_depth_m)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        height, width = depth.shape
        us = np.arange(width, dtype=np.float32) * self.stride
        vs = np.arange(height, dtype=np.float32) * self.stride
        grid_u, grid_v = np.meshgrid(us, vs)

        z = depth[valid]
        x = (grid_u[valid] - cx) * z / fx
        y = (grid_v[valid] - cy) * z / fy
        return np.stack((x, y, z), axis=1)

    def depth_image_to_array(self, msg):
        dtype, scale_m = self.encoding_info(msg.encoding, msg.is_bigendian)
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
        full_array = np.frombuffer(msg.data, dtype=dtype, count=expected_items)
        full_array = full_array.reshape(msg.height, row_items)
        return full_array[:, : msg.width], scale_m

    @staticmethod
    def encoding_info(encoding, is_bigendian):
        endian = ">" if is_bigendian else "<"
        if encoding in ("16UC1", "mono16"):
            return np.dtype(endian + "u2"), 0.001
        if encoding == "32FC1":
            return np.dtype(endian + "f4"), 1.0
        return None, None

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
    node = DepthToPointCloudNode()
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
