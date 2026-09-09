"""ROS subscriptions shared by static generation and its fast fusion clear."""

import time

import numpy as np
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

from rmp_camera.human_static_filter_core import HumanStaticParameters, HumanVoxelHistory
from rmp_camera.vision_geometry import transform_to_matrix


class HumanStaticFilter:
    def __init__(self, node):
        self.node = node
        defaults = {
            "human_static_filter_enabled": False,
            "human_static_points_topic": "/rmp_camera/human_points",
            "human_static_depth_topic": "/camera0/realsense_splitter_node/output/depth",
            "human_static_camera_info_topic": "/camera0/camera/depth/camera_info",
            "human_static_support_topic": "/rmp_camera/static_sphere_support",
            "human_static_free_support_topic": "/rmp_camera/static_sphere_free_support",
        }
        p = HumanStaticParameters()
        defaults.update({"human_static_" + key: value for key, value in vars(p).items()})
        for key, value in defaults.items():
            node.declare_parameter(key, value)
        self.enabled = bool(node.get_parameter("human_static_filter_enabled").value)
        self.support_topic = node.get_parameter("human_static_support_topic").value
        self.free_support_topic = node.get_parameter("human_static_free_support_topic").value
        self.params = HumanStaticParameters(**{
            key: node.get_parameter("human_static_" + key).value for key in vars(p)})
        self.history = HumanVoxelHistory(self.params)
        self.target = node.target_frame
        self.info = None
        self.depth = None
        self.depth_stamp = None
        self.depth_received = 0.0
        self.transform = None
        self.sequence = 0
        self.last_log = 0.0
        self.last_stats = (0, 0)
        if not self.enabled:
            return
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, node)
        self.subscriptions = [
            node.create_subscription(PointCloud2, node.get_parameter(
                "human_static_points_topic").value, self._human, qos_profile_sensor_data),
            node.create_subscription(Image, node.get_parameter(
                "human_static_depth_topic").value, self._depth, qos_profile_sensor_data),
            node.create_subscription(CameraInfo, node.get_parameter(
                "human_static_camera_info_topic").value, self._info, qos_profile_sensor_data),
        ]
        self.jump_handle = node.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1),
                          on_clock_change=True),
            post_callback=self._jump)
        node.get_logger().info("Human static voxel filter enabled: "
            f"match={self.params.match_distance_m:.2f} m, "
            f"history={self.params.history_s:.1f}s, "
            f"ownership={self.params.occlusion_hold_enabled}, "
            f"ownership_hold={self.params.ownership_hold_s:.2f}s")

    def _jump(self, _jump):
        self.history.clear()
        self.depth = None
        self.transform = None
        self.sequence += 1
        reset = getattr(self.node, "_reset_human_static_cache", None)
        if reset is not None:
            reset()

    @staticmethod
    def _stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def _info(self, message):
        self.info = message

    def _human(self, message):
        if message.header.frame_id != self.target:
            return
        try:
            rows = point_cloud2.read_points_numpy(
                message, field_names=("x", "y", "z"), skip_nans=True)
            camera_origin = None
            if self.transform is not None:
                camera_origin = -self.transform[:3, :3].T @ self.transform[:3, 3]
            self.history.add(rows, self._stamp(message), camera_origin)
            self.sequence += 1
        except (ValueError, AssertionError):
            return

    def _depth(self, message):
        if self.info is None or message.header.frame_id != self.info.header.frame_id:
            return
        if message.encoding in ("16UC1", "mono16"):
            dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
            scale = 0.001
        elif message.encoding == "32FC1":
            dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
            scale = 1.0
        else:
            return
        if (message.width != self.info.width or message.height != self.info.height
                or message.step < message.width * dtype.itemsize
                or len(message.data) < message.height * message.step):
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                message.header.frame_id, self.target, Time.from_msg(message.header.stamp))
            self.transform = transform_to_matrix(tf.transform)
        except TransformException:
            return
        self.depth = np.ndarray((message.height, message.width), dtype=dtype,
            buffer=message.data, strides=(message.step, dtype.itemsize)).astype(np.float32) * scale
        self.depth_stamp = self._stamp(message)
        self.depth_received = time.monotonic()
        self.sequence += 1

    def excluded(self, points, now):
        started = time.monotonic()
        points = np.asarray(points).reshape(-1, 3)
        if not self.enabled:
            return np.zeros(len(points), dtype=bool)
        depth = self.depth
        if started - self.depth_received > self.params.depth_max_age_s:
            depth = None
        intrinsics = None if self.info is None else (
            self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5])
        active, cleared = self.history.classify(
            points, now, depth, intrinsics, self.transform, self.depth_stamp)
        self.last_stats = (int(active.sum()), int(cleared.sum()))
        if started - self.last_log >= 2.0 and len(points):
            self.last_log = started
            self.node.get_logger().info(
                f"Human static voxel evidence: candidates={len(points)}, "
                f"active={active.sum()}, depth_free={cleared.sum()}, "
                f"human_owned={self.history.last_occluded_count}, "
                f"history={len(self.history.voxels)}, "
                f"processing={(time.monotonic() - started) * 1000:.1f} ms")
        return active | cleared
