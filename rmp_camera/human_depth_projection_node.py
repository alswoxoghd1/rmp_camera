"""Project PeopleSemSegNet masks onto D435 depth and publish human points."""

from __future__ import annotations

from collections import deque
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

from rmp_camera.human_depth_projection_core import (
    crop_points,
    filter_small_mask_components,
    project_masked_depth,
)
from rmp_camera.vision_geometry import transform_to_matrix, voxel_downsample
from rmp_camera.human_frame_sync import select_latest_depth_mask_pair


class HumanDepthProjectionNode(Node):
    """Approximate-time RGB-mask/depth registration without a second mapper."""

    def __init__(self):
        super().__init__("human_depth_projection_node")
        self._declare_parameters()
        self._read_parameters()
        if self.prefer_latest_frames:
            cv2.setNumThreads(1)
        self.depth_camera_info = None
        self.mask_camera_info = None
        self.depth_frames = deque(maxlen=self.sync_queue_size)
        self.mask_frames = deque(maxlen=2 if self.prefer_latest_frames else self.sync_queue_size)
        self.last_process_stamp_sec = None
        self.last_mask_stamp_sec = None
        self.last_pair_clock_ns = None
        self.last_log_monotonic = 0.0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(2, self.sync_queue_size),
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.depth_info_sub = self.create_subscription(
            CameraInfo, self.depth_camera_info_topic,
            self._depth_info_callback, sensor_qos)
        self.mask_info_sub = self.create_subscription(
            CameraInfo, self.mask_camera_info_topic,
            self._mask_info_callback, sensor_qos)
        self.depth_sub = self.create_subscription(
            Image, self.depth_image_topic, self._depth_callback, sensor_qos)
        # Keep depth history for delayed inference, not a DDS backlog of masks.
        mask_qos = (QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
                    if self.prefer_latest_frames else sensor_qos)
        self.mask_sub = self.create_subscription(
            Image, self.person_mask_topic, self._mask_callback, mask_qos)
        self.cloud_pub = self.create_publisher(
            PointCloud2, self.output_human_points_topic, 2)
        rate_description = (
            "unlimited" if self.max_rate_hz <= 0.0
            else f"<={self.max_rate_hz:.1f} Hz"
        )
        self.get_logger().info(
            "Human depth projection started: "
            f"depth={self.depth_image_topic}, mask={self.person_mask_topic}, "
            f"output={self.output_human_points_topic}, target={self.target_frame}, "
            f"rate={rate_description}, latest_frames={self.prefer_latest_frames}")

    def _declare_parameters(self):
        defaults = {
            "depth_image_topic": "/camera0/realsense_splitter_node/output/depth",
            "depth_camera_info_topic": "/camera0/camera/depth/camera_info",
            "person_mask_topic": "/camera0/segmentation/people_mask",
            "mask_camera_info_topic": "/camera0/segmentation/camera_info_resized",
            "output_human_points_topic": "/rmp_camera/human_points",
            "target_frame": "base_link",
            "max_sync_delta_s": 0.08,
            "sync_queue_size": 12,
            "prefer_latest_frames": True,
            "max_frame_age_s": 0.25,
            "tf_timeout_s": 0.05,
            "max_rate_hz": 15.0,
            "pixel_stride": 2,
            "mask_threshold": 1,
            "mask_min_component_pixels": 0,
            "mask_dilation_pixels": 2,
            "foreground_neighborhood_pixels": 0,
            "foreground_max_local_depth_jump_m": 0.0,
            "foreground_max_component_depth_span_m": 0.0,
            "min_depth_m": 0.15,
            "max_depth_m": 3.0,
            "min_x_m": -1.5,
            "max_x_m": 1.5,
            "min_y_m": -1.5,
            "max_y_m": 1.5,
            "min_z_m": -0.2,
            "max_z_m": 2.5,
            "max_range_from_base_m": 2.0,
            "output_voxel_size_m": 0.025,
            "max_output_points": 80000,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        string_names = (
            "depth_image_topic", "depth_camera_info_topic",
            "person_mask_topic", "mask_camera_info_topic",
            "output_human_points_topic", "target_frame")
        float_names = (
            "max_sync_delta_s", "tf_timeout_s", "max_rate_hz",
            "max_frame_age_s",
            "foreground_max_local_depth_jump_m",
            "foreground_max_component_depth_span_m",
            "min_depth_m", "max_depth_m", "min_x_m", "max_x_m",
            "min_y_m", "max_y_m", "min_z_m", "max_z_m",
            "max_range_from_base_m", "output_voxel_size_m")
        integer_names = (
            "sync_queue_size", "pixel_stride", "mask_threshold",
            "mask_min_component_pixels", "mask_dilation_pixels",
            "foreground_neighborhood_pixels", "max_output_points")
        for name in string_names:
            setattr(self, name, str(self.get_parameter(name).value))
        for name in float_names:
            setattr(self, name, float(self.get_parameter(name).value))
        for name in integer_names:
            setattr(self, name, int(self.get_parameter(name).value))
        self.prefer_latest_frames = bool(self.get_parameter('prefer_latest_frames').value)
        if not math.isfinite(self.max_frame_age_s) or self.max_frame_age_s <= 0:
            raise ValueError('max_frame_age_s must be finite and positive')
        if self.sync_queue_size <= 0 or self.pixel_stride <= 0:
            raise ValueError("sync_queue_size and pixel_stride must be positive")
        if self.max_sync_delta_s < 0.0 or self.tf_timeout_s < 0.0:
            raise ValueError("sync and TF timeouts must be non-negative")
        if self.mask_min_component_pixels < 0:
            raise ValueError("mask_min_component_pixels must be non-negative")
        if self.foreground_neighborhood_pixels < 0:
            raise ValueError("foreground_neighborhood_pixels must be non-negative")
        if (
            self.foreground_max_local_depth_jump_m < 0.0
            or self.foreground_max_component_depth_span_m < 0.0
        ):
            raise ValueError("foreground depth tolerances must be non-negative")
        if self.min_depth_m < 0.0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("invalid depth range")
        if self.max_output_points <= 0:
            raise ValueError("max_output_points must be positive")

    @staticmethod
    def _stamp_sec(message):
        return float(message.header.stamp.sec) + float(
            message.header.stamp.nanosec) * 1e-9

    def _depth_info_callback(self, message):
        self.depth_camera_info = message

    def _mask_info_callback(self, message):
        self.mask_camera_info = message

    def _depth_callback(self, message):
        self._check_pair_clock()
        self.depth_frames.append(message)
        self._process_nearest_pair()

    def _mask_callback(self, message):
        self._check_pair_clock()
        self.mask_frames.append(message)
        self._process_nearest_pair()

    def _check_pair_clock(self):
        if not self.prefer_latest_frames:
            return
        now = self.get_clock().now().nanoseconds
        if self.last_pair_clock_ns is not None and now < self.last_pair_clock_ns:
            self.depth_frames.clear()
            self.mask_frames.clear()
            self.last_process_stamp_sec = None
            self.last_mask_stamp_sec = None
        self.last_pair_clock_ns = now

    def _process_nearest_pair(self):
        if (
            not self.depth_frames or not self.mask_frames
            or self.depth_camera_info is None or self.mask_camera_info is None
        ):
            return
        if self.prefer_latest_frames:
            now = self.get_clock().now().nanoseconds * 1e-9
            for frames, last in ((self.depth_frames, self.last_process_stamp_sec),
                                 (self.mask_frames, self.last_mask_stamp_sec)):
                retained = [m for m in frames if (last is None or self._stamp_sec(m) > last)
                            and now - self.max_frame_age_s <= self._stamp_sec(m)
                            <= now + self.max_sync_delta_s]
                frames.clear()
                frames.extend(retained)
            selected = select_latest_depth_mask_pair(
                [self._stamp_sec(m) for m in self.depth_frames],
                [self._stamp_sec(m) for m in self.mask_frames], self.max_sync_delta_s,
                self.last_process_stamp_sec, self.last_mask_stamp_sec)
            if selected is None:
                return
            depth_index, mask_index, delta_s = selected
        else:
            selected = self._legacy_nearest_pair()
            if selected is None:
                return
            depth_index, mask_index, delta_s = selected
        depth_message = self._pop_index(self.depth_frames, depth_index)
        mask_message = self._pop_index(self.mask_frames, mask_index)
        source_stamp_sec = self._stamp_sec(depth_message)
        if self.max_rate_hz > 0.0 and self.last_process_stamp_sec is not None:
            source_delta_s = source_stamp_sec - self.last_process_stamp_sec
            if 0.0 <= source_delta_s < 1.0 / self.max_rate_hz:
                return
        self.last_process_stamp_sec = source_stamp_sec
        self.last_mask_stamp_sec = self._stamp_sec(mask_message)
        self._project_pair(depth_message, mask_message, delta_s)

    def _legacy_nearest_pair(self):
        pairs = [
            (abs(self._stamp_sec(depth) - self._stamp_sec(mask)), depth_index, mask_index)
            for depth_index, depth in enumerate(self.depth_frames)
            for mask_index, mask in enumerate(self.mask_frames)
        ]
        delta_s, depth_index, mask_index = min(pairs)
        if delta_s > self.max_sync_delta_s:
            oldest_depth = self._stamp_sec(self.depth_frames[0])
            oldest_mask = self._stamp_sec(self.mask_frames[0])
            if oldest_depth + self.max_sync_delta_s < oldest_mask:
                self.depth_frames.popleft()
            elif oldest_mask + self.max_sync_delta_s < oldest_depth:
                self.mask_frames.popleft()
            return
        return depth_index, mask_index, delta_s

    @staticmethod
    def _pop_index(values, index):
        values.rotate(-index)
        result = values.popleft()
        values.rotate(index)
        return result

    def _project_pair(self, depth_message, mask_message, delta_s):
        started = time.monotonic()
        depth_parsed = self._depth_array(depth_message)
        mask = self._mask_array(mask_message)
        if depth_parsed is None or mask is None:
            return
        depth, depth_scale = depth_parsed
        depth_m = depth.astype(np.float32, copy=False) * depth_scale
        mask = filter_small_mask_components(
            mask,
            mask_threshold=self.mask_threshold,
            min_component_pixels=self.mask_min_component_pixels,
        )
        if self.mask_dilation_pixels > 0 and np.any(mask):
            size = 2 * self.mask_dilation_pixels + 1
            mask = cv2.dilate(
                mask, np.ones((size, size), dtype=np.uint8), iterations=1)
        depth_frame = depth_message.header.frame_id or self.depth_camera_info.header.frame_id
        mask_frame = mask_message.header.frame_id or self.mask_camera_info.header.frame_id
        if not depth_frame or not mask_frame:
            self._log_throttled("Depth or mask frame_id is empty.", warn=True)
            return
        if not np.any(mask >= self.mask_threshold):
            self._publish(depth_message.header.stamp, np.empty((0, 3)))
            self._log_result(0, 0, delta_s, started)
            return
        try:
            stamp = Time.from_msg(depth_message.header.stamp)
            timeout = Duration(seconds=self.tf_timeout_s)
            mask_from_depth = self.tf_buffer.lookup_transform(
                mask_frame, depth_frame, stamp, timeout=timeout)
            target_from_depth = self.tf_buffer.lookup_transform(
                self.target_frame, depth_frame, stamp, timeout=timeout)
        except TransformException as exc:
            self._log_throttled(
                f"Human projection TF unavailable: {exc}", warn=True)
            return
        points = project_masked_depth(
            depth_m,
            self._camera_intrinsics(self.depth_camera_info),
            mask,
            self._camera_intrinsics(self.mask_camera_info),
            transform_to_matrix(mask_from_depth.transform),
            transform_to_matrix(target_from_depth.transform),
            stride=self.pixel_stride,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            mask_threshold=self.mask_threshold,
            foreground_neighborhood_pixels=self.foreground_neighborhood_pixels,
            foreground_max_local_depth_jump_m=(
                self.foreground_max_local_depth_jump_m),
            foreground_max_component_depth_span_m=(
                self.foreground_max_component_depth_span_m),
        )
        selected_before_crop = len(points)
        points = crop_points(
            points,
            (self.min_x_m, self.min_y_m, self.min_z_m),
            (self.max_x_m, self.max_y_m, self.max_z_m),
            self.max_range_from_base_m,
        )
        points = voxel_downsample(points, self.output_voxel_size_m)
        if len(points) > self.max_output_points:
            step = int(np.ceil(len(points) / self.max_output_points))
            points = points[::step]
        self._publish(depth_message.header.stamp, points)
        self._log_result(selected_before_crop, len(points), delta_s, started)

    def _publish(self, stamp, points):
        header = Header(stamp=stamp, frame_id=self.target_frame)
        cloud = point_cloud2.create_cloud_xyz32(
            header, np.asarray(points, dtype=np.float32).reshape(-1, 3))
        self.cloud_pub.publish(cloud)

    def _log_result(self, projected, output, delta_s, started):
        self._log_throttled(
            f"Human depth points: projected={projected}, output={output}, "
            f"sync_delta={1000.0 * delta_s:.1f} ms, "
            f"processing={1000.0 * (time.monotonic() - started):.1f} ms")

    def _log_throttled(self, message, warn=False):
        now = time.monotonic()
        if now - self.last_log_monotonic < 2.0:
            return
        self.last_log_monotonic = now
        # rcutils keys a log call site by source line and rejects changing its
        # severity later. Keep WARN and INFO on distinct call sites.
        if warn:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def _depth_array(self, message):
        if message.encoding in ("16UC1", "mono16"):
            dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
            scale = 0.001
        elif message.encoding == "32FC1":
            dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
            scale = 1.0
        else:
            self._log_throttled(
                f"Unsupported depth encoding: {message.encoding}", warn=True)
            return None
        return self._image_array(message, dtype), scale

    def _mask_array(self, message):
        if message.encoding not in ("mono8", "8UC1"):
            self._log_throttled(
                f"Unsupported person mask encoding: {message.encoding}", warn=True)
            return None
        return self._image_array(message, np.dtype("u1"))

    @staticmethod
    def _image_array(message, dtype):
        item_size = dtype.itemsize
        row_items = message.step // item_size
        expected = row_items * message.height
        values = np.frombuffer(message.data, dtype=dtype, count=expected)
        return values.reshape(message.height, row_items)[:, :message.width]

    @staticmethod
    def _camera_intrinsics(camera_info):
        if camera_info.p[0] and camera_info.p[5]:
            return (
                float(camera_info.p[0]), float(camera_info.p[5]),
                float(camera_info.p[2]), float(camera_info.p[6]))
        return (
            float(camera_info.k[0]), float(camera_info.k[4]),
            float(camera_info.k[2]), float(camera_info.k[5]))


def main(args=None):
    rclpy.init(args=args)
    node = HumanDepthProjectionNode()
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
