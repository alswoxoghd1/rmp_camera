"""Low-queue, single-frame depth validation before Nvblox and human projection."""

from array import array
from dataclasses import fields
import math
import time

import cv2
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

from rmp_camera.depth_edge_filter_core import DepthEdgeFilterConfig, depth_edge_rejection_mask
from rmp_camera.robot_depth_filter_core import as_ros_image_data


def depth_image_view(message):
    """Read 16UC1/mono16 (mm) or 32FC1 (m), including endian and row padding."""
    types = {'16UC1': ('u2', .001), 'mono16': ('u2', .001), '32FC1': ('f4', 1.0)}
    if message.encoding not in types:
        raise ValueError(f'unsupported depth encoding: {message.encoding}')
    code, scale = types[message.encoding]
    dtype = np.dtype(('>' if message.is_bigendian else '<') + code)
    if message.width <= 0 or message.height <= 0:
        raise ValueError('depth image dimensions must be positive')
    if message.step < message.width * dtype.itemsize:
        raise ValueError('depth image step is smaller than its pixel row')
    if len(message.data) < message.step * message.height:
        raise ValueError('depth image buffer is truncated')
    view = np.ndarray((message.height, message.width), dtype=dtype,
                      buffer=message.data, strides=(message.step, dtype.itemsize))
    return view, scale


def filter_depth_message(message, config):
    """Preserve timestamp, encoding, layout and all retained pixel bytes."""
    view, scale = depth_image_view(message)
    result = depth_edge_rejection_mask(view.astype(np.float32) * scale, config)
    if not np.any(result.rejected):
        return message, result
    output = Image()
    output.header = message.header
    output.height, output.width = message.height, message.width
    output.encoding, output.is_bigendian = message.encoding, message.is_bigendian
    output.step = message.step
    # Copy the original bytes so even non-item-aligned padding is preserved.
    output.data = array('B', message.data)
    pixels, _ = depth_image_view(output)
    pixels[result.rejected] = 0  # Unknown depth, never a fabricated free ray.
    return output, result


class DepthEdgeFilterNode(Node):
    def __init__(self):
        super().__init__('depth_edge_filter_node')
        cv2.setNumThreads(1)
        defaults = DepthEdgeFilterConfig()
        parameters = {f.name: getattr(defaults, f.name) for f in fields(defaults)}
        parameters.update(
            input_depth_topic='/camera0/realsense_splitter_node/output/depth',
            output_depth_topic='/rmp_camera/edge_filtered_depth/image_rect_raw',
            rejection_mask_topic='/rmp_camera/depth_edge_rejection_mask',
            status_topic='/rmp_camera/depth_edge_filter/status',
            publish_rejection_mask=False, statistics_period_s=2.0)
        # Startup-only parameters; do not silently accept ineffective updates.
        for key, value in parameters.items():
            self.declare_parameter(key, value, ParameterDescriptor(read_only=True))
        values = {key: self.get_parameter(key).value for key in parameters}
        self.config = DepthEdgeFilterConfig(**{f.name: values[f.name] for f in fields(defaults)})
        self.period = float(values['statistics_period_s'])
        if not math.isfinite(self.period) or self.period <= 0:
            raise ValueError('statistics_period_s must be finite and positive')
        source, target = values['input_depth_topic'], values['output_depth_topic']
        if self.resolve_topic_name(source) == self.resolve_topic_name(target):
            raise ValueError('input and output depth topics must differ')
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(Image, target, qos)
        self.mask_publisher = (self.create_publisher(Image, values['rejection_mask_topic'], qos)
                               if values['publish_rejection_mask'] else None)
        self.status_publisher = self.create_publisher(DiagnosticArray, values['status_topic'], 10)
        self.started = time.monotonic()
        self.last_error_log = -math.inf
        self.received = self.published = self.errors = 0
        self.valid = self.candidates = self.rejected = 0
        self.total_ms = self.max_ms = 0.0
        self.subscription = self.create_subscription(Image, source, self.depth_callback, qos)
        self.get_logger().info(
            f'Depth edge filter: {source} -> {target}, {self.config}; '
            'single frame, no robot/TF wait, queue=1, rejected depth=0')

    def depth_callback(self, message):
        start = time.perf_counter()
        self.received += 1
        try:
            output, result = filter_depth_message(message, self.config)
        except ValueError as error:
            self.errors += 1
            now = time.monotonic()
            if now - self.last_error_log >= self.period:
                self.last_error_log = now
                self.get_logger().error(f'Dropping malformed depth frame: {error}')
            self.report(message)
            return
        self.publisher.publish(output)
        self.published += 1
        self.valid += result.valid_pixels
        self.candidates += result.candidate_pixels
        self.rejected += int(np.count_nonzero(result.rejected))
        if self.mask_publisher is not None:
            mask = Image()
            mask.header = message.header
            mask.height, mask.width = message.height, message.width
            mask.encoding, mask.step = 'mono8', message.width
            mask.data = as_ros_image_data(result.rejected.astype(np.uint8) * 255)
            self.mask_publisher.publish(mask)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.total_ms += elapsed_ms
        self.max_ms = max(self.max_ms, elapsed_ms)
        self.report(message)

    def report(self, message):
        now = time.monotonic()
        elapsed = now - self.started
        if elapsed < self.period:
            return
        mean_ms = self.total_ms / max(1, self.published)
        fraction = self.rejected / max(1, self.valid)
        status = DiagnosticStatus()
        status.name = self.get_name()
        status.level = DiagnosticStatus.WARN if self.errors else DiagnosticStatus.OK
        status.message = 'single_frame_depth_edge_filter'
        values = dict(received=self.received, published=self.published, errors=self.errors,
            valid_pixels=self.valid, candidate_pixels=self.candidates, rejected_pixels=self.rejected,
            rejection_fraction=fraction, processing_mean_ms=mean_ms, processing_max_ms=self.max_ms,
            output_rate_hz=self.published / elapsed, interval_wall_s=elapsed)
        status.values = [KeyValue(key=k, value=str(v)) for k, v in values.items()]
        report = DiagnosticArray()
        report.header = message.header
        report.status = [status]
        self.status_publisher.publish(report)
        self.get_logger().info(
            f'Depth edge filter: received={self.received}, published={self.published}, '
            f'errors={self.errors}, rejected={fraction:.2%}, '
            f'processing_mean={mean_ms:.2f} ms, max={self.max_ms:.2f} ms')
        self.started = now
        self.received = self.published = self.errors = 0
        self.valid = self.candidates = self.rejected = 0
        self.total_ms = self.max_ms = 0.0


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DepthEdgeFilterNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
