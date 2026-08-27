"""ROS 2 adapter from native nvblox dynamic points to tracked spheres."""

from __future__ import annotations

import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
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
from rmp_camera.vision_geometry import marker_spheres


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
        self.latest_points_received_monotonic = time.monotonic()
        self.latest_frame = self.target_frame
        self.have_message = False
        self.ever_received = False
        self.pending_points_frame = None
        self.latest_robot_markers = None
        self.robot_marker_buffer = []
        self.latest_robot_marker_delta_s = None
        self.latest_robot_marker_signed_delta_s = None
        self.latest_robot_marker_buffer_span_s = None
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
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.robot_marker_subscription_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.robot_marker_subscription = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self._robot_marker_callback,
            marker_qos,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 2)
        self.sphere_pub = self.create_publisher(PointCloud2, self.sphere_cloud_topic, 2)
        self.voxel_pub = self.create_publisher(PointCloud2, self.dynamic_voxel_cloud_topic, 2)
        self.uncovered_pub = self.create_publisher(
            PointCloud2, self.uncovered_voxel_cloud_topic, 2)
        self.robot_rejected_voxel_pub = self.create_publisher(
            PointCloud2, self.robot_rejected_voxel_cloud_topic, 2)
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
            f"empty_guard={self.dynamic_merge_enable_empty_space_guard}, "
            f"min_k={self.dynamic_enable_min_k_search}, "
            f"beam={self.dynamic_min_k_beam_width}, "
            f"max_states={self.dynamic_min_k_max_states}, "
            f"robot_component_filter={self.robot_component_filter_enabled}, "
            f"robot_overlap_threshold="
            f"{self.robot_component_overlap_threshold:.3f}")

    def _declare_parameters(self):
        defaults = {
            "input_dynamic_points_topic": "/nvblox_node/dynamic_points",
            "target_frame": "base_link",
            "marker_topic": "/rmp_camera/dynamic_obstacle_sphere_markers",
            "sphere_cloud_topic": "/rmp_camera/dynamic_obstacle_sphere_cloud",
            "dynamic_voxel_cloud_topic": "/rmp_camera/dynamic_obstacle_voxels",
            "uncovered_voxel_cloud_topic": "/rmp_camera/dynamic_uncovered_voxels",
            "robot_sphere_marker_topic":
                "/rmp_camera/robot_collision_sphere_markers",
            "robot_rejected_voxel_cloud_topic":
                "/rmp_camera/dynamic_robot_rejected_voxels",
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
            "dynamic_enable_min_k_search": True,
            "dynamic_min_k_beam_width": 8,
            "dynamic_min_k_max_states": 128,
            "dynamic_min_k_enable_overlap_constraint": True,
            "dynamic_max_allowed_overlap_fraction": 0.20,
            "dynamic_min_k_use_output_overlap": True,
            "dynamic_min_k_prefer_lower_overlap": True,
            "dynamic_tracking_enabled": True,
            "dynamic_association_distance_m": 0.20,
            "dynamic_smoothing_alpha": 0.55,
            "dynamic_sphere_ttl_sec": 0.8,
            "dynamic_max_missed_updates": 8,
            "dynamic_update_rate_hz": 10.0,
            "robot_component_filter_enabled": False,
            "robot_component_overlap_threshold": 0.10,
            "robot_sphere_filter_margin_m": 0.0,
            "robot_marker_expected_sphere_count": 35,
            "robot_marker_require_expected_count": True,
            "use_time_synchronized_robot_markers": True,
            "robot_marker_buffer_duration_s": 1.0,
            "robot_marker_max_stamp_delta_s": 0.05,
            "robot_marker_sync_wait_timeout_s": 0.25,
            "robot_marker_subscription_depth": 2,
            "fallback_to_latest_marker_on_time_miss": False,
            "robot_component_filter_fail_closed": True,
            "publish_debug_clouds": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        string_names = (
            "input_dynamic_points_topic", "target_frame", "marker_topic",
            "sphere_cloud_topic", "dynamic_voxel_cloud_topic",
            "uncovered_voxel_cloud_topic", "robot_sphere_marker_topic",
            "robot_rejected_voxel_cloud_topic")
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
            "dynamic_merge_max_gap_m", "dynamic_merge_max_empty_fraction",
            "dynamic_max_allowed_overlap_fraction",
            "robot_component_overlap_threshold", "robot_sphere_filter_margin_m",
            "robot_marker_buffer_duration_s", "robot_marker_max_stamp_delta_s",
            "robot_marker_sync_wait_timeout_s")
        for name in float_names:
            setattr(self, name, float(self.get_parameter(name).value))
        int_names = (
            "dynamic_min_component_voxels", "dynamic_max_component_voxels",
            "dynamic_max_components", "dynamic_max_input_points", "dynamic_connectivity",
            "dynamic_dilation_voxels", "dynamic_closing_iterations",
            "dynamic_max_spheres_per_component", "dynamic_max_iterations_per_component",
            "dynamic_max_total_spheres", "dynamic_max_local_grid_voxels",
            "dynamic_max_missed_updates",
            "dynamic_merge_max_validation_voxels",
            "dynamic_min_k_beam_width", "dynamic_min_k_max_states",
            "robot_marker_expected_sphere_count",
            "robot_marker_subscription_depth")
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
        self.dynamic_enable_min_k_search = bool(
            self.get_parameter("dynamic_enable_min_k_search").value)
        self.dynamic_min_k_enable_overlap_constraint = bool(
            self.get_parameter(
                "dynamic_min_k_enable_overlap_constraint").value)
        self.dynamic_min_k_use_output_overlap = bool(
            self.get_parameter("dynamic_min_k_use_output_overlap").value)
        self.dynamic_min_k_prefer_lower_overlap = bool(
            self.get_parameter("dynamic_min_k_prefer_lower_overlap").value)
        self.dynamic_tracking_enabled = bool(
            self.get_parameter("dynamic_tracking_enabled").value)
        self.robot_component_filter_enabled = bool(
            self.get_parameter("robot_component_filter_enabled").value)
        self.robot_marker_require_expected_count = bool(
            self.get_parameter("robot_marker_require_expected_count").value)
        self.use_time_synchronized_robot_markers = bool(
            self.get_parameter("use_time_synchronized_robot_markers").value)
        self.fallback_to_latest_marker_on_time_miss = bool(
            self.get_parameter("fallback_to_latest_marker_on_time_miss").value)
        self.robot_component_filter_fail_closed = bool(
            self.get_parameter("robot_component_filter_fail_closed").value)
        self.publish_debug_clouds = bool(self.get_parameter("publish_debug_clouds").value)
        if not 0.0 <= self.robot_component_overlap_threshold <= 1.0:
            raise ValueError(
                "robot_component_overlap_threshold must be in [0, 1]")
        if self.robot_sphere_filter_margin_m < 0.0:
            raise ValueError("robot_sphere_filter_margin_m must be non-negative")
        if self.robot_marker_expected_sphere_count <= 0:
            raise ValueError(
                "robot_marker_expected_sphere_count must be positive")
        if self.robot_marker_subscription_depth <= 0:
            raise ValueError("robot_marker_subscription_depth must be positive")
        if self.robot_marker_sync_wait_timeout_s < 0.0:
            raise ValueError(
                "robot_marker_sync_wait_timeout_s must be non-negative")
        self.robot_marker_buffer_duration_ns = int(
            max(0.0, self.robot_marker_buffer_duration_s) * 1e9)
        self.robot_marker_max_stamp_delta_ns = int(
            max(0.0, self.robot_marker_max_stamp_delta_s) * 1e9)

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
            dynamic_enable_min_k_search=self.dynamic_enable_min_k_search,
            dynamic_min_k_beam_width=self.dynamic_min_k_beam_width,
            dynamic_min_k_max_states=self.dynamic_min_k_max_states,
            dynamic_max_allowed_overlap_fraction=(
                self.dynamic_max_allowed_overlap_fraction),
            dynamic_min_k_use_output_overlap=self.dynamic_min_k_use_output_overlap,
            dynamic_min_k_enable_overlap_constraint=(
                self.dynamic_min_k_enable_overlap_constraint),
            dynamic_min_k_prefer_lower_overlap=self.dynamic_min_k_prefer_lower_overlap,
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
        self.latest_points_received_monotonic = time.monotonic()
        self.latest_frame = self.target_frame
        self.have_message = True
        self.ever_received = True

    def _robot_marker_callback(self, message):
        self.latest_robot_markers = message
        stamp_ns = self._marker_stamp_ns(message)
        if stamp_ns == 0:
            stamp_ns = self._stamp_to_ns(self.get_clock().now().to_msg())
        self.robot_marker_buffer.append((stamp_ns, message))
        self._prune_robot_marker_buffer(stamp_ns)

    def _select_robot_markers(self, stamp):
        if not self.use_time_synchronized_robot_markers:
            self.latest_robot_marker_delta_s = None
            self.latest_robot_marker_signed_delta_s = None
            self.latest_robot_marker_buffer_span_s = None
            return self.latest_robot_markers
        if not self.robot_marker_buffer:
            self.latest_robot_marker_delta_s = None
            self.latest_robot_marker_signed_delta_s = None
            self.latest_robot_marker_buffer_span_s = None
            return None
        target_ns = self._stamp_to_ns(stamp)
        if target_ns == 0:
            self.latest_robot_marker_delta_s = None
            self.latest_robot_marker_signed_delta_s = None
            self.latest_robot_marker_buffer_span_s = None
            return self.latest_robot_markers
        best_stamp_ns, best_markers = min(
            self.robot_marker_buffer,
            key=lambda item: abs(item[0] - target_ns),
        )
        signed_delta_ns = best_stamp_ns - target_ns
        delta_ns = abs(signed_delta_ns)
        self.latest_robot_marker_delta_s = delta_ns / 1e9
        self.latest_robot_marker_signed_delta_s = signed_delta_ns / 1e9
        marker_stamps = [item[0] for item in self.robot_marker_buffer]
        self.latest_robot_marker_buffer_span_s = (
            (min(marker_stamps) - target_ns) / 1e9,
            (max(marker_stamps) - target_ns) / 1e9,
        )
        if (
            self.robot_marker_max_stamp_delta_ns > 0
            and delta_ns > self.robot_marker_max_stamp_delta_ns
        ):
            if self.fallback_to_latest_marker_on_time_miss:
                return self.latest_robot_markers
            return None
        return best_markers

    def _robot_sphere_geometry(self, marker_array, stamp):
        if marker_array is None:
            return None
        centers, radii, _ = marker_spheres(marker_array)
        if (
            self.robot_marker_require_expected_count
            and len(centers) != self.robot_marker_expected_sphere_count
        ):
            self._warn_throttled(
                "dropping dynamic frame: expected "
                f"{self.robot_marker_expected_sphere_count} robot spheres, "
                f"received {len(centers)}")
            return None
        if len(centers) == 0:
            return None
        marker_frames = {
            marker.header.frame_id for marker in marker_array.markers
            if marker.type == marker.SPHERE and marker.action == marker.ADD
        }
        if len(marker_frames) != 1 or not next(iter(marker_frames), ""):
            self._warn_throttled(
                "dropping dynamic frame: robot spheres need one valid frame")
            return None
        marker_frame = next(iter(marker_frames))
        if marker_frame != self.target_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    marker_frame,
                    stamp,
                    timeout=Duration(seconds=0.05),
                )
                q = transform.transform.rotation
                t = transform.transform.translation
                centers = centers @ _rotation_matrix(q.x, q.y, q.z, q.w).T
                centers += np.asarray((t.x, t.y, t.z), dtype=np.float64)
            except TransformException as exc:
                self._warn_throttled(
                    "dropping dynamic frame: cannot transform robot spheres "
                    f"{marker_frame}->{self.target_frame}: {exc}")
                return None
        if (
            not np.isfinite(centers).all()
            or not np.isfinite(radii).all()
            or np.any(radii <= 0.0)
        ):
            self._warn_throttled(
                "dropping dynamic frame: invalid robot sphere geometry")
            return None
        return centers, radii

    def _prune_robot_marker_buffer(self, latest_stamp_ns):
        if self.robot_marker_buffer_duration_ns <= 0:
            self.robot_marker_buffer = self.robot_marker_buffer[-1:]
            return
        cutoff = latest_stamp_ns - self.robot_marker_buffer_duration_ns
        self.robot_marker_buffer = [
            item for item in self.robot_marker_buffer if item[0] >= cutoff]

    @classmethod
    def _marker_stamp_ns(cls, marker_array):
        for marker in marker_array.markers:
            stamp_ns = cls._stamp_to_ns(marker.header.stamp)
            if stamp_ns != 0:
                return stamp_ns
        return 0

    @staticmethod
    def _stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _tick(self):
        if not self.ever_received:
            return
        if self.pending_points_frame is None and self.have_message:
            self.pending_points_frame = (
                self.latest_points,
                self.latest_stamp,
                self.latest_points_received_monotonic,
            )
            self.have_message = False
        has_new_points_message = self.pending_points_frame is not None
        if has_new_points_message:
            points, point_stamp, received_monotonic = self.pending_points_frame
        else:
            points = np.empty((0, 3), dtype=np.float64)
            point_stamp = self.latest_stamp
            received_monotonic = time.monotonic()
        robot_centers = None
        robot_radii = None
        robot_sphere_count = 0
        if self.robot_component_filter_enabled and has_new_points_message:
            robot_markers = self._select_robot_markers(point_stamp)
            geometry = self._robot_sphere_geometry(robot_markers, point_stamp)
            if geometry is None:
                if self.robot_component_filter_fail_closed:
                    waited_s = time.monotonic() - received_monotonic
                    if waited_s < self.robot_marker_sync_wait_timeout_s:
                        return
                    delta = self.latest_robot_marker_delta_s
                    delta_text = "unknown" if delta is None else f"{delta:.3f}s"
                    signed_delta = getattr(
                        self, "latest_robot_marker_signed_delta_s", None)
                    signed_delta_text = (
                        "unknown" if signed_delta is None
                        else f"{signed_delta:+.3f}s")
                    span = getattr(
                        self, "latest_robot_marker_buffer_span_s", None)
                    span_text = (
                        "unknown" if span is None
                        else f"[{span[0]:+.3f}s, {span[1]:+.3f}s]")
                    self._warn_throttled(
                        "dropping dynamic frame: synchronized robot spheres "
                        f"unavailable after waiting {waited_s:.3f}s "
                        f"(marker delta={delta_text}, nearest signed delta="
                        f"{signed_delta_text}, buffer relative span={span_text}, "
                        f"buffer size="
                        f"{len(getattr(self, 'robot_marker_buffer', []))})")
                    self.pending_points_frame = None
                    self.tracker.clear()
                    stamp = self.get_clock().now().to_msg()
                    self._publish_sphere_cloud(stamp, [])
                    self._publish_markers(stamp, [])
                    if self.publish_debug_clouds:
                        empty = np.empty((0, 3), dtype=np.float64)
                        self._publish_xyz_cloud(self.voxel_pub, stamp, empty)
                        self._publish_xyz_cloud(self.uncovered_pub, stamp, empty)
                        self._publish_xyz_cloud(
                            self.robot_rejected_voxel_pub, stamp, empty)
                    return
            else:
                robot_centers, robot_radii = geometry
                robot_sphere_count = len(robot_centers)
        elif self.robot_component_filter_enabled:
            # The timer also drives tracker missed-update/TTL behavior.  An
            # empty timer tick is not a pointcloud frame, so reusing the last
            # point timestamp here would create false marker-sync failures.
            self.latest_robot_marker_delta_s = None
            self.latest_robot_marker_signed_delta_s = None
            self.latest_robot_marker_buffer_span_s = None
        if has_new_points_message:
            self.pending_points_frame = None
        result = generate_dynamic_spheres(
            points,
            self._core_parameters(),
            robot_sphere_centers=robot_centers,
            robot_sphere_radii=robot_radii,
            robot_component_overlap_threshold=(
                self.robot_component_overlap_threshold),
            robot_sphere_margin_m=self.robot_sphere_filter_margin_m,
        )
        now = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9
        spheres = result.spheres
        if self.dynamic_tracking_enabled:
            self.tracker.remove_tracks_in_bounds(
                result.robot_rejected_component_bounds,
                self.dynamic_association_distance_m,
            )
            spheres = self.tracker.update(spheres, now_sec)
        stamp = now.to_msg()
        self._publish_sphere_cloud(stamp, spheres)
        self._publish_markers(stamp, spheres)
        if self.publish_debug_clouds:
            self._publish_xyz_cloud(self.voxel_pub, stamp, result.voxel_centers)
            self._publish_xyz_cloud(self.uncovered_pub, stamp, result.uncovered_voxels)
            self._publish_xyz_cloud(
                self.robot_rejected_voxel_pub,
                stamp,
                result.robot_rejected_voxels,
            )
        coverage = 1.0 - len(result.uncovered_voxels) / max(
            1, len(result.voxel_centers))
        pre_merge_count = sum(
            component.pre_merge_sphere_count
            for component in result.components)
        agglomerative_count = sum(
            component.agglomerative_sphere_count
            for component in result.components)
        min_k_count = sum(
            component.min_k_sphere_count
            for component in result.components)
        max_raw_overlap = max(
            (component.max_raw_overlap_fraction
             for component in result.components),
            default=0.0)
        max_output_overlap = max(
            (component.max_output_overlap_fraction
             for component in result.components),
            default=0.0)
        min_k_states = sum(
            component.min_k_states_explored
            for component in result.components)
        min_k_reason = ",".join(sorted({
            component.min_k_search_termination
            for component in result.components
        })) or "not_needed"
        overlap_constraint_ok = all(
            component.overlap_constraint_satisfied
            for component in result.components)
        max_robot_component_overlap = max(
            result.robot_component_overlap_fractions, default=0.0)
        marker_delta_text = (
            "n/a" if self.latest_robot_marker_delta_s is None
            else f"{self.latest_robot_marker_delta_s:.3f}s")
        self._info_throttled(
            f"dynamic points={len(points)}, voxels={len(result.voxel_centers)}, "
            f"pre_merge_spheres={pre_merge_count}, "
            f"agglomerative_spheres={agglomerative_count}, "
            f"min_k_spheres={min_k_count}, "
            f"generated_spheres={len(result.spheres)}, "
            f"tracked_spheres={len(spheres)}, "
            f"components={result.input_component_count}, "
            f"robot_rejected_components="
            f"{result.robot_rejected_component_count}, "
            f"robot_rejected_voxels={result.robot_rejected_voxel_count}, "
            f"robot_spheres={robot_sphere_count}, "
            f"marker_delta={marker_delta_text}, "
            f"max_robot_component_overlap="
            f"{max_robot_component_overlap:.3f}, "
            f"max_raw_overlap={max_raw_overlap:.3f}, "
            f"max_output_overlap={max_output_overlap:.3f}, "
            f"min_k_states={min_k_states}, min_k_reason={min_k_reason}, "
            f"overlap_constraint_ok={overlap_constraint_ok}, "
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
