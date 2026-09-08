"""ROS 2 latest-cache fusion adapter for static and dynamic sphere clouds."""

from __future__ import annotations

import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.clock import JumpThreshold
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.obstacle_sphere_fusion_core import (
    filter_spheres_containing_robot,
    FusionParameters,
    SphereFusionCache,
    spheres_from_field_rows,
)
from rmp_camera.vision_geometry import marker_spheres
from rmp_camera.human_static_filter import HumanStaticFilter
from rmp_camera.human_static_filter_core import fully_excluded_spheres, refit_mixed_static_spheres
from rmp_camera.static_result_status import StaticResultGate, read_result_status
from rmp_camera.pointcloud_numpy import read_numeric_cloud


class ObstacleSphereFusionNode(Node):
    def __init__(self):
        super().__init__("obstacle_sphere_fusion_node")
        self._declare_parameters()
        self._read_parameters()
        self.static_result_gate = StaticResultGate(timeout_s=1.0, capacity=8)
        self.static_clock_jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1),
                          on_clock_change=True), post_callback=self._static_clock_reset)
        self.human_filter = HumanStaticFilter(self)
        self.static_support = {}
        self.static_free_support = {}
        self.static_esdf_cleared = {}
        self.static_generation_stamp = None
        self.source_generation = {}
        self.source_support = {1: {}, 2: {}}
        self.current_fusion_support = {}
        self.static_refit_key = None
        self.static_refit_result = None
        self.static_refit_support = None
        self.last_human_filter_log = 0.0
        self.cache = SphereFusionCache(self._core_parameters())
        self.robot_centers = np.empty((0, 3), dtype=np.float64)
        self.robot_radii = np.empty(0, dtype=np.float64)
        self.robot_marker_stamp_sec = None
        self.previous_marker_keys: set[tuple[str, int]] = set()
        self.clear_markers_on_next_publish = True
        self.last_warn_time = 0.0
        self.last_filter_log_time = 0.0
        self.static_sub = self.create_subscription(
            PointCloud2, self.static_sphere_cloud_topic, self._static_callback, 5)
        self.static_status_sub = self.create_subscription(
            DiagnosticArray, self.static_result_status_topic, self._static_status_callback, 5)
        if self.human_filter.enabled or self.fusion_coverage_guard_enabled:
            self.support_sub = self.create_subscription(
                PointCloud2, self.human_filter.support_topic, self._support_callback, 5)
            self.free_support_sub = self.create_subscription(
                PointCloud2, self.human_filter.free_support_topic,
                self._free_support_callback, 5)
        self.dynamic_sub = self.create_subscription(
            PointCloud2, self.dynamic_sphere_cloud_topic, self._dynamic_callback, 5)
        self.human_sub = self.create_subscription(
            PointCloud2, self.human_sphere_cloud_topic, self._human_callback, 5)
        self.coverage_support_subs = []
        if self.fusion_coverage_guard_enabled:
            for source, topic in ((1, self.dynamic_sphere_cloud_topic), (2, self.human_sphere_cloud_topic)):
                self.coverage_support_subs.append(self.create_subscription(
                    PointCloud2, topic + "/support",
                    lambda msg, source=source: self._source_support_callback(msg, source), 2))
        self.robot_marker_sub = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self._robot_marker_callback,
            5,
        )
        self.cloud_pub = self.create_publisher(PointCloud2, self.combined_sphere_cloud_topic, 2)
        self.marker_pub = self.create_publisher(MarkerArray, self.combined_marker_topic, 2)
        self.coverage_status_pub = self.create_publisher(
            DiagnosticArray, self.combined_sphere_cloud_topic + "/status", 2)
        period = 1.0 if self.fusion_rate_hz <= 0.0 else max(0.01, 1.0 / self.fusion_rate_hz)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"Sphere fusion started: static={self.static_sphere_cloud_topic}, "
            f"dynamic={self.dynamic_sphere_cloud_topic}, "
            f"human={self.human_sphere_cloud_topic}, frame={self.target_frame}, "
            f"robot_filter={'on' if self.robot_overlap_filter_enabled else 'off'}, "
            f"robot_coverage_threshold={self.robot_overlap_coverage_threshold:.2f}")

    def _declare_parameters(self):
        defaults = {
            "static_sphere_cloud_topic": "/rmp_camera/esdf_medial_sphere_cloud",
            "static_result_status_topic": "/rmp_camera/static_sphere_result_status",
            "static_require_result_status": True,
            "static_fast_empty_clear": True,
            "human_static_refit_enabled": True,
            "human_static_refit_min_radius_m": 0.04,
            "human_static_refit_coverage_tolerance_m": 0.02,
            "dynamic_sphere_cloud_topic": "/rmp_camera/dynamic_obstacle_sphere_cloud",
            "human_sphere_cloud_topic": "/rmp_camera/human_sphere_cloud",
            "combined_sphere_cloud_topic": "/rmp_camera/combined_obstacle_sphere_cloud",
            "combined_marker_topic": "/rmp_camera/combined_obstacle_sphere_markers",
            "robot_sphere_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
            "target_frame": "base_link",
            "fusion_rate_hz": 10.0,
            "fusion_overlap_tolerance_m": 0.01,
            "fusion_coverage_guard_enabled": False,
            "fusion_coverage_tolerance_m": .02,
            "fusion_dynamic_priority": True,
            "fusion_hysteresis_sec": 0.5,
            "fusion_max_total_spheres": 512,
            "static_cache_expire_enabled": False,
            "static_empty_confirmation_frames": 3,
            "dynamic_handover_grace_sec": 0.5,
            "keep_dynamic_until_static_overlap": True,
            "dynamic_absolute_max_ttl_sec": 3.0,
            "human_priority": True,
            "human_handover_grace_sec": 0.35,
            "human_absolute_max_ttl_sec": 0.35,
            "human_empty_confirmation_frames": 3,
            "robot_overlap_filter_enabled": False,
            "robot_overlap_coverage_threshold": 0.8,
            "robot_overlap_sample_count": 256,
            "robot_overlap_robot_margin_m": 0.0,
            "robot_overlap_marker_timeout_s": 0.25,
            "static_marker_color": [0.15, 0.55, 1.0],
            "dynamic_marker_color": [1.0, 0.12, 0.04],
            "human_marker_color": [0.20, 1.0, 0.25],
            "marker_alpha": 0.65,
            "marker_lifetime_s": 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        strings = (
            "static_sphere_cloud_topic", "static_result_status_topic", "dynamic_sphere_cloud_topic",
            "human_sphere_cloud_topic",
            "combined_sphere_cloud_topic", "combined_marker_topic",
            "robot_sphere_marker_topic", "target_frame")
        for name in strings:
            setattr(self, name, str(self.get_parameter(name).value))
        floats = (
            "human_static_refit_min_radius_m", "human_static_refit_coverage_tolerance_m",
            "fusion_rate_hz", "fusion_overlap_tolerance_m", "fusion_hysteresis_sec",
            "fusion_coverage_tolerance_m",
            "dynamic_handover_grace_sec", "dynamic_absolute_max_ttl_sec",
            "human_handover_grace_sec", "human_absolute_max_ttl_sec",
            "robot_overlap_coverage_threshold", "robot_overlap_robot_margin_m",
            "robot_overlap_marker_timeout_s", "marker_alpha", "marker_lifetime_s")
        for name in floats:
            setattr(self, name, float(self.get_parameter(name).value))
        self.human_static_refit_enabled = bool(self.get_parameter(
            "human_static_refit_enabled").value)
        if (not np.isfinite(self.human_static_refit_min_radius_m)
                or self.human_static_refit_min_radius_m <= 0
                or not np.isfinite(self.human_static_refit_coverage_tolerance_m)
                or self.human_static_refit_coverage_tolerance_m < 0):
            raise ValueError("invalid human static refit radius/tolerance")
        integers = (
            "fusion_max_total_spheres", "static_empty_confirmation_frames",
            "human_empty_confirmation_frames", "robot_overlap_sample_count")
        for name in integers:
            setattr(self, name, int(self.get_parameter(name).value))
        booleans = (
            "fusion_dynamic_priority", "static_cache_expire_enabled",
            "fusion_coverage_guard_enabled",
            "static_require_result_status", "static_fast_empty_clear",
            "keep_dynamic_until_static_overlap", "human_priority",
            "robot_overlap_filter_enabled")
        for name in booleans:
            setattr(self, name, bool(self.get_parameter(name).value))
        self.static_marker_color = tuple(float(v) for v in self.get_parameter(
            "static_marker_color").value)
        self.dynamic_marker_color = tuple(float(v) for v in self.get_parameter(
            "dynamic_marker_color").value)
        self.human_marker_color = tuple(float(v) for v in self.get_parameter(
            "human_marker_color").value)
        if (
            len(self.static_marker_color) != 3
            or len(self.dynamic_marker_color) != 3
            or len(self.human_marker_color) != 3
        ):
            raise ValueError("marker colors must contain exactly three RGB values")
        if not 0.0 <= self.robot_overlap_coverage_threshold <= 1.0:
            raise ValueError("robot_overlap_coverage_threshold must be in [0, 1]")
        if self.robot_overlap_sample_count <= 0:
            raise ValueError("robot_overlap_sample_count must be positive")
        if self.human_empty_confirmation_frames <= 0:
            raise ValueError("human_empty_confirmation_frames must be positive")
        if self.robot_overlap_robot_margin_m < 0.0:
            raise ValueError("robot_overlap_robot_margin_m must be non-negative")
        if self.robot_overlap_marker_timeout_s < 0.0:
            raise ValueError("robot_overlap_marker_timeout_s must be non-negative")
        if self.marker_lifetime_s < 0.0:
            raise ValueError("marker_lifetime_s must be non-negative")
        if not np.isfinite(self.fusion_coverage_tolerance_m) or self.fusion_coverage_tolerance_m < 0.:
            raise ValueError("fusion coverage tolerance must be finite and non-negative")

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
            dynamic_absolute_max_ttl_sec=self.dynamic_absolute_max_ttl_sec,
            human_priority=self.human_priority,
            human_handover_grace_sec=self.human_handover_grace_sec,
            human_absolute_max_ttl_sec=self.human_absolute_max_ttl_sec,
            human_empty_confirmation_frames=self.human_empty_confirmation_frames,
            coverage_guard_enabled=self.fusion_coverage_guard_enabled,
            coverage_tolerance_m=self.fusion_coverage_tolerance_m)

    @staticmethod
    def _stamp_sec(message):
        return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9

    def _parse_cloud(self, message, source_type):
        names = [field.name for field in message.fields]
        required = {"x", "y", "z", "raw_radius", "output_radius"}
        if not required.issubset(names):
            raise ValueError(f"sphere cloud missing fields: {sorted(required - set(names))}")
        rows = point_cloud2.read_points_list(message, field_names=names, skip_nans=False)
        positional = [[getattr(row, name) for name in names] for row in rows]
        spheres = spheres_from_field_rows(names, positional, source_type)
        if any(not np.isfinite((s.x, s.y, s.z, s.raw_radius, s.output_radius)).all()
               or s.raw_radius <= 0. or s.output_radius < s.raw_radius for s in spheres):
            raise ValueError("invalid sphere geometry; not an empty observation")
        return spheres

    def _static_callback(self, message):
        if not self.static_require_result_status:
            self._accept_static_result(message, None)
            return
        self._join_static_result(message, "cloud", message)

    def _static_status_callback(self, message):
        if not self.static_require_result_status:
            return
        try:
            state = read_result_status(message)
        except ValueError as exc:
            self._warn_throttled(str(exc))
            return
        self._join_static_result(message, "status", state)

    def _join_static_result(self, message, kind, payload):
        if message.header.frame_id != self.target_frame:
            self._warn_throttled("rejecting static result/status with mismatched frame")
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - self._stamp_sec(message)
        if not -0.05 <= age <= 1.0:
            self._warn_throttled("rejecting stale/future static result/status")
            return
        key = self._stamp_key(message)
        if (self.static_result_gate.last_completed is not None
                and key <= self.static_result_gate.last_completed):
            return
        if kind == "status" and payload == "unknown":
            self.cache.invalidate_static_observation()
            self._warn_throttled("static observation UNKNOWN; cached geometry retained")
        joined = self.static_result_gate.add(
            key, kind, payload, time.monotonic())
        if joined is not None:
            self._accept_static_result(*joined)

    def _static_clock_reset(self, _jump):
        self.static_result_gate.reset()
        self._reset_human_static_cache()
        if hasattr(self, "source_support"):
            self.source_support = {1: {}, 2: {}}
            self.source_generation.clear()
        if hasattr(self, "cache"):
            self.cache.invalidate_static_observation()

    def _accept_static_result(self, message, state):
        try:
            declared_count = message.width * message.height
            if state is not None and ((state == "empty") != (declared_count == 0)):
                raise ValueError("static result status disagrees with sphere cloud")
            spheres = self._parse_cloud(message, 0)
            if declared_count != len(spheres):
                raise ValueError("invalid static sphere rows; not an empty observation")
            fast_clear = self.static_fast_empty_clear and state == "empty"
            previous_count = len(self.cache.static_spheres)
            accepted = self.cache.update_static(
                spheres, message.header.frame_id, self._stamp_sec(message),
                confirmed_empty=fast_clear)
            if not accepted:
                self._warn_throttled(self.cache.last_rejection_reason)
            elif spheres or not self.cache.static_spheres:
                self.static_generation_stamp = self._stamp_key(message)
            if accepted and not spheres and not self.cache.static_spheres:
                if previous_count:
                    self.get_logger().info(
                        f"Static confirmed empty: cleared={previous_count}, "
                        f"fast={fast_clear}, generation={self.static_generation_stamp}")
                # Send DELETE now even when bag playback stops advancing /clock.
                self._tick()
        except (ValueError, TypeError) as exc:
            self._warn_throttled(f"rejecting static sphere cloud: {exc}")

    @staticmethod
    def _stamp_key(message):
        return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

    def _support_callback(self, message):
        if message.header.frame_id != self.target_frame:
            return
        try:
            rows = point_cloud2.read_points_list(
                message, field_names=("x", "y", "z", "sphere_index"), skip_nans=False)
            xyz = np.asarray([(r.x, r.y, r.z) for r in rows], dtype=float).reshape(-1, 3)
            ids = np.asarray([r.sphere_index for r in rows], dtype=np.int64)
            if not np.isfinite(xyz).all() or np.any(ids < 0):
                return
            self.static_support[self._stamp_key(message)] = (xyz, ids)
            while len(self.static_support) > 4:
                del self.static_support[next(iter(self.static_support))]
        except (ValueError, TypeError, AssertionError) as exc:
            self._warn_throttled(f"rejecting static sphere support: {exc}")

    def _human_filtered_static(self, now_sec):
        spheres = self.cache.static_spheres
        support = self.static_support.get(self.static_generation_stamp)
        if not self.human_filter.enabled or not spheres or support is None:
            if getattr(self, "fusion_coverage_guard_enabled", False) and support is not None:
                points, indices = support
                for index, sphere in enumerate(spheres):
                    selected = points[indices == index]
                    if len(selected):
                        self.current_fusion_support[sphere] = selected
            return spheres
        points, indices = support
        if np.any(indices >= len(spheres)):
            return spheres
        # Retained/unknown support is never deleted. Mixed balls can be
        # tightened around it before the next full static solve.
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        excluded = self.human_filter.excluded(unique, now_sec)
        excluded = excluded[inverse]
        esdf_free_count = 0
        evidence = self.static_free_support.get(self.static_generation_stamp)
        if evidence is not None:
            evidence_stamp, xyz, ids, free = evidence
            if (-0.05 <= now_sec - evidence_stamp <= 0.25
                    and np.array_equal(points, xyz) and np.array_equal(indices, ids)):
                excluded = excluded | free
                esdf_free_count = int(free.sum())
                # A new map observation supersedes this old generation.
                # Do not revive it when the evidence packet later ages out;
                # newly occupied space must arrive as a new static solve.
                cleared = np.flatnonzero(fully_excluded_spheres(
                    indices, free, len(spheres)))
                self.static_esdf_cleared.setdefault(
                    self.static_generation_stamp, set()).update(cleared.tolist())
                while len(self.static_esdf_cleared) > 4:
                    del self.static_esdf_cleared[next(iter(self.static_esdf_cleared))]
        remove = fully_excluded_spheres(indices, excluded, len(spheres))
        for index in self.static_esdf_cleared.get(self.static_generation_stamp, ()):
            if index < len(remove):
                remove[index] = True
        refit_stats = dict(mixed=0, refitted=0, excluded_support_avoided=0)
        if getattr(self, "human_static_refit_enabled", False):
            key = (self.static_generation_stamp, id(support),
                   np.packbits(excluded).tobytes())
            if key != self.static_refit_key:
                started = time.monotonic()
                self.static_refit_result = refit_mixed_static_spheres(
                    spheres, points, indices, excluded,
                    self.human_static_refit_min_radius_m,
                    self.human_static_refit_coverage_tolerance_m)
                self.static_refit_key = key
                # Retain the paired support object while its identity is in
                # the cache key, preventing Python object-id reuse.
                self.static_refit_support = support
                self.static_refit_ms = (time.monotonic() - started) * 1000.0
            spheres, refit_stats = self.static_refit_result
        if ((remove.any() or refit_stats['mixed'])
                and time.monotonic() - self.last_human_filter_log >= 1.0):
            self.last_human_filter_log = time.monotonic()
            self.get_logger().info(
                f"Human static fast clear: input={len(spheres)}, removed={remove.sum()}, "
                f"active_voxels={self.human_filter.last_stats[0]}, "
                f"depth_free_voxels={self.human_filter.last_stats[1]}, "
                f"esdf_free_support={esdf_free_count}, "
                f"mixed={refit_stats['mixed']}, refitted={refit_stats['refitted']}, "
                f"excluded_support_avoided={refit_stats['excluded_support_avoided']}, "
                f"refit_volume_m3={refit_stats.get('refit_volume_before_m3', 0):.5f}"
                f"->{refit_stats.get('refit_volume_after_m3', 0):.5f}, "
                f"refit_ms={getattr(self, 'static_refit_ms', 0.0):.2f}")
        if getattr(self, "fusion_coverage_guard_enabled", False):
            for index, (sphere, drop) in enumerate(zip(spheres, remove)):
                selected = points[(indices == index) & ~excluded]
                if not drop and len(selected):
                    self.current_fusion_support[sphere] = selected
        return [s for s, drop in zip(spheres, remove) if not drop]

    def _reset_human_static_cache(self):
        self.static_refit_key = None
        self.static_refit_result = None
        self.static_refit_support = None
        if hasattr(self, "static_esdf_cleared"):
            self.static_support.clear()
            self.static_free_support.clear()
            self.static_esdf_cleared.clear()
            self.static_generation_stamp = None

    def _free_support_callback(self, message):
        if message.header.frame_id != self.target_frame:
            return
        try:
            rows = point_cloud2.read_points_list(message, field_names=(
                "x", "y", "z", "sphere_index", "esdf_free", "evidence_stamp"))
            if not rows:
                return
            xyz = np.asarray([(r.x, r.y, r.z) for r in rows], dtype=float)
            ids = np.asarray([r.sphere_index for r in rows], dtype=np.int64)
            flags = np.asarray([r.esdf_free for r in rows])
            stamp = rows[0].evidence_stamp
            if (not np.isfinite(xyz).all() or np.any(ids < 0)
                    or not np.isfinite(stamp) or not np.isin(flags, (0, 1)).all()
                    or any(r.evidence_stamp != stamp for r in rows)):
                return
            key = self._stamp_key(message)
            previous = self.static_free_support.get(key)
            if previous is not None and previous[0] > stamp:
                return
            self.static_free_support[key] = (stamp, xyz, ids, flags.astype(bool))
            while len(self.static_free_support) > 4:
                del self.static_free_support[next(iter(self.static_free_support))]
        except (ValueError, TypeError, AssertionError) as exc:
            self._warn_throttled(f"rejecting static free support: {exc}")

    def _dynamic_callback(self, message):
        try:
            spheres = self._parse_cloud(message, 1)
            accepted = self.cache.update_dynamic(
                spheres, message.header.frame_id, self._stamp_sec(message))
            if not accepted:
                self._warn_throttled(self.cache.last_rejection_reason)
            elif hasattr(self, "source_generation") and (spheres or not self.cache.dynamic_spheres):
                self.source_generation[1] = self._stamp_key(message)
        except (ValueError, TypeError) as exc:
            self._warn_throttled(f"rejecting dynamic sphere cloud: {exc}")

    def _human_callback(self, message):
        try:
            spheres = self._parse_cloud(message, 2)
            accepted = self.cache.update_human(
                spheres, message.header.frame_id, self._stamp_sec(message))
            if not accepted:
                self._warn_throttled(self.cache.last_rejection_reason)
            if accepted and hasattr(self, "source_generation") and (spheres or not self.cache.human_spheres):
                self.source_generation[2] = self._stamp_key(message)
            if accepted and not spheres and not self.cache.human_spheres:
                # Publish DELETE markers from the subscription callback as
                # well as the ROS-time timer. This still clears RViz when a
                # rosbag ends and /clock no longer advances.
                self._tick()
        except (ValueError, TypeError) as exc:
            self._warn_throttled(f"rejecting human sphere cloud: {exc}")

    def _source_support_callback(self, message, source):
        if message.header.frame_id != self.target_frame or message.width * message.height > 200000:
            return
        try:
            values = read_numeric_cloud(message, ('x', 'y', 'z', 'sphere_index', 'evidence_stamp'))
            if not len(values) or not np.isfinite(values).all():
                return
            indices = values[:, 3].astype(np.int64)
            if np.any(indices < 0) or not np.array_equal(indices, values[:, 3]):
                return
            evidence = float(values[0, 4])
            if not np.all(values[:, 4] == evidence):
                return
            packets = self.source_support[source]
            packets[self._stamp_key(message)] = (values[:, :3], indices, evidence)
            while len(packets) > 4:
                del packets[next(iter(packets))]
        except (ValueError, TypeError, BufferError) as exc:
            self._warn_throttled("invalid fusion support: " + str(exc))

    def _paired_source_support(self, now_sec):
        for source, spheres in ((1, self.cache.dynamic_spheres), (2, self.cache.human_spheres)):
            packet = self.source_support[source].get(self.source_generation.get(source))
            if packet is None:
                continue
            points, indices, evidence = packet
            if not -.05 <= now_sec - evidence <= .25 or np.any(indices >= len(spheres)):
                continue
            for index, sphere in enumerate(spheres):
                selected = points[indices == index]
                if len(selected):
                    self.current_fusion_support[sphere] = selected

    def _robot_marker_callback(self, message):
        marker_frame = next((
            marker.header.frame_id for marker in message.markers
            if marker.header.frame_id
        ), "")
        if marker_frame and marker_frame != self.target_frame:
            self._warn_throttled(
                f"rejecting robot sphere markers: expected frame "
                f"{self.target_frame}, got {marker_frame}")
            return
        centers, radii, _ = marker_spheres(message)
        self.robot_centers = centers
        self.robot_radii = radii
        stamps = [
            float(marker.header.stamp.sec)
            + float(marker.header.stamp.nanosec) * 1e-9
            for marker in message.markers
            if marker.header.stamp.sec or marker.header.stamp.nanosec
        ]
        self.robot_marker_stamp_sec = max(stamps) if stamps else None

    def _tick(self):
        now = self.get_clock().now()
        if getattr(self, "fusion_coverage_guard_enabled", False):
            self.current_fusion_support = {}
            static = self._human_filtered_static(now.nanoseconds * 1e-9)
            self._paired_source_support(now.nanoseconds * 1e-9)
            stats = {}
            spheres = self.cache.combined(now.nanoseconds * 1e-9, static_override=static,
                support=self.current_fusion_support, stats=stats)
            if not stats.get("coverage_valid", True):
                self._warn_throttled("Fusion sphere cap exceeded: final coverage NOT guaranteed")
            self.coverage_status_pub.publish(DiagnosticArray(header=Header(
                stamp=now.to_msg(), frame_id=self.target_frame), status=[DiagnosticStatus(
                    name=self.get_name(), level=DiagnosticStatus.OK if stats.get('coverage_valid')
                    else DiagnosticStatus.WARN, message="pre_robot_fusion_coverage",
                    values=[KeyValue(key=key, value=str(value)) for key, value in stats.items()])]))
        else:
            spheres = self.cache.combined(now.nanoseconds * 1e-9,
                static_override=self._human_filtered_static(now.nanoseconds * 1e-9))
        spheres = self._filter_robot_covered(spheres, now.nanoseconds * 1e-9)
        stamp = now.to_msg()
        self._publish_cloud(stamp, spheres)
        self._publish_markers(stamp, spheres)

    def _filter_robot_covered(self, spheres, now_sec):
        if not self.robot_overlap_filter_enabled or not spheres:
            return spheres
        if len(self.robot_centers) == 0:
            self._warn_throttled(
                "Robot-overlap filtering is enabled but no robot spheres are available.")
            return spheres
        if (
            self.robot_overlap_marker_timeout_s > 0.0
            and self.robot_marker_stamp_sec is not None
            and now_sec - self.robot_marker_stamp_sec
            > self.robot_overlap_marker_timeout_s
        ):
            self._warn_throttled(
                "Robot-overlap filtering skipped because robot markers are stale.")
            return spheres
        expanded_radii = self.robot_radii + self.robot_overlap_robot_margin_m
        kept, removed = filter_spheres_containing_robot(
            spheres,
            self.robot_centers,
            expanded_radii,
            self.robot_overlap_coverage_threshold,
            self.robot_overlap_sample_count,
        )
        now_monotonic = time.monotonic()
        if now_monotonic - self.last_filter_log_time >= 2.0:
            self.last_filter_log_time = now_monotonic
            removed_static = sum(item.source_type == 0 for item in removed)
            removed_dynamic = sum(item.source_type == 1 for item in removed)
            removed_human = sum(item.source_type == 2 for item in removed)
            self.get_logger().info(
                f"Robot-containment post-filter: input={len(spheres)}, "
                f"removed_static={removed_static}, "
                f"removed_dynamic={removed_dynamic}, "
                f"removed_human={removed_human}, output={len(kept)}")
        return kept

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
        if self.clear_markers_on_next_publish:
            clear_marker = Marker()
            clear_marker.header = Header(stamp=stamp, frame_id=self.target_frame)
            clear_marker.action = Marker.DELETEALL
            markers.append(clear_marker)
            self.clear_markers_on_next_publish = False
        current = set()
        for index, sphere in enumerate(spheres):
            marker_id = index
            namespace = {
                0: "combined_static",
                1: "combined_dynamic",
                2: "combined_human",
            }.get(sphere.source_type, "combined_unknown")
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
            color = {
                0: self.static_marker_color,
                1: self.dynamic_marker_color,
                2: self.human_marker_color,
            }.get(sphere.source_type, self.dynamic_marker_color)
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
