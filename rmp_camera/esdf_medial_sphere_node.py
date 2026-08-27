"""ROS 2 adapter for dense static-ESDF medial obstacle spheres.

Numeric parameter guidance (all distance values are metres):

* ``update_rate_hz`` controls service requests; high values can monopolize the
  Python executor, while very low values make RViz stale.
* ``aabb_min_*_m`` and ``aabb_size_*_m`` define the query volume; an oversized
  box costs memory/CPU and an undersized box clips obstacles.
* ``unobserved_distance_value`` must match nvblox's sentinel or unobserved
  voxels may be interpreted as deep obstacle interiors.
* ``inside_epsilon_m`` rejects small negative surface noise; large values erode
  thin objects and zero retains sign jitter.
* ``target_coverage`` is a fraction in [0, 1]; values near one need more
  spheres, while low values leave more interior voxels uncovered.
* ``coverage_tolerance_m`` compensates voxel-centre discretization; too large
  inflates measured coverage and too small may require many extra spheres.
* ``plateau_epsilon_m`` joins near-equal adjacent minima; too large merges
  different depths and too small fragments noisy plateaus.
* ``minimum_center_spacing_m`` suppresses duplicate centres; too large blocks
  coverage of long/thin components and too small produces dense spheres.
* ``min_component_voxels`` removes noise; large values discard small objects.
* ``min_raw_sphere_radius_m`` removes tiny medial balls; large values lose thin
  geometry and values near zero retain noisy spheres.
* ``safety_margin_m`` is added only for output; large values over-inflate RViz
  obstacles and do not affect raw coverage.
* ``redundancy_tolerance_m`` relaxes containment; large values attempt more
  removals, each still guarded by the target-coverage test.
* ``max_spheres_per_component``, ``max_iterations_per_component``, and
  ``max_total_spheres`` bound work/output; small limits can prevent coverage,
  while very large limits increase runtime and visualization load.
* ``max_grid_voxels`` is a hard memory guard; too small rejects valid AABBs and
  too large permits allocations unsuitable for Python.
"""

import colorsys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.esdf_medial_sphere_core import generate_medial_spheres
from rmp_camera.vision_geometry import (
    marker_spheres,
    quaternion_to_matrix,
)


class DenseEsdfGrid:
    """Validated dense ``(x, y, z)`` NumPy view of an nvblox response."""

    def __init__(self, response, max_grid_voxels):
        message = response.esdf_and_gradients
        dims = message.layout.dim
        self.valid = False
        self.error = ""
        self.values = np.empty((0, 0, 0), dtype=np.float64)
        self.shape = (0, 0, 0)
        self.voxel_size_m = float(response.voxel_size_m)
        self.origin_m = np.array(
            [response.origin_m.x, response.origin_m.y, response.origin_m.z],
            dtype=np.float64,
        )
        if len(dims) < 3:
            self.error = "ESDF layout has fewer than three dimensions"
            return
        self.shape = tuple(int(dims[axis].size) for axis in range(3))
        if (
            any(size <= 0 for size in self.shape)
            or self.voxel_size_m <= 0.0
            or not np.isfinite(self.origin_m).all()
        ):
            self.error = "ESDF dimensions, origin, or voxel size are invalid"
            return
        voxel_count = int(np.prod(self.shape, dtype=np.int64))
        if voxel_count > int(max_grid_voxels):
            self.error = (
                f"ESDF grid has {voxel_count} voxels, exceeding "
                f"max_grid_voxels={max_grid_voxels}"
            )
            return

        data = np.asarray(message.data, dtype=np.float64)
        stride_y = int(dims[1].stride)
        stride_z = int(dims[2].stride)
        if data.size == 0 or stride_y <= 0 or stride_z <= 0:
            self.error = "ESDF data or layout strides are empty/invalid"
            return
        offsets = (
            np.arange(self.shape[0], dtype=np.int64)[:, None, None] * stride_y
            + np.arange(self.shape[1], dtype=np.int64)[None, :, None] * stride_z
            + np.arange(self.shape[2], dtype=np.int64)[None, None, :]
        )
        if int(offsets.max()) >= data.size:
            self.error = "ESDF layout strides address beyond the response data"
            return
        self.values = data[offsets]
        self.valid = True


class EsdfMedialSphereNode(Node):
    """Query nvblox's static dense signed ESDF and publish medial spheres."""

    def __init__(self):
        super().__init__("esdf_medial_sphere_node")
        self._declare_parameters()
        self._read_parameters()
        self.pending = False
        self.previous_marker_keys = set()
        self.last_info_time = 0.0
        self.last_warn_time = 0.0
        self.latest_robot_markers = None
        self.robot_marker_buffer = []
        self.latest_robot_marker_delta_s = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.client = self.create_client(EsdfAndGradients, self.service_name)
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
        self.sphere_cloud_pub = self.create_publisher(
            PointCloud2, self.sphere_cloud_topic, 2
        )
        self.inside_cloud_pub = self.create_publisher(
            PointCloud2, self.inside_voxel_cloud_topic, 2
        )
        self.uncovered_cloud_pub = self.create_publisher(
            PointCloud2, self.uncovered_voxel_cloud_topic, 2
        )
        self.robot_rejected_voxel_pub = self.create_publisher(
            PointCloud2, self.robot_rejected_voxel_cloud_topic, 2
        )
        period = 1.0 if self.update_rate_hz <= 0.0 else max(
            0.02, 1.0 / self.update_rate_hz
        )
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(
            "Dense ESDF medial sphere node started: "
            f"service={self.service_name}, frame={self.target_frame}, "
            f"rate={self.update_rate_hz:.2f} Hz, "
            f"target_coverage={self.target_coverage:.3f}, "
            f"single={self.enable_single_sphere_replacement}, "
            f"greedy={self.enable_greedy_set_cover}, "
            f"pruning={self.enable_general_coverage_pruning}, "
            f"merge={self.enable_agglomerative_merge}, "
            f"merge_radius<={self.merge_max_radius_m:.3f} m, "
            f"merge_growth<={self.merge_max_radius_growth_ratio:.3f}, "
            f"merge_gap<={self.merge_max_gap_m:.3f} m, "
            f"merge_esdf_guard={self.merge_enable_esdf_guard}, "
            f"merge_samples={self.merge_surface_sample_count}, "
            f"robot_component_filter={self.robot_component_filter_enabled}, "
            f"robot_overlap_threshold="
            f"{self.robot_component_overlap_threshold:.3f}"
        )

    def _declare_parameters(self):
        self.declare_parameter("service_name", "/nvblox_node/get_esdf_and_gradient")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("update_rate_hz", 1.0)
        self.declare_parameter("aabb_min_x_m", -1.5)
        self.declare_parameter("aabb_min_y_m", -1.5)
        self.declare_parameter("aabb_min_z_m", 0.0)
        self.declare_parameter("aabb_size_x_m", 3.0)
        self.declare_parameter("aabb_size_y_m", 3.0)
        self.declare_parameter("aabb_size_z_m", 2.0)
        self.declare_parameter("update_esdf", True)
        self.declare_parameter("visualize_esdf", False)
        self.declare_parameter("unobserved_distance_value", -1000.0)
        self.declare_parameter("inside_epsilon_m", 0.005)
        self.declare_parameter("target_coverage", 0.95)
        self.declare_parameter("coverage_tolerance_m", 0.01)
        self.declare_parameter("plateau_epsilon_m", 0.001)
        self.declare_parameter("minimum_center_spacing_m", 0.05)
        self.declare_parameter("min_component_voxels", 8)
        self.declare_parameter("min_raw_sphere_radius_m", 0.01)
        self.declare_parameter("safety_margin_m", 0.02)
        self.declare_parameter("redundancy_tolerance_m", 0.001)
        self.declare_parameter("max_spheres_per_component", 128)
        self.declare_parameter("max_iterations_per_component", 256)
        self.declare_parameter("max_total_spheres", 256)
        self.declare_parameter("enable_single_sphere_replacement", True)
        self.declare_parameter("enable_greedy_set_cover", True)
        self.declare_parameter("enable_general_coverage_pruning", True)
        self.declare_parameter("enable_surface_shell_guard", True)
        self.declare_parameter("surface_shell_thickness_m", 0.10)
        self.declare_parameter("target_shell_coverage", 0.98)
        self.declare_parameter("shell_coverage_loss_tolerance", 0.005)
        self.declare_parameter("max_optimization_matrix_elements", 20000000)
        self.declare_parameter("enable_agglomerative_merge", True)
        self.declare_parameter("merge_max_radius_m", 0.35)
        self.declare_parameter("merge_max_radius_growth_ratio", 1.45)
        self.declare_parameter("merge_max_gap_m", 0.05)
        self.declare_parameter("merge_enable_esdf_guard", True)
        self.declare_parameter("merge_max_free_space_distance_m", 0.08)
        self.declare_parameter("merge_surface_sample_count", 64)
        self.declare_parameter("merge_min_observed_surface_fraction", 0.70)
        self.declare_parameter("robot_component_filter_enabled", False)
        self.declare_parameter(
            "robot_sphere_marker_topic",
            "/rmp_camera/calibrated_robot_collision_sphere_markers",
        )
        self.declare_parameter("robot_component_overlap_threshold", 0.10)
        self.declare_parameter("robot_sphere_filter_margin_m", 0.0)
        self.declare_parameter("robot_marker_expected_sphere_count", 35)
        self.declare_parameter("robot_marker_require_expected_count", True)
        self.declare_parameter("use_time_synchronized_robot_markers", True)
        self.declare_parameter("robot_marker_buffer_duration_s", 1.0)
        self.declare_parameter("robot_marker_max_stamp_delta_s", 0.10)
        self.declare_parameter("robot_marker_subscription_depth", 2)
        self.declare_parameter("fallback_to_latest_marker_on_time_miss", False)
        self.declare_parameter("robot_component_filter_fail_closed", True)
        self.declare_parameter("max_grid_voxels", 8000000)
        self.declare_parameter(
            "marker_topic", "/rmp_camera/esdf_medial_sphere_markers"
        )
        self.declare_parameter(
            "sphere_cloud_topic", "/rmp_camera/esdf_medial_sphere_cloud"
        )
        self.declare_parameter(
            "inside_voxel_cloud_topic", "/rmp_camera/esdf_medial_inside_voxels"
        )
        self.declare_parameter(
            "uncovered_voxel_cloud_topic",
            "/rmp_camera/esdf_medial_uncovered_voxels",
        )
        self.declare_parameter(
            "robot_rejected_voxel_cloud_topic",
            "/rmp_camera/static_robot_rejected_voxels",
        )
        self.declare_parameter("publish_debug_clouds", True)

    def _read_parameters(self):
        for name in (
            "service_name",
            "target_frame",
            "marker_topic",
            "sphere_cloud_topic",
            "inside_voxel_cloud_topic",
            "uncovered_voxel_cloud_topic",
            "robot_sphere_marker_topic",
            "robot_rejected_voxel_cloud_topic",
        ):
            setattr(self, name, str(self.get_parameter(name).value))
        for name in (
            "update_rate_hz",
            "unobserved_distance_value",
            "inside_epsilon_m",
            "target_coverage",
            "coverage_tolerance_m",
            "plateau_epsilon_m",
            "minimum_center_spacing_m",
            "min_raw_sphere_radius_m",
            "safety_margin_m",
            "redundancy_tolerance_m",
            "surface_shell_thickness_m",
            "target_shell_coverage",
            "shell_coverage_loss_tolerance",
            "merge_max_radius_m",
            "merge_max_radius_growth_ratio",
            "merge_max_gap_m",
            "merge_max_free_space_distance_m",
            "merge_min_observed_surface_fraction",
            "robot_component_overlap_threshold",
            "robot_sphere_filter_margin_m",
            "robot_marker_buffer_duration_s",
            "robot_marker_max_stamp_delta_s",
        ):
            setattr(self, name, float(self.get_parameter(name).value))
        for name in (
            "min_component_voxels",
            "max_spheres_per_component",
            "max_iterations_per_component",
            "max_total_spheres",
            "max_optimization_matrix_elements",
            "merge_surface_sample_count",
            "max_grid_voxels",
            "robot_marker_expected_sphere_count",
            "robot_marker_subscription_depth",
        ):
            setattr(self, name, int(self.get_parameter(name).value))
        self.aabb_min_m = np.array(
            [
                float(self.get_parameter("aabb_min_x_m").value),
                float(self.get_parameter("aabb_min_y_m").value),
                float(self.get_parameter("aabb_min_z_m").value),
            ]
        )
        self.aabb_size_m = np.array(
            [
                float(self.get_parameter("aabb_size_x_m").value),
                float(self.get_parameter("aabb_size_y_m").value),
                float(self.get_parameter("aabb_size_z_m").value),
            ]
        )
        self.update_esdf = self._as_bool(self.get_parameter("update_esdf").value)
        self.visualize_esdf = self._as_bool(
            self.get_parameter("visualize_esdf").value
        )
        self.enable_single_sphere_replacement = self._as_bool(
            self.get_parameter("enable_single_sphere_replacement").value
        )
        self.enable_greedy_set_cover = self._as_bool(
            self.get_parameter("enable_greedy_set_cover").value
        )
        self.enable_general_coverage_pruning = self._as_bool(
            self.get_parameter("enable_general_coverage_pruning").value
        )
        self.enable_surface_shell_guard = self._as_bool(
            self.get_parameter("enable_surface_shell_guard").value
        )
        self.enable_agglomerative_merge = self._as_bool(
            self.get_parameter("enable_agglomerative_merge").value
        )
        self.merge_enable_esdf_guard = self._as_bool(
            self.get_parameter("merge_enable_esdf_guard").value
        )
        self.robot_component_filter_enabled = self._as_bool(
            self.get_parameter("robot_component_filter_enabled").value
        )
        self.robot_marker_require_expected_count = self._as_bool(
            self.get_parameter("robot_marker_require_expected_count").value
        )
        self.use_time_synchronized_robot_markers = self._as_bool(
            self.get_parameter("use_time_synchronized_robot_markers").value
        )
        self.fallback_to_latest_marker_on_time_miss = self._as_bool(
            self.get_parameter("fallback_to_latest_marker_on_time_miss").value
        )
        self.robot_component_filter_fail_closed = self._as_bool(
            self.get_parameter("robot_component_filter_fail_closed").value
        )
        self.publish_debug_clouds = self._as_bool(
            self.get_parameter("publish_debug_clouds").value
        )
        if np.any(self.aabb_size_m <= 0.0):
            raise ValueError("All AABB sizes must be positive")
        if not 0.0 <= self.target_coverage <= 1.0:
            raise ValueError("target_coverage must be in [0, 1]")
        if self.surface_shell_thickness_m < 0.0:
            raise ValueError("surface_shell_thickness_m must be non-negative")
        if not 0.0 <= self.target_shell_coverage <= 1.0:
            raise ValueError("target_shell_coverage must be in [0, 1]")
        if not 0.0 <= self.shell_coverage_loss_tolerance <= 1.0:
            raise ValueError("shell_coverage_loss_tolerance must be in [0, 1]")
        if self.max_optimization_matrix_elements <= 0:
            raise ValueError(
                "max_optimization_matrix_elements must be positive"
            )
        if not np.isfinite((
            self.merge_max_radius_m,
            self.merge_max_radius_growth_ratio,
            self.merge_max_gap_m,
            self.merge_max_free_space_distance_m,
            self.merge_min_observed_surface_fraction,
        )).all():
            raise ValueError("merge parameters must be finite")
        if self.merge_max_radius_m <= 0.0:
            raise ValueError("merge_max_radius_m must be positive")
        if self.merge_max_radius_growth_ratio < 1.0:
            raise ValueError("merge_max_radius_growth_ratio must be at least 1")
        if self.merge_max_gap_m < 0.0:
            raise ValueError("merge_max_gap_m must be non-negative")
        if self.merge_max_free_space_distance_m < 0.0:
            raise ValueError("merge_max_free_space_distance_m must be non-negative")
        if self.merge_surface_sample_count <= 0:
            raise ValueError("merge_surface_sample_count must be positive")
        if not 0.0 <= self.merge_min_observed_surface_fraction <= 1.0:
            raise ValueError(
                "merge_min_observed_surface_fraction must be in [0, 1]")
        if not 0.0 <= self.robot_component_overlap_threshold <= 1.0:
            raise ValueError(
                "robot_component_overlap_threshold must be in [0, 1]")
        if self.robot_sphere_filter_margin_m < 0.0:
            raise ValueError("robot_sphere_filter_margin_m must be non-negative")
        if self.robot_marker_expected_sphere_count <= 0:
            raise ValueError("robot_marker_expected_sphere_count must be positive")
        if self.robot_marker_subscription_depth <= 0:
            raise ValueError("robot_marker_subscription_depth must be positive")
        self.robot_marker_buffer_duration_ns = int(
            max(0.0, self.robot_marker_buffer_duration_s) * 1e9)
        self.robot_marker_max_stamp_delta_ns = int(
            max(0.0, self.robot_marker_max_stamp_delta_s) * 1e9)

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
            return self.latest_robot_markers
        if not self.robot_marker_buffer:
            self.latest_robot_marker_delta_s = None
            return None
        target_ns = self._stamp_to_ns(stamp)
        if target_ns == 0:
            self.latest_robot_marker_delta_s = None
            return self.latest_robot_markers
        best_stamp_ns, best_markers = min(
            self.robot_marker_buffer,
            key=lambda item: abs(item[0] - target_ns),
        )
        delta_ns = abs(best_stamp_ns - target_ns)
        self.latest_robot_marker_delta_s = delta_ns / 1e9
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
                "dropping static sphere update: expected "
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
                "dropping static sphere update: robot spheres need one valid frame")
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
                rotation = quaternion_to_matrix(transform.transform.rotation)
                translation = transform.transform.translation
                centers = centers @ rotation.T
                centers += np.asarray(
                    (translation.x, translation.y, translation.z),
                    dtype=np.float64,
                )
            except TransformException as exc:
                self._warn_throttled(
                    "dropping static sphere update: cannot transform robot spheres "
                    f"{marker_frame}->{self.target_frame}: {exc}")
                return None
        if (
            not np.isfinite(centers).all()
            or not np.isfinite(radii).all()
            or np.any(radii <= 0.0)
        ):
            self._warn_throttled(
                "dropping static sphere update: invalid robot sphere geometry")
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

    def tick(self):
        if self.pending:
            return
        if not self.client.service_is_ready():
            self._warn_throttled(f"ESDF service unavailable: {self.service_name}")
            return
        request = EsdfAndGradients.Request()
        request.update_esdf = self.update_esdf
        request.visualize_esdf = self.visualize_esdf
        request.use_aabb = True
        request.frame_id = self.target_frame
        request.aabb_min_m = Point(
            x=float(self.aabb_min_m[0]),
            y=float(self.aabb_min_m[1]),
            z=float(self.aabb_min_m[2]),
        )
        request.aabb_size_m = Vector3(
            x=float(self.aabb_size_m[0]),
            y=float(self.aabb_size_m[1]),
            z=float(self.aabb_size_m[2]),
        )
        request.spheres_to_clear_center_m = []
        request.spheres_to_clear_radius_m = []
        started_at = time.monotonic()
        self.pending = True
        future = self.client.call_async(request)
        future.add_done_callback(
            lambda done_future, started=started_at: self.handle_response(
                done_future, started
            )
        )

    def handle_response(self, future, service_started_at):
        self.pending = False
        service_elapsed_s = time.monotonic() - service_started_at
        stamp = self.get_clock().now().to_msg()
        try:
            response = future.result()
        except Exception as exc:
            self._warn_throttled(f"ESDF service call failed: {exc}")
            self.publish_empty(stamp)
            return
        if not response.success:
            self._warn_throttled("ESDF service returned success=false")
            self.publish_empty(stamp)
            return

        grid = DenseEsdfGrid(response, self.max_grid_voxels)
        if not grid.valid:
            self._warn_throttled(f"Invalid/empty ESDF grid: {grid.error}")
            self.publish_empty(stamp)
            return
        robot_centers = None
        robot_radii = None
        robot_sphere_count = 0
        if self.robot_component_filter_enabled:
            robot_markers = self._select_robot_markers(stamp)
            geometry = self._robot_sphere_geometry(robot_markers, stamp)
            if geometry is None:
                if self.robot_component_filter_fail_closed:
                    delta_text = (
                        "unknown" if self.latest_robot_marker_delta_s is None
                        else f"{self.latest_robot_marker_delta_s:.3f}s")
                    self._warn_throttled(
                        "dropping static sphere update: synchronized robot "
                        f"spheres unavailable (marker delta={delta_text}, "
                        f"buffer size={len(self.robot_marker_buffer)})")
                    self.publish_empty(stamp)
                    return
            else:
                robot_centers, robot_radii = geometry
                robot_sphere_count = len(robot_centers)
        processing_started_at = time.monotonic()
        sentinel_tolerance = max(
            1e-9, abs(self.unobserved_distance_value) * 1e-12
        )
        observed_mask = np.isfinite(grid.values) & (
            np.abs(grid.values - self.unobserved_distance_value)
            > sentinel_tolerance
        )
        observed_count = int(np.count_nonzero(observed_mask))
        negative_count = int(np.count_nonzero(observed_mask & (grid.values < 0.0)))
        try:
            result = generate_medial_spheres(
                grid.values,
                grid.origin_m,
                grid.voxel_size_m,
                unobserved_distance_value=self.unobserved_distance_value,
                inside_epsilon_m=self.inside_epsilon_m,
                target_coverage=self.target_coverage,
                coverage_tolerance_m=self.coverage_tolerance_m,
                plateau_epsilon_m=self.plateau_epsilon_m,
                minimum_center_spacing_m=self.minimum_center_spacing_m,
                min_component_voxels=self.min_component_voxels,
                min_raw_sphere_radius_m=self.min_raw_sphere_radius_m,
                safety_margin_m=self.safety_margin_m,
                redundancy_tolerance_m=self.redundancy_tolerance_m,
                max_spheres_per_component=self.max_spheres_per_component,
                max_iterations_per_component=self.max_iterations_per_component,
                max_total_spheres=self.max_total_spheres,
                enable_single_sphere_replacement=self.enable_single_sphere_replacement,
                enable_greedy_set_cover=self.enable_greedy_set_cover,
                enable_general_coverage_pruning=self.enable_general_coverage_pruning,
                enable_surface_shell_guard=self.enable_surface_shell_guard,
                surface_shell_thickness_m=self.surface_shell_thickness_m,
                target_shell_coverage=self.target_shell_coverage,
                shell_coverage_loss_tolerance=self.shell_coverage_loss_tolerance,
                max_optimization_matrix_elements=self.max_optimization_matrix_elements,
                enable_agglomerative_merge=self.enable_agglomerative_merge,
                merge_max_radius_m=self.merge_max_radius_m,
                merge_max_radius_growth_ratio=self.merge_max_radius_growth_ratio,
                merge_max_gap_m=self.merge_max_gap_m,
                merge_enable_esdf_guard=self.merge_enable_esdf_guard,
                merge_max_free_space_distance_m=self.merge_max_free_space_distance_m,
                merge_surface_sample_count=self.merge_surface_sample_count,
                merge_min_observed_surface_fraction=self.merge_min_observed_surface_fraction,
                robot_sphere_centers=robot_centers,
                robot_sphere_radii=robot_radii,
                robot_component_overlap_threshold=(
                    self.robot_component_overlap_threshold),
                robot_sphere_margin_m=self.robot_sphere_filter_margin_m,
            )
        except Exception as exc:
            self._warn_throttled(f"ESDF medial sphere processing failed: {exc}")
            self.publish_empty(stamp)
            return

        self.publish_markers(stamp, result.spheres)
        self.publish_sphere_cloud(stamp, result)
        if self.publish_debug_clouds:
            self.publish_inside_cloud(stamp, grid, result)
            self.publish_uncovered_cloud(stamp, grid, result)
            self.publish_robot_rejected_cloud(stamp, grid, result)
        warnings = []
        inside_count = int(np.count_nonzero(result.inside_mask))
        raw_inside_count = inside_count + result.robot_rejected_voxel_count
        if negative_count == 0:
            warnings.append("no negative ESDF voxels; no fallback spheres generated")
        elif raw_inside_count == 0:
            warnings.append(
                "no inside voxels after inside_epsilon_m; no fallback spheres generated"
            )
        if raw_inside_count and result.input_component_count == 0:
            warnings.append("all inside components were below min_component_voxels")
        for component in result.components:
            if component.coverage + 1e-12 < self.target_coverage:
                warnings.append(
                    f"component {component.component_id} coverage="
                    f"{component.coverage:.3f}, stop={component.termination_reason}"
                )
        if result.coverage_lost_component_ids:
            warnings.append(
                "max_total_spheres caused coverage loss in components "
                f"{result.coverage_lost_component_ids}"
            )
        if warnings:
            self._warn_throttled("; ".join(warnings))

        processing_elapsed_s = time.monotonic() - processing_started_at
        coverage_text = ",".join(
            f"{component.component_id}:{component.coverage:.3f}"
            for component in result.components
        )
        marker_delta_text = (
            "n/a" if self.latest_robot_marker_delta_s is None
            else f"{self.latest_robot_marker_delta_s:.3f}s"
        )
        max_robot_component_overlap = max(
            result.robot_component_overlap_fractions, default=0.0)
        self._info_throttled(
            "Dense ESDF medial spheres: "
            f"shape={grid.shape}, voxel={grid.voxel_size_m:.4f} m, "
            f"observed={observed_count}, inside_pre_robot={raw_inside_count}, "
            f"inside={inside_count}, components={result.input_component_count}, "
            f"robot_rejected_components="
            f"{result.robot_rejected_component_count}, "
            f"robot_rejected_voxels={result.robot_rejected_voxel_count}, "
            f"robot_spheres={robot_sphere_count}, "
            f"marker_delta={marker_delta_text}, "
            f"max_robot_component_overlap="
            f"{max_robot_component_overlap:.3f}, "
            f"removed_small={result.removed_small_components}, "
            f"pre_merge_spheres={sum(c.pre_merge_sphere_count for c in result.components)}, "
            f"post_merge_spheres={len(result.spheres)}, coverage=[{coverage_text}], "
            f"processing={processing_elapsed_s * 1000.0:.1f} ms, "
            f"service={service_elapsed_s * 1000.0:.1f} ms"
        )

    def publish_empty(self, stamp):
        self.publish_markers(stamp, [])
        self.sphere_cloud_pub.publish(
            point_cloud2.create_cloud(
                self._header(stamp), self._sphere_fields(), []
            )
        )
        if self.publish_debug_clouds:
            self.inside_cloud_pub.publish(
                point_cloud2.create_cloud(
                    self._header(stamp), self._inside_fields(), []
                )
            )
            self.uncovered_cloud_pub.publish(
                point_cloud2.create_cloud(
                    self._header(stamp), self._uncovered_fields(), []
                )
            )
            self.robot_rejected_voxel_pub.publish(
                point_cloud2.create_cloud(
                    self._header(stamp), self._uncovered_fields(), []
                )
            )

    def publish_markers(self, stamp, spheres):
        message = MarkerArray()
        current_keys = set()
        component_local_ids = {}
        for sphere in spheres:
            local_id = component_local_ids.get(sphere.component_id, 0)
            component_local_ids[sphere.component_id] = local_id + 1
            namespace = f"esdf_medial_component_{sphere.component_id}"
            current_keys.add((namespace, local_id))
            marker = Marker()
            marker.header = self._header(stamp)
            marker.ns = namespace
            marker.id = local_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(sphere.center[0])
            marker.pose.position.y = float(sphere.center[1])
            marker.pose.position.z = float(sphere.center[2])
            marker.pose.orientation.w = 1.0
            diameter = 2.0 * sphere.output_radius
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = diameter
            red, green, blue = self._component_color(sphere.component_id)
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = 0.42
            message.markers.append(marker)
        for namespace, marker_id in sorted(self.previous_marker_keys - current_keys):
            marker = Marker()
            marker.header = self._header(stamp)
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            message.markers.append(marker)
        self.previous_marker_keys = current_keys
        self.marker_pub.publish(message)

    def publish_sphere_cloud(self, stamp, result):
        coverage_by_id = {
            component.component_id: component.coverage
            for component in result.components
        }
        rows = [
            (
                float(sphere.center[0]),
                float(sphere.center[1]),
                float(sphere.center[2]),
                float(sphere.raw_radius),
                float(sphere.output_radius),
                int(sphere.component_id),
                float(coverage_by_id[sphere.component_id]),
                float(-sphere.raw_radius),
            )
            for sphere in result.spheres
        ]
        self.sphere_cloud_pub.publish(
            point_cloud2.create_cloud(
                self._header(stamp), self._sphere_fields(), rows
            )
        )

    def publish_inside_cloud(self, stamp, grid, result):
        covered_by_index = {}
        for component in result.components:
            _, covered = calculate_coverage_for_result(
                component,
                grid.origin_m,
                grid.voxel_size_m,
                self.coverage_tolerance_m,
            )
            for index, is_covered in zip(component.voxel_indices, covered):
                covered_by_index[tuple(int(value) for value in index)] = bool(is_covered)
        rows = []
        for index_array in np.argwhere(result.inside_mask):
            index = tuple(int(value) for value in index_array)
            point = grid.origin_m + (index_array.astype(np.float64) + 0.5) * (
                grid.voxel_size_m
            )
            rows.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(grid.values[index]),
                    int(result.component_labels[index]),
                    int(covered_by_index.get(index, False)),
                )
            )
        self.inside_cloud_pub.publish(
            point_cloud2.create_cloud(
                self._header(stamp), self._inside_fields(), rows
            )
        )

    def publish_uncovered_cloud(self, stamp, grid, result):
        rows = []
        for component in result.components:
            for index_array in component.uncovered_indices:
                index = tuple(int(value) for value in index_array)
                point = grid.origin_m + (index_array.astype(np.float64) + 0.5) * (
                    grid.voxel_size_m
                )
                rows.append(
                    (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        float(grid.values[index]),
                        int(component.component_id),
                    )
                )
        self.uncovered_cloud_pub.publish(
            point_cloud2.create_cloud(
                self._header(stamp), self._uncovered_fields(), rows
            )
        )

    def publish_robot_rejected_cloud(self, stamp, grid, result):
        rows = []
        for index_array in result.robot_rejected_voxel_indices:
            index = tuple(int(value) for value in index_array)
            point = grid.origin_m + (index_array.astype(np.float64) + 0.5) * (
                grid.voxel_size_m
            )
            rows.append(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(grid.values[index]),
                    int(result.component_labels[index]),
                )
            )
        self.robot_rejected_voxel_pub.publish(
            point_cloud2.create_cloud(
                self._header(stamp), self._uncovered_fields(), rows
            )
        )

    def _header(self, stamp):
        return Header(stamp=stamp, frame_id=self.target_frame)

    @staticmethod
    def _sphere_fields():
        return [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="raw_radius", offset=12, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="output_radius",
                offset=16,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="component_id", offset=20, datatype=PointField.INT32, count=1
            ),
            PointField(
                name="component_coverage",
                offset=24,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="esdf_distance",
                offset=28,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]

    @staticmethod
    def _inside_fields():
        return [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="esdf_distance",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="component_id", offset=16, datatype=PointField.INT32, count=1
            ),
            PointField(
                name="covered", offset=20, datatype=PointField.UINT8, count=1
            ),
        ]

    @staticmethod
    def _uncovered_fields():
        return [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="esdf_distance",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="component_id", offset=16, datatype=PointField.INT32, count=1
            ),
        ]

    @staticmethod
    def _component_color(component_id):
        return colorsys.hsv_to_rgb((component_id * 0.61803398875) % 1.0, 0.72, 0.95)

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

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")


def calculate_coverage_for_result(
    component, origin_m, voxel_size_m, coverage_tolerance_m
):
    """Small adapter kept outside the node to make debug publishing concise."""

    from rmp_camera.esdf_medial_sphere_core import calculate_component_coverage

    return calculate_component_coverage(
        component.voxel_indices,
        component.spheres,
        origin_m,
        voxel_size_m,
        coverage_tolerance_m,
    )


def main(args=None):
    rclpy.init(args=args)
    node = EsdfMedialSphereNode()
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
