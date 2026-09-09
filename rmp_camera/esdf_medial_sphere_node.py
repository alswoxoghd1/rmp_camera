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
* ``min_k_beam_width``, ``min_k_max_states``, and
  ``min_k_processing_budget_ms`` bound the search over valid merge orders;
  larger values may find fewer spheres but increase static update latency.
* ``max_grid_voxels`` is a hard memory guard; too small rejects valid AABBs and
  too large permits allocations unsuitable for Python.
"""

import colorsys
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, KeyValue
from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
from rclpy.duration import Duration
from rclpy.clock import JumpThreshold
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.esdf_medial_sphere_core import (
    generate_medial_spheres, has_static_obstacle_component, previous_support_is_observed,
)
from rmp_camera.human_static_filter import HumanStaticFilter
from rmp_camera.human_static_filter_core import esdf_proves_voxels_free
from rmp_camera.static_result_status import make_result_status
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
        self.result_status_pub = self.create_publisher(
            DiagnosticArray, self.get_parameter("result_status_topic").value, 5)
        self.human_filter = HumanStaticFilter(self)
        self.support_pub = self.create_publisher(
            PointCloud2, self.human_filter.support_topic, 2)
        self.free_support_pub = self.create_publisher(
            PointCloud2, self.human_filter.free_support_topic, 2)
        self.support_history = []
        self.last_support_evidence_stamp = None
        self.last_full_solve_stamp = None
        self.last_output_nonempty = False
        self.result_clock_jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1),
                          on_clock_change=True),
            post_callback=lambda _jump: self._reset_human_static_cache())
        self.pending = False
        self.previous_marker_keys = set()
        # Clear markers cached by RViz from a previous node instance on the
        # first result, then rely on finite lifetimes as a missed-DELETE guard.
        self.clear_markers_on_next_publish = True
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
        self.query_bounds_marker_pub = self.create_publisher(
            MarkerArray, self.query_bounds_marker_topic, 2
        )
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
        request_rate = max(self.update_rate_hz if self.update_rate_hz > 0 else 1.0,
                           self.empty_check_rate_hz)
        period = max(0.02, 1.0 / request_rate)
        self.timer = self.create_timer(period, self.tick)
        self.get_logger().info(
            "Dense ESDF medial sphere node started: "
            f"service={self.service_name}, frame={self.target_frame}, "
            f"rate={self.update_rate_hz:.2f} Hz, "
            f"empty_check_rate={self.empty_check_rate_hz:.2f} Hz, "
            f"query_aabb_min={self.aabb_min_m.tolist()} m, "
            f"query_aabb_size={self.aabb_size_m.tolist()} m, "
            f"target_coverage={self.target_coverage:.3f}, "
            f"raw_radius=[{self.min_raw_sphere_radius_m:.3f}, "
            f"{self.max_raw_sphere_radius_m:.3f}] m, "
            f"single={self.enable_single_sphere_replacement}, "
            f"greedy={self.enable_greedy_set_cover}, "
            f"pruning={self.enable_general_coverage_pruning}, "
            f"coarse_cover={self.enable_component_coarse_cover}, "
            f"coarse_min_voxels={self.component_coarse_min_voxels}, "
            f"coarse_scale={self.component_coarse_radius_scale:.3f}, "
            f"merge={self.enable_agglomerative_merge}, "
            f"merge_radius<={self.merge_max_radius_m:.3f} m, "
            f"merge_growth<={self.merge_max_radius_growth_ratio:.3f}, "
            f"merge_gap<={self.merge_max_gap_m:.3f} m, "
            f"merge_esdf_guard={self.merge_enable_esdf_guard}, "
            f"merge_samples={self.merge_surface_sample_count}, "
            f"min_k={self.enable_min_k_search}, "
            f"min_k_beam={self.min_k_beam_width}, "
            f"min_k_states={self.min_k_max_states}, "
            f"min_k_budget={self.min_k_processing_budget_ms:.1f} ms, "
            f"post_overlap_pruning={self.enable_post_merge_overlap_pruning}, "
            f"post_overlap_limit="
            f"{self.post_merge_max_overlap_fraction:.3f}, "
            f"robot_component_filter={self.robot_component_filter_enabled}, "
            f"robot_overlap_threshold="
            f"{self.robot_component_overlap_threshold:.3f}"
        )

    def _declare_parameters(self):
        self.declare_parameter("result_status_topic", "/rmp_camera/static_sphere_result_status")
        self.declare_parameter("service_name", "/nvblox_node/get_esdf_and_gradient")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("update_rate_hz", 1.0)
        self.declare_parameter("empty_check_rate_hz", 0.0)
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
        self.declare_parameter("max_raw_sphere_radius_m", 1.0)
        self.declare_parameter("safety_margin_m", 0.02)
        self.declare_parameter("marker_lifetime_s", 3.0)
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
        self.declare_parameter("enable_component_coarse_cover", False)
        self.declare_parameter("enable_local_width_cover", False)
        self.declare_parameter("local_width_ratio", 1.4)
        self.declare_parameter("local_width_budget_ms", 15.0)
        self.declare_parameter("component_coarse_min_voxels", 80)
        self.declare_parameter("component_coarse_radius_scale", 0.45)
        self.declare_parameter("component_coarse_max_radius_m", 0.32)
        self.declare_parameter("component_coarse_max_empty_fraction", 0.75)
        self.declare_parameter(
            "component_coarse_max_free_space_distance_m", 0.15)
        self.declare_parameter("enable_agglomerative_merge", True)
        self.declare_parameter("merge_max_radius_m", 0.35)
        self.declare_parameter("merge_max_radius_growth_ratio", 1.45)
        self.declare_parameter("merge_max_gap_m", 0.05)
        self.declare_parameter("merge_enable_esdf_guard", True)
        self.declare_parameter("merge_max_free_space_distance_m", 0.08)
        self.declare_parameter("merge_surface_sample_count", 64)
        self.declare_parameter("merge_min_observed_surface_fraction", 0.70)
        self.declare_parameter("enable_min_k_search", False)
        self.declare_parameter("min_k_beam_width", 8)
        self.declare_parameter("min_k_max_states", 128)
        self.declare_parameter("min_k_processing_budget_ms", 200.0)
        self.declare_parameter("enable_post_merge_overlap_pruning", False)
        self.declare_parameter("post_merge_max_overlap_fraction", 0.20)
        self.declare_parameter("post_merge_pruning_max_spheres", 64)
        self.declare_parameter("post_merge_pruning_max_removals", 32)
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
            "query_bounds_marker_topic",
            "/rmp_camera/esdf_medial_query_bounds",
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
        empty_rate = float(self.get_parameter("empty_check_rate_hz").value)
        if not np.isfinite(empty_rate) or empty_rate < 0.0:
            raise ValueError("empty_check_rate_hz must be finite and non-negative")
        for name in (
            "service_name",
            "target_frame",
            "marker_topic",
            "query_bounds_marker_topic",
            "sphere_cloud_topic",
            "inside_voxel_cloud_topic",
            "uncovered_voxel_cloud_topic",
            "robot_sphere_marker_topic",
            "robot_rejected_voxel_cloud_topic",
        ):
            setattr(self, name, str(self.get_parameter(name).value))
        for name in (
            "update_rate_hz",
            "empty_check_rate_hz",
            "unobserved_distance_value",
            "inside_epsilon_m",
            "target_coverage",
            "coverage_tolerance_m",
            "plateau_epsilon_m",
            "minimum_center_spacing_m",
            "min_raw_sphere_radius_m",
            "max_raw_sphere_radius_m",
            "safety_margin_m",
            "marker_lifetime_s",
            "redundancy_tolerance_m",
            "surface_shell_thickness_m",
            "target_shell_coverage",
            "shell_coverage_loss_tolerance",
            "component_coarse_radius_scale",
            "local_width_ratio", "local_width_budget_ms",
            "component_coarse_max_radius_m",
            "component_coarse_max_empty_fraction",
            "component_coarse_max_free_space_distance_m",
            "merge_max_radius_m",
            "merge_max_radius_growth_ratio",
            "merge_max_gap_m",
            "merge_max_free_space_distance_m",
            "merge_min_observed_surface_fraction",
            "min_k_processing_budget_ms",
            "post_merge_max_overlap_fraction",
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
            "component_coarse_min_voxels",
            "merge_surface_sample_count",
            "min_k_beam_width",
            "min_k_max_states",
            "post_merge_pruning_max_spheres",
            "post_merge_pruning_max_removals",
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
        self.enable_component_coarse_cover = self._as_bool(
            self.get_parameter("enable_component_coarse_cover").value
        )
        self.enable_local_width_cover = self._as_bool(
            self.get_parameter("enable_local_width_cover").value)
        self.enable_agglomerative_merge = self._as_bool(
            self.get_parameter("enable_agglomerative_merge").value
        )
        self.merge_enable_esdf_guard = self._as_bool(
            self.get_parameter("merge_enable_esdf_guard").value
        )
        self.enable_min_k_search = self._as_bool(
            self.get_parameter("enable_min_k_search").value
        )
        self.enable_post_merge_overlap_pruning = self._as_bool(
            self.get_parameter("enable_post_merge_overlap_pruning").value
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
        if (
            self.min_raw_sphere_radius_m < 0.0
            or self.max_raw_sphere_radius_m < self.min_raw_sphere_radius_m
        ):
            raise ValueError("invalid static raw-radius limits")
        if self.surface_shell_thickness_m < 0.0:
            raise ValueError("surface_shell_thickness_m must be non-negative")
        if self.marker_lifetime_s < 0.0:
            raise ValueError("marker_lifetime_s must be non-negative")
        if not 0.0 <= self.target_shell_coverage <= 1.0:
            raise ValueError("target_shell_coverage must be in [0, 1]")
        if not 0.0 <= self.shell_coverage_loss_tolerance <= 1.0:
            raise ValueError("shell_coverage_loss_tolerance must be in [0, 1]")
        if self.max_optimization_matrix_elements <= 0:
            raise ValueError(
                "max_optimization_matrix_elements must be positive"
            )
        if self.component_coarse_min_voxels <= 0:
            raise ValueError(
                "component_coarse_min_voxels must be positive")
        if (
            not np.isfinite(self.component_coarse_radius_scale)
            or self.component_coarse_radius_scale < 0.0
        ):
            raise ValueError(
                "component_coarse_radius_scale must be finite and non-negative")
        if (
            not np.isfinite(self.component_coarse_max_radius_m)
            or self.component_coarse_max_radius_m <= 0.0
        ):
            raise ValueError(
                "component_coarse_max_radius_m must be finite and positive")
        if not 0.0 <= self.component_coarse_max_empty_fraction <= 1.0:
            raise ValueError(
                "component_coarse_max_empty_fraction must be in [0, 1]")
        if (
            not np.isfinite(self.component_coarse_max_free_space_distance_m)
            or self.component_coarse_max_free_space_distance_m < 0.0
        ):
            raise ValueError(
                "component coarse free-space distance must be non-negative")
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
        if self.min_k_beam_width <= 0:
            raise ValueError("min_k_beam_width must be positive")
        if self.min_k_max_states <= 0:
            raise ValueError("min_k_max_states must be positive")
        if (
            not np.isfinite(self.min_k_processing_budget_ms)
            or self.min_k_processing_budget_ms <= 0.0
        ):
            raise ValueError(
                "min_k_processing_budget_ms must be finite and positive")
        if not 0.0 <= self.post_merge_max_overlap_fraction <= 1.0:
            raise ValueError(
                "post_merge_max_overlap_fraction must be in [0, 1]")
        if self.post_merge_pruning_max_spheres <= 0:
            raise ValueError(
                "post_merge_pruning_max_spheres must be positive")
        if self.post_merge_pruning_max_removals < 0:
            raise ValueError(
                "post_merge_pruning_max_removals must be non-negative")
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

    def _full_solve_due(self, now_sec):
        solve_period = 1.0 / self.update_rate_hz if self.update_rate_hz > 0 else 1.0
        return (self.empty_check_rate_hz <= 1.0 / solve_period
                or self.last_full_solve_stamp is None
                or now_sec < self.last_full_solve_stamp
                or now_sec - self.last_full_solve_stamp >= solve_period - 1e-6)

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        if self.pending:
            return
        now_sec = stamp.sec + stamp.nanosec * 1e-9
        # Empty output has no geometry to retire. Resume the normal solve
        # cadence instead of polling the ESDF at 5 Hz in an empty scene.
        if not self.last_output_nonempty and not self._full_solve_due(now_sec):
            return
        self.publish_query_bounds_markers(stamp)
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

    def publish_query_bounds_markers(self, stamp):
        """Publish the exact dense-ESDF request AABB as fill and wireframe."""

        minimum = self.aabb_min_m
        maximum = self.aabb_min_m + self.aabb_size_m
        center = minimum + 0.5 * self.aabb_size_m
        message = MarkerArray()

        fill = Marker()
        fill.header = self._header(stamp)
        fill.ns = "esdf_medial_query_bounds"
        fill.id = 0
        fill.type = Marker.CUBE
        fill.action = Marker.ADD
        fill.pose.position = Point(
            x=float(center[0]), y=float(center[1]), z=float(center[2]))
        fill.pose.orientation.w = 1.0
        fill.scale = Vector3(
            x=float(self.aabb_size_m[0]),
            y=float(self.aabb_size_m[1]),
            z=float(self.aabb_size_m[2]),
        )
        fill.color.r = 0.05
        fill.color.g = 0.85
        fill.color.b = 1.0
        fill.color.a = 0.035
        message.markers.append(fill)

        corners = [
            np.array((x, y, z), dtype=np.float64)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
        edges = (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        )
        wireframe = Marker()
        wireframe.header = self._header(stamp)
        wireframe.ns = "esdf_medial_query_bounds"
        wireframe.id = 1
        wireframe.type = Marker.LINE_LIST
        wireframe.action = Marker.ADD
        wireframe.pose.orientation.w = 1.0
        wireframe.scale.x = 0.025
        wireframe.color.r = 0.05
        wireframe.color.g = 0.85
        wireframe.color.b = 1.0
        wireframe.color.a = 0.90
        for start, end in edges:
            for corner_index in (start, end):
                corner = corners[corner_index]
                wireframe.points.append(Point(
                    x=float(corner[0]),
                    y=float(corner[1]),
                    z=float(corner[2]),
                ))
        message.markers.append(wireframe)
        self.query_bounds_marker_pub.publish(message)

    def handle_response(self, future, service_started_at):
        self.pending = False
        service_elapsed_s = time.monotonic() - service_started_at
        stamp = self.get_clock().now().to_msg()
        try:
            response = future.result()
        except Exception as exc:
            self._warn_throttled(f"ESDF service call failed: {exc}")
            self.publish_invalid(stamp, "esdf_service_exception")
            return
        if not response.success:
            self._warn_throttled("ESDF service returned success=false")
            self.publish_invalid(stamp, "esdf_service_failed")
            return

        grid = DenseEsdfGrid(response, self.max_grid_voxels)
        if not grid.valid:
            self._warn_throttled(f"Invalid/empty ESDF grid: {grid.error}")
            self.publish_invalid(stamp, "invalid_esdf_grid")
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
                    self.publish_invalid(stamp, "robot_marker_sync_unavailable")
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
        # Check the original observed mask, before human exclusion changes
        # local distances. An empty result from missing coverage is UNKNOWN.
        previous_support = (self.support_history[-1][1][:, :3] if self.support_history
                            else np.empty((0, 3)))
        empty_observation_valid = previous_support_is_observed(
            previous_support, observed_mask, grid.origin_m, grid.voxel_size_m)
        empty_observation_reason = ("unobserved_esdf" if observed_count == 0
                                    else "unobserved_previous_support")
        if self.human_filter.enabled:
            # A responsive ESDF service does not prove the camera is live.
            depth_age = (None if self.human_filter.depth_stamp is None else
                stamp.sec + stamp.nanosec * 1e-9 - self.human_filter.depth_stamp)
            fresh_depth = (depth_age is not None
                and -0.05 <= depth_age <= self.human_filter.params.depth_max_age_s
                and time.monotonic() - self.human_filter.depth_received
                <= self.human_filter.params.depth_max_age_s)
            if empty_observation_valid and not fresh_depth:
                empty_observation_reason = "stale_or_missing_depth"
            empty_observation_valid = empty_observation_valid and fresh_depth
        negative_count = int(np.count_nonzero(observed_mask & (grid.values < 0.0)))
        human_excluded_count = 0
        if self.human_filter.enabled:
            # Use the untouched map, never the human-excluded local copy,
            # as independent evidence for clearing older fusion geometry.
            self.publish_esdf_free_support(stamp, grid)
            indices = np.argwhere(observed_mask & (grid.values < -self.inside_epsilon_m))
            points = grid.origin_m + (indices + 0.5) * grid.voxel_size_m
            excluded = self.human_filter.excluded(
                points, stamp.sec + stamp.nanosec * 1e-9)
            human_excluded_count = int(excluded.sum())
            # Unknown/excluded, not invented free distance. Keep all other
            # signed distances intact for sphere radii and ESDF merge guards.
            grid.values[tuple(indices[excluded].T)] = self.unobserved_distance_value
        now_sec = stamp.sec + stamp.nanosec * 1e-9
        if not self._full_solve_due(now_sec):
            # Do not run the expensive sphere optimizer on these extra ticks.
            # This check can only clear; it cannot create substitute geometry.
            if not empty_observation_valid:
                self.publish_invalid(stamp, "fast_check_" + empty_observation_reason)
                return
            try:
                occupied = has_static_obstacle_component(grid.values, grid.origin_m,
                    grid.voxel_size_m, unobserved_distance_value=self.unobserved_distance_value,
                    inside_epsilon_m=self.inside_epsilon_m,
                    min_component_voxels=self.min_component_voxels,
                    robot_sphere_centers=robot_centers, robot_sphere_radii=robot_radii,
                    robot_component_overlap_threshold=self.robot_component_overlap_threshold,
                    robot_sphere_margin_m=self.robot_sphere_filter_margin_m)
            except Exception as exc:
                self._warn_throttled(f"static empty check failed: {exc}")
                self.publish_invalid(stamp, "fast_empty_check_failed")
                return
            if not occupied:
                if self.last_output_nonempty:
                    self.get_logger().info(
                        "Static fast empty: no admitted obstacle components, "
                        f"processing={(time.monotonic() - processing_started_at) * 1000:.1f} ms")
                self.publish_empty(stamp, "fast_component_empty")
            return
        self.last_full_solve_stamp = now_sec
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
                max_raw_sphere_radius_m=self.max_raw_sphere_radius_m,
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
                enable_local_width_cover=getattr(self, "enable_local_width_cover", False),
                local_width_ratio=getattr(self, "local_width_ratio", 1.4),
                local_width_budget_ms=getattr(self, "local_width_budget_ms", 15.0),
                enable_component_coarse_cover=(
                    self.enable_component_coarse_cover),
                component_coarse_min_voxels=(
                    self.component_coarse_min_voxels),
                component_coarse_radius_scale=(
                    self.component_coarse_radius_scale),
                component_coarse_max_radius_m=(
                    self.component_coarse_max_radius_m),
                component_coarse_max_empty_fraction=(
                    self.component_coarse_max_empty_fraction),
                component_coarse_max_free_space_distance_m=(
                    self.component_coarse_max_free_space_distance_m),
                enable_agglomerative_merge=self.enable_agglomerative_merge,
                merge_max_radius_m=self.merge_max_radius_m,
                merge_max_radius_growth_ratio=self.merge_max_radius_growth_ratio,
                merge_max_gap_m=self.merge_max_gap_m,
                merge_enable_esdf_guard=self.merge_enable_esdf_guard,
                merge_max_free_space_distance_m=self.merge_max_free_space_distance_m,
                merge_surface_sample_count=self.merge_surface_sample_count,
                merge_min_observed_surface_fraction=self.merge_min_observed_surface_fraction,
                enable_min_k_search=self.enable_min_k_search,
                min_k_beam_width=self.min_k_beam_width,
                min_k_max_states=self.min_k_max_states,
                min_k_processing_budget_ms=self.min_k_processing_budget_ms,
                enable_post_merge_overlap_pruning=(
                    self.enable_post_merge_overlap_pruning),
                post_merge_max_overlap_fraction=(
                    self.post_merge_max_overlap_fraction),
                post_merge_pruning_max_spheres=(
                    self.post_merge_pruning_max_spheres),
                post_merge_pruning_max_removals=(
                    self.post_merge_pruning_max_removals),
                robot_sphere_centers=robot_centers,
                robot_sphere_radii=robot_radii,
                robot_component_overlap_threshold=(
                    self.robot_component_overlap_threshold),
                robot_sphere_margin_m=self.robot_sphere_filter_margin_m,
            )
        except Exception as exc:
            self._warn_throttled(f"ESDF medial sphere processing failed: {exc}")
            self.publish_invalid(stamp, "sphere_processing_failed")
            return

        if not result.spheres and not empty_observation_valid:
            self.publish_invalid(stamp, "empty_result_" + empty_observation_reason)
            return
        status = make_result_status(stamp, self.target_frame,
            "valid" if result.spheres else "empty", "successful_observed_solve")
        status.status[0].values.extend([
            KeyValue(key="local_width_saved", value=str(sum(c.local_width_saved_spheres for c in result.components))),
            KeyValue(key="local_width_ms", value=str(sum(c.local_width_elapsed_ms for c in result.components))),
            KeyValue(key="local_width_reasons", value=','.join(c.local_width_reason for c in result.components)),
        ])
        self.result_status_pub.publish(status)
        # Also retained when the semantic branch is disabled: old occupied
        # locations must remain observed before accepting a later empty solve.
        self.publish_support(stamp, grid, result)
        self.publish_markers(stamp, result.spheres)
        self.publish_sphere_cloud(stamp, result)
        self.last_output_nonempty = bool(result.spheres)
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
        agglomerative_sphere_count = sum(
            component.agglomerative_sphere_count
            for component in result.components)
        min_k_sphere_count = sum(
            component.min_k_sphere_count for component in result.components)
        post_overlap_sphere_count = sum(
            component.post_overlap_sphere_count
            for component in result.components)
        post_overlap_removed_count = sum(
            component.post_overlap_removed_count
            for component in result.components)
        min_k_states_explored = sum(
            component.min_k_states_explored for component in result.components)
        min_k_termination_text = ",".join(
            f"{component.component_id}:{component.min_k_search_termination}"
            for component in result.components
            if component.min_k_search_applied
        ) or "n/a"
        self._info_throttled(
            "Dense ESDF medial spheres: "
            f"shape={grid.shape}, voxel={grid.voxel_size_m:.4f} m, "
            f"observed={observed_count}, inside_pre_robot={raw_inside_count}, "
            f"inside={inside_count}, components={result.input_component_count}, "
            f"robot_rejected_components="
            f"{result.robot_rejected_component_count}, "
            f"robot_rejected_voxels={result.robot_rejected_voxel_count}, "
            f"human_excluded_voxels={human_excluded_count}, "
            f"robot_spheres={robot_sphere_count}, "
            f"marker_delta={marker_delta_text}, "
            f"max_robot_component_overlap="
            f"{max_robot_component_overlap:.3f}, "
            f"removed_small={result.removed_small_components}, "
            f"coarse_candidates="
            f"{sum(c.coarse_candidate_count for c in result.components)}, "
            f"coarse_selected="
            f"{sum(c.coarse_selected_count for c in result.components)}, "
            f"pre_merge_spheres={sum(c.pre_merge_sphere_count for c in result.components)}, "
            f"agglomerative_spheres={agglomerative_sphere_count}, "
            f"min_k_spheres={min_k_sphere_count}, "
            f"min_k_states={min_k_states_explored}, "
            f"min_k_stop=[{min_k_termination_text}], "
            f"post_overlap_spheres={post_overlap_sphere_count}, "
            f"post_overlap_removed={post_overlap_removed_count}, "
            f"post_merge_spheres={len(result.spheres)}, coverage=[{coverage_text}], "
            f"local_width_saved={sum(c.local_width_saved_spheres for c in result.components)}, "
            f"local_width_ms={sum(c.local_width_elapsed_ms for c in result.components):.2f}, "
            f"processing={processing_elapsed_s * 1000.0:.1f} ms, "
            f"service={service_elapsed_s * 1000.0:.1f} ms"
        )

    def publish_invalid(self, stamp, reason):
        self.result_status_pub.publish(make_result_status(
            stamp, self.target_frame, "unknown", reason))
        self._warn_throttled(f"static result UNKNOWN: {reason}; not an empty confirmation")

    def publish_empty(self, stamp, reason="explicit_valid_empty"):
        """Explicitly publish a valid empty result; never use for failures."""
        self.result_status_pub.publish(make_result_status(
            stamp, self.target_frame, "empty", reason))
        self.last_output_nonempty = False
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
        if self.clear_markers_on_next_publish:
            clear_marker = Marker()
            clear_marker.header = self._header(stamp)
            clear_marker.action = Marker.DELETEALL
            message.markers.append(clear_marker)
            self.clear_markers_on_next_publish = False
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
            if self.marker_lifetime_s > 0.0:
                marker.lifetime = Duration(seconds=self.marker_lifetime_s).to_msg()
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

    def publish_support(self, stamp, grid, result):
        """Publish the actual retained voxel support of each generated sphere.

        The exact generation stamp pairs this with sphere_cloud in Fusion;
        a missing support message must never cause geometry to be deleted.
        """
        components = {c.component_id: grid.origin_m + (c.voxel_indices + 0.5)
                      * grid.voxel_size_m for c in result.components}
        rows = []
        for index, sphere in enumerate(result.spheres):
            points = components[sphere.component_id]
            inside = np.linalg.norm(points - sphere.center, axis=1) <= (
                sphere.raw_radius + self.coverage_tolerance_m)
            rows.extend((float(x), float(y), float(z), index)
                        for x, y, z in points[inside])
        fields = [PointField(name=name, offset=i * 4, count=1,
                  datatype=PointField.INT32 if name == "sphere_index" else PointField.FLOAT32)
                  for i, name in enumerate(("x", "y", "z", "sphere_index"))]
        self.support_pub.publish(point_cloud2.create_cloud(self._header(stamp), fields, rows))
        if rows:
            self.support_history.append((stamp, np.asarray(rows, dtype=float)))
            self.support_history = self.support_history[-3:]

    def _reset_human_static_cache(self):
        if hasattr(self, "support_history"):
            self.support_history.clear()
            self.last_support_evidence_stamp = None
            self.last_full_solve_stamp = None
            self.last_output_nonempty = False

    def publish_esdf_free_support(self, stamp, grid):
        """Recheck cached support against a newly observed, unmodified ESDF."""
        now = stamp.sec + stamp.nanosec * 1e-9
        if self.last_support_evidence_stamp is not None and now < self.last_support_evidence_stamp:
            self.support_history.clear()
        self.last_support_evidence_stamp = now
        fields = [PointField(name=name, offset=i * 4, count=1,
                  datatype=PointField.INT32 if i >= 3 else PointField.FLOAT32)
                  for i, name in enumerate(("x", "y", "z", "sphere_index", "esdf_free"))]
        fields.append(PointField(name="evidence_stamp", offset=20,
                                 count=1, datatype=PointField.FLOAT64))
        for generation_stamp, support in self.support_history:
            free = esdf_proves_voxels_free(support[:, :3], grid.values,
                grid.origin_m, grid.voxel_size_m, self.unobserved_distance_value,
                self.human_filter.params.free_clearance_m)
            rows = [(float(x), float(y), float(z), int(index), int(clear), now)
                    for (x, y, z, index), clear in zip(support, free)]
            self.free_support_pub.publish(point_cloud2.create_cloud(
                self._header(generation_stamp), fields, rows))

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
