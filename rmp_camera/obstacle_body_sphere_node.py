import math
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.vision_geometry import voxel_downsample


NEIGHBOR_OFFSETS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


class ObstacleBodySphereNode(Node):
    def __init__(self):
        super().__init__("obstacle_body_sphere_node")

        self.declare_parameter("input_cloud_topic", "/rmp_camera/robot_free_pointcloud")
        self.declare_parameter(
            "robot_sphere_marker_topic",
            "/rmp_camera/robot_collision_sphere_markers",
        )
        self.declare_parameter("obstacle_marker_topic", "/rmp_camera/obstacle_body_spheres")
        self.declare_parameter("obstacle_cloud_topic", "/rmp_camera/obstacle_body_sphere_cloud")
        self.declare_parameter("candidate_cloud_topic", "/rmp_camera/obstacle_body_candidate_cloud")
        self.declare_parameter("cluster_cloud_topic", "/rmp_camera/obstacle_body_cluster_cloud")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("publish_candidate_cloud", True)
        self.declare_parameter("publish_cluster_cloud", True)
        self.declare_parameter("background_subtraction_enabled", True)
        self.declare_parameter("background_voxel_size_m", 0.05)
        self.declare_parameter("background_neighbor_voxels", 1)
        self.declare_parameter("auto_capture_background_on_start", True)
        self.declare_parameter("auto_capture_background_delay_s", 5.0)
        self.declare_parameter("auto_capture_background_sample_count", 20)
        self.declare_parameter("robot_proximity_filter_enabled", False)
        self.declare_parameter("min_robot_clearance_m", 0.20)
        self.declare_parameter("max_robot_clearance_m", 1.0)
        self.declare_parameter("min_sphere_robot_clearance_m", 0.20)

        self.declare_parameter("min_x_m", -1.5)
        self.declare_parameter("max_x_m", 1.6)
        self.declare_parameter("min_y_m", -1.8)
        self.declare_parameter("max_y_m", 1.4)
        self.declare_parameter("min_z_m", 0.05)
        self.declare_parameter("max_z_m", 1.8)
        self.declare_parameter("max_range_from_base_m", 2.2)

        self.declare_parameter("voxel_size_m", 0.03)
        self.declare_parameter("cluster_cell_size_m", 0.10)
        self.declare_parameter("min_cluster_points", 10)
        self.declare_parameter("max_cluster_points", 5000)
        self.declare_parameter("max_cluster_extent_m", 2.2)
        self.declare_parameter("max_clusters", 8)
        self.declare_parameter("max_input_points", 30000)

        self.declare_parameter("sphere_spacing_m", 0.06)
        self.declare_parameter("min_sphere_radius_m", 0.035)
        self.declare_parameter("max_sphere_radius_m", 0.20)
        self.declare_parameter("sphere_padding_m", 0.015)
        self.declare_parameter("max_spheres_per_cluster", 12)
        self.declare_parameter("max_total_spheres", 36)
        self.declare_parameter("max_rate_hz", 5.0)
        self.declare_parameter("marker_lifetime_s", 0.5)
        self.declare_parameter("min_persistent_frames", 2)
        self.declare_parameter("persistence_match_distance_m", 0.14)
        self.declare_parameter("persistence_smoothing_alpha", 0.6)

        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.robot_sphere_marker_topic = self.get_parameter("robot_sphere_marker_topic").value
        self.obstacle_marker_topic = self.get_parameter("obstacle_marker_topic").value
        self.obstacle_cloud_topic = self.get_parameter("obstacle_cloud_topic").value
        self.candidate_cloud_topic = self.get_parameter("candidate_cloud_topic").value
        self.cluster_cloud_topic = self.get_parameter("cluster_cloud_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.publish_candidate_cloud_enabled = bool(self.get_parameter("publish_candidate_cloud").value)
        self.publish_cluster_cloud_enabled = bool(self.get_parameter("publish_cluster_cloud").value)
        self.background_subtraction_enabled = bool(
            self.get_parameter("background_subtraction_enabled").value
        )
        self.background_voxel_size_m = float(self.get_parameter("background_voxel_size_m").value)
        self.background_neighbor_voxels = int(self.get_parameter("background_neighbor_voxels").value)
        self.auto_capture_background_on_start = bool(
            self.get_parameter("auto_capture_background_on_start").value
        )
        self.auto_capture_background_delay_s = float(
            self.get_parameter("auto_capture_background_delay_s").value
        )
        self.auto_capture_background_sample_count = max(
            1,
            int(self.get_parameter("auto_capture_background_sample_count").value),
        )
        self.robot_proximity_filter_enabled = bool(
            self.get_parameter("robot_proximity_filter_enabled").value
        )
        self.min_robot_clearance_m = float(self.get_parameter("min_robot_clearance_m").value)
        self.max_robot_clearance_m = float(self.get_parameter("max_robot_clearance_m").value)
        self.min_sphere_robot_clearance_m = float(
            self.get_parameter("min_sphere_robot_clearance_m").value
        )

        self.min_x_m = float(self.get_parameter("min_x_m").value)
        self.max_x_m = float(self.get_parameter("max_x_m").value)
        self.min_y_m = float(self.get_parameter("min_y_m").value)
        self.max_y_m = float(self.get_parameter("max_y_m").value)
        self.min_z_m = float(self.get_parameter("min_z_m").value)
        self.max_z_m = float(self.get_parameter("max_z_m").value)
        self.max_range_from_base_m = float(self.get_parameter("max_range_from_base_m").value)

        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        self.cluster_cell_size_m = float(self.get_parameter("cluster_cell_size_m").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.max_cluster_points = int(self.get_parameter("max_cluster_points").value)
        self.max_cluster_extent_m = float(self.get_parameter("max_cluster_extent_m").value)
        self.max_clusters = int(self.get_parameter("max_clusters").value)
        self.max_input_points = int(self.get_parameter("max_input_points").value)

        self.sphere_spacing_m = float(self.get_parameter("sphere_spacing_m").value)
        self.min_sphere_radius_m = float(self.get_parameter("min_sphere_radius_m").value)
        self.max_sphere_radius_m = float(self.get_parameter("max_sphere_radius_m").value)
        self.sphere_padding_m = float(self.get_parameter("sphere_padding_m").value)
        self.max_spheres_per_cluster = int(self.get_parameter("max_spheres_per_cluster").value)
        self.max_total_spheres = int(self.get_parameter("max_total_spheres").value)
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self.marker_lifetime_s = float(self.get_parameter("marker_lifetime_s").value)
        self.min_persistent_frames = int(self.get_parameter("min_persistent_frames").value)
        self.persistence_match_distance_m = float(
            self.get_parameter("persistence_match_distance_m").value
        )
        self.persistence_smoothing_alpha = float(
            self.get_parameter("persistence_smoothing_alpha").value
        )

        self.last_process_time = 0.0
        self.last_log_time = 0.0
        self.previous_marker_count = 0
        self.latest_pre_background_points = None
        self.background_keys = set()
        self.background_point_count = 0
        self.background_offsets = self.make_neighbor_offsets(self.background_neighbor_voxels)
        self.node_start_time = time.monotonic()
        self.auto_background_done = False
        self.auto_background_samples = []
        self.sphere_tracks = []
        self.robot_spheres = []

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.input_cloud_topic,
            self.cloud_callback,
            cloud_qos,
        )
        self.robot_sphere_sub = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self.robot_sphere_callback,
            2,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.obstacle_marker_topic, 2)
        self.obstacle_cloud_pub = self.create_publisher(PointCloud2, self.obstacle_cloud_topic, 2)
        self.candidate_cloud_pub = self.create_publisher(PointCloud2, self.candidate_cloud_topic, 2)
        self.cluster_cloud_pub = self.create_publisher(PointCloud2, self.cluster_cloud_topic, 2)
        self.create_service(
            Trigger,
            "/rmp_camera/capture_obstacle_background",
            self.capture_background_callback,
        )
        self.create_service(
            Trigger,
            "/rmp_camera/clear_obstacle_background",
            self.clear_background_callback,
        )
        self.get_logger().info(
            "Obstacle body sphere node started: "
            f"{self.input_cloud_topic} -> {self.obstacle_marker_topic}, "
            f"workspace=x[{self.min_x_m:.1f},{self.max_x_m:.1f}] "
            f"y[{self.min_y_m:.1f},{self.max_y_m:.1f}] "
            f"z[{self.min_z_m:.2f},{self.max_z_m:.1f}], "
            f"background_subtraction={self.background_subtraction_enabled}, "
            f"auto_background={self.auto_capture_background_on_start}, "
            f"auto_delay={self.auto_capture_background_delay_s:.1f}s, "
            f"auto_samples={self.auto_capture_background_sample_count}, "
            f"robot_proximity_filter={self.robot_proximity_filter_enabled}, "
            f"robot_clearance_range=[{self.min_robot_clearance_m:.2f}, "
            f"{self.max_robot_clearance_m:.2f}]m, "
            f"sphere_min_robot_clearance={self.min_sphere_robot_clearance_m:.2f}m"
        )

    def robot_sphere_callback(self, msg):
        spheres = []
        for marker in msg.markers:
            if marker.action == Marker.DELETE or marker.type != Marker.SPHERE:
                continue
            if marker.header.frame_id and marker.header.frame_id != self.target_frame:
                continue
            radius = 0.5 * max(
                float(marker.scale.x),
                float(marker.scale.y),
                float(marker.scale.z),
            )
            if radius <= 0.0:
                continue
            spheres.append(
                (
                    np.array(
                        [
                            marker.pose.position.x,
                            marker.pose.position.y,
                            marker.pose.position.z,
                        ],
                        dtype=np.float64,
                    ),
                    radius,
                )
            )
        self.robot_spheres = spheres

    def cloud_callback(self, msg):
        now = time.monotonic()
        if self.max_rate_hz > 0.0 and now - self.last_process_time < 1.0 / self.max_rate_hz:
            return
        self.last_process_time = now

        if msg.header.frame_id != self.target_frame:
            self.log_throttled(
                f"Skipping cloud in frame {msg.header.frame_id}; expected {self.target_frame}.",
                warn=True,
            )
            return

        points = self.read_xyz_points(msg)
        raw_count = len(points)
        if raw_count == 0:
            self.publish_results(msg.header, [], np.empty((0, 3), dtype=np.float64))
            return

        points = points[np.isfinite(points).all(axis=1)]
        points = self.crop_workspace(points)
        points = voxel_downsample(points, self.voxel_size_m)
        if self.max_input_points > 0 and len(points) > self.max_input_points:
            step = int(np.ceil(len(points) / self.max_input_points))
            points = points[::step]

        self.latest_pre_background_points = points.copy()
        self.update_auto_background(points)
        background_removed = 0
        if self.background_subtraction_enabled and self.background_keys:
            before_background = len(points)
            points = self.subtract_background(points)
            background_removed = before_background - len(points)

        robot_proximity_removed = 0
        if self.robot_proximity_filter_enabled:
            before_robot_proximity = len(points)
            points = self.filter_near_robot(points)
            robot_proximity_removed = before_robot_proximity - len(points)

        clusters = self.cluster_points(points)
        spheres = self.fit_spheres(clusters)
        sphere_robot_removed = 0
        if self.robot_proximity_filter_enabled:
            before_sphere_robot = len(spheres)
            spheres = self.filter_spheres_near_robot(spheres)
            sphere_robot_removed = before_sphere_robot - len(spheres)
        spheres = self.filter_persistent_spheres(spheres)
        clustered_points = self.clustered_points(clusters)
        self.publish_results(msg.header, spheres, points, clustered_points)

        self.log_throttled(
            "Obstacle body spheres: "
            f"raw={raw_count}, candidate={len(points)}, "
            f"background_removed={background_removed}, "
            f"robot_proximity_removed={robot_proximity_removed}, "
            f"sphere_robot_removed={sphere_robot_removed}, "
            f"clusters={len(clusters)}, spheres={len(spheres)}"
        )

    def capture_background_callback(self, request, response):
        del request
        response.success, response.message = self.capture_background_from_latest()
        return response

    def capture_background_from_latest(self):
        points = self.latest_pre_background_points
        if points is None:
            return False, "No processed point cloud is available yet."
        if len(points) == 0:
            return False, "Current processed point cloud is empty; background was not captured."

        self.capture_background_from_point_sets([points])
        self.auto_background_done = True
        self.auto_background_samples = []
        return True, self.background_summary("Captured obstacle background")

    def capture_background_from_point_sets(self, point_sets):
        keys = set()
        point_count = 0
        for points in point_sets:
            if points is None or len(points) == 0:
                continue
            keys.update(self.points_to_key_set(points))
            point_count += int(len(points))
        self.background_keys = keys
        self.background_point_count = point_count

    def background_summary(self, prefix):
        return (
            f"{prefix}: "
            f"{self.background_point_count} points, {len(self.background_keys)} voxels, "
            f"voxel_size={self.background_voxel_size_m:.3f} m, "
            f"neighbor_voxels={self.background_neighbor_voxels}."
        )

    def update_auto_background(self, points):
        if not self.background_subtraction_enabled:
            return
        if not self.auto_capture_background_on_start:
            return
        if self.auto_background_done or self.background_keys:
            return
        if time.monotonic() - self.node_start_time < self.auto_capture_background_delay_s:
            return
        if len(points) == 0:
            return

        self.auto_background_samples.append(points.copy())
        sample_count = len(self.auto_background_samples)
        if sample_count < self.auto_capture_background_sample_count:
            self.log_throttled(
                "Collecting auto obstacle background samples: "
                f"{sample_count}/{self.auto_capture_background_sample_count}"
            )
            return

        self.capture_background_from_point_sets(self.auto_background_samples)
        self.auto_background_done = True
        self.auto_background_samples = []
        self.get_logger().info(
            self.background_summary(
                f"Auto Captured obstacle background from {sample_count} frames"
            )
        )

    def clear_background_callback(self, request, response):
        del request
        self.background_keys = set()
        self.background_point_count = 0
        self.auto_background_done = True
        self.auto_background_samples = []
        response.success = True
        response.message = "Cleared obstacle background."
        return response

    def subtract_background(self, points):
        if len(points) == 0 or not self.background_keys:
            return points
        if self.background_voxel_size_m <= 0.0:
            return points

        keys = np.floor(points / self.background_voxel_size_m).astype(np.int64)
        keep = np.ones(len(points), dtype=bool)
        for index, key_array in enumerate(keys):
            key = tuple(int(value) for value in key_array)
            if self.key_near_background(key):
                keep[index] = False
        return points[keep]

    def key_near_background(self, key):
        kx, ky, kz = key
        for dx, dy, dz in self.background_offsets:
            if (kx + dx, ky + dy, kz + dz) in self.background_keys:
                return True
        return False

    def points_to_key_set(self, points):
        if len(points) == 0 or self.background_voxel_size_m <= 0.0:
            return set()
        keys = np.floor(points / self.background_voxel_size_m).astype(np.int64)
        unique_keys = np.unique(keys, axis=0)
        return {tuple(int(value) for value in key) for key in unique_keys}

    @staticmethod
    def make_neighbor_offsets(radius):
        radius = max(0, int(radius))
        return [
            (dx, dy, dz)
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dz in range(-radius, radius + 1)
        ]

    def filter_persistent_spheres(self, spheres):
        if self.min_persistent_frames <= 1:
            return spheres
        if not spheres:
            self.sphere_tracks = []
            return []

        previous_tracks = self.sphere_tracks
        matched_previous = set()
        new_tracks = []
        persistent_spheres = []

        for sphere in spheres:
            center = np.asarray(sphere["center"], dtype=np.float64)
            best_index = None
            best_distance = self.persistence_match_distance_m
            for index, track in enumerate(previous_tracks):
                if index in matched_previous:
                    continue
                distance = float(np.linalg.norm(center - track["center"]))
                if distance <= best_distance:
                    best_index = index
                    best_distance = distance

            age = 1
            smoothed_center = center
            smoothed_radius = float(sphere["radius"])
            if best_index is not None:
                previous_track = previous_tracks[best_index]
                matched_previous.add(best_index)
                age = int(previous_track["age"]) + 1
                alpha = min(1.0, max(0.0, self.persistence_smoothing_alpha))
                smoothed_center = (
                    alpha * center
                    + (1.0 - alpha) * np.asarray(previous_track["center"], dtype=np.float64)
                )
                smoothed_radius = (
                    alpha * float(sphere["radius"])
                    + (1.0 - alpha) * float(previous_track["radius"])
                )

            tracked_sphere = dict(sphere)
            tracked_sphere["age"] = age
            tracked_sphere["center"] = smoothed_center
            tracked_sphere["radius"] = smoothed_radius
            new_tracks.append(
                {
                    "center": smoothed_center,
                    "radius": smoothed_radius,
                    "age": age,
                }
            )
            if age >= self.min_persistent_frames:
                persistent_spheres.append(tracked_sphere)

        self.sphere_tracks = new_tracks
        return persistent_spheres

    def crop_workspace(self, points):
        if len(points) == 0:
            return points
        mask = (
            (points[:, 0] >= self.min_x_m)
            & (points[:, 0] <= self.max_x_m)
            & (points[:, 1] >= self.min_y_m)
            & (points[:, 1] <= self.max_y_m)
            & (points[:, 2] >= self.min_z_m)
            & (points[:, 2] <= self.max_z_m)
        )
        if self.max_range_from_base_m > 0.0:
            range_xy = np.linalg.norm(points[:, :2], axis=1)
            mask &= range_xy <= self.max_range_from_base_m
        return points[mask]

    def filter_near_robot(self, points):
        if len(points) == 0:
            return points
        if not self.robot_spheres:
            self.log_throttled(
                "Robot proximity filter enabled but no robot collision spheres are available.",
                warn=True,
            )
            return np.empty((0, 3), dtype=np.float64)

        min_clearance = max(0.0, self.min_robot_clearance_m)
        keep = np.zeros(len(points), dtype=bool)
        max_clearance = max(0.0, self.max_robot_clearance_m)
        for center, radius in self.robot_spheres:
            clearance = np.linalg.norm(points - center, axis=1) - radius
            keep |= (clearance >= min_clearance) & (clearance <= max_clearance)
        return points[keep]

    def filter_spheres_near_robot(self, spheres):
        if not spheres:
            return spheres
        if not self.robot_spheres:
            return []

        min_clearance = max(0.0, self.min_sphere_robot_clearance_m)
        max_clearance = max(0.0, self.max_robot_clearance_m)
        kept = []
        for sphere in spheres:
            center = np.asarray(sphere["center"], dtype=np.float64)
            radius = float(sphere["radius"])
            clearances = [
                float(np.linalg.norm(center - robot_center) - robot_radius - radius)
                for robot_center, robot_radius in self.robot_spheres
            ]
            if not clearances:
                continue
            nearest_clearance = min(clearances)
            if nearest_clearance < min_clearance:
                continue
            if max_clearance > 0.0 and nearest_clearance > max_clearance:
                continue
            kept.append(sphere)
        return kept

    def cluster_points(self, points):
        if len(points) == 0:
            return []

        cell_size = max(self.cluster_cell_size_m, 1e-3)
        cell_keys = np.floor(points / cell_size).astype(np.int64)
        cell_to_indices = {}
        for index, key_array in enumerate(cell_keys):
            key = tuple(int(value) for value in key_array)
            cell_to_indices.setdefault(key, []).append(index)

        unvisited = set(cell_to_indices.keys())
        clusters = []
        while unvisited:
            seed = unvisited.pop()
            queue = [seed]
            cluster_cells = [seed]
            while queue:
                cell = queue.pop()
                cx, cy, cz = cell
                for dx, dy, dz in NEIGHBOR_OFFSETS_26:
                    neighbor = (cx + dx, cy + dy, cz + dz)
                    if neighbor not in unvisited:
                        continue
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
                    cluster_cells.append(neighbor)

            indices = []
            for cell in cluster_cells:
                indices.extend(cell_to_indices[cell])
            if len(indices) < self.min_cluster_points:
                continue

            cluster_points = points[np.asarray(indices, dtype=np.int64)]
            extent = np.ptp(cluster_points, axis=0)
            max_extent = float(np.max(extent)) if len(cluster_points) else 0.0
            if self.max_cluster_extent_m > 0.0 and max_extent > self.max_cluster_extent_m:
                continue
            if self.max_cluster_points > 0 and len(cluster_points) > self.max_cluster_points:
                step = int(math.ceil(len(cluster_points) / self.max_cluster_points))
                cluster_points = cluster_points[::step]

            clusters.append(
                {
                    "points": cluster_points,
                    "centroid": np.mean(cluster_points, axis=0),
                    "point_count": len(indices),
                    "extent": extent,
                }
            )

        clusters.sort(key=lambda item: item["point_count"], reverse=True)
        if self.max_clusters > 0:
            clusters = clusters[: self.max_clusters]
        return clusters

    def fit_spheres(self, clusters):
        spheres = []
        for cluster_id, cluster in enumerate(clusters):
            points = cluster["points"]
            if len(points) == 0:
                continue
            extent = np.asarray(cluster.get("extent", np.ptp(points, axis=0)), dtype=np.float64)
            extent_for_count = float(np.max(extent)) if len(extent) else 0.0
            sphere_count = max(1, int(math.ceil(extent_for_count / max(self.sphere_spacing_m, 1e-3))))
            sphere_count = min(sphere_count, max(1, self.max_spheres_per_cluster))

            bins = self.split_indices_by_farthest_points(points, sphere_count)
            for local_id, indices in enumerate(bins):
                if len(indices) == 0:
                    continue
                sphere_points = points[indices]
                center = np.mean(sphere_points, axis=0)
                distances = np.linalg.norm(sphere_points - center, axis=1)
                radius = float(np.percentile(distances, 90.0)) + self.sphere_padding_m
                radius = max(self.min_sphere_radius_m, min(radius, self.max_sphere_radius_m))
                spheres.append(
                    {
                        "center": center,
                        "radius": radius,
                        "cluster_id": cluster_id,
                        "sphere_id": local_id,
                        "point_count": int(len(sphere_points)),
                        "confidence": min(1.0, len(sphere_points) / max(self.min_cluster_points * 3.0, 1.0)),
                    }
                )
                if self.max_total_spheres > 0 and len(spheres) >= self.max_total_spheres:
                    return spheres
        return spheres

    @staticmethod
    def split_indices_by_farthest_points(points, count):
        if count <= 1 or len(points) == 0:
            return [np.arange(len(points), dtype=np.int64)]
        count = min(count, len(points))

        centroid = np.mean(points, axis=0)
        first_index = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))
        center_indices = [first_index]
        min_distances = np.linalg.norm(points - points[first_index], axis=1)

        for _ in range(1, count):
            next_index = int(np.argmax(min_distances))
            if min_distances[next_index] < 1e-9:
                break
            center_indices.append(next_index)
            distances = np.linalg.norm(points - points[next_index], axis=1)
            min_distances = np.minimum(min_distances, distances)

        centers = points[np.asarray(center_indices, dtype=np.int64)]
        distances_to_centers = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances_to_centers, axis=1)
        return [
            np.flatnonzero(labels == label).astype(np.int64, copy=False)
            for label in range(len(center_indices))
        ]

    @staticmethod
    def clustered_points(clusters):
        if not clusters:
            return np.empty((0, 3), dtype=np.float64)
        return np.vstack([cluster["points"] for cluster in clusters])

    @staticmethod
    def principal_axis(points):
        if len(points) < 3:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        centered = points - np.mean(points, axis=0)
        covariance = centered.T @ centered / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return axis / norm

    @staticmethod
    def split_indices_by_projection(projections, count):
        if count <= 1:
            return [np.arange(len(projections), dtype=np.int64)]
        order = np.argsort(projections)
        return [chunk.astype(np.int64, copy=False) for chunk in np.array_split(order, count)]

    def publish_results(self, header, spheres, candidate_points, clustered_points=None):
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        for idx, sphere in enumerate(spheres):
            center = sphere["center"]
            radius = sphere["radius"]
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "obstacle_body_spheres"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = radius * 2.0
            marker.color.r = 0.05
            marker.color.g = 0.65
            marker.color.b = 1.0
            marker.color.a = 0.55
            if self.marker_lifetime_s > 0.0:
                marker.lifetime.sec = int(self.marker_lifetime_s)
                marker.lifetime.nanosec = int(
                    (self.marker_lifetime_s - int(self.marker_lifetime_s)) * 1e9
                )
            marker.text = (
                f"cluster={sphere['cluster_id']};"
                f"sphere={sphere['sphere_id']};"
                f"points={sphere['point_count']}"
            )
            marker_array.markers.append(marker)

        for idx in range(len(spheres), self.previous_marker_count):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "obstacle_body_spheres"
            marker.id = idx
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.previous_marker_count = len(spheres)

        self.marker_pub.publish(marker_array)
        self.publish_sphere_cloud(stamp, spheres)
        if self.publish_candidate_cloud_enabled:
            self.publish_candidate_cloud(stamp, candidate_points)
        if self.publish_cluster_cloud_enabled:
            if clustered_points is None:
                clustered_points = np.empty((0, 3), dtype=np.float64)
            self.publish_cluster_cloud(stamp, clustered_points)

    def publish_sphere_cloud(self, stamp, spheres):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="radius", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="cluster_id", offset=16, datatype=PointField.INT32, count=1),
            PointField(name="sphere_id", offset=20, datatype=PointField.INT32, count=1),
            PointField(name="point_count", offset=24, datatype=PointField.INT32, count=1),
            PointField(name="confidence", offset=28, datatype=PointField.FLOAT32, count=1),
        ]
        rows = []
        for sphere in spheres:
            center = sphere["center"]
            rows.append(
                [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(sphere["radius"]),
                    int(sphere["cluster_id"]),
                    int(sphere["sphere_id"]),
                    int(sphere["point_count"]),
                    float(sphere["confidence"]),
                ]
            )
        cloud_header = Header()
        cloud_header.stamp = stamp
        cloud_header.frame_id = self.target_frame
        cloud = point_cloud2.create_cloud(cloud_header, fields, rows)
        self.obstacle_cloud_pub.publish(cloud)

    def publish_candidate_cloud(self, stamp, points):
        cloud_header = Header()
        cloud_header.stamp = stamp
        cloud_header.frame_id = self.target_frame
        cloud = point_cloud2.create_cloud_xyz32(
            cloud_header,
            points.astype(np.float32, copy=False),
        )
        self.candidate_cloud_pub.publish(cloud)

    def publish_cluster_cloud(self, stamp, points):
        cloud_header = Header()
        cloud_header.stamp = stamp
        cloud_header.frame_id = self.target_frame
        cloud = point_cloud2.create_cloud_xyz32(
            cloud_header,
            points.astype(np.float32, copy=False),
        )
        self.cluster_cloud_pub.publish(cloud)

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
    node = ObstacleBodySphereNode()
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
