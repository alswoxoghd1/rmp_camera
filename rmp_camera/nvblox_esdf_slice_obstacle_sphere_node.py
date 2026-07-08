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
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.vision_geometry import marker_spheres

NEIGHBOR_OFFSETS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


class NvbloxEsdfSliceObstacleSphereNode(Node):
    def __init__(self):
        super().__init__("nvblox_esdf_slice_obstacle_sphere_node")

        self.declare_parameter("input_cloud_topic", "/nvblox_node/static_esdf_pointcloud")
        self.declare_parameter("obstacle_marker_topic", "/rmp_camera/camera_obstacle_spheres")
        self.declare_parameter("obstacle_cloud_topic", "/rmp_camera/camera_obstacle_sphere_cloud")
        self.declare_parameter(
            "surface_cloud_topic",
            "/rmp_camera/esdf_slice_obstacle_surface_points",
        )
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("publish_surface_cloud", True)
        self.declare_parameter("self_filter_robot_enabled", True)
        self.declare_parameter("robot_sphere_marker_topic", "/rmp_camera/robot_collision_sphere_markers")
        self.declare_parameter("robot_self_filter_margin_m", 0.05)

        self.declare_parameter("surface_min_distance_m", -0.02)
        self.declare_parameter("surface_max_distance_m", 0.08)
        self.declare_parameter("danger_distance_m", 0.02)

        self.declare_parameter("min_x_m", -1.5)
        self.declare_parameter("max_x_m", 1.6)
        self.declare_parameter("min_y_m", -1.8)
        self.declare_parameter("max_y_m", 1.4)
        self.declare_parameter("min_z_m", 0.05)
        self.declare_parameter("max_z_m", 1.8)
        self.declare_parameter("max_range_from_base_m", 2.2)

        self.declare_parameter("voxel_size_m", 0.05)
        self.declare_parameter("cluster_cell_size_m", 0.12)
        self.declare_parameter("min_cluster_points", 4)
        self.declare_parameter("max_cluster_points", 5000)
        self.declare_parameter("max_cluster_extent_m", 2.2)
        self.declare_parameter("max_clusters", 8)
        self.declare_parameter("max_input_points", 20000)

        self.declare_parameter("sphere_spacing_m", 0.18)
        self.declare_parameter("min_sphere_radius_m", 0.05)
        self.declare_parameter("max_sphere_radius_m", 0.18)
        self.declare_parameter("sphere_padding_m", 0.025)
        self.declare_parameter("max_spheres_per_cluster", 6)
        self.declare_parameter("max_total_spheres", 24)
        self.declare_parameter("max_rate_hz", 5.0)

        self.declare_parameter("min_persistent_frames", 2)
        self.declare_parameter("persistence_match_distance_m", 0.16)
        self.declare_parameter("persistence_smoothing_alpha", 0.65)

        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.obstacle_marker_topic = self.get_parameter("obstacle_marker_topic").value
        self.obstacle_cloud_topic = self.get_parameter("obstacle_cloud_topic").value
        self.surface_cloud_topic = self.get_parameter("surface_cloud_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.publish_surface_cloud_enabled = bool(
            self.get_parameter("publish_surface_cloud").value
        )
        self.self_filter_robot_enabled = bool(
            self.get_parameter("self_filter_robot_enabled").value
        )
        self.robot_sphere_marker_topic = self.get_parameter("robot_sphere_marker_topic").value
        self.robot_self_filter_margin_m = float(
            self.get_parameter("robot_self_filter_margin_m").value
        )

        self.surface_min_distance_m = float(self.get_parameter("surface_min_distance_m").value)
        self.surface_max_distance_m = float(self.get_parameter("surface_max_distance_m").value)
        self.danger_distance_m = float(self.get_parameter("danger_distance_m").value)

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
        self.sphere_tracks = []
        self.latest_robot_markers = None

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        marker_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2)
        self.robot_marker_sub = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self.robot_marker_callback,
            marker_qos,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.input_cloud_topic,
            self.cloud_callback,
            cloud_qos,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.obstacle_marker_topic, 2)
        self.obstacle_cloud_pub = self.create_publisher(PointCloud2, self.obstacle_cloud_topic, 2)
        self.surface_cloud_pub = self.create_publisher(PointCloud2, self.surface_cloud_topic, 2)

        self.get_logger().info(
            "Nvblox ESDF slice obstacle sphere node started: "
            f"{self.input_cloud_topic} -> {self.obstacle_marker_topic}, "
            f"surface_distance=[{self.surface_min_distance_m:.3f}, "
            f"{self.surface_max_distance_m:.3f}] m, "
            f"max_spheres={self.max_total_spheres}, "
            f"robot_self_filter={self.self_filter_robot_enabled}"
        )

    def robot_marker_callback(self, msg):
        self.latest_robot_markers = msg

    def cloud_callback(self, msg):
        now = time.monotonic()
        if self.max_rate_hz > 0.0 and now - self.last_process_time < 1.0 / self.max_rate_hz:
            return
        self.last_process_time = now

        if msg.header.frame_id != self.target_frame:
            self.log_throttled(
                f"Skipping ESDF slice cloud in frame {msg.header.frame_id}; "
                f"expected {self.target_frame}.",
                warn=True,
            )
            return

        points, distances = self.read_esdf_points(msg)
        raw_count = len(points)
        if raw_count == 0:
            self.publish_results(msg.header, [], points, distances)
            return

        finite = np.isfinite(points).all(axis=1) & np.isfinite(distances)
        points = points[finite]
        distances = distances[finite]

        surface_mask = (
            (distances >= self.surface_min_distance_m)
            & (distances <= self.surface_max_distance_m)
        )
        points = points[surface_mask]
        distances = distances[surface_mask]

        points, distances = self.crop_workspace(points, distances)
        points, distances, removed_robot_count = self.filter_robot_points(points, distances)
        points, distances = self.downsample_with_distance(points, distances)
        if self.max_input_points > 0 and len(points) > self.max_input_points:
            step = int(np.ceil(len(points) / self.max_input_points))
            points = points[::step]
            distances = distances[::step]

        clusters = self.cluster_points(points, distances)
        spheres = self.fit_spheres(clusters)
        spheres = self.filter_persistent_spheres(spheres)
        self.publish_results(msg.header, spheres, points, distances)

        min_distance = float(np.min(distances)) if len(distances) else float("nan")
        self.log_throttled(
            "Nvblox ESDF slice obstacle spheres: "
            f"raw={raw_count}, surface={len(points)}, clusters={len(clusters)}, "
            f"removed_robot={removed_robot_count}, spheres={len(spheres)}, "
            f"min_esdf={min_distance:.3f} m"
        )

    def read_esdf_points(self, msg):
        field_names = {field.name for field in msg.fields}
        if "intensity" not in field_names:
            self.log_throttled(
                f"Skipping ESDF slice cloud without intensity field: {sorted(field_names)}",
                warn=True,
            )
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

        rows = point_cloud2.read_points_numpy(
            msg,
            field_names=("x", "y", "z", "intensity"),
            skip_nans=True,
        )
        rows = np.asarray(rows)
        if rows.dtype.fields:
            points = np.vstack([rows["x"], rows["y"], rows["z"]]).T
            distances = np.asarray(rows["intensity"])
        else:
            if rows.ndim == 1 and rows.size:
                rows = rows.reshape(-1, 4)
            if rows.size == 0:
                return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
            points = rows[:, :3]
            distances = rows[:, 3]
        return points.astype(np.float64, copy=False), distances.astype(np.float64, copy=False)

    def crop_workspace(self, points, distances):
        if len(points) == 0:
            return points, distances
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
        return points[mask], distances[mask]

    def filter_robot_points(self, points, distances):
        if not self.self_filter_robot_enabled or len(points) == 0:
            return points, distances, 0
        if self.latest_robot_markers is None:
            self.log_throttled(
                "Robot self-filter is enabled but no robot collision sphere markers are available yet.",
                warn=True,
            )
            return points, distances, 0

        marker_frame = self.latest_robot_markers.markers[0].header.frame_id if self.latest_robot_markers.markers else ""
        if marker_frame and marker_frame != self.target_frame:
            self.log_throttled(
                f"Skipping robot self-filter because marker frame is {marker_frame}; "
                f"expected {self.target_frame}.",
                warn=True,
            )
            return points, distances, 0

        centers, radii, _ = marker_spheres(self.latest_robot_markers)
        if len(centers) == 0:
            self.log_throttled(
                "Robot self-filter is enabled but marker array has no sphere markers.",
                warn=True,
            )
            return points, distances, 0

        keep_mask = np.ones(len(points), dtype=bool)
        expanded_radii = radii + self.robot_self_filter_margin_m
        for center, radius in zip(centers, expanded_radii):
            squared_distance = np.sum((points - center) ** 2, axis=1)
            keep_mask &= squared_distance >= radius * radius

        removed_count = len(points) - int(np.count_nonzero(keep_mask))
        return points[keep_mask], distances[keep_mask], removed_count

    def downsample_with_distance(self, points, distances):
        if self.voxel_size_m <= 0.0 or len(points) == 0:
            return points, distances
        keys = np.floor(points / self.voxel_size_m).astype(np.int64)
        order = np.lexsort((distances, keys[:, 2], keys[:, 1], keys[:, 0]))
        sorted_keys = keys[order]
        keep_sorted = np.ones(len(order), dtype=bool)
        if len(order) > 1:
            keep_sorted[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        keep_indices = order[keep_sorted]
        keep_indices.sort()
        return points[keep_indices], distances[keep_indices]

    def cluster_points(self, points, distances):
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

            index_array = np.asarray(indices, dtype=np.int64)
            cluster_points = points[index_array]
            cluster_distances = distances[index_array]
            extent = np.ptp(cluster_points, axis=0)
            max_extent = float(np.max(extent)) if len(cluster_points) else 0.0
            if self.max_cluster_extent_m > 0.0 and max_extent > self.max_cluster_extent_m:
                continue
            if self.max_cluster_points > 0 and len(cluster_points) > self.max_cluster_points:
                step = int(math.ceil(len(cluster_points) / self.max_cluster_points))
                cluster_points = cluster_points[::step]
                cluster_distances = cluster_distances[::step]

            clusters.append(
                {
                    "points": cluster_points,
                    "distances": cluster_distances,
                    "centroid": np.mean(cluster_points, axis=0),
                    "point_count": len(indices),
                    "extent": extent,
                    "min_distance": float(np.min(cluster_distances)),
                }
            )

        clusters.sort(key=lambda item: (item["min_distance"], -item["point_count"]))
        if self.max_clusters > 0:
            clusters = clusters[: self.max_clusters]
        return clusters

    def fit_spheres(self, clusters):
        spheres = []
        for cluster_id, cluster in enumerate(clusters):
            points = cluster["points"]
            distances = cluster["distances"]
            if len(points) == 0:
                continue

            extent = np.asarray(cluster.get("extent", np.ptp(points, axis=0)), dtype=np.float64)
            extent_for_count = float(np.max(extent)) if len(extent) else 0.0
            sphere_count = max(
                1,
                int(math.ceil(extent_for_count / max(self.sphere_spacing_m, 1e-3))),
            )
            sphere_count = min(sphere_count, max(1, self.max_spheres_per_cluster))

            bins = self.split_indices_by_farthest_points(points, sphere_count)
            for local_id, indices in enumerate(bins):
                if len(indices) == 0:
                    continue
                sphere_points = points[indices]
                sphere_distances = distances[indices]
                center = np.mean(sphere_points, axis=0)
                point_distances = np.linalg.norm(sphere_points - center, axis=1)
                radius = float(np.percentile(point_distances, 90.0)) + self.sphere_padding_m
                radius = max(self.min_sphere_radius_m, min(radius, self.max_sphere_radius_m))
                min_esdf = float(np.min(sphere_distances))
                spheres.append(
                    {
                        "center": center,
                        "radius": radius,
                        "cluster_id": cluster_id,
                        "sphere_id": local_id,
                        "point_count": int(len(sphere_points)),
                        "min_esdf_distance": min_esdf,
                        "mean_esdf_distance": float(np.mean(sphere_distances)),
                        "confidence": self.confidence_for_points(len(sphere_points), min_esdf),
                        "risk_score": self.surface_score(min_esdf),
                    }
                )
                if self.max_total_spheres > 0 and len(spheres) >= self.max_total_spheres:
                    return spheres
        spheres.sort(key=lambda item: (-item["risk_score"], item["min_esdf_distance"]))
        return spheres

    def confidence_for_points(self, point_count, min_distance):
        point_score = min(1.0, point_count / max(self.min_cluster_points * 3.0, 1.0))
        return float(point_score * (0.5 + 0.5 * self.surface_score(min_distance)))

    def surface_score(self, distance):
        if distance <= self.danger_distance_m:
            return 1.0
        denominator = max(1e-6, self.surface_max_distance_m - self.danger_distance_m)
        score = (self.surface_max_distance_m - distance) / denominator
        return float(np.clip(score, 0.0, 1.0) ** 2)

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

    def publish_results(self, header, spheres, surface_points, surface_distances):
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        for idx, sphere in enumerate(spheres):
            center = sphere["center"]
            radius = sphere["radius"]
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "camera_obstacle_spheres"
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
            marker.color.r, marker.color.g, marker.color.b = self.color_for_score(
                sphere["risk_score"]
            )
            marker.color.a = 0.72
            marker.text = (
                f"source=esdf_slice;"
                f"cluster={sphere['cluster_id']};"
                f"sphere={sphere['sphere_id']};"
                f"min_esdf={sphere['min_esdf_distance']:.3f};"
                f"points={sphere['point_count']}"
            )
            marker_array.markers.append(marker)

        for idx in range(len(spheres), self.previous_marker_count):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "camera_obstacle_spheres"
            marker.id = idx
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.previous_marker_count = len(spheres)

        self.marker_pub.publish(marker_array)
        self.publish_sphere_cloud(stamp, spheres)
        if self.publish_surface_cloud_enabled:
            self.publish_surface_cloud(stamp, surface_points, surface_distances)

    def publish_sphere_cloud(self, stamp, spheres):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="radius", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="min_esdf_distance", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="mean_esdf_distance", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="cluster_id", offset=24, datatype=PointField.INT32, count=1),
            PointField(name="sphere_id", offset=28, datatype=PointField.INT32, count=1),
            PointField(name="point_count", offset=32, datatype=PointField.INT32, count=1),
            PointField(name="confidence", offset=36, datatype=PointField.FLOAT32, count=1),
            PointField(name="risk_score", offset=40, datatype=PointField.FLOAT32, count=1),
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
                    float(sphere["min_esdf_distance"]),
                    float(sphere["mean_esdf_distance"]),
                    int(sphere["cluster_id"]),
                    int(sphere["sphere_id"]),
                    int(sphere["point_count"]),
                    float(sphere["confidence"]),
                    float(sphere["risk_score"]),
                ]
            )
        cloud_header = Header()
        cloud_header.stamp = stamp
        cloud_header.frame_id = self.target_frame
        self.obstacle_cloud_pub.publish(point_cloud2.create_cloud(cloud_header, fields, rows))

    def publish_surface_cloud(self, stamp, points, distances):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="esdf_distance", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        rows = [
            [
                float(point[0]),
                float(point[1]),
                float(point[2]),
                float(distance),
            ]
            for point, distance in zip(points, distances)
        ]
        cloud_header = Header()
        cloud_header.stamp = stamp
        cloud_header.frame_id = self.target_frame
        self.surface_cloud_pub.publish(point_cloud2.create_cloud(cloud_header, fields, rows))

    @staticmethod
    def color_for_score(score):
        if score >= 0.7:
            return (1.0, 0.85, 0.02)
        if score >= 0.3:
            return (0.1, 0.85, 1.0)
        return (0.2, 0.9, 0.35)

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
    node = NvbloxEsdfSliceObstacleSphereNode()
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
