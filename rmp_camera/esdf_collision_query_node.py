import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.vision_geometry import marker_spheres


class EsdfCollisionQueryNode(Node):
    def __init__(self):
        super().__init__("esdf_collision_query_node")

        self.declare_parameter("service_name", "/nvblox_node/get_esdf_and_gradient")
        self.declare_parameter(
            "robot_sphere_marker_topic",
            "/rmp_camera/robot_collision_sphere_markers",
        )
        self.declare_parameter("collision_sample_topic", "/rmp_camera/esdf_collision_samples")
        self.declare_parameter("collision_marker_topic", "/rmp_camera/esdf_collision_markers")
        self.declare_parameter("surface_cloud_topic", "/rmp_camera/esdf_obstacle_surface_points")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("max_rate_hz", 5.0)
        self.declare_parameter("query_padding_m", 1.0)
        self.declare_parameter("min_aabb_size_m", 0.5)
        self.declare_parameter("robot_clear_margin_m", 0.08)
        self.declare_parameter("self_filter_margin_m", 0.03)
        self.declare_parameter("safety_margin_m", 0.03)
        self.declare_parameter("max_publish_clearance_m", 1.0)
        self.declare_parameter("publish_all_valid_spheres", False)
        self.declare_parameter("update_esdf", True)
        self.declare_parameter("visualize_esdf", False)
        self.declare_parameter("unobserved_distance_value", -1000.0)
        self.declare_parameter("min_gradient_norm", 1e-3)
        self.declare_parameter("arrow_length_m", 0.18)
        self.declare_parameter("shell_sample_offset_m", 0.03)
        self.declare_parameter("closest_obstacle_radius_m", 0.04)
        self.declare_parameter("publish_closest_obstacle_spheres", True)
        self.declare_parameter("warning_clearance_m", 0.25)
        self.declare_parameter("publish_surface_cloud", True)
        self.declare_parameter("surface_min_distance_m", -0.02)
        self.declare_parameter("surface_max_distance_m", 0.10)
        self.declare_parameter("surface_max_robot_clearance_m", 1.0)
        self.declare_parameter("surface_stride", 1)
        self.declare_parameter("max_surface_points", 2000)

        self.service_name = self.get_parameter("service_name").value
        self.robot_sphere_marker_topic = self.get_parameter("robot_sphere_marker_topic").value
        self.collision_sample_topic = self.get_parameter("collision_sample_topic").value
        self.collision_marker_topic = self.get_parameter("collision_marker_topic").value
        self.surface_cloud_topic = self.get_parameter("surface_cloud_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        self.query_padding_m = float(self.get_parameter("query_padding_m").value)
        self.min_aabb_size_m = float(self.get_parameter("min_aabb_size_m").value)
        self.robot_clear_margin_m = float(self.get_parameter("robot_clear_margin_m").value)
        self.self_filter_margin_m = float(self.get_parameter("self_filter_margin_m").value)
        self.safety_margin_m = float(self.get_parameter("safety_margin_m").value)
        self.max_publish_clearance_m = float(self.get_parameter("max_publish_clearance_m").value)
        self.publish_all_valid_spheres = self._as_bool(
            self.get_parameter("publish_all_valid_spheres").value
        )
        self.update_esdf = self._as_bool(self.get_parameter("update_esdf").value)
        self.visualize_esdf = self._as_bool(self.get_parameter("visualize_esdf").value)
        self.unobserved_distance_value = float(
            self.get_parameter("unobserved_distance_value").value
        )
        self.min_gradient_norm = float(self.get_parameter("min_gradient_norm").value)
        self.arrow_length_m = float(self.get_parameter("arrow_length_m").value)
        self.shell_sample_offset_m = float(self.get_parameter("shell_sample_offset_m").value)
        self.closest_obstacle_radius_m = float(
            self.get_parameter("closest_obstacle_radius_m").value
        )
        self.publish_closest_obstacle_spheres = self._as_bool(
            self.get_parameter("publish_closest_obstacle_spheres").value
        )
        self.warning_clearance_m = float(self.get_parameter("warning_clearance_m").value)
        self.publish_surface_cloud = self._as_bool(
            self.get_parameter("publish_surface_cloud").value
        )
        self.surface_min_distance_m = float(self.get_parameter("surface_min_distance_m").value)
        self.surface_max_distance_m = float(self.get_parameter("surface_max_distance_m").value)
        self.surface_max_robot_clearance_m = float(
            self.get_parameter("surface_max_robot_clearance_m").value
        )
        self.surface_stride = max(1, int(self.get_parameter("surface_stride").value))
        self.max_surface_points = int(self.get_parameter("max_surface_points").value)
        self.shell_directions = self.make_shell_directions()

        self.latest_centers = None
        self.latest_radii = None
        self.latest_ids = None
        self.latest_marker_frame = ""
        self.previous_marker_count = 0
        self.pending = False
        self.last_log_time = 0.0
        self.last_warn_time = 0.0

        marker_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2)
        self.marker_sub = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self.marker_callback,
            marker_qos,
        )
        self.sample_pub = self.create_publisher(PointCloud2, self.collision_sample_topic, 2)
        self.surface_pub = self.create_publisher(PointCloud2, self.surface_cloud_topic, 2)
        self.marker_pub = self.create_publisher(MarkerArray, self.collision_marker_topic, 2)
        self.client = self.create_client(EsdfAndGradients, self.service_name)

        period = 0.2
        if self.max_rate_hz > 0.0:
            period = max(0.02, 1.0 / self.max_rate_hz)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            "ESDF collision query node started: "
            f"markers={self.robot_sphere_marker_topic}, service={self.service_name}, "
            f"sample_topic={self.collision_sample_topic}, frame={self.target_frame}, "
            f"surface_topic={self.surface_cloud_topic}, "
            f"clear_margin={self.robot_clear_margin_m:.3f} m, "
            f"self_filter_margin={self.self_filter_margin_m:.3f} m, "
            f"shell_offset={self.shell_sample_offset_m:.3f} m"
        )

    def marker_callback(self, msg):
        centers, radii, ids = marker_spheres(msg)
        if len(centers) == 0:
            return
        frame_id = ""
        for marker in msg.markers:
            if marker.type == marker.SPHERE and marker.action == marker.ADD:
                frame_id = marker.header.frame_id
                break
        self.latest_centers = centers
        self.latest_radii = radii
        self.latest_ids = ids
        self.latest_marker_frame = frame_id

    def tick(self):
        if self.pending:
            return
        if self.latest_centers is None or len(self.latest_centers) == 0:
            self.log_throttled("Waiting for robot collision sphere markers.", warn=True)
            return
        if self.latest_marker_frame and self.latest_marker_frame != self.target_frame:
            self.log_throttled(
                f"Robot sphere markers are in {self.latest_marker_frame}; "
                f"expected {self.target_frame}.",
                warn=True,
            )
            return
        if not self.client.service_is_ready():
            self.log_throttled(f"Waiting for ESDF service {self.service_name}", warn=True)
            return

        centers = self.latest_centers.copy()
        radii = self.latest_radii.copy()
        ids = list(self.latest_ids)
        request = self.make_request(centers, radii)
        started_at = time.monotonic()
        future = self.client.call_async(request)
        future.add_done_callback(
            lambda done_future, c=centers, r=radii, i=ids, t=started_at: self.handle_response(
                done_future, c, r, i, t
            )
        )
        self.pending = True

    def make_request(self, centers, radii):
        clear_radii = radii + self.robot_clear_margin_m
        padding = np.maximum(clear_radii + self.query_padding_m, self.min_aabb_size_m * 0.5)
        aabb_min = np.min(centers - padding[:, None], axis=0)
        aabb_max = np.max(centers + padding[:, None], axis=0)
        size = np.maximum(aabb_max - aabb_min, self.min_aabb_size_m)

        request = EsdfAndGradients.Request()
        request.update_esdf = self.update_esdf
        request.visualize_esdf = self.visualize_esdf
        request.use_aabb = True
        request.frame_id = self.target_frame
        request.aabb_min_m = Point(x=float(aabb_min[0]), y=float(aabb_min[1]), z=float(aabb_min[2]))
        request.aabb_size_m = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
        request.spheres_to_clear_center_m = [
            Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            for center in centers
        ]
        request.spheres_to_clear_radius_m = [float(radius) for radius in clear_radii]
        return request

    def handle_response(self, future, centers, radii, ids, started_at):
        self.pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.log_throttled(f"ESDF query failed: {exc}", warn=True)
            return
        if not response.success:
            self.log_throttled("ESDF query returned success=false.", warn=True)
            return

        grid = EsdfGrid(response, self.unobserved_distance_value)
        if not grid.valid:
            self.log_throttled("ESDF query returned an empty or invalid grid.", warn=True)
            return

        samples = []
        for center, radius, sphere_id in zip(centers, radii, ids):
            sample = self.best_shell_sample(
                grid,
                center,
                float(radius),
                int(sphere_id),
                centers,
                radii,
            )
            if sample is None:
                continue
            if (
                not self.publish_all_valid_spheres
                and self.max_publish_clearance_m > 0.0
                and sample["clearance"] > self.max_publish_clearance_m
            ):
                continue
            samples.append(sample)

        stamp = self.get_clock().now().to_msg()
        self.publish_sample_cloud(stamp, samples)
        surface_points = []
        if self.publish_surface_cloud:
            surface_points = self.extract_surface_points(grid, centers, radii, ids)
            self.publish_surface_cloud_msg(stamp, surface_points)
        self.publish_markers(stamp, samples)
        self.log_status(samples, len(centers), time.monotonic() - started_at, len(surface_points))

    def is_observed_distance(self, distance):
        return (
            distance is not None
            and math.isfinite(distance)
            and distance > self.unobserved_distance_value + 1.0
        )

    def best_shell_sample(self, grid, center, radius, sphere_id, robot_centers, robot_radii):
        clear_radius = radius + self.robot_clear_margin_m
        sample_radius = clear_radius + self.shell_sample_offset_m
        best = None
        for direction in self.shell_directions:
            query_point = center + direction * sample_radius
            distance = grid.value_at_point(query_point)
            if not self.is_observed_distance(distance):
                continue

            raw_gradient = grid.gradient_at_point(query_point)
            gradient_norm = float(np.linalg.norm(raw_gradient))
            gradient_valid = bool(
                np.isfinite(raw_gradient).all() and gradient_norm >= self.min_gradient_norm
            )
            if gradient_valid:
                esdf_gradient = raw_gradient / gradient_norm
                if distance >= 0.0:
                    closest_point = query_point - esdf_gradient * float(distance)
                    away_vector = center - closest_point
                    away_norm = float(np.linalg.norm(away_vector))
                    if away_norm >= self.min_gradient_norm:
                        push_gradient = away_vector / away_norm
                        clearance = away_norm - radius - self.safety_margin_m
                    else:
                        push_gradient = esdf_gradient
                        clearance = float(distance) - self.safety_margin_m
                else:
                    push_gradient = esdf_gradient
                    closest_point = query_point.copy()
                    clearance = float(distance) - self.safety_margin_m

                if self.closest_point_matches_robot(
                    closest_point,
                    robot_centers,
                    robot_radii,
                ):
                    continue
            else:
                # Fallback for flat or boundary cells. The direction is less exact than
                # an ESDF gradient, but still gives a useful obstacle-sphere candidate.
                push_gradient = -direction
                if distance >= 0.0:
                    closest_point = query_point + direction * float(distance)
                else:
                    closest_point = query_point.copy()
                clearance = float(distance) + sample_radius - radius - self.safety_margin_m

                if self.closest_point_matches_robot(
                    closest_point,
                    robot_centers,
                    robot_radii,
                ):
                    continue

            sample = {
                "center": center,
                "robot_radius": radius,
                "query_point": query_point,
                "closest_point": closest_point,
                "distance": float(distance),
                "clearance": float(clearance),
                "gradient": push_gradient,
                "raw_gradient_norm": gradient_norm,
                "gradient_valid": 1 if gradient_valid else 0,
                "sphere_id": sphere_id,
                "observed": 1,
            }
            if best is None or sample["clearance"] < best["clearance"]:
                best = sample
        return best

    def closest_point_matches_robot(self, closest_point, robot_centers, robot_radii):
        if not np.isfinite(closest_point).all():
            return False
        clear_radii = robot_radii + self.robot_clear_margin_m + self.self_filter_margin_m
        distances = np.linalg.norm(robot_centers - closest_point, axis=1)
        return bool(np.any(distances <= clear_radii))

    def extract_surface_points(self, grid, robot_centers, robot_radii, robot_ids):
        rows = []
        stride = self.surface_stride
        for ix in range(0, int(grid.size[0]), stride):
            for iy in range(0, int(grid.size[1]), stride):
                for iz in range(0, int(grid.size[2]), stride):
                    index = np.array([ix, iy, iz], dtype=np.int64)
                    distance = grid.value_at_index(index)
                    if not self.is_observed_distance(distance):
                        continue
                    if (
                        distance < self.surface_min_distance_m
                        or distance > self.surface_max_distance_m
                    ):
                        continue

                    point = grid.index_to_point(index)
                    if self.closest_point_matches_robot(point, robot_centers, robot_radii):
                        continue

                    center_distances = np.linalg.norm(robot_centers - point, axis=1)
                    clearances = center_distances - robot_radii - self.safety_margin_m
                    nearest_index = int(np.argmin(clearances))
                    clearance = float(clearances[nearest_index])
                    if (
                        self.surface_max_robot_clearance_m > 0.0
                        and clearance > self.surface_max_robot_clearance_m
                    ):
                        continue

                    away_vector = robot_centers[nearest_index] - point
                    away_norm = float(np.linalg.norm(away_vector))
                    if away_norm > self.min_gradient_norm:
                        normal = away_vector / away_norm
                    else:
                        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

                    distance_confidence = 1.0 - min(
                        1.0,
                        abs(float(distance))
                        / max(1e-6, self.surface_max_distance_m - self.surface_min_distance_m),
                    )
                    rows.append(
                        (
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            float(distance),
                            float(clearance),
                            float(normal[0]),
                            float(normal[1]),
                            float(normal[2]),
                            int(robot_ids[nearest_index]),
                            float(max(0.2, distance_confidence)),
                        )
                    )

        rows.sort(key=lambda row: row[4])
        if self.max_surface_points > 0 and len(rows) > self.max_surface_points:
            rows = rows[: self.max_surface_points]
        return rows

    def publish_surface_cloud_msg(self, stamp, rows):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="esdf_distance", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="clearance", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_x", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_y", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_z", offset=28, datatype=PointField.FLOAT32, count=1),
            PointField(name="robot_sphere_id", offset=32, datatype=PointField.INT32, count=1),
            PointField(name="confidence", offset=36, datatype=PointField.FLOAT32, count=1),
        ]
        header = Header()
        header.stamp = stamp
        header.frame_id = self.target_frame
        self.surface_pub.publish(point_cloud2.create_cloud(header, fields, rows))

    def publish_sample_cloud(self, stamp, samples):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="robot_radius", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="esdf_distance", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="clearance", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="gradient_x", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="gradient_y", offset=28, datatype=PointField.FLOAT32, count=1),
            PointField(name="gradient_z", offset=32, datatype=PointField.FLOAT32, count=1),
            PointField(name="raw_gradient_norm", offset=36, datatype=PointField.FLOAT32, count=1),
            PointField(name="sphere_id", offset=40, datatype=PointField.INT32, count=1),
            PointField(name="observed", offset=44, datatype=PointField.INT32, count=1),
            PointField(name="gradient_valid", offset=48, datatype=PointField.INT32, count=1),
            PointField(name="closest_x", offset=52, datatype=PointField.FLOAT32, count=1),
            PointField(name="closest_y", offset=56, datatype=PointField.FLOAT32, count=1),
            PointField(name="closest_z", offset=60, datatype=PointField.FLOAT32, count=1),
        ]
        rows = []
        for sample in samples:
            center = sample["center"]
            gradient = sample["gradient"]
            closest = sample["closest_point"]
            rows.append(
                (
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(sample["robot_radius"]),
                    float(sample["distance"]),
                    float(sample["clearance"]),
                    float(gradient[0]),
                    float(gradient[1]),
                    float(gradient[2]),
                    float(sample["raw_gradient_norm"]),
                    int(sample["sphere_id"]),
                    int(sample["observed"]),
                    int(sample["gradient_valid"]),
                    float(closest[0]),
                    float(closest[1]),
                    float(closest[2]),
                )
            )
        header = Header()
        header.stamp = stamp
        header.frame_id = self.target_frame
        self.sample_pub.publish(point_cloud2.create_cloud(header, fields, rows))

    def publish_markers(self, stamp, samples):
        marker_array = MarkerArray()
        for index, sample in enumerate(samples):
            center = sample["center"]
            radius = sample["robot_radius"]
            gradient = sample["gradient"]
            color = self.color_for_clearance(sample["clearance"])

            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = self.target_frame
            sphere.ns = "esdf_collision_query_spheres"
            sphere.id = index
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(center[0])
            sphere.pose.position.y = float(center[1])
            sphere.pose.position.z = float(center[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = radius * 2.0
            sphere.scale.y = radius * 2.0
            sphere.scale.z = radius * 2.0
            sphere.color.r = color[0]
            sphere.color.g = color[1]
            sphere.color.b = color[2]
            sphere.color.a = 0.45
            marker_array.markers.append(sphere)

            arrow = Marker()
            arrow.header.stamp = stamp
            arrow.header.frame_id = self.target_frame
            arrow.ns = "esdf_collision_gradients"
            arrow.id = index
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            start = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
            end_point = center + gradient * self.arrow_length_m
            end = Point(x=float(end_point[0]), y=float(end_point[1]), z=float(end_point[2]))
            arrow.points = [start, end]
            arrow.scale.x = 0.012
            arrow.scale.y = 0.035
            arrow.scale.z = 0.06
            arrow.color.r = color[0]
            arrow.color.g = color[1]
            arrow.color.b = color[2]
            arrow.color.a = 0.95
            marker_array.markers.append(arrow)

            closest = sample["closest_point"]
            if (
                self.publish_closest_obstacle_spheres
                and np.isfinite(closest).all()
            ):
                obstacle = Marker()
                obstacle.header.stamp = stamp
                obstacle.header.frame_id = self.target_frame
                obstacle.ns = "esdf_closest_obstacle_spheres"
                obstacle.id = index
                obstacle.type = Marker.SPHERE
                obstacle.action = Marker.ADD
                obstacle.pose.position.x = float(closest[0])
                obstacle.pose.position.y = float(closest[1])
                obstacle.pose.position.z = float(closest[2])
                obstacle.pose.orientation.w = 1.0
                obstacle.scale.x = self.closest_obstacle_radius_m * 2.0
                obstacle.scale.y = self.closest_obstacle_radius_m * 2.0
                obstacle.scale.z = self.closest_obstacle_radius_m * 2.0
                obstacle.color.r = 1.0
                obstacle.color.g = 0.95
                obstacle.color.b = 0.05
                obstacle.color.a = 0.85 if sample["gradient_valid"] else 0.45
                marker_array.markers.append(obstacle)

        for index in range(len(samples), self.previous_marker_count):
            for namespace in (
                "esdf_collision_query_spheres",
                "esdf_collision_gradients",
                "esdf_closest_obstacle_spheres",
            ):
                marker = Marker()
                marker.header.stamp = stamp
                marker.header.frame_id = self.target_frame
                marker.ns = namespace
                marker.id = index
                marker.action = Marker.DELETE
                marker_array.markers.append(marker)
        self.previous_marker_count = len(samples)
        self.marker_pub.publish(marker_array)

    def color_for_clearance(self, clearance):
        if clearance <= 0.0:
            return (1.0, 0.05, 0.02)
        if clearance <= self.warning_clearance_m:
            return (1.0, 0.62, 0.02)
        return (0.15, 0.85, 0.35)

    def log_status(self, samples, robot_sphere_count, elapsed_s, surface_point_count=0):
        now = time.monotonic()
        if now - self.last_log_time < 2.0:
            return
        self.last_log_time = now
        if samples:
            min_clearance = min(sample["clearance"] for sample in samples)
            min_distance = min(sample["distance"] for sample in samples)
            self.get_logger().info(
                "ESDF collision samples: "
                f"robot_spheres={robot_sphere_count}, published={len(samples)}, "
                f"surface_points={surface_point_count}, "
                f"min_distance={min_distance:.3f} m, "
                f"min_clearance={min_clearance:.3f} m, "
                f"service_time={elapsed_s * 1000.0:.1f} ms"
            )
        else:
            self.get_logger().info(
                "ESDF collision samples: "
                f"robot_spheres={robot_sphere_count}, published=0, "
                f"surface_points={surface_point_count}, "
                f"service_time={elapsed_s * 1000.0:.1f} ms"
            )

    def log_throttled(self, message, warn=False):
        now = time.monotonic()
        if now - self.last_warn_time < 2.0:
            return
        self.last_warn_time = now
        if warn:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def make_shell_directions():
        directions = []
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                for dz in (-1.0, 0.0, 1.0):
                    direction = np.array([dx, dy, dz], dtype=np.float64)
                    norm = float(np.linalg.norm(direction))
                    if norm > 1e-9:
                        directions.append(direction / norm)
        return directions


class EsdfGrid:
    def __init__(self, response, unobserved_distance_value):
        self.data = response.esdf_and_gradients.data
        self.dims = response.esdf_and_gradients.layout.dim
        self.voxel_size = float(response.voxel_size_m)
        self.unobserved_distance_value = float(unobserved_distance_value)
        self.origin = np.array(
            [response.origin_m.x, response.origin_m.y, response.origin_m.z],
            dtype=np.float64,
        )
        self.valid = len(self.dims) >= 3 and len(self.data) > 0 and self.voxel_size > 0.0
        if self.valid:
            self.size = np.array(
                [int(self.dims[0].size), int(self.dims[1].size), int(self.dims[2].size)],
                dtype=np.int64,
            )
            self.stride_y = int(self.dims[1].stride)
            self.stride_z = int(self.dims[2].stride)
        else:
            self.size = np.zeros(3, dtype=np.int64)
            self.stride_y = 0
            self.stride_z = 0

    def point_to_index(self, point):
        index = np.rint((np.asarray(point, dtype=np.float64) - self.origin) / self.voxel_size - 0.5)
        index = index.astype(np.int64)
        if np.any(index < 0) or np.any(index >= self.size):
            return None
        return index

    def index_to_point(self, index):
        return self.origin + (np.asarray(index, dtype=np.float64) + 0.5) * self.voxel_size

    def value_at_point(self, point):
        index = self.point_to_index(point)
        if index is None:
            return None
        return self.value_at_index(index)

    def value_at_index(self, index):
        ix, iy, iz = int(index[0]), int(index[1]), int(index[2])
        return float(self.data[ix * self.stride_y + iy * self.stride_z + iz])

    def observed(self, value):
        return math.isfinite(value) and value > self.unobserved_distance_value + 1.0

    def gradient_at_point(self, point):
        index = self.point_to_index(point)
        if index is None:
            return np.array([float("nan"), float("nan"), float("nan")], dtype=np.float64)
        return np.array([self.axis_gradient(index, axis) for axis in range(3)], dtype=np.float64)

    def axis_gradient(self, index, axis):
        if self.size[axis] < 2:
            return float("nan")
        minus = index.copy()
        plus = index.copy()
        if index[axis] <= 0:
            plus[axis] += 1
            center = self.value_at_index(index)
            plus_value = self.value_at_index(plus)
            if not self.observed(center) or not self.observed(plus_value):
                return float("nan")
            return (plus_value - center) / self.voxel_size
        if index[axis] >= self.size[axis] - 1:
            minus[axis] -= 1
            center = self.value_at_index(index)
            minus_value = self.value_at_index(minus)
            if not self.observed(center) or not self.observed(minus_value):
                return float("nan")
            return (center - minus_value) / self.voxel_size
        minus[axis] -= 1
        plus[axis] += 1
        minus_value = self.value_at_index(minus)
        plus_value = self.value_at_index(plus)
        if not self.observed(minus_value) or not self.observed(plus_value):
            return float("nan")
        return (plus_value - minus_value) / (2.0 * self.voxel_size)


def main(args=None):
    rclpy.init(args=args)
    node = EsdfCollisionQueryNode()
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
