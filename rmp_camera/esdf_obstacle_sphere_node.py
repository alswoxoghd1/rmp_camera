import math
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


SAMPLE_FIELDS = (
    "x",
    "y",
    "z",
    "robot_radius",
    "esdf_distance",
    "clearance",
    "gradient_x",
    "gradient_y",
    "gradient_z",
    "sphere_id",
    "observed",
    "gradient_valid",
    "closest_x",
    "closest_y",
    "closest_z",
)

SURFACE_FIELDS = (
    "x",
    "y",
    "z",
    "esdf_distance",
    "clearance",
    "normal_x",
    "normal_y",
    "normal_z",
    "robot_sphere_id",
    "confidence",
)


class EsdfObstacleSphereNode(Node):
    def __init__(self):
        super().__init__("esdf_obstacle_sphere_node")

        self.declare_parameter("sample_topic", "/rmp_camera/esdf_obstacle_surface_points")
        self.declare_parameter("obstacle_marker_topic", "/rmp_camera/camera_obstacle_spheres")
        self.declare_parameter("obstacle_cloud_topic", "/rmp_camera/camera_obstacle_sphere_cloud")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("active_range_m", 1.0)
        self.declare_parameter("danger_clearance_m", 0.05)
        self.declare_parameter("cluster_radius_m", 0.14)
        self.declare_parameter("min_cluster_points", 3)
        self.declare_parameter("min_sphere_radius_m", 0.04)
        self.declare_parameter("max_sphere_radius_m", 0.14)
        self.declare_parameter("sphere_padding_m", 0.02)
        self.declare_parameter("max_obstacle_spheres", 5)
        self.declare_parameter("smoothing_alpha", 0.6)
        self.declare_parameter("smoothing_match_distance_m", 0.18)
        self.declare_parameter("min_persistent_frames", 3)
        self.declare_parameter("max_rate_hz", 10.0)

        self.sample_topic = self.get_parameter("sample_topic").value
        self.obstacle_marker_topic = self.get_parameter("obstacle_marker_topic").value
        self.obstacle_cloud_topic = self.get_parameter("obstacle_cloud_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.active_range_m = float(self.get_parameter("active_range_m").value)
        self.danger_clearance_m = float(self.get_parameter("danger_clearance_m").value)
        self.cluster_radius_m = float(self.get_parameter("cluster_radius_m").value)
        self.min_cluster_points = int(self.get_parameter("min_cluster_points").value)
        self.min_sphere_radius_m = float(self.get_parameter("min_sphere_radius_m").value)
        self.max_sphere_radius_m = float(self.get_parameter("max_sphere_radius_m").value)
        self.sphere_padding_m = float(self.get_parameter("sphere_padding_m").value)
        self.max_obstacle_spheres = int(self.get_parameter("max_obstacle_spheres").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.smoothing_match_distance_m = float(
            self.get_parameter("smoothing_match_distance_m").value
        )
        self.min_persistent_frames = int(self.get_parameter("min_persistent_frames").value)
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)

        self.last_process_time = 0.0
        self.last_log_time = 0.0
        self.previous_marker_count = 0
        self.previous_spheres = []
        self.cluster_tracks = []

        sample_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.sample_sub = self.create_subscription(
            PointCloud2,
            self.sample_topic,
            self.sample_callback,
            sample_qos,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.obstacle_marker_topic, 2)
        self.cloud_pub = self.create_publisher(PointCloud2, self.obstacle_cloud_topic, 2)

        self.get_logger().info(
            "ESDF obstacle sphere node started: "
            f"{self.sample_topic} -> {self.obstacle_cloud_topic}, "
            f"markers={self.obstacle_marker_topic}, active_range={self.active_range_m:.2f} m, "
            f"max_spheres={self.max_obstacle_spheres}, "
            f"min_cluster_points={self.min_cluster_points}, "
            f"min_persistent_frames={self.min_persistent_frames}"
        )

    def sample_callback(self, msg):
        now = time.monotonic()
        if self.max_rate_hz > 0.0 and now - self.last_process_time < 1.0 / self.max_rate_hz:
            return
        self.last_process_time = now

        if msg.header.frame_id != self.target_frame:
            self.log_throttled(
                f"Skipping ESDF samples in frame {msg.header.frame_id}; expected {self.target_frame}.",
                warn=True,
            )
            return

        candidates = self.read_candidates(msg)
        clusters = self.cluster_candidates(candidates)
        clusters = self.smooth_clusters(clusters)
        raw_cluster_count = len(clusters)
        clusters = self.filter_persistent_clusters(clusters)
        self.publish_results(msg.header, clusters)

        min_clearance = min((item["clearance"] for item in candidates), default=float("nan"))
        self.log_throttled(
            "ESDF obstacle spheres: "
            f"candidates={len(candidates)}, raw_clusters={raw_cluster_count}, "
            f"spheres={len(clusters)}, min_clearance={min_clearance:.3f} m"
        )

    def read_candidates(self, msg):
        candidates = []
        available_fields = {field.name for field in msg.fields}
        if "closest_x" in available_fields:
            return self.read_collision_sample_candidates(msg)
        if "robot_sphere_id" in available_fields and "normal_x" in available_fields:
            return self.read_surface_candidates(msg)

        self.log_throttled(
            "Skipping ESDF obstacle input with unsupported fields: "
            f"{sorted(available_fields)}",
            warn=True,
        )
        return candidates

    def read_collision_sample_candidates(self, msg):
        candidates = []
        for row in point_cloud2.read_points(msg, field_names=SAMPLE_FIELDS, skip_nans=False):
            (
                robot_x,
                robot_y,
                robot_z,
                robot_radius,
                _esdf_distance,
                clearance,
                gradient_x,
                gradient_y,
                gradient_z,
                sphere_id,
                observed,
                gradient_valid,
                closest_x,
                closest_y,
                closest_z,
            ) = row

            if int(observed) == 0:
                continue
            point = np.array([closest_x, closest_y, closest_z], dtype=np.float64)
            robot_center = np.array([robot_x, robot_y, robot_z], dtype=np.float64)
            normal = np.array([gradient_x, gradient_y, gradient_z], dtype=np.float64)
            if not np.isfinite(point).all() or not np.isfinite(robot_center).all():
                continue
            if not math.isfinite(float(clearance)):
                continue
            if float(clearance) > self.active_range_m:
                continue

            normal_norm = float(np.linalg.norm(normal))
            if not math.isfinite(normal_norm) or normal_norm < 1e-6:
                vector = robot_center - point
                vector_norm = float(np.linalg.norm(vector))
                if vector_norm < 1e-6:
                    continue
                normal = vector / vector_norm
            else:
                normal = normal / normal_norm

            score = self.risk_score(float(clearance))
            candidates.append(
                {
                    "point": point,
                    "robot_center": robot_center,
                    "robot_radius": float(robot_radius),
                    "clearance": float(clearance),
                    "robot_sphere_id": int(sphere_id),
                    "normal": normal,
                    "risk_score": score,
                    "gradient_valid": int(gradient_valid),
                }
            )

        candidates.sort(key=lambda item: (item["clearance"], -item["risk_score"]))
        return candidates

    def read_surface_candidates(self, msg):
        candidates = []
        for row in point_cloud2.read_points(msg, field_names=SURFACE_FIELDS, skip_nans=True):
            (
                x,
                y,
                z,
                _esdf_distance,
                clearance,
                normal_x,
                normal_y,
                normal_z,
                robot_sphere_id,
                confidence,
            ) = row

            point = np.array([x, y, z], dtype=np.float64)
            normal = np.array([normal_x, normal_y, normal_z], dtype=np.float64)
            if not np.isfinite(point).all() or not math.isfinite(float(clearance)):
                continue
            if float(clearance) > self.active_range_m:
                continue

            normal_norm = float(np.linalg.norm(normal))
            if math.isfinite(normal_norm) and normal_norm > 1e-6:
                normal = normal / normal_norm
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            score = self.risk_score(float(clearance))
            candidates.append(
                {
                    "point": point,
                    "robot_center": np.zeros(3, dtype=np.float64),
                    "robot_radius": 0.0,
                    "clearance": float(clearance),
                    "robot_sphere_id": int(robot_sphere_id),
                    "normal": normal,
                    "risk_score": score,
                    "gradient_valid": 1,
                    "input_confidence": float(confidence),
                }
            )

        candidates.sort(key=lambda item: (item["clearance"], -item["risk_score"]))
        return candidates

    def cluster_candidates(self, candidates):
        clusters = []
        for candidate in candidates:
            best_cluster = None
            best_distance = float("inf")
            for cluster in clusters:
                distance = float(np.linalg.norm(candidate["point"] - cluster["center"]))
                if distance <= self.cluster_radius_m and distance < best_distance:
                    best_cluster = cluster
                    best_distance = distance
            if best_cluster is None:
                clusters.append(self.make_cluster(candidate))
            else:
                self.add_to_cluster(best_cluster, candidate)

        finished = []
        for cluster in clusters:
            if len(cluster["points"]) < self.min_cluster_points:
                continue
            finished.append(self.finalize_cluster(cluster))

        finished.sort(key=lambda item: (-item["risk_score"], item["min_clearance"]))
        if self.max_obstacle_spheres > 0:
            finished = finished[: self.max_obstacle_spheres]
        return finished

    @staticmethod
    def make_cluster(candidate):
        return {
            "center": candidate["point"].copy(),
            "points": [candidate["point"]],
            "clearances": [candidate["clearance"]],
            "robot_sphere_ids": [candidate["robot_sphere_id"]],
            "normals": [candidate["normal"]],
            "risk_scores": [candidate["risk_score"]],
            "gradient_valid_count": int(candidate["gradient_valid"]),
            "input_confidences": [float(candidate.get("input_confidence", 1.0))],
        }

    @staticmethod
    def add_to_cluster(cluster, candidate):
        cluster["points"].append(candidate["point"])
        cluster["clearances"].append(candidate["clearance"])
        cluster["robot_sphere_ids"].append(candidate["robot_sphere_id"])
        cluster["normals"].append(candidate["normal"])
        cluster["risk_scores"].append(candidate["risk_score"])
        cluster["gradient_valid_count"] += int(candidate["gradient_valid"])
        cluster["input_confidences"].append(float(candidate.get("input_confidence", 1.0)))
        weights = np.maximum(np.asarray(cluster["risk_scores"], dtype=np.float64), 0.05)
        points = np.asarray(cluster["points"], dtype=np.float64)
        cluster["center"] = np.average(points, axis=0, weights=weights)

    def finalize_cluster(self, cluster):
        points = np.asarray(cluster["points"], dtype=np.float64)
        weights = np.maximum(np.asarray(cluster["risk_scores"], dtype=np.float64), 0.05)
        center = np.average(points, axis=0, weights=weights)
        point_radius = 0.0
        if len(points) > 1:
            point_radius = float(np.max(np.linalg.norm(points - center, axis=1)))
        radius = np.clip(
            point_radius + self.sphere_padding_m,
            self.min_sphere_radius_m,
            self.max_sphere_radius_m,
        )

        normals = np.asarray(cluster["normals"], dtype=np.float64)
        normal = np.average(normals, axis=0, weights=weights)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-6:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        clearances = np.asarray(cluster["clearances"], dtype=np.float64)
        min_index = int(np.argmin(clearances))
        return {
            "center": center,
            "radius": float(radius),
            "min_clearance": float(clearances[min_index]),
            "risk_score": float(np.max(cluster["risk_scores"])),
            "confidence": min(1.0, len(points) / max(1.0, float(self.min_cluster_points + 2)))
            * (0.5 + 0.5 * cluster["gradient_valid_count"] / max(1, len(points)))
            * float(np.mean(cluster["input_confidences"])),
            "robot_sphere_id": int(cluster["robot_sphere_ids"][min_index]),
            "normal": normal,
            "point_count": int(len(points)),
        }

    def smooth_clusters(self, clusters):
        if not self.previous_spheres:
            self.previous_spheres = [self.copy_cluster(cluster) for cluster in clusters]
            return clusters

        alpha = min(max(self.smoothing_alpha, 0.0), 1.0)
        used_previous = set()
        smoothed = []
        for cluster in clusters:
            match_index = self.find_previous_match(cluster, used_previous)
            if match_index is not None:
                previous = self.previous_spheres[match_index]
                cluster["center"] = alpha * cluster["center"] + (1.0 - alpha) * previous["center"]
                cluster["radius"] = alpha * cluster["radius"] + (1.0 - alpha) * previous["radius"]
                cluster["risk_score"] = max(cluster["risk_score"], previous["risk_score"] * 0.8)
                used_previous.add(match_index)
            smoothed.append(cluster)

        self.previous_spheres = [self.copy_cluster(cluster) for cluster in smoothed]
        return smoothed

    def find_previous_match(self, cluster, used_previous):
        best_index = None
        best_distance = float("inf")
        for index, previous in enumerate(self.previous_spheres):
            if index in used_previous:
                continue
            distance = float(np.linalg.norm(cluster["center"] - previous["center"]))
            if distance <= self.smoothing_match_distance_m and distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    @staticmethod
    def copy_cluster(cluster):
        copied = dict(cluster)
        copied["center"] = np.asarray(cluster["center"], dtype=np.float64).copy()
        copied["normal"] = np.asarray(cluster["normal"], dtype=np.float64).copy()
        return copied

    def filter_persistent_clusters(self, clusters):
        if self.min_persistent_frames <= 1:
            self.cluster_tracks = [
                {"center": np.asarray(cluster["center"], dtype=np.float64).copy(), "age": 1}
                for cluster in clusters
            ]
            return clusters
        if not clusters:
            self.cluster_tracks = []
            return []

        previous_tracks = self.cluster_tracks
        matched_previous = set()
        new_tracks = []
        persistent_clusters = []

        for cluster in clusters:
            center = np.asarray(cluster["center"], dtype=np.float64)
            best_index = None
            best_distance = self.smoothing_match_distance_m
            for index, track in enumerate(previous_tracks):
                if index in matched_previous:
                    continue
                distance = float(np.linalg.norm(center - track["center"]))
                if distance <= best_distance:
                    best_index = index
                    best_distance = distance

            age = 1
            if best_index is not None:
                matched_previous.add(best_index)
                age = int(previous_tracks[best_index]["age"]) + 1

            new_tracks.append({"center": center.copy(), "age": age})
            if age >= self.min_persistent_frames:
                persistent_clusters.append(cluster)

        self.cluster_tracks = new_tracks
        return persistent_clusters

    def risk_score(self, clearance):
        if clearance <= self.danger_clearance_m:
            return 1.0
        denominator = max(1e-6, self.active_range_m - self.danger_clearance_m)
        score = (self.active_range_m - clearance) / denominator
        return float(np.clip(score, 0.0, 1.0) ** 2)

    def publish_results(self, header, clusters):
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()
        for index, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "camera_obstacle_spheres"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(cluster["center"][0])
            marker.pose.position.y = float(cluster["center"][1])
            marker.pose.position.z = float(cluster["center"][2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = cluster["radius"] * 2.0
            marker.scale.y = cluster["radius"] * 2.0
            marker.scale.z = cluster["radius"] * 2.0
            marker.color.r, marker.color.g, marker.color.b = self.color_for_score(
                cluster["risk_score"]
            )
            marker.color.a = 0.85
            marker.text = (
                f"clearance={cluster['min_clearance']:.3f};"
                f"risk={cluster['risk_score']:.2f};"
                f"robot_sphere={cluster['robot_sphere_id']};"
                f"points={cluster['point_count']}"
            )
            marker_array.markers.append(marker)

        for index in range(len(clusters), self.previous_marker_count):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "camera_obstacle_spheres"
            marker.id = index
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self.previous_marker_count = len(clusters)

        cloud_header = header
        cloud_header.stamp = stamp
        cloud_header.frame_id = self.target_frame
        self.marker_pub.publish(marker_array)
        self.publish_obstacle_cloud(cloud_header, clusters)

    def publish_obstacle_cloud(self, header, clusters):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="radius", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="clearance", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="robot_sphere_id", offset=20, datatype=PointField.INT32, count=1),
            PointField(name="confidence", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="risk_score", offset=28, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_x", offset=32, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_y", offset=36, datatype=PointField.FLOAT32, count=1),
            PointField(name="normal_z", offset=40, datatype=PointField.FLOAT32, count=1),
            PointField(name="point_count", offset=44, datatype=PointField.INT32, count=1),
        ]
        rows = []
        for cluster in clusters:
            rows.append(
                (
                    float(cluster["center"][0]),
                    float(cluster["center"][1]),
                    float(cluster["center"][2]),
                    float(cluster["radius"]),
                    float(cluster["min_clearance"]),
                    int(cluster["robot_sphere_id"]),
                    float(cluster["confidence"]),
                    float(cluster["risk_score"]),
                    float(cluster["normal"][0]),
                    float(cluster["normal"][1]),
                    float(cluster["normal"][2]),
                    int(cluster["point_count"]),
                )
            )
        self.cloud_pub.publish(point_cloud2.create_cloud(header, fields, rows))

    @staticmethod
    def color_for_score(score):
        if score >= 0.7:
            return (1.0, 0.05, 0.02)
        if score >= 0.3:
            return (1.0, 0.62, 0.02)
        return (0.15, 0.85, 0.35)

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
    node = EsdfObstacleSphereNode()
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
