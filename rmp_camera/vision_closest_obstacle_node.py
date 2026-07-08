import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.vision_geometry import marker_spheres, voxel_downsample


class VisionClosestObstacleNode(Node):
    def __init__(self):
        super().__init__("vision_closest_obstacle_node")

        self.declare_parameter("input_cloud_topic", "/rmp_camera/robot_free_pointcloud")
        self.declare_parameter("robot_sphere_marker_topic", "/rmp_camera/robot_collision_sphere_markers")
        self.declare_parameter("obstacle_sphere_marker_topic", "/rmp_camera/camera_obstacle_spheres")
        self.declare_parameter("closest_point_marker_topic", "/rmp_camera/closest_obstacle_points")
        self.declare_parameter("obstacle_pose_topic", "/rmp_camera/camera_obstacle_sphere_poses")
        self.declare_parameter("obstacle_cloud_topic", "/rmp_camera/camera_obstacle_sphere_cloud")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("active_range_m", 1.0)
        self.declare_parameter("min_clearance_m", 0.20)
        self.declare_parameter("obstacle_radius_m", 0.06)
        self.declare_parameter("cluster_radius_m", 0.14)
        self.declare_parameter("max_obstacle_spheres", 5)
        self.declare_parameter("voxel_size_m", 0.02)
        self.declare_parameter("max_points", 80000)
        self.declare_parameter("max_rate_hz", 10.0)

        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.robot_sphere_marker_topic = self.get_parameter("robot_sphere_marker_topic").value
        self.obstacle_sphere_marker_topic = self.get_parameter("obstacle_sphere_marker_topic").value
        self.closest_point_marker_topic = self.get_parameter("closest_point_marker_topic").value
        self.obstacle_pose_topic = self.get_parameter("obstacle_pose_topic").value
        self.obstacle_cloud_topic = self.get_parameter("obstacle_cloud_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.active_range_m = float(self.get_parameter("active_range_m").value)
        self.min_clearance_m = float(self.get_parameter("min_clearance_m").value)
        self.obstacle_radius_m = float(self.get_parameter("obstacle_radius_m").value)
        self.cluster_radius_m = float(self.get_parameter("cluster_radius_m").value)
        self.max_obstacle_spheres = int(self.get_parameter("max_obstacle_spheres").value)
        self.voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.max_rate_hz = float(self.get_parameter("max_rate_hz").value)

        self.latest_markers = None
        self.last_process_time = 0.0
        self.last_log_time = 0.0
        self.previous_obstacle_count = 0
        self.previous_closest_count = 0

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        marker_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2)
        self.marker_sub = self.create_subscription(
            MarkerArray,
            self.robot_sphere_marker_topic,
            self.marker_callback,
            marker_qos,
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            self.input_cloud_topic,
            self.cloud_callback,
            cloud_qos,
        )
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray,
            self.obstacle_sphere_marker_topic,
            2,
        )
        self.closest_marker_pub = self.create_publisher(
            MarkerArray,
            self.closest_point_marker_topic,
            2,
        )
        self.pose_pub = self.create_publisher(PoseArray, self.obstacle_pose_topic, 2)
        self.obstacle_cloud_pub = self.create_publisher(PointCloud2, self.obstacle_cloud_topic, 2)

        self.get_logger().info(
            "Vision closest obstacle node started: "
            f"{self.input_cloud_topic} -> {self.obstacle_sphere_marker_topic}, "
            f"control_topic={self.obstacle_cloud_topic}, "
            f"clearance_range=[{self.min_clearance_m:.2f}, {self.active_range_m:.2f}] m, "
            f"max_obstacles={self.max_obstacle_spheres}"
        )

    def marker_callback(self, msg):
        self.latest_markers = msg

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
        if self.latest_markers is None:
            self.log_throttled("Waiting for robot collision sphere markers.", warn=True)
            return

        points = self.read_xyz_points(msg)
        if len(points) == 0:
            self.publish_results(msg.header, [], [])
            return
        points = points[np.isfinite(points).all(axis=1)]
        points = voxel_downsample(points, self.voxel_size_m)
        if self.max_points > 0 and len(points) > self.max_points:
            step = int(np.ceil(len(points) / self.max_points))
            points = points[::step]

        centers, radii, robot_ids = marker_spheres(self.latest_markers)
        if len(centers) == 0:
            self.log_throttled("Robot marker array has no sphere markers.", warn=True)
            return

        candidates = []
        for sphere_index, (center, radius) in enumerate(zip(centers, radii)):
            distances_to_center = np.linalg.norm(points - center, axis=1)
            clearances = distances_to_center - radius
            mask = (clearances >= self.min_clearance_m) & (clearances <= self.active_range_m)
            if not np.any(mask):
                continue
            masked_indices = np.flatnonzero(mask)
            best_local = int(np.argmin(clearances[mask]))
            best_index = int(masked_indices[best_local])
            candidates.append(
                {
                    "point": points[best_index],
                    "clearance": float(clearances[best_index]),
                    "robot_sphere_index": sphere_index,
                    "robot_marker_id": robot_ids[sphere_index],
                }
            )

        candidates.sort(key=lambda item: item["clearance"])
        clusters = self.cluster_candidates(candidates)
        self.publish_results(msg.header, clusters, candidates)
        min_clearance = candidates[0]["clearance"] if candidates else float("nan")
        self.log_throttled(
            "Vision obstacles: "
            f"points={len(points)}, candidates={len(candidates)}, clusters={len(clusters)}, "
            f"min_clearance={min_clearance:.3f} m"
        )

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

    def cluster_candidates(self, candidates):
        clusters = []
        for candidate in candidates:
            point = candidate["point"]
            assigned = False
            for cluster in clusters:
                if np.linalg.norm(point - cluster["center"]) <= self.cluster_radius_m:
                    cluster["points"].append(point)
                    cluster["clearances"].append(candidate["clearance"])
                    cluster["robot_sphere_indices"].append(candidate["robot_sphere_index"])
                    cluster["center"] = np.mean(cluster["points"], axis=0)
                    assigned = True
                    break
            if assigned:
                continue
            clusters.append(
                {
                    "center": point.copy(),
                    "points": [point],
                    "clearances": [candidate["clearance"]],
                    "robot_sphere_indices": [candidate["robot_sphere_index"]],
                }
            )
            if len(clusters) >= self.max_obstacle_spheres:
                break

        for cluster in clusters:
            points = np.asarray(cluster["points"], dtype=np.float64)
            if len(points) > 1:
                radius_from_points = float(np.max(np.linalg.norm(points - cluster["center"], axis=1)))
            else:
                radius_from_points = 0.0
            cluster["radius"] = max(self.obstacle_radius_m, radius_from_points + self.obstacle_radius_m)
            cluster["min_clearance"] = float(np.min(cluster["clearances"]))
        return clusters

    def publish_results(self, header, clusters, candidates):
        obstacle_markers = MarkerArray()
        closest_markers = MarkerArray()
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.target_frame

        stamp = pose_array.header.stamp
        for idx, cluster in enumerate(clusters):
            center = cluster["center"]
            radius = cluster["radius"]

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
            marker.color.r = 1.0
            marker.color.g = 0.25
            marker.color.b = 0.05
            marker.color.a = 0.8
            marker.text = (
                f"min_clearance={cluster['min_clearance']:.3f};"
                f"robot_spheres={cluster['robot_sphere_indices']}"
            )
            obstacle_markers.markers.append(marker)

            pose = Pose()
            pose.position = marker.pose.position
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        for idx in range(len(clusters), self.previous_obstacle_count):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "camera_obstacle_spheres"
            marker.id = idx
            marker.action = Marker.DELETE
            obstacle_markers.markers.append(marker)
        self.previous_obstacle_count = len(clusters)

        for idx, candidate in enumerate(candidates):
            point = candidate["point"]
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "closest_obstacle_points"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = float(point[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.035
            marker.scale.y = 0.035
            marker.scale.z = 0.035
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.2
            marker.color.a = 0.9
            marker.text = (
                f"clearance={candidate['clearance']:.3f};"
                f"robot_sphere={candidate['robot_sphere_index']}"
            )
            closest_markers.markers.append(marker)

        for idx in range(len(candidates), self.previous_closest_count):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.target_frame
            marker.ns = "closest_obstacle_points"
            marker.id = idx
            marker.action = Marker.DELETE
            closest_markers.markers.append(marker)
        self.previous_closest_count = len(candidates)

        self.obstacle_marker_pub.publish(obstacle_markers)
        self.closest_marker_pub.publish(closest_markers)
        self.pose_pub.publish(pose_array)
        self.publish_obstacle_cloud(pose_array.header, clusters)

    def publish_obstacle_cloud(self, header, clusters):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="radius", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="clearance", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="robot_sphere_id", offset=20, datatype=PointField.INT32, count=1),
            PointField(name="confidence", offset=24, datatype=PointField.FLOAT32, count=1),
        ]
        rows = []
        for cluster in clusters:
            center = cluster["center"]
            nearest_robot_sphere = (
                int(cluster["robot_sphere_indices"][0])
                if cluster["robot_sphere_indices"]
                else -1
            )
            # Confidence is intentionally simple for the first integration:
            # more candidate points in a cluster means the obstacle is more stable.
            confidence = min(1.0, len(cluster["points"]) / 3.0)
            rows.append(
                [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(cluster["radius"]),
                    float(cluster["min_clearance"]),
                    nearest_robot_sphere,
                    confidence,
                ]
            )
        cloud = point_cloud2.create_cloud(header, fields, rows)
        self.obstacle_cloud_pub.publish(cloud)

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
    node = VisionClosestObstacleNode()
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
