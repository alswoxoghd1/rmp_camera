import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass
class SphereSpec:
    name: str
    link: str
    offset: np.ndarray
    radius: float


def translation_matrix(xyz):
    transform = np.eye(4)
    transform[:3, 3] = xyz
    return transform


def rotation_x(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )


def rotation_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=float,
    )


def rotation_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def rotation_matrix_from_rpy(rpy):
    roll, pitch, yaw = rpy
    rotation = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def rotation_matrix_about_axis(axis, angle):
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(4)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def origin_matrix(xyz, rpy):
    return translation_matrix(xyz) @ rotation_matrix_from_rpy(rpy)


def parse_vector(text, default):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(value) for value in text.split()], dtype=float)


def parse_urdf_joints(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint_elem in root.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        joints.append(
            JointSpec(
                name=joint_elem.attrib["name"],
                joint_type=joint_elem.attrib.get("type", "fixed"),
                parent=parent_elem.attrib["link"],
                child=child_elem.attrib["link"],
                xyz=parse_vector(
                    origin_elem.attrib.get("xyz")
                    if origin_elem is not None else None,
                    [0.0, 0.0, 0.0],
                ),
                rpy=parse_vector(
                    origin_elem.attrib.get("rpy")
                    if origin_elem is not None else None,
                    [0.0, 0.0, 0.0],
                ),
                axis=parse_vector(
                    axis_elem.attrib.get("xyz")
                    if axis_elem is not None else None,
                    [0.0, 0.0, 0.0],
                ),
            )
        )
    return joints


class RobotCollisionSphereNode(Node):
    def __init__(self):
        super().__init__("robot_collision_sphere_node")

        package_share = Path(get_package_share_directory("rmp_camera"))
        self.declare_parameter(
            "urdf_path", str(package_share / "urdf" / "rb10_1300e.urdf"))
        self.declare_parameter(
            "sphere_config_path",
            str(package_share / "config" / "collision_spheres.yaml"),
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("stamp_from_joint_state", True)
        self.declare_parameter("require_joint_state", False)
        self.declare_parameter("require_complete_joint_state", False)

        self.urdf_path = Path(self.get_parameter("urdf_path").value)
        self.sphere_config_path = Path(self.get_parameter("sphere_config_path").value)
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.stamp_from_joint_state = bool(self.get_parameter("stamp_from_joint_state").value)
        self.require_joint_state = bool(
            self.get_parameter("require_joint_state").value)
        self.require_complete_joint_state = bool(
            self.get_parameter("require_complete_joint_state").value)

        config = self._load_config(self.sphere_config_path)
        self.base_frame = config.get("base_frame", "base_link")
        self.root_link = config.get("urdf_root_link", "link0")
        self.sphere_topic = config.get("sphere_topic", "/rmp_camera/robot_collision_spheres")
        self.marker_topic = config.get(
            "marker_topic", "/rmp_camera/robot_collision_sphere_markers")
        self.spheres = self._load_spheres(config)

        self.joints = parse_urdf_joints(self.urdf_path)
        self.movable_joint_names = {
            joint.name
            for joint in self.joints
            if joint.joint_type in ("revolute", "continuous", "prismatic")
        }
        self.children_by_parent = {}
        for joint in self.joints:
            self.children_by_parent.setdefault(joint.parent, []).append(joint)

        self.joint_positions = {}
        self.latest_joint_stamp = None
        self.received_joint_state = False
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._joint_state_callback,
            10,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.pose_pub = self.create_publisher(PoseArray, self.sphere_topic, 10)
        self.timer = self.create_timer(1.0 / max(self.publish_rate_hz, 1.0), self._publish)

        self.get_logger().info(
            f"Loaded {len(self.spheres)} collision spheres from {self.sphere_config_path}"
        )
        self.get_logger().info(
            f"Using URDF {self.urdf_path}, root_link={self.root_link}, "
            f"base_frame={self.base_frame}"
        )

    def _load_config(self, path):
        with path.open("r", encoding="utf-8") as config_file:
            return yaml.safe_load(config_file)

    def _load_spheres(self, config):
        spheres = []
        for item in config.get("collision_spheres", []):
            spheres.append(
                SphereSpec(
                    name=str(item["name"]),
                    link=str(item["link"]),
                    offset=np.array(item["offset"], dtype=float),
                    radius=float(item["radius"]),
                )
            )
        return spheres

    def _joint_state_callback(self, msg):
        self.received_joint_state = True
        for name, position in zip(msg.name, msg.position):
            self.joint_positions[name] = float(position)
        if msg.header.stamp.sec != 0 or msg.header.stamp.nanosec != 0:
            self.latest_joint_stamp = msg.header.stamp

    def _compute_link_transforms(self):
        transforms = {self.root_link: np.eye(4)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_transform = transforms[parent]
            for joint in self.children_by_parent.get(parent, []):
                joint_transform = origin_matrix(joint.xyz, joint.rpy)
                if joint.joint_type in ("revolute", "continuous"):
                    q = self.joint_positions.get(joint.name, 0.0)
                    joint_transform = joint_transform @ rotation_matrix_about_axis(joint.axis, q)
                elif joint.joint_type == "prismatic":
                    q = self.joint_positions.get(joint.name, 0.0)
                    joint_transform = joint_transform @ translation_matrix(joint.axis * q)
                transforms[joint.child] = parent_transform @ joint_transform
                stack.append(joint.child)
        return transforms

    def _publish(self):
        if self.require_joint_state and not self.received_joint_state:
            self.get_logger().warn(
                "Waiting for the first robot joint state before publishing collision spheres.",
                throttle_duration_sec=2.0,
            )
            return
        if self.require_complete_joint_state:
            missing = sorted(self.movable_joint_names - self.joint_positions.keys())
            if missing:
                self.get_logger().warn(
                    "Waiting for a complete robot joint state; missing: "
                    + ", ".join(missing),
                    throttle_duration_sec=2.0,
                )
                return
        if self.stamp_from_joint_state and self.latest_joint_stamp is not None:
            now = self.latest_joint_stamp
        else:
            now = self.get_clock().now().to_msg()
        transforms = self._compute_link_transforms()

        pose_array = PoseArray()
        pose_array.header.stamp = now
        pose_array.header.frame_id = self.base_frame

        marker_array = MarkerArray()
        for idx, sphere in enumerate(self.spheres):
            if sphere.link not in transforms:
                self.get_logger().warn(
                    f"Sphere {sphere.name} references unknown link {sphere.link}",
                    throttle_duration_sec=5.0,
                )
                continue
            local_point = np.array([sphere.offset[0], sphere.offset[1], sphere.offset[2], 1.0])
            center = transforms[sphere.link] @ local_point

            pose = Pose()
            pose.position.x = float(center[0])
            pose.position.y = float(center[1])
            pose.position.z = float(center[2])
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = self.base_frame
            marker.ns = "robot_collision_spheres"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose
            marker.scale.x = sphere.radius * 2.0
            marker.scale.y = sphere.radius * 2.0
            marker.scale.z = sphere.radius * 2.0
            marker.color.r = 0.1
            marker.color.g = 0.6
            marker.color.b = 1.0
            marker.color.a = 0.35
            marker_array.markers.append(marker)

        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RobotCollisionSphereNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
