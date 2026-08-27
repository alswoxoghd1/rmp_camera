"""Apply a configurable rigid correction to robot sphere MarkerArrays."""

from copy import deepcopy

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from rmp_camera.robot_sphere_marker_correction_core import (
    correction_in_marker_frame,
    matrix_to_quaternion,
    rpy_correction_matrix,
    transform_pose_matrix,
)
from rmp_camera.vision_geometry import quaternion_to_matrix, transform_to_matrix


class RobotSphereMarkerCorrectionNode(Node):
    """Relay collision sphere markers after applying a bag-safe correction."""

    def __init__(self):
        super().__init__("robot_sphere_marker_correction_node")
        defaults = {
            "input_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
            "output_marker_topic":
                "/rmp_camera/calibrated_robot_collision_sphere_markers",
            "correction_frame": "camera0_depth_optical_frame",
            "translation_x_m": 0.0,
            "translation_y_m": 0.0,
            "translation_z_m": 0.0,
            "rotation_roll_rad": 0.0,
            "rotation_pitch_rad": 0.0,
            "rotation_yaw_rad": 0.0,
            "tf_timeout_s": 0.05,
            "passthrough_on_tf_failure": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.input_marker_topic = str(
            self.get_parameter("input_marker_topic").value)
        self.output_marker_topic = str(
            self.get_parameter("output_marker_topic").value)
        self.correction_frame = str(
            self.get_parameter("correction_frame").value).strip()
        self.tf_timeout_s = max(
            0.0, float(self.get_parameter("tf_timeout_s").value))
        self.passthrough_on_tf_failure = bool(
            self.get_parameter("passthrough_on_tf_failure").value)
        translation = [float(self.get_parameter(name).value) for name in (
            "translation_x_m", "translation_y_m", "translation_z_m")]
        rotation = [float(self.get_parameter(name).value) for name in (
            "rotation_roll_rad", "rotation_pitch_rad", "rotation_yaw_rad")]
        self.correction = rpy_correction_matrix(translation, rotation)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.subscription = self.create_subscription(
            MarkerArray, self.input_marker_topic, self._marker_callback, input_qos)
        self.publisher = self.create_publisher(
            MarkerArray, self.output_marker_topic, output_qos)
        self.get_logger().info(
            "Robot sphere marker correction started: "
            f"input={self.input_marker_topic}, output={self.output_marker_topic}, "
            f"frame={self.correction_frame or '<marker frame>'}, "
            f"translation={translation}, rpy={rotation}")

    def _marker_callback(self, message):
        corrected = deepcopy(message)
        corrections = {}
        try:
            for marker in corrected.markers:
                if marker.action != Marker.ADD:
                    continue
                frame = marker.header.frame_id.strip()
                if not frame:
                    raise ValueError("marker has an empty frame_id")
                stamp_ns = (
                    int(marker.header.stamp.sec) * 1_000_000_000
                    + int(marker.header.stamp.nanosec))
                key = (frame, stamp_ns)
                if key not in corrections:
                    corrections[key] = self._correction_for_marker(marker)
                self._apply_pose_correction(marker, corrections[key])
        except (TransformException, ValueError) as exc:
            self.get_logger().warn(
                f"Cannot correct robot sphere markers: {exc}",
                throttle_duration_sec=2.0)
            if self.passthrough_on_tf_failure:
                self.publisher.publish(message)
            return
        self.publisher.publish(corrected)

    def _correction_for_marker(self, marker):
        marker_frame = marker.header.frame_id.strip()
        if not self.correction_frame or self.correction_frame == marker_frame:
            return self.correction
        transform = self.tf_buffer.lookup_transform(
            self.correction_frame,
            marker_frame,
            rclpy.time.Time.from_msg(marker.header.stamp),
            timeout=Duration(seconds=self.tf_timeout_s),
        )
        marker_to_correction = transform_to_matrix(transform.transform)
        return correction_in_marker_frame(marker_to_correction, self.correction)

    @staticmethod
    def _apply_pose_correction(marker, marker_frame_correction):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quaternion_to_matrix(marker.pose.orientation)
        pose[:3, 3] = [
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
        ]
        corrected = transform_pose_matrix(pose, marker_frame_correction)
        marker.pose.position.x = float(corrected[0, 3])
        marker.pose.position.y = float(corrected[1, 3])
        marker.pose.position.z = float(corrected[2, 3])
        quaternion = matrix_to_quaternion(corrected[:3, :3])
        marker.pose.orientation.x = float(quaternion[0])
        marker.pose.orientation.y = float(quaternion[1])
        marker.pose.orientation.z = float(quaternion[2])
        marker.pose.orientation.w = float(quaternion[3])


def main(args=None):
    rclpy.init(args=args)
    node = RobotSphereMarkerCorrectionNode()
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
