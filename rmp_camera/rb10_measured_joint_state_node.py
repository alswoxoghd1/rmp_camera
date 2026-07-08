import math
import socket
import struct

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
SYSTEM_STATE_FORMAT = (
    "fffffffffffffffffffffffffffffffffffffff"
    "iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii"
    "ffffffiiiififiiffffffiiiiiiiiiiiffiiiifiiiiiiiiiiiffffff"
)
SYSTEM_STATE = struct.Struct(SYSTEM_STATE_FORMAT)
PACKET_HEADER_SIZE = 4
PAYLOAD_SIZE = SYSTEM_STATE.size
PACKET_SIZE = PACKET_HEADER_SIZE + PAYLOAD_SIZE


class Rb10MeasuredJointStateNode(Node):
    def __init__(self):
        super().__init__("rb10_measured_joint_state_node")
        self.declare_parameter("robot_ip", "192.168.111.50")
        self.declare_parameter("data_port", 5001)
        self.declare_parameter("publish_topic", "/joint_states")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("socket_timeout_s", 0.05)
        self.declare_parameter("reconnect_period_s", 1.0)

        self.robot_ip = self.get_parameter("robot_ip").value
        self.data_port = int(self.get_parameter("data_port").value)
        self.publish_topic = self.get_parameter("publish_topic").value
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.socket_timeout_s = float(self.get_parameter("socket_timeout_s").value)
        self.reconnect_period_s = float(self.get_parameter("reconnect_period_s").value)

        self.publisher = self.create_publisher(JointState, self.publish_topic, 10)
        self.sock = None
        self.buffer = bytearray()
        self.last_connect_attempt = self.get_clock().now()
        self.warned_waiting = False
        self.received_first_packet = False

        self.timer = self.create_timer(
            1.0 / max(self.publish_rate_hz, 1.0),
            self._tick,
        )
        self.get_logger().info(
            f"Publishing measured RB10 joint states from {self.robot_ip}:{self.data_port} "
            f"to {self.publish_topic}"
        )

    def _tick(self):
        if self.sock is None:
            self._connect_if_due()
            return

        try:
            self.sock.sendall(b"reqdata")
            data = self.sock.recv(4096)
        except (OSError, socket.timeout) as exc:
            if isinstance(exc, socket.timeout):
                return
            self.get_logger().warn(f"RB10 data socket error: {exc}; reconnecting")
            self._close_socket()
            return

        if not data:
            self.get_logger().warn("RB10 data socket closed; reconnecting")
            self._close_socket()
            return

        self.buffer.extend(data)
        self._parse_buffer()

    def _connect_if_due(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_connect_attempt).nanoseconds / 1e9
        if elapsed < self.reconnect_period_s:
            return

        self.last_connect_attempt = now
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.socket_timeout_s)
            sock.connect((self.robot_ip, self.data_port))
        except OSError as exc:
            if not self.warned_waiting:
                self.warned_waiting = True
                self.get_logger().warn(
                    f"Waiting for RB10 data socket at {self.robot_ip}:{self.data_port}: {exc}"
                )
            return

        self.sock = sock
        self.buffer.clear()
        self.warned_waiting = False
        self.received_first_packet = False
        self.get_logger().info(f"Connected to RB10 data socket at {self.robot_ip}:{self.data_port}")

    def _parse_buffer(self):
        while True:
            start = self.buffer.find(b"$")
            if start < 0:
                self.buffer.clear()
                return
            if start:
                del self.buffer[:start]
            if len(self.buffer) < PACKET_SIZE:
                return

            packet = bytes(self.buffer[:PACKET_SIZE])
            del self.buffer[:PACKET_SIZE]
            if packet[3] != 3:
                continue

            try:
                values = SYSTEM_STATE.unpack(packet[PACKET_HEADER_SIZE:PACKET_SIZE])
            except struct.error as exc:
                self.get_logger().warn(f"Failed to parse RB10 state packet: {exc}")
                continue

            self._publish_joint_state(values[7:13])

    def _publish_joint_state(self, joint_degrees):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [math.radians(float(position)) for position in joint_degrees]
        self.publisher.publish(msg)

        if not self.received_first_packet:
            self.received_first_packet = True
            rounded = ", ".join(f"{position:.2f}" for position in joint_degrees)
            self.get_logger().info(f"Receiving measured RB10 joints in degrees: [{rounded}]")

    def _close_socket(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.buffer.clear()

    def destroy_node(self):
        self._close_socket()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Rb10MeasuredJointStateNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
