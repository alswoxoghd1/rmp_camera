import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


FIELD_NAMES = (
    "x",
    "y",
    "z",
    "radius",
    "clearance",
    "robot_sphere_id",
    "confidence",
)

OPTIONAL_FIELD_NAMES = (
    "risk_score",
    "normal_x",
    "normal_y",
    "normal_z",
    "point_count",
)


class CameraObstacleSphereEcho(Node):
    def __init__(self):
        super().__init__("camera_obstacle_sphere_echo")

        self.declare_parameter("topic", "/rmp_camera/camera_obstacle_sphere_cloud")
        self.declare_parameter("once", False)
        self.declare_parameter("print_rate_hz", 2.0)

        self.topic = self.get_parameter("topic").value
        self.once = bool(self.get_parameter("once").value)
        self.print_period = 1.0 / max(0.1, float(self.get_parameter("print_rate_hz").value))
        self.last_print_time = 0.0
        self.done = False

        self.create_subscription(PointCloud2, self.topic, self.callback, 10)
        self.get_logger().info(f"Decoding camera obstacle spheres from {self.topic}")

    def callback(self, msg):
        now = time.monotonic()
        if not self.once and now - self.last_print_time < self.print_period:
            return
        self.last_print_time = now

        lines = [f"frame={msg.header.frame_id} stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} count={msg.width}"]
        available_fields = {field.name for field in msg.fields}
        optional_fields = tuple(name for name in OPTIONAL_FIELD_NAMES if name in available_fields)
        field_names = FIELD_NAMES + optional_fields
        for idx, row in enumerate(point_cloud2.read_points(msg, field_names=field_names, skip_nans=False)):
            base_count = len(FIELD_NAMES)
            x, y, z, radius, clearance, robot_sphere_id, confidence = row[:base_count]
            extras = dict(zip(optional_fields, row[base_count:]))
            suffix = ""
            if "risk_score" in extras:
                suffix += f" risk={extras['risk_score']:.2f}"
            if {"normal_x", "normal_y", "normal_z"}.issubset(extras):
                suffix += (
                    f" normal=({extras['normal_x']:.2f}, "
                    f"{extras['normal_y']:.2f}, {extras['normal_z']:.2f})"
                )
            if "point_count" in extras:
                suffix += f" points={int(extras['point_count'])}"
            lines.append(
                f"{idx}: center=({x:.3f}, {y:.3f}, {z:.3f}) "
                f"radius={radius:.3f} clearance={clearance:.3f} "
                f"robot_sphere_id={robot_sphere_id} confidence={confidence:.2f}{suffix}"
            )
        self.get_logger().info("\n".join(lines))
        if self.once:
            self.done = True


def main(args=None):
    rclpy.init(args=args)
    node = CameraObstacleSphereEcho()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
