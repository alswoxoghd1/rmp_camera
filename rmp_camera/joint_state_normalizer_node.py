import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


CANONICAL_JOINT_NAMES = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
JOINT_NAME_MAP = {
    "joint_1": "base",
    "joint_2": "shoulder",
    "joint_3": "elbow",
    "joint_4": "wrist1",
    "joint_5": "wrist2",
    "joint_6": "wrist3",
}
DEFAULT_SOURCE_PRIORITY = [
    "/rmp_camera/rb10_measured_joint_states",
    "/rbpodo/joint_states",
    "/joint_states_mapped",
    "/joint_states_relayed",
    "/joint_states",
]


class JointStateNormalizerNode(Node):
    def __init__(self):
        super().__init__("joint_state_normalizer_node")
        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter(
            "extra_input_topics",
            [
                "/rmp_camera/rb10_measured_joint_states",
                "/rbpodo/joint_states",
                "/joint_states_mapped",
                "/joint_states_relayed",
            ],
        )
        self.declare_parameter("source_priority", DEFAULT_SOURCE_PRIORITY)
        self.declare_parameter("source_timeout_s", 1.0)
        self.declare_parameter("output_topic", "/rmp_camera/joint_states_urdf")
        self.declare_parameter("fallback_publish_rate_hz", 10.0)
        self.declare_parameter("publish_zero_fallback", True)
        self.declare_parameter("republish_latest", True)

        self.input_topics = self._input_topics_from_parameters()
        self.source_priority = self._source_priority_from_parameters()
        self.source_timeout_s = float(self.get_parameter("source_timeout_s").value)
        self.output_topic = self.get_parameter("output_topic").value
        self.fallback_publish_rate_hz = float(
            self.get_parameter("fallback_publish_rate_hz").value
        )
        self.publish_zero_fallback = bool(
            self.get_parameter("publish_zero_fallback").value)
        self.republish_latest = bool(
            self.get_parameter("republish_latest").value)

        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.joint_state_subscriptions = []
        for topic in self.input_topics:
            self._create_joint_state_subscriptions(topic)

        self.latest_msg = self._zero_joint_state()
        self.live_sources = set()
        self.active_source = None
        self.active_source_last_time = None
        self.warned_no_live_source = False
        self.timer = None
        if self.republish_latest:
            self.timer = self.create_timer(
                1.0 / max(self.fallback_publish_rate_hz, 1.0),
                self._publish_latest,
            )
        self.get_logger().info(
            "Normalizing joint states from "
            f"{', '.join(self.input_topics)} to {self.output_topic}"
        )

    def _input_topics_from_parameters(self):
        topics = []
        input_topic = self.get_parameter("input_topic").value
        extra_input_topics = self.get_parameter("extra_input_topics").value
        if isinstance(extra_input_topics, (list, tuple)):
            raw_topics = [input_topic, *extra_input_topics]
        else:
            raw_topics = [input_topic, extra_input_topics]

        for raw_topic in raw_topics:
            if isinstance(raw_topic, str):
                candidates = raw_topic.split(",")
            else:
                candidates = [raw_topic]
            for candidate in candidates:
                topic = str(candidate).strip()
                if topic and topic not in topics:
                    topics.append(topic)

        return topics

    def _source_priority_from_parameters(self):
        priority = []
        raw_priority = self.get_parameter("source_priority").value
        if not isinstance(raw_priority, (list, tuple)):
            raw_priority = str(raw_priority).split(",")

        for raw_topic in raw_priority:
            topic = str(raw_topic).strip()
            if topic and topic not in priority:
                priority.append(topic)

        for topic in self.input_topics:
            if topic not in priority:
                priority.append(topic)

        return {topic: index for index, topic in enumerate(priority)}

    def _create_joint_state_subscriptions(self, topic):
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.joint_state_subscriptions.append(
            self.create_subscription(
                JointState,
                topic,
                lambda msg, source=topic: self._joint_state_callback(msg, source),
                qos_profile,
            )
        )

    def _joint_state_callback(self, msg, source_topic):
        if not msg.name:
            return
        if not self._should_accept_source(source_topic):
            return

        normalized = JointState()
        normalized.header = msg.header

        for index, raw_name in enumerate(msg.name):
            name = JOINT_NAME_MAP.get(raw_name, raw_name)
            if name not in CANONICAL_JOINT_NAMES:
                continue

            normalized.name.append(name)
            if index < len(msg.position):
                normalized.position.append(msg.position[index])
            if index < len(msg.velocity):
                normalized.velocity.append(msg.velocity[index])
            if index < len(msg.effort):
                normalized.effort.append(msg.effort[index])

        if normalized.name and normalized.position:
            self.latest_msg = normalized
            self.active_source = source_topic
            self.active_source_last_time = self.get_clock().now()
            if source_topic not in self.live_sources:
                self.live_sources.add(source_topic)
                self.get_logger().info(
                    f"Receiving live joint states from {source_topic}"
                )
            self.publisher.publish(normalized)

    def _should_accept_source(self, source_topic):
        if self.active_source is None:
            return True

        source_rank = self.source_priority.get(source_topic, len(self.source_priority))
        active_rank = self.source_priority.get(
            self.active_source, len(self.source_priority)
        )
        if source_rank < active_rank:
            self.get_logger().info(
                f"Switching joint state source from {self.active_source} to {source_topic}"
            )
            return True

        if source_topic == self.active_source:
            return True

        if self.active_source_last_time is None:
            return False

        elapsed = (self.get_clock().now() - self.active_source_last_time).nanoseconds / 1e9
        if elapsed > self.source_timeout_s:
            self.get_logger().warn(
                f"Joint state source {self.active_source} timed out; switching to {source_topic}"
            )
            return True

        return False

    def _zero_joint_state(self):
        msg = JointState()
        msg.name = list(CANONICAL_JOINT_NAMES)
        msg.position = [0.0] * len(CANONICAL_JOINT_NAMES)
        return msg

    def _publish_latest(self):
        if not self.live_sources and not self.publish_zero_fallback:
            if not self.warned_no_live_source:
                self.warned_no_live_source = True
                self.get_logger().warn(
                    "No live joint state source received yet; suppressing zero pose fallback."
                )
            return
        self.latest_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.latest_msg)
        if not self.live_sources and not self.warned_no_live_source:
            self.warned_no_live_source = True
            self.get_logger().warn(
                "No live joint state source received yet; publishing zero pose fallback."
            )


def main(args=None):
    rclpy.init(args=args)
    node = JointStateNormalizerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
