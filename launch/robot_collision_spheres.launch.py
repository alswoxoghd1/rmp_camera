from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def generate_launch_description():
    urdf_path = LaunchConfiguration("urdf_path")
    sphere_config_path = LaunchConfiguration("sphere_config_path")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    measured_joint_state_topic = LaunchConfiguration("measured_joint_state_topic")
    normalized_joint_state_topic = LaunchConfiguration("normalized_joint_state_topic")
    robot_ip = LaunchConfiguration("robot_ip")
    robot_data_port = LaunchConfiguration("robot_data_port")
    robot_joint_state_publish_rate_hz = LaunchConfiguration(
        "robot_joint_state_publish_rate_hz"
    )
    fallback_joint_state_publish_rate_hz = LaunchConfiguration("fallback_joint_state_publish_rate_hz")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    run_rb10_measured_joint_state_publisher = LaunchConfiguration(
        "run_rb10_measured_joint_state_publisher"
    )
    run_joint_state_normalizer = LaunchConfiguration("run_joint_state_normalizer")
    run_robot_state_publisher = LaunchConfiguration("run_robot_state_publisher")
    publish_robot_base_tf = LaunchConfiguration("publish_robot_base_tf")
    base_frame = LaunchConfiguration("base_frame")
    robot_root_frame = LaunchConfiguration("robot_root_frame")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "urdf_path",
                default_value=package_file("rmp_camera", "urdf/rb10_1300e.urdf"),
            ),
            DeclareLaunchArgument(
                "sphere_config_path",
                default_value=package_file("rmp_camera", "config/collision_spheres.yaml"),
            ),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            DeclareLaunchArgument(
                "measured_joint_state_topic",
                default_value="/rmp_camera/rb10_measured_joint_states",
            ),
            DeclareLaunchArgument(
                "normalized_joint_state_topic",
                default_value="/rmp_camera/joint_states_urdf",
            ),
            DeclareLaunchArgument("robot_ip", default_value="192.168.111.50"),
            DeclareLaunchArgument("robot_data_port", default_value="5001"),
            DeclareLaunchArgument("robot_joint_state_publish_rate_hz", default_value="50.0"),
            DeclareLaunchArgument("fallback_joint_state_publish_rate_hz", default_value="10.0"),
            DeclareLaunchArgument("publish_rate_hz", default_value="30.0"),
            DeclareLaunchArgument(
                "run_rb10_measured_joint_state_publisher",
                default_value="True",
            ),
            DeclareLaunchArgument("run_joint_state_normalizer", default_value="True"),
            DeclareLaunchArgument("run_robot_state_publisher", default_value="True"),
            DeclareLaunchArgument("publish_robot_base_tf", default_value="False"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("robot_root_frame", default_value="link0"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_robot_root_static_tf",
                output="screen",
                arguments=[
                    "--x",
                    "0.0",
                    "--y",
                    "0.0",
                    "--z",
                    "0.0",
                    "--roll",
                    "0.0",
                    "--pitch",
                    "0.0",
                    "--yaw",
                    "0.0",
                    "--frame-id",
                    base_frame,
                    "--child-frame-id",
                    robot_root_frame,
                ],
                condition=IfCondition(publish_robot_base_tf),
            ),
            Node(
                package="rmp_camera",
                executable="rb10_measured_joint_state_node",
                name="rb10_measured_joint_state_node",
                output="screen",
                parameters=[
                    {
                        "robot_ip": robot_ip,
                        "data_port": robot_data_port,
                        "publish_topic": measured_joint_state_topic,
                        "publish_rate_hz": robot_joint_state_publish_rate_hz,
                    }
                ],
                condition=IfCondition(run_rb10_measured_joint_state_publisher),
            ),
            Node(
                package="rmp_camera",
                executable="joint_state_normalizer_node",
                name="joint_state_normalizer_node",
                output="screen",
                parameters=[
                    {
                        "input_topic": joint_state_topic,
                        "output_topic": normalized_joint_state_topic,
                        "fallback_publish_rate_hz": fallback_joint_state_publish_rate_hz,
                    }
                ],
                condition=IfCondition(run_joint_state_normalizer),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="rmp_camera_robot_state_publisher",
                output="screen",
                parameters=[
                    {
                        "robot_description": ParameterValue(
                            Command(["cat ", urdf_path]),
                            value_type=str,
                        )
                    }
                ],
                remappings=[
                    ("joint_states", normalized_joint_state_topic),
                ],
                condition=IfCondition(run_robot_state_publisher),
            ),
            Node(
                package="rmp_camera",
                executable="robot_collision_sphere_node",
                name="robot_collision_sphere_node",
                output="screen",
                parameters=[
                    {
                        "urdf_path": urdf_path,
                        "sphere_config_path": sphere_config_path,
                        "joint_state_topic": normalized_joint_state_topic,
                        "publish_rate_hz": publish_rate_hz,
                    }
                ],
            ),
        ]
    )
