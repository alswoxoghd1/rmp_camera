from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("service_name", default_value="/nvblox_node/get_esdf_and_gradient"),
            DeclareLaunchArgument("frame_id", default_value="base_link"),
            DeclareLaunchArgument(
                "points",
                default_value="0.0,0.0,0.5;0.3,0.0,0.5;0.6,0.0,0.5",
            ),
            DeclareLaunchArgument("aabb_size_m", default_value="0.25"),
            DeclareLaunchArgument("repeat", default_value="False"),
            Node(
                package="rmp_camera",
                executable="esdf_query_debug_node",
                name="esdf_query_debug_node",
                output="screen",
                parameters=[
                    {
                        "service_name": LaunchConfiguration("service_name"),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "points": LaunchConfiguration("points"),
                        "aabb_size_m": LaunchConfiguration("aabb_size_m"),
                        "repeat": LaunchConfiguration("repeat"),
                    }
                ],
            ),
        ]
    )
