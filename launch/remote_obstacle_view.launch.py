from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def generate_launch_description():
    run_rviz = LaunchConfiguration("run_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument("run_rviz", default_value="True"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=package_file("rmp_camera", "config/remote_obstacle_view.rviz"),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="remote_obstacle_view_rviz",
                arguments=["-d", rviz_config],
                output="screen",
                condition=IfCondition(run_rviz),
            ),
        ]
    )
