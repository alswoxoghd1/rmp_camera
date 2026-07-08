from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    default_bag_dir = Path.home() / "bags"
    default_bag_dir.mkdir(parents=True, exist_ok=True)
    default_bag_name = "d435_esdf_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    bag_dir = LaunchConfiguration("bag_dir")
    bag_name = LaunchConfiguration("bag_name")

    output_path = PathJoinSubstitution([bag_dir, bag_name])

    topics = [
        "/camera0/camera/depth/image_rect_raw",
        "/camera0/camera/depth/camera_info",
        "/camera0/camera/color/image_raw",
        "/camera0/camera/color/camera_info",
        "/tf",
        "/tf_static",
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag_dir", default_value=str(default_bag_dir)),
            DeclareLaunchArgument("bag_name", default_value=default_bag_name),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "-o",
                    output_path,
                    *topics,
                ],
                output="screen",
            ),
        ]
    )
