from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "image_topic",
                default_value="/camera0/camera/color/image_raw",
            ),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "run",
                    "rqt_image_view",
                    "rqt_image_view",
                    image_topic,
                ],
                output="screen",
            ),
        ]
    )
