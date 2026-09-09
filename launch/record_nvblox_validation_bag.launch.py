"""Record live inputs needed to replay self-filtering and Nvblox mapping."""

from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Start the live pipeline and record self-filter/Nvblox replay inputs."""
    default_bag_dir = Path.home() / "bags" / "nvblox_validation"
    default_bag_dir.mkdir(parents=True, exist_ok=True)
    default_bag_name = "nvblox_inputs_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    bag_dir = LaunchConfiguration("bag_dir")
    bag_name = LaunchConfiguration("bag_name")
    start_pipeline = LaunchConfiguration("start_pipeline")
    run_rviz = LaunchConfiguration("run_rviz")
    output_path = PathJoinSubstitution([bag_dir, bag_name])

    pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("rmp_camera"),
                    "launch",
                    "d435_nvblox_static.launch.py",
                ]
            )
        ),
        condition=IfCondition(start_pipeline),
        launch_arguments={
            "source": "camera",
            "run_realsense": "true",
            "run_rviz": run_rviz,
            "run_collision_spheres": "true",
            "run_rb10_measured_joint_state_publisher": "true",
            "run_joint_state_normalizer": "true",
            "run_robot_state_publisher": "true",
            "run_robot_depth_mask": "true",
            "robot_depth_filter_mode": "surface_depth",
            "run_depth_to_pointcloud": "false",
            "run_vision_obstacle_pipeline": "false",
            "publish_obstacle_spheres": "false",
            "run_obstacle_body_spheres": "false",
            "run_esdf_collision_query": "false",
            "run_esdf_obstacle_spheres": "false",
            "run_esdf_medial_spheres": "false",
            "run_esdf_slice_obstacle_spheres": "false",
        }.items(),
    )

    # Record upstream inputs only. Masked depth and Nvblox outputs are regenerated
    # after code or parameter changes, avoiding duplicate publishers during replay.
    topics = [
        "/camera0/camera/depth/image_rect_raw",
        "/camera0/camera/depth/camera_info",
        "/camera0/camera/color/image_raw",
        "/camera0/camera/color/camera_info",
        "/rmp_camera/rb10_measured_joint_states",
        "/rmp_camera/joint_states_urdf",
        "/rmp_camera/robot_collision_sphere_markers",
        "/tf",
        "/tf_static",
    ]

    recorder = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", output_path, *topics],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag_dir", default_value=str(default_bag_dir)),
            DeclareLaunchArgument("bag_name", default_value=default_bag_name),
            DeclareLaunchArgument(
                "start_pipeline",
                default_value="true",
                description="Start the live D435/RB10 self-filter pipeline.",
            ),
            DeclareLaunchArgument(
                "run_rviz",
                default_value="false",
                description="Show RViz while recording; disabled by default to reduce load.",
            ),
            LogInfo(msg=["Recording validation inputs to: ", output_path]),
            pipeline,
            recorder,
        ]
    )
