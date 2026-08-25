"""Record repeatable live inputs for Nvblox and robot self-filter tests."""

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
    """Start the live pipeline and record only reproducible input topics."""
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
            PathJoinSubstitution([
                FindPackageShare("rmp_camera"),
                "launch",
                "d435_nvblox_dynamic_spheres.launch.py",
            ])
        ),
        condition=IfCondition(start_pipeline),
        launch_arguments={
            "source": "camera",
            "run_realsense": "true",
            "run_rviz": run_rviz,
            "run_esdf_medial_spheres": "false",
            "run_dynamic_obstacle_spheres": "false",
            "run_obstacle_sphere_fusion": "false",
            "run_robot_self_filter": "true",
            "run_robot_collision_spheres": "true",
            "run_rb10_joint_state_source": "true",
        }.items(),
    )

    # Keep this bag input-only. Derived self-filter and Nvblox outputs must be
    # regenerated after each code/config change for a fair comparison.
    topics = [
        "/camera0/realsense_splitter_node/output/depth",
        "/camera0/camera/depth/camera_info",
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

    return LaunchDescription([
        DeclareLaunchArgument("bag_dir", default_value=str(default_bag_dir)),
        DeclareLaunchArgument("bag_name", default_value=default_bag_name),
        DeclareLaunchArgument(
            "start_pipeline",
            default_value="true",
            description="Start the live D435/RB10 pipeline before recording.",
        ),
        DeclareLaunchArgument(
            "run_rviz",
            default_value="false",
            description="Show RViz while recording (disabled to reduce load).",
        ),
        LogInfo(msg=["Recording validation inputs to: ", output_path]),
        pipeline,
        recorder,
    ])
