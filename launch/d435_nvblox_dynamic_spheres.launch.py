"""D435 dynamic Nvblox experiment with static/dynamic sphere fusion.

All numeric experiment settings come from one YAML.  This launch starts one
RealSense camera, its emitter splitter, and one Nvblox component only; none of
the robot-arm/RMPflow pipeline is part of this experiment.
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _experiment_section(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    try:
        section = data["/experiment"]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{config_path} must contain /experiment/ros__parameters") from exc
    required = (
        "global_frame", "camera_frame", "camera_serial_number",
        "static_tf_x_m", "static_tf_y_m", "static_tf_z_m",
        "static_tf_roll_rad", "static_tf_pitch_rad", "static_tf_yaw_rad",
        "rviz_config", "source", "use_sim_time")
    missing = [name for name in required if name not in section]
    if missing:
        raise RuntimeError(f"experiment config missing: {', '.join(missing)}")
    return section


def _launch_setup(context):
    experiment_config = LaunchConfiguration("experiment_config").perform(context)
    experiment = _experiment_section(experiment_config)
    source_override = LaunchConfiguration("source").perform(context).strip()
    sim_override = LaunchConfiguration("use_sim_time").perform(context).strip()
    source = source_override or str(experiment["source"])
    use_sim_time = _as_bool(sim_override) if sim_override else _as_bool(experiment["use_sim_time"])
    if source == "bag":
        use_sim_time = True
    if source not in ("camera", "bag"):
        raise RuntimeError("source must be 'camera' or 'bag'")

    run_realsense = _as_bool(LaunchConfiguration("run_realsense").perform(context)) and source == "camera"
    run_rviz = _as_bool(LaunchConfiguration("run_rviz").perform(context))
    run_static = _as_bool(LaunchConfiguration("run_esdf_medial_spheres").perform(context))
    run_dynamic = _as_bool(LaunchConfiguration("run_dynamic_obstacle_spheres").perform(context))
    run_fusion = _as_bool(LaunchConfiguration("run_obstacle_sphere_fusion").perform(context))
    log_level = LaunchConfiguration("log_level").perform(context)
    container_name = "rmp_camera_dynamic_nvblox_container"

    base_config = package_file(
        "nvblox_examples_bringup", "config/nvblox/nvblox_base.yaml")
    realsense_config = package_file(
        "nvblox_examples_bringup",
        "config/nvblox/specializations/nvblox_realsense.yaml")
    dynamics_config = package_file(
        "nvblox_examples_bringup",
        "config/nvblox/specializations/nvblox_dynamics.yaml")
    realsense_emitter_config = package_file(
        "nvblox_examples_bringup", "config/sensors/realsense_emitter_flashing.yaml")
    rviz_choice = str(experiment["rviz_config"])
    rviz_config = rviz_choice if Path(rviz_choice).is_absolute() else package_file(
        "rmp_camera", f"config/{rviz_choice}")
    effective = {"use_sim_time": use_sim_time}

    actions = []
    input_actions = []
    if source == "bag":
        bag_path = str(experiment.get("bag_path", ""))
        if not bag_path:
            raise RuntimeError("/experiment bag_path must be set when source=bag")
        input_actions.append(ExecuteProcess(
            cmd=["ros2", "bag", "play", bag_path, "--clock", "--loop", "--rate",
                 str(experiment.get("bag_rate", 1.0))], output="screen"))

    if run_realsense:
        camera_parameters = [
            realsense_emitter_config,
            {"camera_name": "camera0"},
            {
                "decimation_filter.enable": _as_bool(
                    experiment.get("realsense_decimation_filter_enabled", False)),
                "spatial_filter.enable": _as_bool(
                    experiment.get("realsense_spatial_filter_enabled", True)),
                "temporal_filter.enable": _as_bool(
                    experiment.get("realsense_temporal_filter_enabled", True)),
                "hole_filling_filter.enable": _as_bool(
                    experiment.get("realsense_hole_filling_filter_enabled", False)),
            },
        ]
        camera_serial = str(experiment["camera_serial_number"]).strip()
        if camera_serial:
            camera_parameters.append({"serial_no": camera_serial})
        input_actions.append(LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=[
                ComposableNode(
                    namespace="camera0", package="realsense2_camera",
                    plugin="realsense2_camera::RealSenseNodeFactory",
                    parameters=camera_parameters),
                ComposableNode(
                    namespace="camera0", name="realsense_splitter_node",
                    package="realsense_splitter",
                    plugin="nvblox::RealsenseSplitterNode",
                    parameters=[{"input_qos": "SENSOR_DATA", "output_qos": "SENSOR_DATA"}],
                    remappings=[
                        ("input/infra_1", "/camera0/camera/infra1/image_rect_raw"),
                        ("input/infra_1_metadata", "/camera0/camera/infra1/metadata"),
                        ("input/infra_2", "/camera0/camera/infra2/image_rect_raw"),
                        ("input/infra_2_metadata", "/camera0/camera/infra2/metadata"),
                        ("input/depth", "/camera0/camera/depth/image_rect_raw"),
                        ("input/depth_metadata", "/camera0/camera/depth/metadata"),
                        ("input/pointcloud", "/camera0/camera/depth/color/points"),
                        ("input/pointcloud_metadata", "/camera0/camera/depth/metadata"),
                    ]),
            ]))

    actions.extend([
        Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="base_to_d435_dynamic_experiment_tf", output="screen",
            arguments=[
                "--x", str(experiment["static_tf_x_m"]),
                "--y", str(experiment["static_tf_y_m"]),
                "--z", str(experiment["static_tf_z_m"]),
                "--roll", str(experiment["static_tf_roll_rad"]),
                "--pitch", str(experiment["static_tf_pitch_rad"]),
                "--yaw", str(experiment["static_tf_yaw_rad"]),
                "--frame-id", str(experiment["global_frame"]),
                "--child-frame-id", str(experiment["camera_frame"])]),
        ComposableNodeContainer(
            name=container_name, namespace="", package="rclcpp_components",
            executable="component_container_mt", output="screen",
            arguments=["--ros-args", "--log-level", log_level]),
    ])
    actions.extend(input_actions)
    actions.append(
        LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=[ComposableNode(
                name="nvblox_node", package="nvblox_ros", plugin="nvblox::NvbloxNode",
                remappings=[
                    ("camera_0/depth/image", "/camera0/realsense_splitter_node/output/depth"),
                    ("camera_0/depth/camera_info", "/camera0/camera/depth/camera_info"),
                    ("camera_0/color/image", "/camera0/camera/color/image_raw"),
                    ("camera_0/color/camera_info", "/camera0/camera/color/camera_info")],
                # Last file wins: base -> RealSense -> dynamics -> experiment.
                parameters=[base_config, realsense_config, dynamics_config,
                            experiment_config, effective])]),
    )
    if run_static:
        actions.append(Node(
            package="rmp_camera", executable="esdf_medial_sphere_node",
            name="esdf_medial_sphere_node", output="screen",
            parameters=[experiment_config, effective],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_dynamic:
        actions.append(Node(
            package="rmp_camera", executable="dynamic_obstacle_sphere_node",
            name="dynamic_obstacle_sphere_node", output="screen",
            parameters=[experiment_config, effective],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_fusion:
        actions.append(Node(
            package="rmp_camera", executable="obstacle_sphere_fusion_node",
            name="obstacle_sphere_fusion_node", output="screen",
            parameters=[experiment_config, effective],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_rviz:
        actions.append(Node(
            package="rviz2", executable="rviz2", name="dynamic_spheres_rviz",
            output="screen", arguments=["-d", rviz_config, "--ros-args", "--log-level", log_level],
            parameters=[effective]))
    return actions


def generate_launch_description():
    # Match the runtime dependency paths already used by the existing static
    # launch.  Prepending preserves any system paths supplied by the user.
    rosdeps_nv_lib = str(
        Path.home() / ".local/opt/rosdeps_nv/root/usr/lib/x86_64-linux-gnu")
    cuda12_runtime_lib = str(
        Path.home() / ".local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib")
    default_config = package_file(
        "rmp_camera", "config/d435_nvblox_dynamic_experiment.yaml")
    return LaunchDescription([
        SetEnvironmentVariable(
            name="LD_LIBRARY_PATH",
            value=[
                cuda12_runtime_lib,
                ":",
                rosdeps_nv_lib,
                ":",
                EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
            ],
        ),
        DeclareLaunchArgument(
            "experiment_config", default_value=default_config,
            description="Single YAML controlling camera, Nvblox, sphere, and fusion settings"),
        DeclareLaunchArgument(
            "source", default_value="",
            description="Optional camera|bag override; empty uses /experiment source"),
        DeclareLaunchArgument(
            "use_sim_time", default_value="",
            description="Optional boolean override; bag source always enables simulated time"),
        DeclareLaunchArgument("run_realsense", default_value="true"),
        DeclareLaunchArgument("run_rviz", default_value="true"),
        DeclareLaunchArgument("run_esdf_medial_spheres", default_value="true"),
        DeclareLaunchArgument("run_dynamic_obstacle_spheres", default_value="true"),
        DeclareLaunchArgument("run_obstacle_sphere_fusion", default_value="true"),
        DeclareLaunchArgument("log_level", default_value="info"),
        OpaqueFunction(function=_launch_setup),
    ])
