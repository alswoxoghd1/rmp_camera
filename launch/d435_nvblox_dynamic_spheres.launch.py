"""
D435 dynamic Nvblox experiment with static/dynamic sphere fusion.

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
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _bool_arg(value):
    return "true" if value else "false"


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
    bag_path_override = LaunchConfiguration("bag_path").perform(context).strip()
    bag_rate_override = LaunchConfiguration("bag_rate").perform(context).strip()
    repeat_bag = _as_bool(LaunchConfiguration("repeat_bag").perform(context))
    depth_image_topic = LaunchConfiguration(
        "depth_image_topic").perform(context).strip()
    bag_cycle = _as_bool(LaunchConfiguration("_bag_cycle").perform(context))
    source = source_override or str(experiment["source"])
    use_sim_time = _as_bool(sim_override) if sim_override else _as_bool(experiment["use_sim_time"])
    if source == "bag":
        use_sim_time = True
    if source not in ("camera", "bag"):
        raise RuntimeError("source must be 'camera' or 'bag'")

    run_realsense = (
        _as_bool(LaunchConfiguration("run_realsense").perform(context))
        and source == "camera"
    )
    run_rviz = _as_bool(LaunchConfiguration("run_rviz").perform(context))
    run_static = _as_bool(LaunchConfiguration("run_esdf_medial_spheres").perform(context))
    run_dynamic = _as_bool(LaunchConfiguration("run_dynamic_obstacle_spheres").perform(context))
    run_fusion = _as_bool(LaunchConfiguration("run_obstacle_sphere_fusion").perform(context))
    run_robot_self_filter = _as_bool(
        LaunchConfiguration("run_robot_self_filter").perform(context))
    run_robot_collision_spheres = (
        run_robot_self_filter
        and _as_bool(LaunchConfiguration("run_robot_collision_spheres").perform(context))
    )
    run_rb10_joint_state_source = (
        run_robot_self_filter
        and source == "camera"
        and _as_bool(LaunchConfiguration("run_rb10_joint_state_source").perform(context))
    )
    robot_joint_state_topic = LaunchConfiguration(
        "robot_joint_state_topic").perform(context).strip()
    measured_joint_state_topic = LaunchConfiguration(
        "measured_joint_state_topic").perform(context).strip()
    robot_ip = LaunchConfiguration("robot_ip").perform(context).strip()
    robot_data_port = LaunchConfiguration("robot_data_port").perform(context).strip()
    robot_joint_state_publish_rate_hz = LaunchConfiguration(
        "robot_joint_state_publish_rate_hz").perform(context).strip()
    robot_urdf_path = LaunchConfiguration("robot_urdf_path").perform(context).strip()
    robot_sphere_config_path = LaunchConfiguration(
        "robot_sphere_config_path").perform(context).strip()
    self_filter_output_depth_topic = LaunchConfiguration(
        "self_filter_output_depth_topic").perform(context).strip()
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

    bag_path = ""
    bag_rate = ""
    if source == "bag":
        bag_path = bag_path_override or str(experiment.get("bag_path", "")).strip()
        if not bag_path:
            raise RuntimeError(
                "bag_path launch argument or /experiment bag_path must be set "
                "when source=bag")
        bag_path = str(Path(bag_path).expanduser())
        if not Path(bag_path).is_dir():
            raise RuntimeError(f"rosbag directory does not exist: {bag_path}")
        bag_rate = bag_rate_override or str(experiment.get("bag_rate", 1.0))

    # Rewinding /clock with ros2 bag --loop leaves the accumulated Nvblox map
    # and sphere tracks alive. When repetition is explicitly requested, run
    # one bag per child launch and respawn the complete processing graph so
    # every pass starts from clean state. A normal bag run stays in this launch
    # and exits cleanly after one pass without a restart gap.
    if source == "bag" and repeat_bag and not bag_cycle:
        cycle_process = ExecuteProcess(
            cmd=[
                "ros2", "launch", "rmp_camera",
                "d435_nvblox_dynamic_spheres.launch.py",
                f"experiment_config:={experiment_config}",
                "source:=bag",
                "use_sim_time:=true",
                f"bag_path:={bag_path}",
                f"bag_rate:={bag_rate}",
                "repeat_bag:=false",
                f"depth_image_topic:={depth_image_topic}",
                "run_realsense:=false",
                "run_rviz:=false",
                f"run_esdf_medial_spheres:={_bool_arg(run_static)}",
                f"run_dynamic_obstacle_spheres:={_bool_arg(run_dynamic)}",
                f"run_obstacle_sphere_fusion:={_bool_arg(run_fusion)}",
                f"run_robot_self_filter:={_bool_arg(run_robot_self_filter)}",
                f"run_robot_collision_spheres:={_bool_arg(run_robot_collision_spheres)}",
                f"run_rb10_joint_state_source:={_bool_arg(run_rb10_joint_state_source)}",
                f"robot_joint_state_topic:={robot_joint_state_topic}",
                f"measured_joint_state_topic:={measured_joint_state_topic}",
                f"robot_ip:={robot_ip}",
                f"robot_data_port:={robot_data_port}",
                f"robot_joint_state_publish_rate_hz:={robot_joint_state_publish_rate_hz}",
                f"robot_urdf_path:={robot_urdf_path}",
                f"robot_sphere_config_path:={robot_sphere_config_path}",
                f"self_filter_output_depth_topic:={self_filter_output_depth_topic}",
                f"log_level:={log_level}",
                "_bag_cycle:=true",
            ],
            output="screen",
            respawn=True,
            respawn_delay=1.0,
        )
        parent_actions = [cycle_process]
        if run_rviz:
            parent_actions.append(Node(
                package="rviz2", executable="rviz2", name="dynamic_spheres_rviz",
                output="screen",
                arguments=["-d", rviz_config, "--ros-args", "--log-level", log_level],
                parameters=[effective]))
        return parent_actions

    actions = []
    input_actions = []
    if source == "bag":
        bag_play = ExecuteProcess(
            cmd=["ros2", "bag", "play", bag_path, "--clock", "--rate",
                 bag_rate], output="screen")
        input_actions.extend([
            RegisterEventHandler(OnProcessExit(
                target_action=bag_play,
                on_exit=[EmitEvent(event=Shutdown(
                    reason="rosbag pass completed"))])),
            bag_play,
        ])

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
    if run_rb10_joint_state_source:
        actions.extend([
            Node(
                package="rmp_camera", executable="rb10_measured_joint_state_node",
                name="rb10_measured_joint_state_node", output="screen",
                parameters=[{
                    "robot_ip": robot_ip,
                    "data_port": int(robot_data_port),
                    "publish_topic": measured_joint_state_topic,
                    "publish_rate_hz": float(robot_joint_state_publish_rate_hz),
                    "use_sim_time": use_sim_time,
                }],
                arguments=["--ros-args", "--log-level", log_level]),
            Node(
                package="rmp_camera", executable="joint_state_normalizer_node",
                name="joint_state_normalizer_node", output="screen",
                parameters=[{
                    "input_topic": measured_joint_state_topic,
                    "source_priority": [measured_joint_state_topic],
                    "output_topic": robot_joint_state_topic,
                    "fallback_publish_rate_hz": 50.0,
                    "publish_zero_fallback": False,
                    "republish_latest": False,
                    "use_sim_time": use_sim_time,
                }],
                arguments=["--ros-args", "--log-level", log_level]),
        ])
    if run_robot_collision_spheres:
        actions.append(Node(
            package="rmp_camera", executable="robot_collision_sphere_node",
            name="robot_collision_sphere_node", output="screen",
            parameters=[
                experiment_config,
                {
                    "urdf_path": robot_urdf_path,
                    "sphere_config_path": robot_sphere_config_path,
                    "joint_state_topic": robot_joint_state_topic,
                    "use_sim_time": use_sim_time,
                },
            ],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_robot_self_filter:
        actions.append(Node(
            package="rmp_camera", executable="robot_depth_mask_node",
            name="robot_depth_mask_node", output="screen",
            parameters=[
                experiment_config,
                {
                    "input_depth_topic": depth_image_topic,
                    "camera_info_topic": "/camera0/camera/depth/camera_info",
                    "robot_sphere_marker_topic":
                        "/rmp_camera/robot_collision_sphere_markers",
                    "output_depth_topic": self_filter_output_depth_topic,
                    "use_sim_time": use_sim_time,
                },
            ],
            arguments=["--ros-args", "--log-level", log_level]))
    nvblox_depth_image_topic = (
        self_filter_output_depth_topic if run_robot_self_filter
        else depth_image_topic
    )
    actions.append(
        LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=[ComposableNode(
                name="nvblox_node", package="nvblox_ros", plugin="nvblox::NvbloxNode",
                remappings=[
                    ("camera_0/depth/image", nvblox_depth_image_topic),
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
    default_robot_urdf = package_file("rmp_camera", "urdf/rb10_1300e.urdf")
    default_robot_spheres = package_file(
        "rmp_camera", "config/collision_spheres.yaml")
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
        DeclareLaunchArgument(
            "bag_path", default_value="",
            description="Optional rosbag directory override for source=bag"),
        DeclareLaunchArgument(
            "bag_rate", default_value="",
            description="Optional rosbag playback-rate override"),
        DeclareLaunchArgument(
            "repeat_bag", default_value="false",
            description=(
                "Restart the complete processing graph after every bag pass; "
                "false exits cleanly after one pass")),
        DeclareLaunchArgument(
            "depth_image_topic",
            default_value="/camera0/realsense_splitter_node/output/depth",
            description=(
                "Depth image consumed by Nvblox; use raw depth when a bag "
                "has no splitter output")),
        DeclareLaunchArgument(
            "_bag_cycle", default_value="false",
            description="Internal flag for one resettable rosbag playback pass"),
        DeclareLaunchArgument("run_realsense", default_value="true"),
        DeclareLaunchArgument("run_rviz", default_value="true"),
        DeclareLaunchArgument("run_esdf_medial_spheres", default_value="true"),
        DeclareLaunchArgument("run_dynamic_obstacle_spheres", default_value="true"),
        DeclareLaunchArgument("run_obstacle_sphere_fusion", default_value="true"),
        DeclareLaunchArgument(
            "run_robot_self_filter", default_value="false",
            description="Filter robot-matching depth before Nvblox integration"),
        DeclareLaunchArgument(
            "run_robot_collision_spheres", default_value="true",
            description=(
                "Generate robot spheres from URDF and joint states; disable "
                "when replaying recorded markers")),
        DeclareLaunchArgument(
            "run_rb10_joint_state_source", default_value="true",
            description="Read measured RB10 joints directly in live-camera self-filter mode"),
        DeclareLaunchArgument(
            "robot_joint_state_topic",
            default_value="/rmp_camera/joint_states_urdf"),
        DeclareLaunchArgument(
            "measured_joint_state_topic",
            default_value="/rmp_camera/rb10_measured_joint_states"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.111.50"),
        DeclareLaunchArgument("robot_data_port", default_value="5001"),
        DeclareLaunchArgument(
            "robot_joint_state_publish_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("robot_urdf_path", default_value=default_robot_urdf),
        DeclareLaunchArgument(
            "robot_sphere_config_path", default_value=default_robot_spheres),
        DeclareLaunchArgument(
            "self_filter_output_depth_topic",
            default_value="/rmp_camera/robot_surface_filtered_depth/image_rect_raw"),
        DeclareLaunchArgument("log_level", default_value="info"),
        OpaqueFunction(function=_launch_setup),
    ])
