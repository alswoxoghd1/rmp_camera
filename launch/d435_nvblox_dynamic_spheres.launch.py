"""
D435 dynamic Nvblox experiment with static/dynamic sphere fusion.

All numeric experiment settings come from one YAML.  This launch starts one
RealSense camera, its emitter splitter, and one Nvblox component only; none of
the robot-arm/RMPflow pipeline is part of this experiment.
"""

from datetime import datetime
import math
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode


DEBUG_RECORD_TOPICS = (
    # Exact depth input to the self-filter and its diagnostic outputs.
    "/camera0/realsense_splitter_node/output/depth",
    "/camera0/camera/depth/camera_info",
    "/rmp_camera/edge_filtered_depth/image_rect_raw",
    "/rmp_camera/depth_edge_rejection_mask",
    "/rmp_camera/depth_edge_filter/status",
    "/rmp_camera/robot_surface_filtered_depth/image_rect_raw",
    "/rmp_camera/robot_predicted_depth/image_rect_raw",
    "/rmp_camera/robot_depth_mask_removed_points",
    # Robot state and both raw/calibrated collision geometry.
    "/rmp_camera/rb10_measured_joint_states",
    "/rmp_camera/joint_states_urdf",
    "/rmp_camera/robot_collision_sphere_markers",
    "/rmp_camera/calibrated_robot_collision_sphere_markers",
    "/tf",
    "/tf_static",
    # Nvblox detection before sphere generation.
    "/nvblox_node/dynamic_points",
    # Static sphere generation: source voxels, rejected voxels and result.
    "/rmp_camera/esdf_medial_inside_voxels",
    "/rmp_camera/esdf_medial_uncovered_voxels",
    "/rmp_camera/static_robot_rejected_voxels",
    "/rmp_camera/esdf_medial_sphere_markers",
    "/rmp_camera/esdf_medial_sphere_cloud",
    "/rmp_camera/static_sphere_support",
    "/rmp_camera/static_sphere_free_support",
    "/rmp_camera/static_sphere_result_status",
    "/rmp_camera/esdf_medial_query_bounds",
    # Dynamic generation: retained/rejected voxels, tracks and fused result.
    "/rmp_camera/dynamic_obstacle_voxels",
    "/rmp_camera/dynamic_uncovered_voxels",
    "/rmp_camera/dynamic_robot_rejected_voxels",
    "/rmp_camera/dynamic_obstacle_sphere_markers",
    "/rmp_camera/dynamic_obstacle_sphere_cloud",
    "/rmp_camera/dynamic_obstacle_sphere_cloud/status",
    "/rmp_camera/dynamic_obstacle_sphere_cloud/support",
    # Semantic-human branch: mask, registered points, independent spheres.
    "/camera0/segmentation/people_mask",
    "/camera0/segmentation/camera_info_resized",
    "/rmp_camera/human_points",
    "/rmp_camera/human_obstacle_sphere_markers",
    "/rmp_camera/human_sphere_cloud",
    "/rmp_camera/human_sphere_cloud/status",
    "/rmp_camera/human_sphere_cloud/support",
    "/rmp_camera/combined_obstacle_sphere_markers",
    "/rmp_camera/combined_obstacle_sphere_cloud",
    "/rmp_camera/combined_obstacle_sphere_cloud/status",
    # Throttled algorithm/synchronization diagnostics.
    "/rosout",
    "/parameter_events",
)

DEBUG_COLOR_TOPICS = (
    "/camera0/camera/color/image_raw",
    "/camera0/camera/color/camera_info",
)

# These publishers use sensor-data (best-effort) QoS.  The recorder is
# intentionally started before the camera/components so it does not miss the
# beginning of a run, but an unpublished topic would otherwise get a reliable
# subscription and become incompatible once these publishers appear.
DEBUG_BEST_EFFORT_TOPICS = (
    "/camera0/realsense_splitter_node/output/depth",
    "/camera0/camera/depth/camera_info",
    "/rmp_camera/edge_filtered_depth/image_rect_raw",
    "/rmp_camera/depth_edge_rejection_mask",
    "/rmp_camera/robot_surface_filtered_depth/image_rect_raw",
    "/rmp_camera/robot_predicted_depth/image_rect_raw",
    "/nvblox_node/dynamic_points",
    "/camera0/segmentation/people_mask",
    "/camera0/segmentation/camera_info_resized",
    "/rmp_camera/human_points",
    "/camera0/camera/color/image_raw",
    "/camera0/camera/color/camera_info",
)


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _bool_arg(value):
    return "true" if value else "false"


def _depth_processing_topics(raw_topic, edge_enabled, robot_enabled, robot_output):
    """Human/visibility receive validated pre-robot depth; Nvblox may mask robot."""
    validated = "/rmp_camera/edge_filtered_depth/image_rect_raw" if edge_enabled else raw_topic
    if edge_enabled and raw_topic == validated:
        raise RuntimeError("depth_image_topic must be RAW input, not edge filter output")
    if robot_enabled and robot_output == validated:
        raise RuntimeError("robot self-filter input and output topics must differ")
    return validated, robot_output if robot_enabled else validated


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


def _node_parameter_section(config_path, node_name):
    with open(config_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    try:
        return data[node_name]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"{config_path} must contain {node_name}/ros__parameters") from exc


def _trimmed_esdf_aabb(static_parameters, trims_cm):
    axes = ("x", "y", "z")
    minimum = {
        axis: float(static_parameters[f"aabb_min_{axis}_m"])
        for axis in axes
    }
    size = {
        axis: float(static_parameters[f"aabb_size_{axis}_m"])
        for axis in axes
    }
    result = {}
    for axis in axes:
        trim_min_m = 0.01 * float(trims_cm[f"esdf_trim_{axis}_min_cm"])
        trim_max_m = 0.01 * float(trims_cm[f"esdf_trim_{axis}_max_cm"])
        if (
            not math.isfinite(trim_min_m)
            or not math.isfinite(trim_max_m)
            or trim_min_m < 0.0
            or trim_max_m < 0.0
        ):
            raise RuntimeError(
                f"ESDF {axis}-face trims must be finite and non-negative")
        trimmed_size = size[axis] - trim_min_m - trim_max_m
        if trimmed_size <= 0.0:
            raise RuntimeError(
                f"ESDF {axis}-face trims remove the complete {axis} extent "
                f"({size[axis]:.3f} m)")
        result[f"aabb_min_{axis}_m"] = minimum[axis] + trim_min_m
        result[f"aabb_size_{axis}_m"] = trimmed_size
    return result


def _launch_setup(context):
    experiment_config = LaunchConfiguration("experiment_config").perform(context)
    experiment = _experiment_section(experiment_config)
    edge_override = LaunchConfiguration("run_depth_edge_filter").perform(context).strip()
    run_depth_edge_filter = _as_bool(edge_override if edge_override else
                                    experiment.get("run_depth_edge_filter", True))
    human_latency_override = LaunchConfiguration('human_low_latency').perform(context).strip()
    human_low_latency = _as_bool(human_latency_override if human_latency_override else
                                 experiment.get('human_low_latency', True))
    human_compute_env = ({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                          'MKL_NUM_THREADS': '1'} if human_low_latency else {})
    refinement_override = LaunchConfiguration('async_sphere_refinement').perform(context).strip()
    async_refinement = _as_bool(refinement_override if refinement_override else
                               experiment.get('async_sphere_refinement', False))
    improvement_flags = {}
    for key in ('latest_sphere_input', 'component_sphere_tracking', 'fusion_coverage_guard',
                'local_width_sphere_cover', 'integrated_sphere_cover'):
        override = LaunchConfiguration(key).perform(context).strip()
        improvement_flags[key] = _as_bool(override if override else experiment.get(key, False))
    latest_input = improvement_flags['latest_sphere_input']
    component_tracking = improvement_flags['component_sphere_tracking']
    coverage_guard = improvement_flags['fusion_coverage_guard']
    local_width_cover = improvement_flags['local_width_sphere_cover']
    integrated_cover = improvement_flags['integrated_sphere_cover']
    observation_override = LaunchConfiguration('sphere_observation_guard').perform(context).strip()
    observation_guard = observation_override or str(experiment.get('sphere_observation_guard','off'))
    if observation_guard not in ('off','audit','observed'):
        raise ValueError('sphere_observation_guard must be off, audit or observed')
    sphere_compute_env = ({'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                           'MKL_NUM_THREADS': '1'} if async_refinement else {})
    static_parameters = _node_parameter_section(
        experiment_config, "/esdf_medial_sphere_node")
    empty_check_override = LaunchConfiguration("static_empty_check_rate_hz").perform(context).strip()
    empty_check_rate = (float(empty_check_override) if empty_check_override
                        else float(static_parameters.get("empty_check_rate_hz", 0.0)))
    fusion_parameters = _node_parameter_section(experiment_config, "/obstacle_sphere_fusion_node")
    fast_empty_override = LaunchConfiguration("static_fast_empty_clear").perform(context).strip()
    fast_empty_clear = _as_bool(fast_empty_override if fast_empty_override else
                               fusion_parameters.get("static_fast_empty_clear", True))
    refit_override = LaunchConfiguration("human_static_refit_enabled").perform(context).strip()
    human_static_refit_enabled = _as_bool(refit_override if refit_override else
        fusion_parameters.get("human_static_refit_enabled", True))
    dynamic_parameters = _node_parameter_section(
        experiment_config, "/dynamic_obstacle_sphere_node")
    human_projection_parameters = _node_parameter_section(
        experiment_config, "/human_depth_projection_node")
    human_sphere_parameters = _node_parameter_section(
        experiment_config, "/human_obstacle_sphere_node")
    human_static_parameters = _node_parameter_section(
        experiment_config, "/human_static_filter")
    occlusion_override = LaunchConfiguration("human_static_occlusion_hold_enabled").perform(context).strip()
    human_occlusion_hold = _as_bool(occlusion_override if occlusion_override else
        human_static_parameters.get("human_static_occlusion_hold_enabled", False))
    human_static_parameters["human_static_occlusion_hold_enabled"] = human_occlusion_hold
    esdf_trim_names = tuple(
        f"esdf_trim_{axis}_{side}_cm"
        for axis in ("x", "y", "z")
        for side in ("min", "max")
    )
    esdf_trims_cm = {
        name: LaunchConfiguration(name).perform(context).strip()
        for name in esdf_trim_names
    }
    esdf_bounds_override = _trimmed_esdf_aabb(
        static_parameters, esdf_trims_cm)
    source_override = LaunchConfiguration("source").perform(context).strip()
    sim_override = LaunchConfiguration("use_sim_time").perform(context).strip()
    bag_path_override = LaunchConfiguration("bag_path").perform(context).strip()
    bag_rate_override = LaunchConfiguration("bag_rate").perform(context).strip()
    repeat_bag = _as_bool(LaunchConfiguration("repeat_bag").perform(context))
    record_debug_data = _as_bool(
        LaunchConfiguration("record_debug_data").perform(context))
    debug_record_color = _as_bool(
        LaunchConfiguration("debug_record_color").perform(context))
    debug_bag_dir = Path(
        LaunchConfiguration("debug_bag_dir").perform(context).strip()
    ).expanduser()
    debug_bag_name = LaunchConfiguration(
        "debug_bag_name").perform(context).strip()
    debug_bag_max_size_mb = int(LaunchConfiguration(
        "debug_bag_max_size_mb").perform(context).strip())
    if debug_bag_max_size_mb <= 0:
        raise RuntimeError("debug_bag_max_size_mb must be positive")
    if not debug_bag_name or Path(debug_bag_name).name != debug_bag_name:
        raise RuntimeError("debug_bag_name must be one non-empty directory name")
    depth_image_topic = LaunchConfiguration(
        "depth_image_topic").perform(context).strip()
    bag_cycle = _as_bool(LaunchConfiguration("_bag_cycle").perform(context))
    source = source_override or str(experiment["source"])
    use_sim_time = _as_bool(sim_override) if sim_override else _as_bool(experiment["use_sim_time"])
    if source == "bag":
        use_sim_time = True
    if source not in ("camera", "bag"):
        raise RuntimeError("source must be 'camera' or 'bag'")
    if record_debug_data and source == "bag" and repeat_bag and not bag_cycle:
        raise RuntimeError(
            "record_debug_data cannot be combined with repeat_bag; record "
            "one deterministic bag playback instead")

    run_realsense = (
        _as_bool(LaunchConfiguration("run_realsense").perform(context))
        and source == "camera"
    )
    run_rviz = _as_bool(LaunchConfiguration("run_rviz").perform(context))
    run_static = _as_bool(LaunchConfiguration("run_esdf_medial_spheres").perform(context))
    run_dynamic = _as_bool(LaunchConfiguration("run_dynamic_obstacle_spheres").perform(context))
    run_people_segmentation = _as_bool(
        LaunchConfiguration("run_people_segmentation").perform(context))
    run_human_spheres = _as_bool(
        LaunchConfiguration("run_human_obstacle_spheres").perform(context))
    run_human_static_filter = run_human_spheres and _as_bool(
        LaunchConfiguration("run_human_static_filter").perform(context))
    run_fusion = _as_bool(LaunchConfiguration("run_obstacle_sphere_fusion").perform(context))
    validation_only = _as_bool(
        LaunchConfiguration("offline_self_filter_validation_only").perform(context)
    )
    run_offline_validation = validation_only or _as_bool(
        LaunchConfiguration("run_offline_self_filter_validation").perform(context)
    )
    if run_offline_validation and source != "bag":
        raise RuntimeError("offline self-filter validation requires source=bag")
    run_robot_self_filter = _as_bool(
        LaunchConfiguration("run_robot_self_filter").perform(context))
    run_robot_obstacle_filters = _as_bool(
        LaunchConfiguration("run_robot_obstacle_filters").perform(context))
    run_robot_sphere_marker_correction = _as_bool(
        LaunchConfiguration("run_robot_sphere_marker_correction").perform(context))
    # The collision-sphere stream is also used by the static/dynamic component
    # filters.  Keep it independent from the expensive per-pixel depth mask so
    # run_robot_self_filter:=false does not make those filters fail closed.
    run_robot_collision_spheres = _as_bool(
        LaunchConfiguration("run_robot_collision_spheres").perform(context))
    run_rb10_joint_state_source = (
        run_robot_collision_spheres
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
    validated_depth_topic, nvblox_depth_image_topic = _depth_processing_topics(
        depth_image_topic, run_depth_edge_filter, run_robot_self_filter,
        self_filter_output_depth_topic)
    raw_robot_marker_topic = "/rmp_camera/robot_collision_sphere_markers"
    calibrated_robot_marker_topic = (
        "/rmp_camera/calibrated_robot_collision_sphere_markers")
    robot_marker_topic = (
        calibrated_robot_marker_topic
        if run_robot_sphere_marker_correction
        else raw_robot_marker_topic
    )
    log_level = LaunchConfiguration("log_level").perform(context)
    people_model_path = str(Path(LaunchConfiguration(
        "people_model_path").perform(context).strip()).expanduser())
    if run_people_segmentation and not Path(people_model_path).is_file():
        raise RuntimeError(
            f"PeopleSemSegNet TensorRT engine does not exist: {people_model_path}")
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

    validation_action = None
    if run_offline_validation:
        validation_cmd = [
            "ros2", "run", "rmp_camera", "validate_robot_self_filter_bag",
            bag_path,
            "--sample-every",
            LaunchConfiguration("offline_self_filter_sample_every").perform(context),
            "--max-frames",
            LaunchConfiguration("offline_self_filter_max_frames").perform(context),
            "--front-tolerance-m",
            LaunchConfiguration(
                "offline_self_filter_front_tolerance_m").perform(context),
            "--back-tolerance-m",
            LaunchConfiguration(
                "offline_self_filter_back_tolerance_m").perform(context),
            "--surface-sphere-padding-m",
            LaunchConfiguration(
                "offline_self_filter_surface_sphere_padding_m").perform(context),
        ]
        debug_dir = LaunchConfiguration(
            "offline_self_filter_debug_dir").perform(context).strip()
        json_report = LaunchConfiguration(
            "offline_self_filter_json_report").perform(context).strip()
        if debug_dir:
            validation_cmd.extend(["--debug-dir", debug_dir])
        if json_report:
            validation_cmd.extend(["--json-report", json_report])
        validation_action = ExecuteProcess(cmd=validation_cmd, output="screen")
        if validation_only:
            return [validation_action]

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
                f"run_depth_edge_filter:={_bool_arg(run_depth_edge_filter)}",
                f"human_low_latency:={_bool_arg(human_low_latency)}",
                f"async_sphere_refinement:={_bool_arg(async_refinement)}",
                *[f"{key}:={_bool_arg(value)}" for key, value in improvement_flags.items()],
                f"sphere_observation_guard:={observation_guard}",
                "run_realsense:=false",
                "run_rviz:=false",
                f"run_esdf_medial_spheres:={_bool_arg(run_static)}",
                f"run_dynamic_obstacle_spheres:={_bool_arg(run_dynamic)}",
                f"run_people_segmentation:={_bool_arg(run_people_segmentation)}",
                f"run_human_obstacle_spheres:={_bool_arg(run_human_spheres)}",
                f"run_human_static_filter:={_bool_arg(run_human_static_filter)}",
                f"human_static_refit_enabled:={_bool_arg(human_static_refit_enabled)}",
                f"human_static_occlusion_hold_enabled:={_bool_arg(human_occlusion_hold)}",
                f"static_fast_empty_clear:={_bool_arg(fast_empty_clear)}",
                f"static_empty_check_rate_hz:={empty_check_rate}",
                f"people_model_path:={people_model_path}",
                f"run_obstacle_sphere_fusion:={_bool_arg(run_fusion)}",
                f"run_robot_self_filter:={_bool_arg(run_robot_self_filter)}",
                f"run_robot_obstacle_filters:="
                f"{_bool_arg(run_robot_obstacle_filters)}",
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
                *[
                    f"{name}:={esdf_trims_cm[name]}"
                    for name in esdf_trim_names
                ],
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
    if validation_action is not None:
        actions.append(validation_action)
    if record_debug_data:
        debug_bag_dir.mkdir(parents=True, exist_ok=True)
        debug_bag_path = debug_bag_dir / debug_bag_name
        if debug_bag_path.exists():
            raise RuntimeError(
                f"debug bag output already exists: {debug_bag_path}")
        debug_topics = list(DEBUG_RECORD_TOPICS)
        if debug_record_color:
            debug_topics.extend(DEBUG_COLOR_TOPICS)
        manifest_path = debug_bag_dir / f"{debug_bag_name}_manifest.yaml"
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "bag_path": str(debug_bag_path),
            "experiment_config": str(Path(experiment_config).resolve()),
            "source": source,
            "use_sim_time": use_sim_time,
            "run_robot_self_filter": run_robot_self_filter,
            "run_depth_edge_filter": run_depth_edge_filter,
            "human_low_latency": human_low_latency,
            "async_sphere_refinement": async_refinement,
            **improvement_flags,
            'sphere_observation_guard': observation_guard,
            "validated_depth_topic": validated_depth_topic,
            "nvblox_depth_topic": nvblox_depth_image_topic,
            "run_robot_obstacle_filters": run_robot_obstacle_filters,
            "run_people_segmentation": run_people_segmentation,
            "run_human_obstacle_spheres": run_human_spheres,
            "human_static_refit_enabled": human_static_refit_enabled,
            "human_static_occlusion_hold_enabled": human_occlusion_hold,
            "static_fast_empty_clear": fast_empty_clear,
            "static_empty_check_rate_hz": empty_check_rate,
            "people_model_path": people_model_path,
            "record_color": debug_record_color,
            "max_bag_file_size_mb": debug_bag_max_size_mb,
            "esdf_trim_cm": esdf_trims_cm,
            "effective_esdf_aabb_m": esdf_bounds_override,
            "topics": debug_topics,
        }
        with open(experiment_config, "r", encoding="utf-8") as stream:
            manifest["experiment_config_snapshot"] = yaml.safe_load(stream) or {}
        with open(manifest_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)
        qos_overrides_path = (
            debug_bag_dir / f"{debug_bag_name}_qos_overrides.yaml")
        qos_overrides = {
            topic: {
                "history": "keep_last",
                "depth": 10,
                "reliability": "best_effort",
                "durability": "volatile",
            }
            for topic in DEBUG_BEST_EFFORT_TOPICS
            if topic in debug_topics
        }
        with open(qos_overrides_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(qos_overrides, stream, sort_keys=False)
        recorder_cmd = [
            "ros2", "bag", "record",
            "--output", str(debug_bag_path),
            "--max-bag-size", str(debug_bag_max_size_mb * 1024 * 1024),
            "--include-unpublished-topics",
            "--qos-profile-overrides-path", str(qos_overrides_path),
        ]
        if use_sim_time:
            recorder_cmd.append("--use-sim-time")
        recorder_cmd.extend(debug_topics)
        actions.extend([
            LogInfo(msg=[
                "Recording dynamic-sphere diagnostics to: ",
                str(debug_bag_path),
                " (stop soon after reproduction; available disk is limited)",
            ]),
            ExecuteProcess(cmd=recorder_cmd, output="screen"),
        ])
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
                # realsense2_camera 4.57.x stream-profile parameter names.
                # Explicit overrides take precedence over the legacy profile
                # keys in the nvblox emitter-flashing YAML.
                "depth_module.depth_profile": str(experiment.get(
                    "realsense_depth_profile", "848x480x60")),
                "rgb_camera.color_profile": str(experiment.get(
                    "realsense_color_profile", "1280x720x30")),
                # Only publish depth and RGB. The NVIDIA depth splitter only
                # needs the depth image and its metadata; IR/IMU are not needed.
                "enable_depth": _as_bool(
                    experiment.get("realsense_enable_depth", True)),
                "enable_color": _as_bool(
                    experiment.get("realsense_enable_color", True)),
                "enable_infra": _as_bool(
                    experiment.get("realsense_enable_infra", False)),
                "enable_infra1": _as_bool(
                    experiment.get("realsense_enable_infra1", False)),
                "enable_infra2": _as_bool(
                    experiment.get("realsense_enable_infra2", False)),
                "enable_accel": _as_bool(
                    experiment.get("realsense_enable_accel", False)),
                "enable_gyro": _as_bool(
                    experiment.get("realsense_enable_gyro", False)),
                "enable_motion": _as_bool(
                    experiment.get("realsense_enable_motion", False)),
                "pointcloud.enable": _as_bool(
                    experiment.get("realsense_enable_pointcloud", False)),
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
    if run_people_segmentation:
        segmentation_launch = package_file(
            "nvblox_examples_bringup", "launch/perception/segmentation.launch.py")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(segmentation_launch),
            launch_arguments={
                "people_segmentation": "peoplesemsegnet_vanilla",
                "num_cameras": "1",
                "namespace_list": '["camera0"]',
                "input_topic_list": '["/camera0/camera/color/image_raw"]',
                "input_camera_info_topic_list":
                    '["/camera0/camera/color/camera_info"]',
                "output_resized_image_topic_list":
                    '["/camera0/segmentation/image_resized"]',
                "output_resized_camera_info_topic_list":
                    '["/camera0/segmentation/camera_info_resized"]',
                "vanilla_engine_file_path": people_model_path,
                "container_name": "rmp_camera_people_segmentation_container",
                "one_container_per_camera": "True",
                "log_level": log_level,
            }.items(),
        ))
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
    if run_robot_sphere_marker_correction:
        actions.append(Node(
            package="rmp_camera",
            executable="robot_sphere_marker_correction_node",
            name="robot_sphere_marker_correction_node",
            output="screen",
            parameters=[experiment_config, effective],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_depth_edge_filter:
        actions.append(Node(
            package="rmp_camera", executable="depth_edge_filter_node",
            name="depth_edge_filter_node", output="screen",
            parameters=[experiment_config, effective, {
                "input_depth_topic": depth_image_topic,
                "output_depth_topic": validated_depth_topic,
                "publish_rejection_mask": record_debug_data,
            }],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_robot_self_filter:
        actions.append(Node(
            package="rmp_camera", executable="robot_depth_mask_node",
            name="robot_depth_mask_node", output="screen",
            parameters=[
                experiment_config,
                {
                    "input_depth_topic": validated_depth_topic,
                    "camera_info_topic": "/camera0/camera/depth/camera_info",
                    "robot_sphere_marker_topic": robot_marker_topic,
                    "output_depth_topic": self_filter_output_depth_topic,
                    "predicted_depth_topic":
                        "/rmp_camera/robot_predicted_depth/image_rect_raw",
                    "removed_points_topic":
                        "/rmp_camera/robot_depth_mask_removed_points",
                    "publish_removed_points": _as_bool(experiment.get(
                        "robot_depth_publish_removed_points", True))
                        or record_debug_data,
                    "publish_predicted_depth": _as_bool(experiment.get(
                        "robot_depth_publish_predicted_depth", True))
                        or record_debug_data,
                    "filter_mode": "volume",
                    "surface_sphere_padding_m": float(experiment.get(
                        "robot_depth_surface_sphere_padding_m", 0.0)),
                    "surface_front_tolerance_m": float(experiment.get(
                        "robot_depth_surface_front_tolerance_m", 0.02)),
                    "surface_back_tolerance_m": float(experiment.get(
                        "robot_depth_surface_back_tolerance_m", 0.03)),
                    "mask_shadow_behind_robot": _as_bool(experiment.get(
                        "robot_depth_mask_shadow_behind_robot", False)),
                    "max_rate_hz": float(experiment.get(
                        "robot_depth_mask_rate_hz", 30.0)),
                    "use_sim_time": use_sim_time,
                },
            ],
            arguments=["--ros-args", "--log-level", log_level]))
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
            parameters=[experiment_config, effective, human_static_parameters, {
                **esdf_bounds_override,
                "empty_check_rate_hz": empty_check_rate,
                "enable_local_width_cover": local_width_cover,
                "human_static_filter_enabled": run_human_static_filter,
                "human_static_depth_topic": validated_depth_topic,
                "robot_sphere_marker_topic": robot_marker_topic,
                "robot_component_filter_enabled": run_robot_obstacle_filters,
                "publish_debug_clouds": record_debug_data or _as_bool(
                    static_parameters.get("publish_debug_clouds", True)),
            }],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_dynamic:
        actions.append(Node(
            package="rmp_camera", executable="dynamic_obstacle_sphere_node",
            name="dynamic_obstacle_sphere_node", output="screen",
            additional_env=sphere_compute_env,
            parameters=[experiment_config, effective, {
                "dynamic_async_refinement_enabled": async_refinement,
                "dynamic_local_width_cover_enabled": local_width_cover,
                "dynamic_local_width_integrated": integrated_cover,
                "dynamic_observation_guard_mode": observation_guard,
                "dynamic_observation_depth_topic": validated_depth_topic,
                "dynamic_latest_input_enabled": latest_input,
                "dynamic_component_tracking_enabled": component_tracking,
                "publish_fusion_support": coverage_guard,
                "robot_sphere_marker_topic": robot_marker_topic,
                "robot_component_filter_enabled": run_robot_obstacle_filters,
                "publish_debug_clouds": record_debug_data or _as_bool(
                    dynamic_parameters.get("publish_debug_clouds", True)),
            }],
            arguments=["--ros-args", "--log-level", log_level]))
    if run_human_spheres:
        actions.extend([
            Node(
                package="rmp_camera", executable="human_depth_projection_node",
                name="human_depth_projection_node", output="screen",
                additional_env=human_compute_env,
                parameters=[human_projection_parameters, effective, {
                    "depth_image_topic": validated_depth_topic,
                    "prefer_latest_frames": human_low_latency,
                }],
                arguments=["--ros-args", "--log-level", log_level]),
            Node(
                package="rmp_camera", executable="dynamic_obstacle_sphere_node",
                name="human_obstacle_sphere_node", output="screen",
                additional_env={**human_compute_env, **sphere_compute_env},
                parameters=[human_sphere_parameters, effective, {
                    "dynamic_async_refinement_enabled": async_refinement,
                    "dynamic_local_width_cover_enabled": local_width_cover,
                    "dynamic_local_width_integrated": integrated_cover,
                    "dynamic_observation_guard_mode": observation_guard,
                    "dynamic_observation_depth_topic": validated_depth_topic,
                    "dynamic_latest_input_enabled": latest_input,
                    "dynamic_component_tracking_enabled": component_tracking,
                    "publish_fusion_support": coverage_guard,
                    "robot_sphere_marker_topic": robot_marker_topic,
                    "robot_component_filter_enabled": run_robot_obstacle_filters,
                    "publish_debug_clouds": record_debug_data or _as_bool(
                        human_sphere_parameters.get("publish_debug_clouds", False)),
                }],
                arguments=["--ros-args", "--log-level", log_level]),
        ])
    if run_fusion:
        actions.append(Node(
            package="rmp_camera", executable="obstacle_sphere_fusion_node",
            name="obstacle_sphere_fusion_node", output="screen",
            parameters=[experiment_config, effective, human_static_parameters, {
                "fusion_coverage_guard_enabled": coverage_guard,
                "human_static_filter_enabled": run_human_static_filter,
                "human_static_depth_topic": validated_depth_topic,
                "static_fast_empty_clear": fast_empty_clear,
                "human_static_refit_enabled": human_static_refit_enabled,
                "human_static_refit_min_radius_m": float(static_parameters.get(
                    "min_raw_sphere_radius_m", 0.04)),
                "human_static_refit_coverage_tolerance_m": float(static_parameters.get(
                    "coverage_tolerance_m", 0.02)),
                "robot_sphere_marker_topic": robot_marker_topic,
                "robot_overlap_filter_enabled": run_robot_obstacle_filters,
            }],
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
    isaac_ros_workspace = str(Path.home() / "rb_ws")
    default_config = package_file(
        "rmp_camera", "config/d435_nvblox_dynamic_experiment.yaml")
    default_robot_urdf = package_file("rmp_camera", "urdf/rb10_1300e.urdf")
    default_robot_spheres = package_file(
        "rmp_camera", "config/collision_spheres.yaml")
    return LaunchDescription([
        DeclareLaunchArgument(
            "run_human_static_filter", default_value="true",
            description="Exclude semantic human voxels from static spheres and clear depth-confirmed ghosts"),
        DeclareLaunchArgument(
            "static_fast_empty_clear", default_value="",
            description="Apply valid observed empty static results immediately; errors never clear obstacles"),
        DeclareLaunchArgument(
            "human_static_refit_enabled", default_value="",
            description="Tighten mixed static balls around retained nonhuman support without adding spheres"),
        DeclareLaunchArgument(
            "human_static_occlusion_hold_enabled", default_value="",
            description="Keep recent human labels with reoccupation veto; older history requires current human occlusion"),
        DeclareLaunchArgument(
            "static_empty_check_rate_hz", default_value="",
            description="Override cheap static disappearance-check rate; full sphere optimization rate is unchanged"),
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
        SetEnvironmentVariable(name="ISAAC_ROS_WS", value=isaac_ros_workspace),
        DeclareLaunchArgument(
            "experiment_config", default_value=default_config,
            description="Single YAML controlling camera, Nvblox, sphere, and fusion settings"),
        DeclareLaunchArgument(
            "run_depth_edge_filter", default_value="",
            description="Single-frame intermediate-depth rejection; empty uses /experiment setting"),
        DeclareLaunchArgument(
            "human_low_latency", default_value="",
            description="Prefer fresh synchronized human input and bound per-node CPU threads"),
        DeclareLaunchArgument(
            "async_sphere_refinement", default_value="",
            description="Refine shapes in bounded workers; reuse only after current-voxel validation"),
        DeclareLaunchArgument("latest_sphere_input", default_value="",
            description="Decode the latest synchronized point frame with NumPy, without a stale backlog"),
        DeclareLaunchArgument("component_sphere_tracking", default_value="",
            description="Associate component shape before spheres; gate incompatible radius changes"),
        DeclareLaunchArgument("fusion_coverage_guard", default_value="",
            description="Remove overlapping cross-source copies only when their support remains covered"),
        DeclareLaunchArgument("local_width_sphere_cover", default_value="",
            description="Reconstruct fewer variable-radius spheres from local voxel widths with coverage guards"),
        DeclareLaunchArgument("integrated_sphere_cover", default_value="",
            description="Share current coverage masks; choose local candidates before a single refinement pass"),
        DeclareLaunchArgument("sphere_observation_guard", default_value="",
            description="off|audit|observed: compare registered current-depth evidence with voxel density"),
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
            "record_debug_data", default_value="false",
            description=(
                "Record depth, robot geometry, Nvblox detections and all "
                "sphere stages for offline diagnosis")),
        DeclareLaunchArgument(
            "debug_bag_dir",
            default_value=str(Path.home() / "bags" / "nvblox_debug"),
            description="Parent directory for diagnostic bags"),
        DeclareLaunchArgument(
            "debug_bag_name",
            default_value=(
                "dynamic_spheres_debug_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")),
            description="Unique directory name for this diagnostic bag"),
        DeclareLaunchArgument(
            "debug_bag_max_size_mb", default_value="2048",
            description="Split diagnostic SQLite files at this size in MB"),
        DeclareLaunchArgument(
            "debug_record_color", default_value="true",
            description=(
                "Record raw color images for visual validation; consumes "
                "about 2.8 GB/min at the configured 960x540x30 profile")),
        DeclareLaunchArgument(
            "depth_image_topic",
            default_value="/camera0/realsense_splitter_node/output/depth",
            description=(
                "Raw depth input before optional edge/robot filters; use raw depth when a bag "
                "has no splitter output")),
        DeclareLaunchArgument(
            "_bag_cycle", default_value="false",
            description="Internal flag for one resettable rosbag playback pass"),
        DeclareLaunchArgument("run_realsense", default_value="true"),
        DeclareLaunchArgument("run_rviz", default_value="true"),
        DeclareLaunchArgument("run_esdf_medial_spheres", default_value="true"),
        DeclareLaunchArgument(
            "esdf_trim_x_min_cm", default_value="100.0",
            description="Trim centimetres from the ESDF AABB -X face"),
        DeclareLaunchArgument(
            "esdf_trim_x_max_cm", default_value="10.0",
            description="Trim centimetres from the ESDF AABB +X face"),
        DeclareLaunchArgument(
            "esdf_trim_y_min_cm", default_value="50.0",
            description="Trim centimetres from the ESDF AABB -Y face"),
        DeclareLaunchArgument(
            "esdf_trim_y_max_cm", default_value="50.0",
            description="Trim centimetres from the ESDF AABB +Y face"),
        DeclareLaunchArgument(
            "esdf_trim_z_min_cm", default_value="0.0",
            description="Trim centimetres from the ESDF AABB bottom face"),
        DeclareLaunchArgument(
            "esdf_trim_z_max_cm", default_value="20.0",
            description="Trim centimetres from the ESDF AABB top face"),
        DeclareLaunchArgument("run_dynamic_obstacle_spheres", default_value="true"),
        DeclareLaunchArgument(
            "run_people_segmentation", default_value="false",
            description="Run GPU PeopleSemSegNet on the D435 RGB stream"),
        DeclareLaunchArgument(
            "run_human_obstacle_spheres", default_value="false",
            description=(
                "Project semantic people masks onto depth and publish an "
                "independent tracked human sphere stream")),
        DeclareLaunchArgument(
            "people_model_path",
            default_value=str(
                Path.home() / "rb_ws" / "isaac_ros_assets" / "models"
                / "peoplesemsegnet"
                / "deployable_quantized_vanilla_unet_onnx_v2.0"
                / "1" / "model.plan"),
            description="PeopleSemSegNet TensorRT engine path"),
        DeclareLaunchArgument("run_obstacle_sphere_fusion", default_value="true"),
        DeclareLaunchArgument(
            "run_robot_self_filter", default_value="false",
            description="Filter robot-matching depth before Nvblox integration"),
        DeclareLaunchArgument(
            "run_robot_obstacle_filters", default_value="false",
            description=(
                "Filter static, dynamic and fused obstacle spheres against "
                "robot collision spheres; disable for obstacle-only runs "
                "without a robot joint-state source")),
        DeclareLaunchArgument(
            "run_robot_collision_spheres", default_value="false",
            description=(
                "Generate robot spheres from URDF and joint states; disable "
                "when replaying recorded markers")),
        DeclareLaunchArgument(
            "run_robot_sphere_marker_correction", default_value="false",
            description=(
                "Optionally apply a non-zero rigid correction to robot markers")),
        DeclareLaunchArgument(
            "run_rb10_joint_state_source", default_value="false",
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
        DeclareLaunchArgument(
            "run_offline_self_filter_validation", default_value="false",
            description="Run recorded-TF validation alongside bag playback"),
        DeclareLaunchArgument(
            "offline_self_filter_validation_only", default_value="false",
            description="Run only offline validation; requires source=bag"),
        DeclareLaunchArgument(
            "offline_self_filter_sample_every", default_value="10"),
        DeclareLaunchArgument(
            "offline_self_filter_max_frames", default_value="0"),
        DeclareLaunchArgument(
            "offline_self_filter_front_tolerance_m", default_value="0.02"),
        DeclareLaunchArgument(
            "offline_self_filter_back_tolerance_m", default_value="0.03"),
        DeclareLaunchArgument(
            "offline_self_filter_surface_sphere_padding_m", default_value="0.0"),
        DeclareLaunchArgument(
            "offline_self_filter_debug_dir",
            default_value="/tmp/rmp_camera_self_filter_validation"),
        DeclareLaunchArgument(
            "offline_self_filter_json_report",
            default_value="/tmp/rmp_camera_self_filter_validation/report.json"),
        DeclareLaunchArgument("log_level", default_value="info"),
        OpaqueFunction(function=_launch_setup),
    ])
