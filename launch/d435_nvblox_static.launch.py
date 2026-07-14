from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    EnvironmentVariable,
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def generate_launch_description():
    rosdeps_nv_lib = str(
        Path.home() / ".local/opt/rosdeps_nv/root/usr/lib/x86_64-linux-gnu"
    )
    cuda12_runtime_lib = str(
        Path.home() / ".local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    )

    container_name = LaunchConfiguration("container_name")
    log_level = LaunchConfiguration("log_level")
    source = LaunchConfiguration("source")
    bag_path = LaunchConfiguration("bag_path")
    bag_rate = LaunchConfiguration("bag_rate")
    use_sim_time = LaunchConfiguration("use_sim_time")
    effective_use_sim_time = PythonExpression(
        ["'", source, "' == 'bag' or '", use_sim_time, "'.lower() in ('true', '1')"]
    )
    esdf_mode = LaunchConfiguration("esdf_mode")
    run_realsense = LaunchConfiguration("run_realsense")
    run_realsense_for_source = PythonExpression(
        ["'", source, "' == 'camera' and '", run_realsense, "'.lower() in ('true', '1')"]
    )
    run_collision_spheres = LaunchConfiguration("run_collision_spheres")
    run_rb10_measured_joint_state_publisher = LaunchConfiguration(
        "run_rb10_measured_joint_state_publisher"
    )
    run_rb10_measured_joint_state_for_source = PythonExpression(
        [
            "'",
            source,
            "' == 'camera' and '",
            run_rb10_measured_joint_state_publisher,
            "'.lower() in ('true', '1')",
        ]
    )
    run_joint_state_normalizer = LaunchConfiguration("run_joint_state_normalizer")
    run_robot_state_publisher = LaunchConfiguration("run_robot_state_publisher")
    run_robot_depth_mask = LaunchConfiguration("run_robot_depth_mask")
    run_depth_to_pointcloud = LaunchConfiguration("run_depth_to_pointcloud")
    run_vision_obstacle_pipeline = LaunchConfiguration("run_vision_obstacle_pipeline")
    run_depth_to_pointcloud_with_vision = PythonExpression(
        [
            "'",
            run_depth_to_pointcloud,
            "'.lower() in ('true', '1') and '",
            run_vision_obstacle_pipeline,
            "'.lower() in ('true', '1')",
        ]
    )
    run_camera_closest_obstacles = LaunchConfiguration("run_camera_closest_obstacles")
    run_obstacle_body_spheres = LaunchConfiguration("run_obstacle_body_spheres")
    run_esdf_collision_query = LaunchConfiguration("run_esdf_collision_query")
    run_esdf_obstacle_spheres = LaunchConfiguration("run_esdf_obstacle_spheres")
    run_esdf_slice_obstacle_spheres = LaunchConfiguration(
        "run_esdf_slice_obstacle_spheres"
    )
    esdf_slice_obstacle_self_filter_robot_enabled = LaunchConfiguration(
        "esdf_slice_obstacle_self_filter_robot_enabled"
    )
    publish_obstacle_spheres = LaunchConfiguration("publish_obstacle_spheres")
    run_camera_closest_obstacles_with_publish = PythonExpression(
        [
            "'",
            publish_obstacle_spheres,
            "'.lower() in ('true', '1') and '",
            run_camera_closest_obstacles,
            "'.lower() in ('true', '1')",
        ]
    )
    run_obstacle_body_spheres_with_publish = PythonExpression(
        [
            "'",
            publish_obstacle_spheres,
            "'.lower() in ('true', '1') and '",
            run_obstacle_body_spheres,
            "'.lower() in ('true', '1')",
        ]
    )
    run_esdf_obstacle_spheres_with_publish = PythonExpression(
        [
            "'",
            publish_obstacle_spheres,
            "'.lower() in ('true', '1') and '",
            run_esdf_obstacle_spheres,
            "'.lower() in ('true', '1')",
        ]
    )
    run_esdf_slice_obstacle_spheres_with_publish = PythonExpression(
        [
            "'",
            publish_obstacle_spheres,
            "'.lower() in ('true', '1') and '",
            run_esdf_slice_obstacle_spheres,
            "'.lower() in ('true', '1')",
        ]
    )
    publish_obstacle_body_debug_clouds = LaunchConfiguration("publish_obstacle_body_debug_clouds")
    camera_serial_number = LaunchConfiguration("camera_serial_number")
    realsense_decimation_filter_enabled = LaunchConfiguration(
        "realsense_decimation_filter_enabled"
    )
    realsense_spatial_filter_enabled = LaunchConfiguration("realsense_spatial_filter_enabled")
    realsense_temporal_filter_enabled = LaunchConfiguration("realsense_temporal_filter_enabled")
    realsense_hole_filling_filter_enabled = LaunchConfiguration(
        "realsense_hole_filling_filter_enabled"
    )
    publish_static_tf = LaunchConfiguration("publish_static_tf")
    publish_odom_tf = LaunchConfiguration("publish_odom_tf")
    publish_robot_base_tf = LaunchConfiguration("publish_robot_base_tf")
    run_rviz = LaunchConfiguration("run_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    global_frame = LaunchConfiguration("global_frame")
    camera_frame = LaunchConfiguration("camera_frame")
    robot_root_frame = LaunchConfiguration("robot_root_frame")
    robot_urdf_path = LaunchConfiguration("robot_urdf_path")
    sphere_config_path = LaunchConfiguration("sphere_config_path")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    measured_joint_state_topic = LaunchConfiguration("measured_joint_state_topic")
    robot_ip = LaunchConfiguration("robot_ip")
    robot_data_port = LaunchConfiguration("robot_data_port")
    robot_joint_state_publish_rate_hz = LaunchConfiguration(
        "robot_joint_state_publish_rate_hz"
    )
    normalized_joint_state_topic = LaunchConfiguration("normalized_joint_state_topic")
    fallback_joint_state_publish_rate_hz = LaunchConfiguration("fallback_joint_state_publish_rate_hz")
    collision_sphere_publish_rate_hz = LaunchConfiguration("collision_sphere_publish_rate_hz")
    collision_sphere_stamp_from_joint_state = LaunchConfiguration(
        "collision_sphere_stamp_from_joint_state"
    )
    raw_depth_image_topic = LaunchConfiguration("raw_depth_image_topic")
    depth_camera_info_topic = LaunchConfiguration("depth_camera_info_topic")
    masked_depth_image_topic = LaunchConfiguration("masked_depth_image_topic")
    publish_robot_removal_debug = LaunchConfiguration("publish_robot_removal_debug")
    robot_depth_mask_removed_points_topic = LaunchConfiguration(
        "robot_depth_mask_removed_points_topic"
    )
    robot_pointcloud_removed_points_topic = LaunchConfiguration(
        "robot_pointcloud_removed_points_topic"
    )
    depth_pointcloud_topic = LaunchConfiguration("depth_pointcloud_topic")
    obstacle_input_cloud_topic = LaunchConfiguration("obstacle_input_cloud_topic")
    nvblox_depth_image_topic = LaunchConfiguration("nvblox_depth_image_topic")
    robot_free_cloud_topic = LaunchConfiguration("robot_free_cloud_topic")
    obstacle_cloud_topic = LaunchConfiguration("obstacle_cloud_topic")
    obstacle_body_marker_topic = LaunchConfiguration("obstacle_body_marker_topic")
    obstacle_body_cloud_topic = LaunchConfiguration("obstacle_body_cloud_topic")
    obstacle_body_candidate_cloud_topic = LaunchConfiguration("obstacle_body_candidate_cloud_topic")
    obstacle_body_cluster_cloud_topic = LaunchConfiguration("obstacle_body_cluster_cloud_topic")
    robot_margin_m = LaunchConfiguration("robot_margin_m")
    robot_depth_mask_margin_m = LaunchConfiguration("robot_depth_mask_margin_m")
    robot_pointcloud_filter_margin_m = LaunchConfiguration("robot_pointcloud_filter_margin_m")
    use_time_synchronized_robot_markers = LaunchConfiguration(
        "use_time_synchronized_robot_markers"
    )
    robot_marker_buffer_duration_s = LaunchConfiguration("robot_marker_buffer_duration_s")
    robot_marker_max_stamp_delta_s = LaunchConfiguration("robot_marker_max_stamp_delta_s")
    fallback_to_latest_marker_on_time_miss = LaunchConfiguration(
        "fallback_to_latest_marker_on_time_miss"
    )
    depth_pointcloud_stride = LaunchConfiguration("depth_pointcloud_stride")
    depth_pointcloud_max_points = LaunchConfiguration("depth_pointcloud_max_points")
    depth_pointcloud_min_depth_m = LaunchConfiguration("depth_pointcloud_min_depth_m")
    depth_pointcloud_max_depth_m = LaunchConfiguration("depth_pointcloud_max_depth_m")
    vision_filter_voxel_size_m = LaunchConfiguration("vision_filter_voxel_size_m")
    vision_filter_max_input_points = LaunchConfiguration("vision_filter_max_input_points")
    vision_filter_max_publish_points = LaunchConfiguration("vision_filter_max_publish_points")
    vision_closest_voxel_size_m = LaunchConfiguration("vision_closest_voxel_size_m")
    vision_closest_max_points = LaunchConfiguration("vision_closest_max_points")
    camera_obstacle_active_range_m = LaunchConfiguration("camera_obstacle_active_range_m")
    camera_obstacle_min_clearance_m = LaunchConfiguration("camera_obstacle_min_clearance_m")
    camera_obstacle_radius_m = LaunchConfiguration("camera_obstacle_radius_m")
    camera_obstacle_cluster_radius_m = LaunchConfiguration("camera_obstacle_cluster_radius_m")
    max_camera_obstacle_spheres = LaunchConfiguration("max_camera_obstacle_spheres")
    obstacle_body_min_x_m = LaunchConfiguration("obstacle_body_min_x_m")
    obstacle_body_max_x_m = LaunchConfiguration("obstacle_body_max_x_m")
    obstacle_body_min_y_m = LaunchConfiguration("obstacle_body_min_y_m")
    obstacle_body_max_y_m = LaunchConfiguration("obstacle_body_max_y_m")
    obstacle_body_min_z_m = LaunchConfiguration("obstacle_body_min_z_m")
    obstacle_body_max_z_m = LaunchConfiguration("obstacle_body_max_z_m")
    obstacle_body_max_range_from_base_m = LaunchConfiguration("obstacle_body_max_range_from_base_m")
    obstacle_body_voxel_size_m = LaunchConfiguration("obstacle_body_voxel_size_m")
    obstacle_body_cluster_cell_size_m = LaunchConfiguration("obstacle_body_cluster_cell_size_m")
    obstacle_body_min_cluster_points = LaunchConfiguration("obstacle_body_min_cluster_points")
    obstacle_body_max_cluster_extent_m = LaunchConfiguration("obstacle_body_max_cluster_extent_m")
    obstacle_body_max_clusters = LaunchConfiguration("obstacle_body_max_clusters")
    obstacle_body_sphere_spacing_m = LaunchConfiguration("obstacle_body_sphere_spacing_m")
    obstacle_body_min_sphere_radius_m = LaunchConfiguration("obstacle_body_min_sphere_radius_m")
    obstacle_body_max_sphere_radius_m = LaunchConfiguration("obstacle_body_max_sphere_radius_m")
    obstacle_body_sphere_padding_m = LaunchConfiguration("obstacle_body_sphere_padding_m")
    obstacle_body_max_spheres_per_cluster = LaunchConfiguration("obstacle_body_max_spheres_per_cluster")
    obstacle_body_max_total_spheres = LaunchConfiguration("obstacle_body_max_total_spheres")
    obstacle_body_marker_lifetime_s = LaunchConfiguration("obstacle_body_marker_lifetime_s")
    obstacle_body_min_persistent_frames = LaunchConfiguration("obstacle_body_min_persistent_frames")
    obstacle_body_persistence_match_distance_m = LaunchConfiguration(
        "obstacle_body_persistence_match_distance_m"
    )
    obstacle_body_persistence_smoothing_alpha = LaunchConfiguration(
        "obstacle_body_persistence_smoothing_alpha"
    )
    obstacle_background_subtraction_enabled = LaunchConfiguration(
        "obstacle_background_subtraction_enabled"
    )
    obstacle_background_voxel_size_m = LaunchConfiguration("obstacle_background_voxel_size_m")
    obstacle_background_neighbor_voxels = LaunchConfiguration("obstacle_background_neighbor_voxels")
    obstacle_auto_capture_background_on_start = LaunchConfiguration(
        "obstacle_auto_capture_background_on_start"
    )
    obstacle_auto_capture_background_delay_s = LaunchConfiguration(
        "obstacle_auto_capture_background_delay_s"
    )
    obstacle_auto_capture_background_sample_count = LaunchConfiguration(
        "obstacle_auto_capture_background_sample_count"
    )
    obstacle_body_robot_proximity_filter_enabled = LaunchConfiguration(
        "obstacle_body_robot_proximity_filter_enabled"
    )
    obstacle_body_min_robot_clearance_m = LaunchConfiguration(
        "obstacle_body_min_robot_clearance_m"
    )
    obstacle_body_min_sphere_robot_clearance_m = LaunchConfiguration(
        "obstacle_body_min_sphere_robot_clearance_m"
    )
    obstacle_body_max_robot_clearance_m = LaunchConfiguration(
        "obstacle_body_max_robot_clearance_m"
    )
    esdf_collision_sample_topic = LaunchConfiguration("esdf_collision_sample_topic")
    esdf_collision_marker_topic = LaunchConfiguration("esdf_collision_marker_topic")
    esdf_obstacle_surface_cloud_topic = LaunchConfiguration("esdf_obstacle_surface_cloud_topic")
    esdf_slice_obstacle_input_topic = LaunchConfiguration("esdf_slice_obstacle_input_topic")
    esdf_slice_obstacle_surface_cloud_topic = LaunchConfiguration(
        "esdf_slice_obstacle_surface_cloud_topic"
    )
    esdf_collision_service_name = LaunchConfiguration("esdf_collision_service_name")
    esdf_collision_query_padding_m = LaunchConfiguration("esdf_collision_query_padding_m")
    esdf_collision_robot_clear_margin_m = LaunchConfiguration(
        "esdf_collision_robot_clear_margin_m"
    )
    esdf_collision_self_filter_margin_m = LaunchConfiguration(
        "esdf_collision_self_filter_margin_m"
    )
    esdf_collision_safety_margin_m = LaunchConfiguration("esdf_collision_safety_margin_m")
    esdf_collision_max_publish_clearance_m = LaunchConfiguration(
        "esdf_collision_max_publish_clearance_m"
    )
    esdf_collision_rate_hz = LaunchConfiguration("esdf_collision_rate_hz")
    esdf_collision_publish_all_valid_spheres = LaunchConfiguration(
        "esdf_collision_publish_all_valid_spheres"
    )
    esdf_obstacle_marker_topic = LaunchConfiguration("esdf_obstacle_marker_topic")
    esdf_obstacle_cloud_topic = LaunchConfiguration("esdf_obstacle_cloud_topic")
    esdf_obstacle_active_range_m = LaunchConfiguration("esdf_obstacle_active_range_m")
    esdf_obstacle_danger_clearance_m = LaunchConfiguration("esdf_obstacle_danger_clearance_m")
    esdf_obstacle_cluster_radius_m = LaunchConfiguration("esdf_obstacle_cluster_radius_m")
    esdf_obstacle_min_cluster_points = LaunchConfiguration("esdf_obstacle_min_cluster_points")
    esdf_obstacle_min_sphere_radius_m = LaunchConfiguration("esdf_obstacle_min_sphere_radius_m")
    esdf_obstacle_max_sphere_radius_m = LaunchConfiguration("esdf_obstacle_max_sphere_radius_m")
    esdf_obstacle_sphere_padding_m = LaunchConfiguration("esdf_obstacle_sphere_padding_m")
    esdf_obstacle_max_spheres = LaunchConfiguration("esdf_obstacle_max_spheres")
    esdf_obstacle_smoothing_alpha = LaunchConfiguration("esdf_obstacle_smoothing_alpha")
    esdf_obstacle_smoothing_match_distance_m = LaunchConfiguration(
        "esdf_obstacle_smoothing_match_distance_m"
    )
    esdf_obstacle_min_persistent_frames = LaunchConfiguration(
        "esdf_obstacle_min_persistent_frames"
    )
    esdf_surface_min_distance_m = LaunchConfiguration("esdf_surface_min_distance_m")
    esdf_surface_max_distance_m = LaunchConfiguration("esdf_surface_max_distance_m")
    esdf_surface_max_robot_clearance_m = LaunchConfiguration(
        "esdf_surface_max_robot_clearance_m"
    )
    esdf_surface_stride = LaunchConfiguration("esdf_surface_stride")
    esdf_surface_max_points = LaunchConfiguration("esdf_surface_max_points")
    esdf_slice_height_m = LaunchConfiguration("esdf_slice_height_m")
    esdf_slice_min_height_m = LaunchConfiguration("esdf_slice_min_height_m")
    esdf_slice_max_height_m = LaunchConfiguration("esdf_slice_max_height_m")
    vision_pipeline_rate_hz = LaunchConfiguration("vision_pipeline_rate_hz")

    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    roll = LaunchConfiguration("roll")
    pitch = LaunchConfiguration("pitch")
    yaw = LaunchConfiguration("yaw")

    nvblox_base_config = package_file(
        "nvblox_examples_bringup", "config/nvblox/nvblox_base.yaml"
    )
    nvblox_realsense_config = package_file(
        "nvblox_examples_bringup", "config/nvblox/specializations/nvblox_realsense.yaml"
    )
    nvblox_static_config = package_file("rmp_camera", "config/nvblox_d435_static.yaml")
    default_rviz_config = package_file("rmp_camera", "config/d435_nvblox_static.rviz")
    default_robot_urdf_path = package_file("rmp_camera", "urdf/rb10_1300e.urdf")
    default_sphere_config_path = package_file("rmp_camera", "config/collision_spheres.yaml")
    default_bag_path = str(Path.home() / "bags" / "d435_esdf_20260707_140735")

    bag_play = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            bag_path,
            "--clock",
            "--loop",
            "--rate",
            bag_rate,
        ],
        output="screen",
        condition=IfCondition(PythonExpression(["'", source, "' == 'bag'"])),
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            package_file("nvblox_examples_bringup", "launch/sensors/realsense.launch.py")
        ),
        launch_arguments={
            "container_name": container_name,
            "run_standalone": "False",
            "num_cameras": "1",
            "camera_serial_numbers": camera_serial_number,
            # Passed through so they land on the realsense2_camera node if the
            # included launch file forwards unknown launch arguments to it;
            # harmless no-ops otherwise. Verify exact arg names against the
            # realsense2_camera version actually installed before relying on
            # them (see DeclareLaunchArgument descriptions below).
            "decimation_filter.enable": realsense_decimation_filter_enabled,
            "spatial_filter.enable": realsense_spatial_filter_enabled,
            "temporal_filter.enable": realsense_temporal_filter_enabled,
            "hole_filling_filter.enable": realsense_hole_filling_filter_enabled,
        }.items(),
        condition=IfCondition(run_realsense_for_source),
    )

    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            package_file("nvblox_examples_bringup", "launch/visualization/rviz.launch.py")
        ),
        launch_arguments={
            "mode": "static",
            "camera": "realsense",
            "rviz_config": rviz_config,
        }.items(),
        condition=IfCondition(run_rviz),
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_d435_static_tf",
        output="screen",
        arguments=[
            "--x",
            x,
            "--y",
            y,
            "--z",
            z,
            "--roll",
            roll,
            "--pitch",
            pitch,
            "--yaw",
            yaw,
            "--frame-id",
            global_frame,
            "--child-frame-id",
            camera_frame,
        ],
        condition=IfCondition(publish_static_tf),
    )

    odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odom_to_global_static_tf",
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
            "odom",
            "--child-frame-id",
            global_frame,
        ],
        condition=IfCondition(publish_odom_tf),
    )

    robot_base_tf = Node(
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
            global_frame,
            "--child-frame-id",
            robot_root_frame,
        ],
        condition=IfCondition(publish_robot_base_tf),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="rmp_camera_robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": ParameterValue(
                    Command(["cat ", robot_urdf_path]),
                    value_type=str,
                ),
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        remappings=[
            ("joint_states", normalized_joint_state_topic),
        ],
        condition=IfCondition(run_robot_state_publisher),
    )

    joint_state_normalizer = Node(
        package="rmp_camera",
        executable="joint_state_normalizer_node",
        name="joint_state_normalizer_node",
        output="screen",
        parameters=[
            {
                "input_topic": joint_state_topic,
                "output_topic": normalized_joint_state_topic,
                "fallback_publish_rate_hz": fallback_joint_state_publish_rate_hz,
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_joint_state_normalizer),
    )

    rb10_measured_joint_state_publisher = Node(
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
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_rb10_measured_joint_state_for_source),
    )

    robot_collision_spheres = Node(
        package="rmp_camera",
        executable="robot_collision_sphere_node",
        name="robot_collision_sphere_node",
        output="screen",
        parameters=[
            {
                "urdf_path": robot_urdf_path,
                "sphere_config_path": sphere_config_path,
                "joint_state_topic": normalized_joint_state_topic,
                "publish_rate_hz": collision_sphere_publish_rate_hz,
                "stamp_from_joint_state": ParameterValue(
                    collision_sphere_stamp_from_joint_state,
                    value_type=bool,
                ),
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_collision_spheres),
    )

    robot_pointcloud_filter = Node(
        package="rmp_camera",
        executable="robot_pointcloud_filter_node",
        name="robot_pointcloud_filter_node",
        output="screen",
        parameters=[
            {
                "input_cloud_topic": obstacle_input_cloud_topic,
                "robot_sphere_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
                "output_cloud_topic": robot_free_cloud_topic,
                "removed_cloud_topic": robot_pointcloud_removed_points_topic,
                "publish_removed_cloud": ParameterValue(
                    publish_robot_removal_debug,
                    value_type=bool,
                ),
                "target_frame": global_frame,
                "robot_margin_m": robot_pointcloud_filter_margin_m,
                "voxel_size_m": vision_filter_voxel_size_m,
                "max_input_points": vision_filter_max_input_points,
                "max_publish_points": vision_filter_max_publish_points,
                "max_rate_hz": vision_pipeline_rate_hz,
                "passthrough_on_missing_robot": True,
                "use_time_synchronized_robot_markers": ParameterValue(
                    use_time_synchronized_robot_markers,
                    value_type=bool,
                ),
                "robot_marker_buffer_duration_s": robot_marker_buffer_duration_s,
                "robot_marker_max_stamp_delta_s": robot_marker_max_stamp_delta_s,
                "fallback_to_latest_marker_on_time_miss": ParameterValue(
                    fallback_to_latest_marker_on_time_miss,
                    value_type=bool,
                ),
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_vision_obstacle_pipeline),
    )

    robot_depth_mask = Node(
        package="rmp_camera",
        executable="robot_depth_mask_node",
        name="robot_depth_mask_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
                "input_depth_topic": raw_depth_image_topic,
                "camera_info_topic": depth_camera_info_topic,
                "robot_sphere_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
                "output_depth_topic": masked_depth_image_topic,
                "removed_points_topic": robot_depth_mask_removed_points_topic,
                "publish_removed_points": ParameterValue(
                    publish_robot_removal_debug,
                    value_type=bool,
                ),
                "robot_margin_m": robot_depth_mask_margin_m,
                "max_rate_hz": vision_pipeline_rate_hz,
                "passthrough_on_missing_robot": True,
                "use_time_synchronized_robot_markers": ParameterValue(
                    use_time_synchronized_robot_markers,
                    value_type=bool,
                ),
                "robot_marker_buffer_duration_s": robot_marker_buffer_duration_s,
                "robot_marker_max_stamp_delta_s": robot_marker_max_stamp_delta_s,
                "fallback_to_latest_marker_on_time_miss": ParameterValue(
                    fallback_to_latest_marker_on_time_miss,
                    value_type=bool,
                ),
            }
        ],
        condition=IfCondition(run_robot_depth_mask),
    )

    depth_to_pointcloud = Node(
        package="rmp_camera",
        executable="depth_to_pointcloud_node",
        name="depth_to_pointcloud_node",
        output="screen",
        parameters=[
            {
                "input_depth_topic": masked_depth_image_topic,
                "camera_info_topic": depth_camera_info_topic,
                "output_cloud_topic": depth_pointcloud_topic,
                "min_depth_m": depth_pointcloud_min_depth_m,
                "max_depth_m": depth_pointcloud_max_depth_m,
                "stride": depth_pointcloud_stride,
                "max_points": depth_pointcloud_max_points,
                "max_rate_hz": vision_pipeline_rate_hz,
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_depth_to_pointcloud_with_vision),
    )

    vision_closest_obstacle = Node(
        package="rmp_camera",
        executable="vision_closest_obstacle_node",
        name="vision_closest_obstacle_node",
        output="screen",
        parameters=[
            {
                "input_cloud_topic": robot_free_cloud_topic,
                "robot_sphere_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
                "target_frame": global_frame,
                "active_range_m": camera_obstacle_active_range_m,
                "min_clearance_m": camera_obstacle_min_clearance_m,
                "obstacle_radius_m": camera_obstacle_radius_m,
                "cluster_radius_m": camera_obstacle_cluster_radius_m,
                "max_obstacle_spheres": max_camera_obstacle_spheres,
                "obstacle_cloud_topic": obstacle_cloud_topic,
                "voxel_size_m": vision_closest_voxel_size_m,
                "max_points": vision_closest_max_points,
                "max_rate_hz": vision_pipeline_rate_hz,
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_camera_closest_obstacles_with_publish),
    )

    obstacle_body_spheres = Node(
        package="rmp_camera",
        executable="obstacle_body_sphere_node",
        name="obstacle_body_sphere_node",
        output="screen",
        parameters=[
            {
                "input_cloud_topic": robot_free_cloud_topic,
                "robot_sphere_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
                "obstacle_marker_topic": obstacle_body_marker_topic,
                "obstacle_cloud_topic": obstacle_body_cloud_topic,
                "candidate_cloud_topic": obstacle_body_candidate_cloud_topic,
                "cluster_cloud_topic": obstacle_body_cluster_cloud_topic,
                "target_frame": global_frame,
                "publish_candidate_cloud": ParameterValue(
                    publish_obstacle_body_debug_clouds,
                    value_type=bool,
                ),
                "publish_cluster_cloud": ParameterValue(
                    publish_obstacle_body_debug_clouds,
                    value_type=bool,
                ),
                "background_subtraction_enabled": ParameterValue(
                    obstacle_background_subtraction_enabled,
                    value_type=bool,
                ),
                "background_voxel_size_m": obstacle_background_voxel_size_m,
                "background_neighbor_voxels": obstacle_background_neighbor_voxels,
                "auto_capture_background_on_start": ParameterValue(
                    obstacle_auto_capture_background_on_start,
                    value_type=bool,
                ),
                "auto_capture_background_delay_s": obstacle_auto_capture_background_delay_s,
                "auto_capture_background_sample_count": obstacle_auto_capture_background_sample_count,
                "robot_proximity_filter_enabled": ParameterValue(
                    obstacle_body_robot_proximity_filter_enabled,
                    value_type=bool,
                ),
                "min_robot_clearance_m": obstacle_body_min_robot_clearance_m,
                "min_sphere_robot_clearance_m": obstacle_body_min_sphere_robot_clearance_m,
                "max_robot_clearance_m": obstacle_body_max_robot_clearance_m,
                "min_x_m": obstacle_body_min_x_m,
                "max_x_m": obstacle_body_max_x_m,
                "min_y_m": obstacle_body_min_y_m,
                "max_y_m": obstacle_body_max_y_m,
                "min_z_m": obstacle_body_min_z_m,
                "max_z_m": obstacle_body_max_z_m,
                "max_range_from_base_m": obstacle_body_max_range_from_base_m,
                "voxel_size_m": obstacle_body_voxel_size_m,
                "cluster_cell_size_m": obstacle_body_cluster_cell_size_m,
                "min_cluster_points": obstacle_body_min_cluster_points,
                "max_cluster_extent_m": obstacle_body_max_cluster_extent_m,
                "max_clusters": obstacle_body_max_clusters,
                "sphere_spacing_m": obstacle_body_sphere_spacing_m,
                "min_sphere_radius_m": obstacle_body_min_sphere_radius_m,
                "max_sphere_radius_m": obstacle_body_max_sphere_radius_m,
                "sphere_padding_m": obstacle_body_sphere_padding_m,
                "max_spheres_per_cluster": obstacle_body_max_spheres_per_cluster,
                "max_total_spheres": obstacle_body_max_total_spheres,
                "marker_lifetime_s": obstacle_body_marker_lifetime_s,
                "min_persistent_frames": obstacle_body_min_persistent_frames,
                "persistence_match_distance_m": obstacle_body_persistence_match_distance_m,
                "persistence_smoothing_alpha": obstacle_body_persistence_smoothing_alpha,
                "max_rate_hz": vision_pipeline_rate_hz,
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_obstacle_body_spheres_with_publish),
    )

    esdf_collision_query = Node(
        package="rmp_camera",
        executable="esdf_collision_query_node",
        name="esdf_collision_query_node",
        output="screen",
        parameters=[
            {
                "service_name": esdf_collision_service_name,
                "robot_sphere_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
                "collision_sample_topic": esdf_collision_sample_topic,
                "collision_marker_topic": esdf_collision_marker_topic,
                "surface_cloud_topic": esdf_obstacle_surface_cloud_topic,
                "target_frame": global_frame,
                "max_rate_hz": esdf_collision_rate_hz,
                "query_padding_m": esdf_collision_query_padding_m,
                "robot_clear_margin_m": esdf_collision_robot_clear_margin_m,
                "self_filter_margin_m": esdf_collision_self_filter_margin_m,
                "safety_margin_m": esdf_collision_safety_margin_m,
                "max_publish_clearance_m": esdf_collision_max_publish_clearance_m,
                "publish_all_valid_spheres": ParameterValue(
                    esdf_collision_publish_all_valid_spheres,
                    value_type=bool,
                ),
                "publish_surface_cloud": True,
                "surface_min_distance_m": esdf_surface_min_distance_m,
                "surface_max_distance_m": esdf_surface_max_distance_m,
                "surface_max_robot_clearance_m": esdf_surface_max_robot_clearance_m,
                "surface_stride": esdf_surface_stride,
                "max_surface_points": esdf_surface_max_points,
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_esdf_collision_query),
    )

    esdf_obstacle_spheres = Node(
        package="rmp_camera",
        executable="esdf_obstacle_sphere_node",
        name="esdf_obstacle_sphere_node",
        output="screen",
        parameters=[
            {
                "sample_topic": esdf_obstacle_surface_cloud_topic,
                "obstacle_marker_topic": esdf_obstacle_marker_topic,
                "obstacle_cloud_topic": esdf_obstacle_cloud_topic,
                "target_frame": global_frame,
                "active_range_m": esdf_obstacle_active_range_m,
                "danger_clearance_m": esdf_obstacle_danger_clearance_m,
                "cluster_radius_m": esdf_obstacle_cluster_radius_m,
                "min_cluster_points": esdf_obstacle_min_cluster_points,
                "min_sphere_radius_m": esdf_obstacle_min_sphere_radius_m,
                "max_sphere_radius_m": esdf_obstacle_max_sphere_radius_m,
                "sphere_padding_m": esdf_obstacle_sphere_padding_m,
                "max_obstacle_spheres": esdf_obstacle_max_spheres,
                "smoothing_alpha": esdf_obstacle_smoothing_alpha,
                "smoothing_match_distance_m": esdf_obstacle_smoothing_match_distance_m,
                "min_persistent_frames": esdf_obstacle_min_persistent_frames,
                "max_rate_hz": esdf_collision_rate_hz,
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_esdf_obstacle_spheres_with_publish),
    )

    esdf_slice_obstacle_spheres = Node(
        package="rmp_camera",
        executable="nvblox_esdf_slice_obstacle_sphere_node",
        name="nvblox_esdf_slice_obstacle_sphere_node",
        output="screen",
        parameters=[
            {
                "input_cloud_topic": esdf_slice_obstacle_input_topic,
                "obstacle_marker_topic": esdf_obstacle_marker_topic,
                "obstacle_cloud_topic": esdf_obstacle_cloud_topic,
                "surface_cloud_topic": esdf_slice_obstacle_surface_cloud_topic,
                "target_frame": global_frame,
                "publish_surface_cloud": True,
                "self_filter_robot_enabled": ParameterValue(
                    esdf_slice_obstacle_self_filter_robot_enabled,
                    value_type=bool,
                ),
                "robot_sphere_marker_topic": "/rmp_camera/robot_collision_sphere_markers",
                "robot_self_filter_margin_m": robot_margin_m,
                "surface_min_distance_m": esdf_surface_min_distance_m,
                "surface_max_distance_m": esdf_surface_max_distance_m,
                "danger_distance_m": esdf_obstacle_danger_clearance_m,
                "min_x_m": obstacle_body_min_x_m,
                "max_x_m": obstacle_body_max_x_m,
                "min_y_m": obstacle_body_min_y_m,
                "max_y_m": obstacle_body_max_y_m,
                "min_z_m": obstacle_body_min_z_m,
                "max_z_m": obstacle_body_max_z_m,
                "max_range_from_base_m": obstacle_body_max_range_from_base_m,
                "voxel_size_m": obstacle_body_voxel_size_m,
                "cluster_cell_size_m": obstacle_body_cluster_cell_size_m,
                "min_cluster_points": obstacle_body_min_cluster_points,
                "max_cluster_extent_m": obstacle_body_max_cluster_extent_m,
                "max_clusters": obstacle_body_max_clusters,
                "sphere_spacing_m": obstacle_body_sphere_spacing_m,
                "min_sphere_radius_m": obstacle_body_min_sphere_radius_m,
                "max_sphere_radius_m": obstacle_body_max_sphere_radius_m,
                "sphere_padding_m": obstacle_body_sphere_padding_m,
                "max_spheres_per_cluster": obstacle_body_max_spheres_per_cluster,
                "max_total_spheres": obstacle_body_max_total_spheres,
                "min_persistent_frames": obstacle_body_min_persistent_frames,
                "persistence_match_distance_m": obstacle_body_persistence_match_distance_m,
                "persistence_smoothing_alpha": obstacle_body_persistence_smoothing_alpha,
                "max_rate_hz": esdf_collision_rate_hz,
                "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
            }
        ],
        condition=IfCondition(run_esdf_slice_obstacle_spheres_with_publish),
    )

    container = ComposableNodeContainer(
        name=container_name,
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        arguments=["--ros-args", "--log-level", log_level],
    )

    nvblox_node = LoadComposableNodes(
        target_container=container_name,
        composable_node_descriptions=[
            ComposableNode(
                name="nvblox_node",
                package="nvblox_ros",
                plugin="nvblox::NvbloxNode",
                remappings=[
                    ("camera_0/depth/image", nvblox_depth_image_topic),
                    ("camera_0/depth/camera_info", depth_camera_info_topic),
                    ("camera_0/color/image", "/camera0/camera/color/image_raw"),
                    ("camera_0/color/camera_info", "/camera0/camera/color/camera_info"),
                ],
                parameters=[
                    nvblox_base_config,
                    nvblox_realsense_config,
                    nvblox_static_config,
                    {
                        "use_sim_time": ParameterValue(effective_use_sim_time, value_type=bool),
                        "esdf_mode": esdf_mode,
                        "global_frame": global_frame,
                        "pose_frame": global_frame,
                        "map_clearing_frame_id": global_frame,
                        "esdf_slice_bounds_visualization_attachment_frame_id": global_frame,
                        "workspace_height_bounds_visualization_attachment_frame_id": global_frame,
                        "static_mapper.esdf_slice_height": esdf_slice_height_m,
                        "static_mapper.esdf_slice_min_height": esdf_slice_min_height_m,
                        "static_mapper.esdf_slice_max_height": esdf_slice_max_height_m,
                    },
                ],
            )
        ],
    )

    return LaunchDescription(
        [
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
            DeclareLaunchArgument("container_name", default_value="rmp_camera_nvblox_container"),
            DeclareLaunchArgument("log_level", default_value="info"),
            DeclareLaunchArgument(
                "source",
                default_value="camera",
                choices=["camera", "bag"],
                description="Input source. camera starts RealSense; bag starts ros2 bag play.",
            ),
            DeclareLaunchArgument("bag_path", default_value=default_bag_path),
            DeclareLaunchArgument("bag_rate", default_value="1.0"),
            DeclareLaunchArgument("use_sim_time", default_value="False"),
            DeclareLaunchArgument("esdf_mode", default_value="3d"),
            DeclareLaunchArgument("run_realsense", default_value="True"),
            DeclareLaunchArgument("run_collision_spheres", default_value="True"),
            DeclareLaunchArgument(
                "run_rb10_measured_joint_state_publisher",
                default_value="True",
            ),
            DeclareLaunchArgument("run_joint_state_normalizer", default_value="True"),
            DeclareLaunchArgument("run_robot_state_publisher", default_value="True"),
            DeclareLaunchArgument("run_robot_depth_mask", default_value="True"),
            DeclareLaunchArgument(
                "run_depth_to_pointcloud",
                default_value="True",
                description="Project robot-masked depth into the pointcloud used by the obstacle sphere pipeline.",
            ),
            DeclareLaunchArgument("run_vision_obstacle_pipeline", default_value="True"),
            DeclareLaunchArgument(
                "publish_obstacle_spheres",
                default_value="True",
                description="Publish camera/obstacle sphere outputs. Set false to keep mapping/filtering but stop obstacle sphere topics.",
            ),
            DeclareLaunchArgument("run_camera_closest_obstacles", default_value="False"),
            DeclareLaunchArgument("run_obstacle_body_spheres", default_value="True"),
            DeclareLaunchArgument("run_esdf_collision_query", default_value="False"),
            DeclareLaunchArgument("run_esdf_obstacle_spheres", default_value="False"),
            DeclareLaunchArgument("run_esdf_slice_obstacle_spheres", default_value="False"),
            DeclareLaunchArgument(
                "esdf_slice_obstacle_self_filter_robot_enabled",
                default_value="True",
            ),
            DeclareLaunchArgument("publish_obstacle_body_debug_clouds", default_value="False"),
            DeclareLaunchArgument("camera_serial_number", default_value="117322070340"),
            DeclareLaunchArgument(
                "realsense_decimation_filter_enabled",
                default_value="True",
                description="Enable the RealSense driver-level decimation filter "
                "(depth post-processing, reduces noisy/floating points before nvblox "
                "integration). Argument name assumes realsense2_camera's "
                "'decimation_filter.enable' convention (realsense-ros >= 4.51); verify "
                "against the installed realsense2_camera version and adjust the parameter "
                "name passed into realsense_launch below if it differs.",
            ),
            DeclareLaunchArgument(
                "realsense_spatial_filter_enabled",
                default_value="True",
                description="Enable the RealSense driver-level spatial (edge-preserving) "
                "filter. See realsense_decimation_filter_enabled for the naming caveat.",
            ),
            DeclareLaunchArgument(
                "realsense_temporal_filter_enabled",
                default_value="True",
                description="Enable the RealSense driver-level temporal filter. See "
                "realsense_decimation_filter_enabled for the naming caveat.",
            ),
            DeclareLaunchArgument(
                "realsense_hole_filling_filter_enabled",
                default_value="True",
                description="Enable the RealSense driver-level hole-filling filter. See "
                "realsense_decimation_filter_enabled for the naming caveat.",
            ),
            DeclareLaunchArgument("publish_static_tf", default_value="True"),
            DeclareLaunchArgument("publish_odom_tf", default_value="True"),
            DeclareLaunchArgument("publish_robot_base_tf", default_value="False"),
            DeclareLaunchArgument("run_rviz", default_value="True"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            DeclareLaunchArgument("global_frame", default_value="base_link"),
            DeclareLaunchArgument("camera_frame", default_value="camera0_link"),
            DeclareLaunchArgument("robot_root_frame", default_value="link0"),
            DeclareLaunchArgument("robot_urdf_path", default_value=default_robot_urdf_path),
            DeclareLaunchArgument("sphere_config_path", default_value=default_sphere_config_path),
            DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
            DeclareLaunchArgument(
                "measured_joint_state_topic",
                default_value="/rmp_camera/rb10_measured_joint_states",
            ),
            DeclareLaunchArgument("robot_ip", default_value="192.168.111.50"),
            DeclareLaunchArgument("robot_data_port", default_value="5001"),
            DeclareLaunchArgument("robot_joint_state_publish_rate_hz", default_value="50.0"),
            DeclareLaunchArgument(
                "normalized_joint_state_topic",
                default_value="/rmp_camera/joint_states_urdf",
            ),
            DeclareLaunchArgument("fallback_joint_state_publish_rate_hz", default_value="10.0"),
            DeclareLaunchArgument("collision_sphere_publish_rate_hz", default_value="60.0"),
            DeclareLaunchArgument("collision_sphere_stamp_from_joint_state", default_value="True"),
            DeclareLaunchArgument(
                "raw_depth_image_topic",
                default_value="/camera0/camera/depth/image_rect_raw",
            ),
            DeclareLaunchArgument(
                "depth_camera_info_topic",
                default_value="/camera0/camera/depth/camera_info",
            ),
            DeclareLaunchArgument(
                "masked_depth_image_topic",
                default_value="/rmp_camera/robot_masked_depth/image_rect_raw",
            ),
            DeclareLaunchArgument("publish_robot_removal_debug", default_value="True"),
            DeclareLaunchArgument(
                "robot_depth_mask_removed_points_topic",
                default_value="/rmp_camera/robot_depth_mask_removed_points",
            ),
            DeclareLaunchArgument(
                "robot_pointcloud_removed_points_topic",
                default_value="/rmp_camera/robot_pointcloud_removed_points",
            ),
            DeclareLaunchArgument(
                "depth_pointcloud_topic",
                default_value="/rmp_camera/robot_masked_depth/points",
            ),
            DeclareLaunchArgument(
                "obstacle_input_cloud_topic",
                default_value="/rmp_camera/robot_masked_depth/points",
                description="PointCloud2 input for robot-free filtering and obstacle sphere generation.",
            ),
            DeclareLaunchArgument(
                "nvblox_depth_image_topic",
                default_value="/rmp_camera/robot_masked_depth/image_rect_raw",
            ),
            DeclareLaunchArgument("robot_free_cloud_topic", default_value="/rmp_camera/robot_free_pointcloud"),
            DeclareLaunchArgument(
                "obstacle_cloud_topic",
                default_value="/rmp_camera/camera_obstacle_sphere_cloud",
            ),
            DeclareLaunchArgument("obstacle_body_marker_topic", default_value="/rmp_camera/obstacle_body_spheres"),
            DeclareLaunchArgument("obstacle_body_cloud_topic", default_value="/rmp_camera/obstacle_body_sphere_cloud"),
            DeclareLaunchArgument(
                "obstacle_body_candidate_cloud_topic",
                default_value="/rmp_camera/obstacle_body_candidate_cloud",
            ),
            DeclareLaunchArgument(
                "obstacle_body_cluster_cloud_topic",
                default_value="/rmp_camera/obstacle_body_cluster_cloud",
            ),
            DeclareLaunchArgument("robot_margin_m", default_value="0.08"),
            DeclareLaunchArgument("robot_depth_mask_margin_m", default_value="0.06"),
            DeclareLaunchArgument("robot_pointcloud_filter_margin_m", default_value="0.06"),
            DeclareLaunchArgument("use_time_synchronized_robot_markers", default_value="True"),
            DeclareLaunchArgument("robot_marker_buffer_duration_s", default_value="1.0"),
            DeclareLaunchArgument("robot_marker_max_stamp_delta_s", default_value="0.15"),
            DeclareLaunchArgument("fallback_to_latest_marker_on_time_miss", default_value="True"),
            DeclareLaunchArgument("depth_pointcloud_stride", default_value="3"),
            DeclareLaunchArgument("depth_pointcloud_max_points", default_value="50000"),
            DeclareLaunchArgument("depth_pointcloud_min_depth_m", default_value="0.05"),
            DeclareLaunchArgument("depth_pointcloud_max_depth_m", default_value="4.0"),
            DeclareLaunchArgument("vision_filter_voxel_size_m", default_value="0.025"),
            DeclareLaunchArgument("vision_filter_max_input_points", default_value="120000"),
            DeclareLaunchArgument("vision_filter_max_publish_points", default_value="40000"),
            DeclareLaunchArgument("vision_closest_voxel_size_m", default_value="0.035"),
            DeclareLaunchArgument("vision_closest_max_points", default_value="30000"),
            DeclareLaunchArgument("camera_obstacle_min_clearance_m", default_value="0.30"),
            DeclareLaunchArgument("camera_obstacle_active_range_m", default_value="1.0"),
            DeclareLaunchArgument("camera_obstacle_radius_m", default_value="0.06"),
            DeclareLaunchArgument("camera_obstacle_cluster_radius_m", default_value="0.14"),
            DeclareLaunchArgument("max_camera_obstacle_spheres", default_value="5"),
            DeclareLaunchArgument("obstacle_body_min_x_m", default_value="-1.5"),
            DeclareLaunchArgument("obstacle_body_max_x_m", default_value="1.6"),
            DeclareLaunchArgument("obstacle_body_min_y_m", default_value="-1.8"),
            DeclareLaunchArgument("obstacle_body_max_y_m", default_value="1.4"),
            DeclareLaunchArgument("obstacle_body_min_z_m", default_value="0.05"),
            DeclareLaunchArgument("obstacle_body_max_z_m", default_value="1.8"),
            DeclareLaunchArgument("obstacle_body_max_range_from_base_m", default_value="2.2"),
            DeclareLaunchArgument("obstacle_body_voxel_size_m", default_value="0.035"),
            DeclareLaunchArgument("obstacle_body_cluster_cell_size_m", default_value="0.11"),
            DeclareLaunchArgument("obstacle_body_min_cluster_points", default_value="18"),
            DeclareLaunchArgument("obstacle_body_max_cluster_extent_m", default_value="2.2"),
            DeclareLaunchArgument("obstacle_body_max_clusters", default_value="6"),
            DeclareLaunchArgument("obstacle_body_sphere_spacing_m", default_value="0.08"),
            DeclareLaunchArgument("obstacle_body_min_sphere_radius_m", default_value="0.04"),
            DeclareLaunchArgument("obstacle_body_max_sphere_radius_m", default_value="0.20"),
            DeclareLaunchArgument("obstacle_body_sphere_padding_m", default_value="0.015"),
            DeclareLaunchArgument("obstacle_body_max_spheres_per_cluster", default_value="6"),
            DeclareLaunchArgument("obstacle_body_max_total_spheres", default_value="16"),
            DeclareLaunchArgument("obstacle_body_marker_lifetime_s", default_value="0.35"),
            DeclareLaunchArgument("obstacle_body_min_persistent_frames", default_value="3"),
            DeclareLaunchArgument("obstacle_body_persistence_match_distance_m", default_value="0.12"),
            DeclareLaunchArgument("obstacle_body_persistence_smoothing_alpha", default_value="0.6"),
            DeclareLaunchArgument("obstacle_background_subtraction_enabled", default_value="True"),
            DeclareLaunchArgument("obstacle_background_voxel_size_m", default_value="0.05"),
            DeclareLaunchArgument("obstacle_background_neighbor_voxels", default_value="1"),
            DeclareLaunchArgument("obstacle_auto_capture_background_on_start", default_value="True"),
            DeclareLaunchArgument("obstacle_auto_capture_background_delay_s", default_value="1.0"),
            DeclareLaunchArgument("obstacle_auto_capture_background_sample_count", default_value="10"),
            DeclareLaunchArgument(
                "obstacle_body_robot_proximity_filter_enabled",
                default_value="True",
            ),
            DeclareLaunchArgument("obstacle_body_min_robot_clearance_m", default_value="0.02"),
            DeclareLaunchArgument("obstacle_body_min_sphere_robot_clearance_m", default_value="0.02"),
            DeclareLaunchArgument("obstacle_body_max_robot_clearance_m", default_value="1.0"),
            DeclareLaunchArgument(
                "esdf_collision_sample_topic",
                default_value="/rmp_camera/esdf_collision_samples",
            ),
            DeclareLaunchArgument(
                "esdf_collision_marker_topic",
                default_value="/rmp_camera/esdf_collision_markers",
            ),
            DeclareLaunchArgument(
                "esdf_obstacle_surface_cloud_topic",
                default_value="/rmp_camera/esdf_obstacle_surface_points",
            ),
            DeclareLaunchArgument(
                "esdf_slice_obstacle_input_topic",
                default_value="/nvblox_node/static_esdf_pointcloud",
            ),
            DeclareLaunchArgument(
                "esdf_slice_obstacle_surface_cloud_topic",
                default_value="/rmp_camera/esdf_slice_obstacle_surface_points",
            ),
            DeclareLaunchArgument(
                "esdf_collision_service_name",
                default_value="/nvblox_node/get_esdf_and_gradient",
            ),
            DeclareLaunchArgument("esdf_collision_query_padding_m", default_value="1.0"),
            DeclareLaunchArgument("esdf_collision_robot_clear_margin_m", default_value="0.08"),
            DeclareLaunchArgument("esdf_collision_self_filter_margin_m", default_value="0.03"),
            DeclareLaunchArgument("esdf_collision_safety_margin_m", default_value="0.03"),
            DeclareLaunchArgument("esdf_collision_max_publish_clearance_m", default_value="1.0"),
            DeclareLaunchArgument("esdf_collision_rate_hz", default_value="5.0"),
            DeclareLaunchArgument("esdf_collision_publish_all_valid_spheres", default_value="True"),
            DeclareLaunchArgument(
                "esdf_obstacle_marker_topic",
                default_value="/rmp_camera/camera_obstacle_spheres",
            ),
            DeclareLaunchArgument(
                "esdf_obstacle_cloud_topic",
                default_value="/rmp_camera/camera_obstacle_sphere_cloud",
            ),
            DeclareLaunchArgument("esdf_obstacle_active_range_m", default_value="1.0"),
            DeclareLaunchArgument("esdf_obstacle_danger_clearance_m", default_value="0.05"),
            DeclareLaunchArgument("esdf_obstacle_cluster_radius_m", default_value="0.14"),
            DeclareLaunchArgument("esdf_obstacle_min_cluster_points", default_value="3"),
            DeclareLaunchArgument("esdf_obstacle_min_sphere_radius_m", default_value="0.04"),
            DeclareLaunchArgument("esdf_obstacle_max_sphere_radius_m", default_value="0.14"),
            DeclareLaunchArgument("esdf_obstacle_sphere_padding_m", default_value="0.02"),
            DeclareLaunchArgument("esdf_obstacle_max_spheres", default_value="5"),
            DeclareLaunchArgument("esdf_obstacle_smoothing_alpha", default_value="0.6"),
            DeclareLaunchArgument(
                "esdf_obstacle_smoothing_match_distance_m",
                default_value="0.18",
            ),
            DeclareLaunchArgument(
                "esdf_obstacle_min_persistent_frames",
                default_value="3",
                description="Consecutive frames an ESDF obstacle cluster must be seen in "
                "(matched within esdf_obstacle_smoothing_match_distance_m) before it is "
                "published as an obstacle sphere. Set to 1 to disable persistence gating.",
            ),
            DeclareLaunchArgument("esdf_surface_min_distance_m", default_value="-0.02"),
            DeclareLaunchArgument("esdf_surface_max_distance_m", default_value="0.12"),
            DeclareLaunchArgument("esdf_surface_max_robot_clearance_m", default_value="1.0"),
            DeclareLaunchArgument("esdf_surface_stride", default_value="1"),
            DeclareLaunchArgument("esdf_surface_max_points", default_value="2000"),
            DeclareLaunchArgument("esdf_slice_height_m", default_value="0.5"),
            DeclareLaunchArgument("esdf_slice_min_height_m", default_value="0.35"),
            DeclareLaunchArgument("esdf_slice_max_height_m", default_value="0.75"),
            DeclareLaunchArgument("vision_pipeline_rate_hz", default_value="10.0"),
            DeclareLaunchArgument("x", default_value="1.403303227"),
            DeclareLaunchArgument("y", default_value="-1.733655308"),
            DeclareLaunchArgument("z", default_value="0.912738743"),
            DeclareLaunchArgument("roll", default_value="-0.108681179"),
            DeclareLaunchArgument("pitch", default_value="0.233046210"),
            DeclareLaunchArgument("yaw", default_value="1.782183343"),
            container,
            static_tf,
            odom_tf,
            robot_base_tf,
            rb10_measured_joint_state_publisher,
            joint_state_normalizer,
            robot_state_publisher,
            bag_play,
            realsense_launch,
            nvblox_node,
            robot_collision_spheres,
            robot_depth_mask,
            depth_to_pointcloud,
            robot_pointcloud_filter,
            vision_closest_obstacle,
            obstacle_body_spheres,
            esdf_collision_query,
            esdf_obstacle_spheres,
            esdf_slice_obstacle_spheres,
            rviz_launch,
        ]
    )
