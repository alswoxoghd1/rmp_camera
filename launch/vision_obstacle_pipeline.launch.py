from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    run_robot_pointcloud_filter = LaunchConfiguration("run_robot_pointcloud_filter")
    run_vision_closest_obstacle = LaunchConfiguration("run_vision_closest_obstacle")
    run_obstacle_body_spheres = LaunchConfiguration("run_obstacle_body_spheres")
    publish_obstacle_spheres = LaunchConfiguration("publish_obstacle_spheres")
    run_vision_closest_obstacle_with_publish = PythonExpression(
        [
            "'",
            publish_obstacle_spheres,
            "'.lower() in ('true', '1') and '",
            run_vision_closest_obstacle,
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
    input_cloud_topic = LaunchConfiguration("input_cloud_topic")
    robot_free_cloud_topic = LaunchConfiguration("robot_free_cloud_topic")
    publish_robot_removal_debug = LaunchConfiguration("publish_robot_removal_debug")
    robot_pointcloud_removed_points_topic = LaunchConfiguration(
        "robot_pointcloud_removed_points_topic"
    )
    robot_sphere_marker_topic = LaunchConfiguration("robot_sphere_marker_topic")
    target_frame = LaunchConfiguration("target_frame")
    robot_margin_m = LaunchConfiguration("robot_margin_m")
    use_time_synchronized_robot_markers = LaunchConfiguration(
        "use_time_synchronized_robot_markers"
    )
    robot_marker_buffer_duration_s = LaunchConfiguration("robot_marker_buffer_duration_s")
    robot_marker_max_stamp_delta_s = LaunchConfiguration("robot_marker_max_stamp_delta_s")
    fallback_to_latest_marker_on_time_miss = LaunchConfiguration(
        "fallback_to_latest_marker_on_time_miss"
    )
    filter_voxel_size_m = LaunchConfiguration("filter_voxel_size_m")
    filter_max_input_points = LaunchConfiguration("filter_max_input_points")
    filter_max_publish_points = LaunchConfiguration("filter_max_publish_points")
    closest_voxel_size_m = LaunchConfiguration("closest_voxel_size_m")
    closest_max_points = LaunchConfiguration("closest_max_points")
    active_range_m = LaunchConfiguration("active_range_m")
    min_clearance_m = LaunchConfiguration("min_clearance_m")
    obstacle_radius_m = LaunchConfiguration("obstacle_radius_m")
    cluster_radius_m = LaunchConfiguration("cluster_radius_m")
    max_obstacle_spheres = LaunchConfiguration("max_obstacle_spheres")
    obstacle_cloud_topic = LaunchConfiguration("obstacle_cloud_topic")
    obstacle_body_marker_topic = LaunchConfiguration("obstacle_body_marker_topic")
    obstacle_body_cloud_topic = LaunchConfiguration("obstacle_body_cloud_topic")
    obstacle_body_candidate_cloud_topic = LaunchConfiguration("obstacle_body_candidate_cloud_topic")
    obstacle_body_cluster_cloud_topic = LaunchConfiguration("obstacle_body_cluster_cloud_topic")
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
    obstacle_body_max_spheres_per_cluster = LaunchConfiguration("obstacle_body_max_spheres_per_cluster")
    obstacle_body_max_total_spheres = LaunchConfiguration("obstacle_body_max_total_spheres")
    obstacle_body_marker_lifetime_s = LaunchConfiguration("obstacle_body_marker_lifetime_s")
    publish_obstacle_body_debug_clouds = LaunchConfiguration("publish_obstacle_body_debug_clouds")
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
    obstacle_body_min_sphere_radius_m = LaunchConfiguration("obstacle_body_min_sphere_radius_m")
    obstacle_body_max_sphere_radius_m = LaunchConfiguration("obstacle_body_max_sphere_radius_m")
    obstacle_body_sphere_padding_m = LaunchConfiguration("obstacle_body_sphere_padding_m")
    obstacle_body_min_persistent_frames = LaunchConfiguration("obstacle_body_min_persistent_frames")
    obstacle_body_persistence_match_distance_m = LaunchConfiguration(
        "obstacle_body_persistence_match_distance_m"
    )
    obstacle_body_persistence_smoothing_alpha = LaunchConfiguration(
        "obstacle_body_persistence_smoothing_alpha"
    )
    max_rate_hz = LaunchConfiguration("max_rate_hz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("run_robot_pointcloud_filter", default_value="True"),
            DeclareLaunchArgument(
                "publish_obstacle_spheres",
                default_value="True",
                description="Publish camera/obstacle sphere outputs. Set false to keep pointcloud filtering but stop obstacle sphere topics.",
            ),
            DeclareLaunchArgument("run_vision_closest_obstacle", default_value="False"),
            DeclareLaunchArgument("run_obstacle_body_spheres", default_value="True"),
            DeclareLaunchArgument(
                "input_cloud_topic",
                default_value="/nvblox_node/back_projected_depth/camera0_depth_optical_frame",
            ),
            DeclareLaunchArgument("robot_free_cloud_topic", default_value="/rmp_camera/robot_free_pointcloud"),
            DeclareLaunchArgument("publish_robot_removal_debug", default_value="True"),
            DeclareLaunchArgument(
                "robot_pointcloud_removed_points_topic",
                default_value="/rmp_camera/robot_pointcloud_removed_points",
            ),
            DeclareLaunchArgument(
                "robot_sphere_marker_topic",
                default_value="/rmp_camera/robot_collision_sphere_markers",
            ),
            DeclareLaunchArgument("target_frame", default_value="base_link"),
            DeclareLaunchArgument("robot_margin_m", default_value="0.12"),
            DeclareLaunchArgument("use_time_synchronized_robot_markers", default_value="True"),
            DeclareLaunchArgument("robot_marker_buffer_duration_s", default_value="1.0"),
            DeclareLaunchArgument("robot_marker_max_stamp_delta_s", default_value="0.15"),
            DeclareLaunchArgument("fallback_to_latest_marker_on_time_miss", default_value="True"),
            DeclareLaunchArgument("filter_voxel_size_m", default_value="0.025"),
            DeclareLaunchArgument("filter_max_input_points", default_value="120000"),
            DeclareLaunchArgument("filter_max_publish_points", default_value="40000"),
            DeclareLaunchArgument("closest_voxel_size_m", default_value="0.035"),
            DeclareLaunchArgument("closest_max_points", default_value="30000"),
            DeclareLaunchArgument("min_clearance_m", default_value="0.30"),
            DeclareLaunchArgument("active_range_m", default_value="1.0"),
            DeclareLaunchArgument("obstacle_radius_m", default_value="0.06"),
            DeclareLaunchArgument("cluster_radius_m", default_value="0.14"),
            DeclareLaunchArgument("max_obstacle_spheres", default_value="5"),
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
            DeclareLaunchArgument("publish_obstacle_body_debug_clouds", default_value="False"),
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
            DeclareLaunchArgument("max_rate_hz", default_value="10.0"),
            Node(
                package="rmp_camera",
                executable="robot_pointcloud_filter_node",
                name="robot_pointcloud_filter_node",
                output="screen",
                parameters=[
                    {
                        "input_cloud_topic": input_cloud_topic,
                        "robot_sphere_marker_topic": robot_sphere_marker_topic,
                        "output_cloud_topic": robot_free_cloud_topic,
                        "removed_cloud_topic": robot_pointcloud_removed_points_topic,
                        "publish_removed_cloud": ParameterValue(
                            publish_robot_removal_debug,
                            value_type=bool,
                        ),
                        "target_frame": target_frame,
                        "robot_margin_m": robot_margin_m,
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
                        "voxel_size_m": filter_voxel_size_m,
                        "max_input_points": filter_max_input_points,
                        "max_publish_points": filter_max_publish_points,
                        "max_rate_hz": max_rate_hz,
                    }
                ],
                condition=IfCondition(run_robot_pointcloud_filter),
            ),
            Node(
                package="rmp_camera",
                executable="vision_closest_obstacle_node",
                name="vision_closest_obstacle_node",
                output="screen",
                parameters=[
                    {
                        "input_cloud_topic": robot_free_cloud_topic,
                        "robot_sphere_marker_topic": robot_sphere_marker_topic,
                        "target_frame": target_frame,
                        "active_range_m": active_range_m,
                        "min_clearance_m": min_clearance_m,
                        "obstacle_radius_m": obstacle_radius_m,
                        "cluster_radius_m": cluster_radius_m,
                        "max_obstacle_spheres": max_obstacle_spheres,
                        "obstacle_cloud_topic": obstacle_cloud_topic,
                        "voxel_size_m": closest_voxel_size_m,
                        "max_points": closest_max_points,
                        "max_rate_hz": max_rate_hz,
                    }
                ],
                condition=IfCondition(run_vision_closest_obstacle_with_publish),
            ),
            Node(
                package="rmp_camera",
                executable="obstacle_body_sphere_node",
                name="obstacle_body_sphere_node",
                output="screen",
                parameters=[
                    {
                        "input_cloud_topic": robot_free_cloud_topic,
                        "robot_sphere_marker_topic": robot_sphere_marker_topic,
                        "obstacle_marker_topic": obstacle_body_marker_topic,
                        "obstacle_cloud_topic": obstacle_body_cloud_topic,
                        "candidate_cloud_topic": obstacle_body_candidate_cloud_topic,
                        "cluster_cloud_topic": obstacle_body_cluster_cloud_topic,
                        "target_frame": target_frame,
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
                        "max_rate_hz": max_rate_hz,
                    }
                ],
                condition=IfCondition(run_obstacle_body_spheres_with_publish),
            ),
        ]
    )
