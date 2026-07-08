from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def generate_launch_description():
    default_bag_path = str(Path.home() / "bags" / "d435_esdf_20260707_140735")
    base_launch = PythonLaunchDescriptionSource(
        package_file("rmp_camera", "launch/d435_nvblox_static.launch.py")
    )

    def base_launch_arguments(run_realsense, use_sim_time):
        return {
            "run_rviz": LaunchConfiguration("run_rviz"),
            "rviz_config": package_file(
                "rmp_camera",
                "config/d435_nvblox_esdf_map.rviz",
            ),
            "camera_serial_number": LaunchConfiguration("camera_serial_number"),
            "global_frame": LaunchConfiguration("global_frame"),
            "camera_frame": LaunchConfiguration("camera_frame"),
            "x": LaunchConfiguration("x"),
            "y": LaunchConfiguration("y"),
            "z": LaunchConfiguration("z"),
            "roll": LaunchConfiguration("roll"),
            "pitch": LaunchConfiguration("pitch"),
            "yaw": LaunchConfiguration("yaw"),
            "esdf_slice_height_m": LaunchConfiguration("esdf_slice_height_m"),
            "esdf_slice_min_height_m": LaunchConfiguration("esdf_slice_min_height_m"),
            "esdf_slice_max_height_m": LaunchConfiguration("esdf_slice_max_height_m"),
            "run_realsense": run_realsense,
            "use_sim_time": use_sim_time,
            "esdf_mode": LaunchConfiguration("esdf_mode"),
            "publish_static_tf": "True",
            "publish_odom_tf": "True",
            "publish_robot_base_tf": "False",
            "run_collision_spheres": LaunchConfiguration("run_collision_spheres"),
            "run_robot_depth_mask": LaunchConfiguration("run_robot_depth_mask"),
            "run_rb10_measured_joint_state_publisher": LaunchConfiguration(
                "run_rb10_measured_joint_state_publisher"
            ),
            "run_joint_state_normalizer": LaunchConfiguration("run_joint_state_normalizer"),
            "run_robot_state_publisher": LaunchConfiguration("run_robot_state_publisher"),
            "run_vision_obstacle_pipeline": "True",
            "publish_obstacle_spheres": LaunchConfiguration("publish_obstacle_spheres"),
            "run_camera_closest_obstacles": LaunchConfiguration(
                "run_pointcloud_closest_obstacles"
            ),
            "run_obstacle_body_spheres": LaunchConfiguration("run_3d_obstacle_spheres"),
            "run_esdf_collision_query": "False",
            "run_esdf_obstacle_spheres": "False",
            "run_esdf_slice_obstacle_spheres": LaunchConfiguration(
                "run_esdf_slice_obstacle_spheres"
            ),
            "raw_depth_image_topic": LaunchConfiguration("raw_depth_image_topic"),
            "depth_camera_info_topic": LaunchConfiguration("depth_camera_info_topic"),
            "masked_depth_image_topic": LaunchConfiguration("masked_depth_image_topic"),
            "nvblox_depth_image_topic": LaunchConfiguration("nvblox_depth_image_topic"),
            "esdf_slice_obstacle_self_filter_robot_enabled": LaunchConfiguration(
                "esdf_slice_obstacle_self_filter_robot_enabled"
            ),
            "robot_margin_m": LaunchConfiguration("robot_margin_m"),
            "camera_obstacle_min_clearance_m": LaunchConfiguration(
                "camera_obstacle_min_clearance_m"
            ),
            "obstacle_body_marker_topic": "/rmp_camera/camera_obstacle_spheres",
            "obstacle_body_cloud_topic": "/rmp_camera/camera_obstacle_sphere_cloud",
            "obstacle_background_subtraction_enabled": LaunchConfiguration(
                "obstacle_background_subtraction_enabled"
            ),
            "obstacle_auto_capture_background_on_start": LaunchConfiguration(
                "obstacle_auto_capture_background_on_start"
            ),
            "obstacle_auto_capture_background_delay_s": LaunchConfiguration(
                "obstacle_auto_capture_background_delay_s"
            ),
            "obstacle_auto_capture_background_sample_count": LaunchConfiguration(
                "obstacle_auto_capture_background_sample_count"
            ),
            "obstacle_background_voxel_size_m": LaunchConfiguration(
                "obstacle_background_voxel_size_m"
            ),
            "obstacle_background_neighbor_voxels": LaunchConfiguration(
                "obstacle_background_neighbor_voxels"
            ),
            "obstacle_body_robot_proximity_filter_enabled": LaunchConfiguration(
                "obstacle_body_robot_proximity_filter_enabled"
            ),
            "obstacle_body_min_robot_clearance_m": LaunchConfiguration(
                "obstacle_body_min_robot_clearance_m"
            ),
            "obstacle_body_max_robot_clearance_m": LaunchConfiguration(
                "obstacle_body_max_robot_clearance_m"
            ),
            "esdf_surface_max_distance_m": "0.08",
            "obstacle_body_min_z_m": LaunchConfiguration("obstacle_body_min_z_m"),
            "obstacle_body_max_range_from_base_m": LaunchConfiguration(
                "obstacle_body_max_range_from_base_m"
            ),
            "obstacle_body_voxel_size_m": LaunchConfiguration("obstacle_body_voxel_size_m"),
            "obstacle_body_cluster_cell_size_m": LaunchConfiguration(
                "obstacle_body_cluster_cell_size_m"
            ),
            "obstacle_body_min_cluster_points": LaunchConfiguration(
                "obstacle_body_min_cluster_points"
            ),
            "obstacle_body_max_cluster_extent_m": "0.0",
            "obstacle_body_sphere_spacing_m": LaunchConfiguration(
                "obstacle_body_sphere_spacing_m"
            ),
            "obstacle_body_max_spheres_per_cluster": LaunchConfiguration(
                "obstacle_body_max_spheres_per_cluster"
            ),
            "obstacle_body_max_total_spheres": LaunchConfiguration(
                "obstacle_body_max_total_spheres"
            ),
            "obstacle_body_min_persistent_frames": LaunchConfiguration(
                "obstacle_body_min_persistent_frames"
            ),
            "publish_obstacle_body_debug_clouds": "False",
        }.items()

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "source",
                default_value="camera",
                choices=["camera", "bag"],
                description="Input source. camera uses the physical RealSense; bag starts ros2 bag play.",
            ),
            DeclareLaunchArgument("bag_path", default_value=default_bag_path),
            DeclareLaunchArgument("bag_rate", default_value="1.0"),
            DeclareLaunchArgument("run_rviz", default_value="True"),
            DeclareLaunchArgument("run_realsense", default_value="True"),
            DeclareLaunchArgument("use_sim_time", default_value="False"),
            DeclareLaunchArgument("esdf_mode", default_value="2d"),
            DeclareLaunchArgument("run_pointcloud_closest_obstacles", default_value="True"),
            DeclareLaunchArgument("run_3d_obstacle_spheres", default_value="False"),
            DeclareLaunchArgument("run_esdf_slice_obstacle_spheres", default_value="False"),
            DeclareLaunchArgument("esdf_slice_obstacle_self_filter_robot_enabled", default_value="True"),
            DeclareLaunchArgument("run_collision_spheres", default_value="True"),
            DeclareLaunchArgument("run_robot_depth_mask", default_value="True"),
            DeclareLaunchArgument("run_rb10_measured_joint_state_publisher", default_value="False"),
            DeclareLaunchArgument("run_joint_state_normalizer", default_value="True"),
            DeclareLaunchArgument("run_robot_state_publisher", default_value="True"),
            DeclareLaunchArgument("robot_margin_m", default_value="0.10"),
            DeclareLaunchArgument("publish_obstacle_spheres", default_value="True"),
            DeclareLaunchArgument("camera_obstacle_min_clearance_m", default_value="0.20"),
            DeclareLaunchArgument("obstacle_background_subtraction_enabled", default_value="True"),
            DeclareLaunchArgument("obstacle_auto_capture_background_on_start", default_value="True"),
            DeclareLaunchArgument("obstacle_auto_capture_background_delay_s", default_value="2.0"),
            DeclareLaunchArgument("obstacle_auto_capture_background_sample_count", default_value="10"),
            DeclareLaunchArgument("obstacle_background_voxel_size_m", default_value="0.06"),
            DeclareLaunchArgument("obstacle_background_neighbor_voxels", default_value="1"),
            DeclareLaunchArgument(
                "obstacle_body_robot_proximity_filter_enabled",
                default_value="True",
            ),
            DeclareLaunchArgument("obstacle_body_min_robot_clearance_m", default_value="0.20"),
            DeclareLaunchArgument("obstacle_body_max_robot_clearance_m", default_value="1.0"),
            DeclareLaunchArgument("obstacle_body_min_z_m", default_value="0.10"),
            DeclareLaunchArgument("obstacle_body_max_range_from_base_m", default_value="2.2"),
            DeclareLaunchArgument("obstacle_body_voxel_size_m", default_value="0.06"),
            DeclareLaunchArgument("obstacle_body_cluster_cell_size_m", default_value="0.16"),
            DeclareLaunchArgument("obstacle_body_min_cluster_points", default_value="20"),
            DeclareLaunchArgument("obstacle_body_sphere_spacing_m", default_value="0.28"),
            DeclareLaunchArgument("obstacle_body_max_spheres_per_cluster", default_value="4"),
            DeclareLaunchArgument("obstacle_body_max_total_spheres", default_value="8"),
            DeclareLaunchArgument("obstacle_body_min_persistent_frames", default_value="4"),
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
            DeclareLaunchArgument(
                "nvblox_depth_image_topic",
                default_value="/rmp_camera/robot_masked_depth/image_rect_raw",
            ),
            DeclareLaunchArgument("camera_serial_number", default_value="117322070340"),
            DeclareLaunchArgument("global_frame", default_value="base_link"),
            DeclareLaunchArgument("camera_frame", default_value="camera0_link"),
            DeclareLaunchArgument("esdf_slice_height_m", default_value="0.5"),
            DeclareLaunchArgument("esdf_slice_min_height_m", default_value="0.35"),
            DeclareLaunchArgument("esdf_slice_max_height_m", default_value="0.75"),
            DeclareLaunchArgument("x", default_value="1.403303227"),
            DeclareLaunchArgument("y", default_value="-1.733655308"),
            DeclareLaunchArgument("z", default_value="0.912738743"),
            DeclareLaunchArgument("roll", default_value="-0.108681179"),
            DeclareLaunchArgument("pitch", default_value="0.233046210"),
            DeclareLaunchArgument("yaw", default_value="1.782183343"),
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "play",
                    LaunchConfiguration("bag_path"),
                    "--clock",
                    "--loop",
                    "--rate",
                    LaunchConfiguration("bag_rate"),
                ],
                output="screen",
                condition=LaunchConfigurationEquals("source", "bag"),
            ),
            IncludeLaunchDescription(
                base_launch,
                launch_arguments=base_launch_arguments(
                    LaunchConfiguration("run_realsense"),
                    LaunchConfiguration("use_sim_time"),
                ),
                condition=LaunchConfigurationEquals("source", "camera"),
            ),
            IncludeLaunchDescription(
                base_launch,
                launch_arguments=base_launch_arguments("False", "True"),
                condition=LaunchConfigurationEquals("source", "bag"),
            ),
        ]
    )
