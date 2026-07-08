from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def package_file(package_name, relative_path):
    return str(Path(get_package_share_directory(package_name)) / relative_path)


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("run_rviz", default_value="True"),
            DeclareLaunchArgument("camera_serial_number", default_value="117322070340"),
            DeclareLaunchArgument("robot_ip", default_value="192.168.111.50"),
            DeclareLaunchArgument("esdf_collision_robot_clear_margin_m", default_value="0.12"),
            DeclareLaunchArgument("esdf_collision_self_filter_margin_m", default_value="0.06"),
            DeclareLaunchArgument("esdf_obstacle_active_range_m", default_value="1.0"),
            DeclareLaunchArgument("esdf_obstacle_max_spheres", default_value="5"),
            DeclareLaunchArgument("esdf_surface_max_distance_m", default_value="0.12"),
            DeclareLaunchArgument("esdf_surface_max_points", default_value="2000"),
            DeclareLaunchArgument("esdf_slice_height_m", default_value="0.5"),
            DeclareLaunchArgument("esdf_slice_min_height_m", default_value="0.35"),
            DeclareLaunchArgument("esdf_slice_max_height_m", default_value="0.75"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    package_file("rmp_camera", "launch/d435_nvblox_static.launch.py")
                ),
                launch_arguments={
                    "run_rviz": LaunchConfiguration("run_rviz"),
                    "camera_serial_number": LaunchConfiguration("camera_serial_number"),
                    "robot_ip": LaunchConfiguration("robot_ip"),
                    "run_esdf_collision_query": "True",
                    "run_esdf_obstacle_spheres": "True",
                    "run_vision_obstacle_pipeline": "False",
                    "run_camera_closest_obstacles": "False",
                    "run_obstacle_body_spheres": "False",
                    "publish_obstacle_body_debug_clouds": "False",
                    "esdf_collision_publish_all_valid_spheres": "True",
                    "esdf_collision_robot_clear_margin_m": LaunchConfiguration(
                        "esdf_collision_robot_clear_margin_m"
                    ),
                    "esdf_collision_self_filter_margin_m": LaunchConfiguration(
                        "esdf_collision_self_filter_margin_m"
                    ),
                    "esdf_obstacle_active_range_m": LaunchConfiguration(
                        "esdf_obstacle_active_range_m"
                    ),
                    "esdf_obstacle_max_spheres": LaunchConfiguration("esdf_obstacle_max_spheres"),
                    "esdf_surface_max_distance_m": LaunchConfiguration(
                        "esdf_surface_max_distance_m"
                    ),
                    "esdf_surface_max_points": LaunchConfiguration("esdf_surface_max_points"),
                    "esdf_slice_height_m": LaunchConfiguration("esdf_slice_height_m"),
                    "esdf_slice_min_height_m": LaunchConfiguration("esdf_slice_min_height_m"),
                    "esdf_slice_max_height_m": LaunchConfiguration("esdf_slice_max_height_m"),
                }.items(),
            ),
        ]
    )
