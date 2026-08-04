from glob import glob

from setuptools import setup

package_name = "rmp_camera"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.rviz")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/urdf", glob("urdf/*.urdf")),
        (f"share/{package_name}/urdf/stl", glob("urdf/stl/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="son_rb",
    maintainer_email="user@example.com",
    description="Camera-side obstacle processing package for RMPflow.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "robot_collision_sphere_node = rmp_camera.robot_collision_sphere_node:main",
            "esdf_query_debug_node = rmp_camera.esdf_query_debug_node:main",
            "esdf_collision_query_node = rmp_camera.esdf_collision_query_node:main",
            "esdf_obstacle_sphere_node = rmp_camera.esdf_obstacle_sphere_node:main",
            "esdf_medial_sphere_node = rmp_camera.esdf_medial_sphere_node:main",
            "dynamic_obstacle_sphere_node = rmp_camera.dynamic_obstacle_sphere_node:main",
            "obstacle_sphere_fusion_node = rmp_camera.obstacle_sphere_fusion_node:main",
            "nvblox_esdf_slice_obstacle_sphere_node = rmp_camera.nvblox_esdf_slice_obstacle_sphere_node:main",
            "joint_state_normalizer_node = rmp_camera.joint_state_normalizer_node:main",
            "rb10_measured_joint_state_node = rmp_camera.rb10_measured_joint_state_node:main",
            "charuco_board_generator = rmp_camera.charuco_board_generator:main",
            "charuco_eye_to_hand_calibrator_node = rmp_camera.charuco_eye_to_hand_calibrator_node:main",
            "robot_pointcloud_filter_node = rmp_camera.robot_pointcloud_filter_node:main",
            "robot_depth_mask_node = rmp_camera.robot_depth_mask_node:main",
            "depth_to_pointcloud_node = rmp_camera.depth_to_pointcloud_node:main",
            "vision_closest_obstacle_node = rmp_camera.vision_closest_obstacle_node:main",
            "obstacle_body_sphere_node = rmp_camera.obstacle_body_sphere_node:main",
            "camera_obstacle_sphere_echo = rmp_camera.camera_obstacle_sphere_echo:main",
        ],
    },
)
