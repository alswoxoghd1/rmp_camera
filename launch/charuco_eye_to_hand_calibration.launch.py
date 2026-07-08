from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    base_frame = LaunchConfiguration("base_frame")
    gripper_frame = LaunchConfiguration("gripper_frame")
    output_camera_frame = LaunchConfiguration("output_camera_frame")
    dictionary = LaunchConfiguration("dictionary")
    squares_x = LaunchConfiguration("squares_x")
    squares_y = LaunchConfiguration("squares_y")
    square_length_m = LaunchConfiguration("square_length_m")
    marker_length_m = LaunchConfiguration("marker_length_m")
    min_charuco_corners = LaunchConfiguration("min_charuco_corners")
    sample_path = LaunchConfiguration("sample_path")
    result_path = LaunchConfiguration("result_path")
    display = LaunchConfiguration("display")
    auto_capture = LaunchConfiguration("auto_capture")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_z = LaunchConfiguration("initial_z")
    initial_roll = LaunchConfiguration("initial_roll")
    initial_pitch = LaunchConfiguration("initial_pitch")
    initial_yaw = LaunchConfiguration("initial_yaw")

    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera0/camera/color/image_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera0/camera/color/camera_info"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("gripper_frame", default_value="tcp"),
            DeclareLaunchArgument("output_camera_frame", default_value="camera0_link"),
            DeclareLaunchArgument("dictionary", default_value="DICT_5X5_100"),
            DeclareLaunchArgument("squares_x", default_value="5"),
            DeclareLaunchArgument("squares_y", default_value="7"),
            DeclareLaunchArgument("square_length_m", default_value="0.04"),
            DeclareLaunchArgument("marker_length_m", default_value="0.03"),
            DeclareLaunchArgument("min_charuco_corners", default_value="8"),
            DeclareLaunchArgument("sample_path", default_value="/tmp/rmp_camera_charuco_samples.yaml"),
            DeclareLaunchArgument("result_path", default_value="/tmp/rmp_camera_charuco_result.yaml"),
            DeclareLaunchArgument("display", default_value="False"),
            DeclareLaunchArgument("auto_capture", default_value="False"),
            DeclareLaunchArgument("initial_x", default_value="-1.4"),
            DeclareLaunchArgument("initial_y", default_value="-1.3"),
            DeclareLaunchArgument("initial_z", default_value="0.9"),
            DeclareLaunchArgument("initial_roll", default_value="0.0"),
            DeclareLaunchArgument("initial_pitch", default_value="0.0"),
            DeclareLaunchArgument("initial_yaw", default_value="0.0"),
            Node(
                package="rmp_camera",
                executable="charuco_eye_to_hand_calibrator_node",
                name="charuco_eye_to_hand_calibrator",
                output="screen",
                parameters=[
                    {
                        "image_topic": image_topic,
                        "camera_info_topic": camera_info_topic,
                        "base_frame": base_frame,
                        "gripper_frame": gripper_frame,
                        "output_camera_frame": output_camera_frame,
                        "dictionary": dictionary,
                        "squares_x": squares_x,
                        "squares_y": squares_y,
                        "square_length_m": square_length_m,
                        "marker_length_m": marker_length_m,
                        "min_charuco_corners": min_charuco_corners,
                        "sample_path": sample_path,
                        "result_path": result_path,
                        "display": display,
                        "auto_capture": auto_capture,
                        "initial_x": initial_x,
                        "initial_y": initial_y,
                        "initial_z": initial_z,
                        "initial_roll": initial_roll,
                        "initial_pitch": initial_pitch,
                        "initial_yaw": initial_yaw,
                    }
                ],
            ),
        ]
    )
