import math
import os
import shutil
import time
import warnings

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
import tf2_ros
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="A NumPy version.*")
    from scipy.optimize import least_squares

from rmp_camera.charuco_common import (
    aruco_dictionary,
    invert_transform,
    list_from_matrix,
    matrix_from_quaternion,
    rpy_from_matrix,
    rotvec_from_matrix,
    transform_from_rt,
    transform_from_rotvec_xyz,
    transform_from_rvec_tvec,
    transform_from_xyz_rpy,
)


def stamp_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def parameter_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class ChArUcoEyeToHandCalibrator(Node):
    def __init__(self):
        super().__init__("charuco_eye_to_hand_calibrator")

        self.declare_parameter("image_topic", "/camera0/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera0/camera/color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("gripper_frame", "tcp")
        self.declare_parameter("output_camera_frame", "camera0_link")
        self.declare_parameter("dictionary", "DICT_5X5_100")
        self.declare_parameter("squares_x", 5)
        self.declare_parameter("squares_y", 7)
        self.declare_parameter("square_length_m", 0.04)
        self.declare_parameter("marker_length_m", 0.03)
        self.declare_parameter("min_charuco_corners", 8)
        self.declare_parameter("sample_path", "/tmp/rmp_camera_charuco_samples.yaml")
        self.declare_parameter("result_path", "/tmp/rmp_camera_charuco_result.yaml")
        self.declare_parameter("display", False)
        self.declare_parameter("auto_capture", False)
        self.declare_parameter("auto_capture_interval_s", 3.0)
        self.declare_parameter("min_motion_translation_m", 0.04)
        self.declare_parameter("min_motion_rotation_deg", 8.0)
        self.declare_parameter("tf_lookup_timeout_s", 0.2)
        self.declare_parameter("initial_x", -1.4)
        self.declare_parameter("initial_y", -1.3)
        self.declare_parameter("initial_z", 0.9)
        self.declare_parameter("initial_roll", 0.0)
        self.declare_parameter("initial_pitch", 0.0)
        self.declare_parameter("initial_yaw", 0.0)
        self.declare_parameter("rotation_residual_weight", 1.0)
        self.declare_parameter("translation_residual_weight", 1.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.gripper_frame = self.get_parameter("gripper_frame").value
        self.output_camera_frame = self.get_parameter("output_camera_frame").value
        self.dictionary_name = self.get_parameter("dictionary").value
        self.squares_x = int(self.get_parameter("squares_x").value)
        self.squares_y = int(self.get_parameter("squares_y").value)
        self.square_length_m = float(self.get_parameter("square_length_m").value)
        self.marker_length_m = float(self.get_parameter("marker_length_m").value)
        self.min_charuco_corners = int(self.get_parameter("min_charuco_corners").value)
        self.sample_path = os.path.abspath(os.path.expanduser(self.get_parameter("sample_path").value))
        self.result_path = os.path.abspath(os.path.expanduser(self.get_parameter("result_path").value))
        self.display = parameter_bool(self.get_parameter("display").value)
        self.auto_capture = parameter_bool(self.get_parameter("auto_capture").value)
        self.auto_capture_interval_s = float(self.get_parameter("auto_capture_interval_s").value)
        self.min_motion_translation_m = float(self.get_parameter("min_motion_translation_m").value)
        self.min_motion_rotation_rad = math.radians(
            float(self.get_parameter("min_motion_rotation_deg").value)
        )
        self.tf_lookup_timeout = Duration(seconds=float(self.get_parameter("tf_lookup_timeout_s").value))
        self.rotation_residual_weight = float(self.get_parameter("rotation_residual_weight").value)
        self.translation_residual_weight = float(self.get_parameter("translation_residual_weight").value)

        self.dictionary = aruco_dictionary(self.dictionary_name)
        self.board = cv2.aruco.CharucoBoard_create(
            self.squares_x,
            self.squares_y,
            self.square_length_m,
            self.marker_length_m,
            self.dictionary,
        )

        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_detection = None
        self.samples = []
        self.last_status_log_time = 0.0
        self.last_auto_capture_time = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        service_prefix = self.get_fully_qualified_name()
        self.capture_service_name = f"{service_prefix}/capture"
        self.solve_service_name = f"{service_prefix}/solve"
        self.clear_service_name = f"{service_prefix}/clear_samples"
        self.create_service(Trigger, self.capture_service_name, self.capture_service_callback)
        self.create_service(Trigger, self.solve_service_name, self.solve_service_callback)
        self.create_service(Trigger, self.clear_service_name, self.clear_service_callback)
        self.create_timer(0.5, self.auto_capture_timer_callback)

        self.get_logger().info(
            "ChArUco eye-to-hand calibrator ready. "
            f"Image={self.image_topic}, camera_info={self.camera_info_topic}, "
            f"base={self.base_frame}, gripper={self.gripper_frame}, "
            f"output_camera={self.output_camera_frame}"
        )
        self.get_logger().info(
            "Move the robot to varied poses and call "
            f"`ros2 service call {self.capture_service_name} std_srvs/srv/Trigger {{}}`. "
            f"After 15-25 samples, call `{self.solve_service_name}`."
        )

    def camera_info_callback(self, msg):
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            self.log_status("Waiting for camera_info...")
            return

        try:
            gray, color = self.ros_image_to_gray(msg)
        except Exception as exc:
            self.log_status(f"Image conversion failed: {exc}", warn=True)
            return

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if ids is None or len(ids) == 0:
            self.last_detection = None
            self.log_status("No ChArUco markers detected.")
            self.show_debug_image(color, corners, ids, None, None, None, False)
            return

        count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            self.board,
            self.camera_matrix,
            self.dist_coeffs,
        )
        if charuco_ids is None or count < self.min_charuco_corners:
            self.last_detection = None
            self.log_status(
                f"Detected markers, but only {int(count)} ChArUco corners "
                f"(need {self.min_charuco_corners})."
            )
            self.show_debug_image(color, corners, ids, charuco_corners, charuco_ids, None, False)
            return

        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self.board,
            self.camera_matrix,
            self.dist_coeffs,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
        )
        if not ok:
            self.last_detection = None
            self.log_status("ChArUco pose estimation failed.", warn=True)
            self.show_debug_image(color, corners, ids, charuco_corners, charuco_ids, None, False)
            return

        camera_T_board = transform_from_rvec_tvec(rvec, tvec)
        self.last_detection = {
            "stamp": msg.header.stamp,
            "stamp_float": stamp_to_float(msg.header.stamp),
            "camera_frame": msg.header.frame_id,
            "camera_T_board": camera_T_board,
            "charuco_corner_count": int(count),
            "marker_count": int(len(ids)),
        }
        self.log_status(
            f"Detected ChArUco: {int(count)} corners, {len(ids)} markers, "
            f"frame={msg.header.frame_id}, samples={len(self.samples)}"
        )
        self.show_debug_image(color, corners, ids, charuco_corners, charuco_ids, (rvec, tvec), True)

    def ros_image_to_gray(self, msg):
        encoding = msg.encoding.lower()
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding in ("rgb8", "bgr8"):
            channels = 3
            row_pixels = msg.step // channels
            image = data.reshape((msg.height, row_pixels, channels))[:, : msg.width, :]
            image = np.ascontiguousarray(image)
            if encoding == "rgb8":
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                color = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                color = image.copy()
            return gray, color

        if encoding in ("rgba8", "bgra8"):
            channels = 4
            row_pixels = msg.step // channels
            image = data.reshape((msg.height, row_pixels, channels))[:, : msg.width, :]
            image = np.ascontiguousarray(image)
            if encoding == "rgba8":
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                color = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                color = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return gray, color

        if encoding in ("mono8", "8uc1"):
            row_pixels = msg.step
            gray = data.reshape((msg.height, row_pixels))[:, : msg.width]
            gray = np.ascontiguousarray(gray)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray, color

        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    def show_debug_image(self, color, marker_corners, marker_ids, charuco_corners, charuco_ids, pose, ok):
        if not self.display:
            return
        try:
            debug = color.copy()
            if marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
            if charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
            if ok and pose is not None:
                rvec, tvec = pose
                cv2.drawFrameAxes(
                    debug,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.square_length_m,
                )
            cv2.imshow("rmp_camera ChArUco calibration", debug)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().warn(f"Debug display failed: {exc}")
            self.display = False

    def log_status(self, message, warn=False):
        now = time.monotonic()
        if now - self.last_status_log_time < 2.0:
            return
        self.last_status_log_time = now
        if warn:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def auto_capture_timer_callback(self):
        if not self.auto_capture:
            return
        now = time.monotonic()
        if now - self.last_auto_capture_time < self.auto_capture_interval_s:
            return
        success, message = self.capture_current(reason="auto")
        if success:
            self.last_auto_capture_time = now
            self.get_logger().info(message)

    def capture_service_callback(self, request, response):
        del request
        response.success, response.message = self.capture_current(reason="manual")
        return response

    def solve_service_callback(self, request, response):
        del request
        response.success, response.message = self.solve()
        return response

    def clear_service_callback(self, request, response):
        del request
        backup_path = None
        if self.samples and os.path.exists(self.sample_path):
            root, ext = os.path.splitext(self.sample_path)
            backup_path = f"{root}.backup_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.yaml'}"
            shutil.copy2(self.sample_path, backup_path)
        self.samples = []
        self.write_samples()
        response.success = True
        if backup_path:
            response.message = f"Cleared all ChArUco calibration samples. Backup: {backup_path}"
        else:
            response.message = "Cleared all ChArUco calibration samples."
        return response

    def capture_current(self, reason):
        detection = self.last_detection
        if detection is None:
            return False, "No valid ChArUco pose is currently visible."

        try:
            base_T_gripper = self.lookup_transform_matrix(
                self.base_frame,
                self.gripper_frame,
                detection["stamp"],
            )
        except Exception as exc:
            return False, f"Could not read TF {self.base_frame}->{self.gripper_frame}: {exc}"

        if self.samples and reason == "auto":
            last = np.asarray(self.samples[-1]["base_T_gripper"], dtype=np.float64)
            relative = invert_transform(last) @ base_T_gripper
            translation_delta = float(np.linalg.norm(relative[:3, 3]))
            rotation_delta = float(np.linalg.norm(rotvec_from_matrix(relative[:3, :3])))
            if (
                translation_delta < self.min_motion_translation_m
                and rotation_delta < self.min_motion_rotation_rad
            ):
                return False, "Robot motion since last sample is too small."

        sample = {
            "index": len(self.samples) + 1,
            "stamp": detection["stamp_float"],
            "camera_frame": detection["camera_frame"],
            "base_frame": self.base_frame,
            "gripper_frame": self.gripper_frame,
            "charuco_corner_count": detection["charuco_corner_count"],
            "marker_count": detection["marker_count"],
            "base_T_gripper": list_from_matrix(base_T_gripper),
            "camera_T_board": list_from_matrix(detection["camera_T_board"]),
        }
        self.samples.append(sample)
        self.write_samples()
        return (
            True,
            f"Saved ChArUco sample {len(self.samples)} "
            f"({sample['charuco_corner_count']} corners) to {self.sample_path}",
        )

    def lookup_transform_matrix(self, target_frame, source_frame, stamp=None):
        if stamp is None:
            time_obj = rclpy.time.Time()
        else:
            time_obj = rclpy.time.Time.from_msg(stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time_obj,
                timeout=self.tf_lookup_timeout,
            )
        except Exception:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=self.tf_lookup_timeout,
            )
        return self.transform_msg_to_matrix(transform.transform)

    @staticmethod
    def transform_msg_to_matrix(transform):
        translation = np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=np.float64,
        )
        quaternion = np.array(
            [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ],
            dtype=np.float64,
        )
        return transform_from_rt(matrix_from_quaternion(quaternion), translation)

    def write_samples(self):
        os.makedirs(os.path.dirname(self.sample_path) or ".", exist_ok=True)
        data = {
            "board": {
                "dictionary": self.dictionary_name,
                "squares_x": self.squares_x,
                "squares_y": self.squares_y,
                "square_length_m": self.square_length_m,
                "marker_length_m": self.marker_length_m,
            },
            "base_frame": self.base_frame,
            "gripper_frame": self.gripper_frame,
            "output_camera_frame": self.output_camera_frame,
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        with open(self.sample_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)

    def solve(self):
        if len(self.samples) < 4:
            return False, f"Need at least 4 samples; current sample count is {len(self.samples)}."

        camera_frames = {sample["camera_frame"] for sample in self.samples}
        if len(camera_frames) != 1:
            return False, f"Samples contain multiple camera frames: {sorted(camera_frames)}"
        observation_camera_frame = next(iter(camera_frames))

        try:
            output_T_observation = self.get_output_to_observation_transform(observation_camera_frame)
        except Exception as exc:
            return False, f"Could not read TF {self.output_camera_frame}->{observation_camera_frame}: {exc}"

        base_T_gripper = [
            np.asarray(sample["base_T_gripper"], dtype=np.float64) for sample in self.samples
        ]
        observation_T_board = [
            np.asarray(sample["camera_T_board"], dtype=np.float64) for sample in self.samples
        ]

        initial_base_T_output = transform_from_xyz_rpy(
            [
                float(self.get_parameter("initial_x").value),
                float(self.get_parameter("initial_y").value),
                float(self.get_parameter("initial_z").value),
            ],
            [
                float(self.get_parameter("initial_roll").value),
                float(self.get_parameter("initial_pitch").value),
                float(self.get_parameter("initial_yaw").value),
            ],
        )
        initial_base_T_observation = initial_base_T_output @ output_T_observation
        initial_gripper_T_board = (
            invert_transform(base_T_gripper[0])
            @ initial_base_T_observation
            @ observation_T_board[0]
        )
        p0 = np.concatenate(
            [
                rotvec_from_matrix(initial_base_T_observation[:3, :3]),
                initial_base_T_observation[:3, 3],
                rotvec_from_matrix(initial_gripper_T_board[:3, :3]),
                initial_gripper_T_board[:3, 3],
            ]
        )

        def residual(params):
            base_T_observation = transform_from_rotvec_xyz(params[0:3], params[3:6])
            gripper_T_board = transform_from_rotvec_xyz(params[6:9], params[9:12])
            residuals = []
            for b_T_g, c_T_board in zip(base_T_gripper, observation_T_board):
                expected_base_T_board = b_T_g @ gripper_T_board
                observed_base_T_board = base_T_observation @ c_T_board
                error = invert_transform(expected_base_T_board) @ observed_base_T_board
                residuals.extend(
                    (rotvec_from_matrix(error[:3, :3]) * self.rotation_residual_weight).tolist()
                )
                residuals.extend((error[:3, 3] * self.translation_residual_weight).tolist())
            return np.asarray(residuals, dtype=np.float64)

        result = least_squares(
            residual,
            p0,
            method="trf",
            loss="soft_l1",
            f_scale=0.01,
            max_nfev=2000,
        )

        base_T_observation = transform_from_rotvec_xyz(result.x[0:3], result.x[3:6])
        gripper_T_board = transform_from_rotvec_xyz(result.x[6:9], result.x[9:12])
        base_T_output = base_T_observation @ invert_transform(output_T_observation)

        per_sample_errors = self.compute_sample_errors(
            base_T_gripper,
            observation_T_board,
            base_T_observation,
            gripper_T_board,
        )
        mean_translation_error = float(np.mean([error["translation_m"] for error in per_sample_errors]))
        mean_rotation_error = float(np.mean([error["rotation_deg"] for error in per_sample_errors]))
        max_translation_error = float(np.max([error["translation_m"] for error in per_sample_errors]))
        max_rotation_error = float(np.max([error["rotation_deg"] for error in per_sample_errors]))

        roll, pitch, yaw = rpy_from_matrix(base_T_output[:3, :3])
        x, y, z = base_T_output[:3, 3].tolist()
        static_tf_command = (
            "ros2 run tf2_ros static_transform_publisher "
            f"--x {x:.9f} --y {y:.9f} --z {z:.9f} "
            f"--roll {roll:.9f} --pitch {pitch:.9f} --yaw {yaw:.9f} "
            f"--frame-id {self.base_frame} --child-frame-id {self.output_camera_frame}"
        )
        launch_arguments = (
            f"x:={x:.9f} y:={y:.9f} z:={z:.9f} "
            f"roll:={roll:.9f} pitch:={pitch:.9f} yaw:={yaw:.9f}"
        )

        output = {
            "success": bool(result.success),
            "message": result.message,
            "sample_count": len(self.samples),
            "base_frame": self.base_frame,
            "observation_camera_frame": observation_camera_frame,
            "output_camera_frame": self.output_camera_frame,
            "base_T_observation_camera": list_from_matrix(base_T_observation),
            "base_T_output_camera": list_from_matrix(base_T_output),
            "gripper_T_charuco_board": list_from_matrix(gripper_T_board),
            "launch_arguments": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            },
            "launch_arguments_text": launch_arguments,
            "static_transform_publisher": static_tf_command,
            "mean_translation_error_m": mean_translation_error,
            "mean_rotation_error_deg": mean_rotation_error,
            "max_translation_error_m": max_translation_error,
            "max_rotation_error_deg": max_rotation_error,
            "per_sample_errors": per_sample_errors,
        }
        os.makedirs(os.path.dirname(self.result_path) or ".", exist_ok=True)
        with open(self.result_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(output, stream, sort_keys=False)

        self.get_logger().info("ChArUco calibration result:")
        self.get_logger().info(f"  launch args: {launch_arguments}")
        self.get_logger().info(
            f"  mean error: {mean_translation_error * 1000.0:.1f} mm, "
            f"{mean_rotation_error:.3f} deg"
        )
        self.get_logger().info(
            f"  max error: {max_translation_error * 1000.0:.1f} mm, "
            f"{max_rotation_error:.3f} deg"
        )
        self.get_logger().info(f"  result file: {self.result_path}")

        return (
            bool(result.success),
            f"Wrote calibration result to {self.result_path}. Launch args: {launch_arguments}",
        )

    def get_output_to_observation_transform(self, observation_camera_frame):
        if self.output_camera_frame == observation_camera_frame:
            return np.eye(4, dtype=np.float64)
        return self.lookup_transform_matrix(
            self.output_camera_frame,
            observation_camera_frame,
            None,
        )

    @staticmethod
    def compute_sample_errors(base_T_gripper, observation_T_board, base_T_observation, gripper_T_board):
        errors = []
        for index, (b_T_g, c_T_board) in enumerate(zip(base_T_gripper, observation_T_board), start=1):
            expected_base_T_board = b_T_g @ gripper_T_board
            observed_base_T_board = base_T_observation @ c_T_board
            error = invert_transform(expected_base_T_board) @ observed_base_T_board
            rotation_error_deg = math.degrees(float(np.linalg.norm(rotvec_from_matrix(error[:3, :3]))))
            translation_error_m = float(np.linalg.norm(error[:3, 3]))
            errors.append(
                {
                    "index": index,
                    "translation_m": translation_error_m,
                    "rotation_deg": rotation_error_deg,
                }
            )
        return errors


def main(args=None):
    rclpy.init(args=args)
    node = ChArUcoEyeToHandCalibrator()
    try:
        rclpy.spin(node)
    finally:
        if node.display:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
