"""Refine an eye-to-hand camera transform from manually clicked TCP pixels.

The bag supplies synchronized robot kinematics and color images.  A user clicks
the same, kinematically known TCP point in a set of stationary robot poses, and
the tool minimizes robust pixel reprojection error.  Collection is resumable
and the resulting transform is written as a candidate; configuration files are
never changed by this command.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
import warnings

import cv2
import numpy as np
import yaml

from rmp_camera.charuco_common import invert_transform, list_from_matrix
from rmp_camera.robot_depth_extrinsic_calibrator import (
    DEFAULT_JOINTS,
    candidate_base_from_camera,
    load_tf_snapshots,
    nearest_index,
    select_stable_pose_targets,
    transform_launch_arguments,
)
from rmp_camera.robot_self_filter_bag_validator import (
    open_bag_reader,
    read_first_camera_info,
    stamp_to_ns,
)


@dataclass
class ClickFrame:
    """One stationary color frame and the matching base-to-TCP pose."""

    pose_index: int
    target_stamp_ns: int
    image_stamp_ns: int
    image_delta_ms: float
    image_bgr: np.ndarray
    base_from_tcp: np.ndarray


def decode_color_image(message):
    """Decode common uncompressed ROS color encodings to BGR8."""
    encoding = message.encoding.lower()
    data = np.frombuffer(message.data, dtype=np.uint8)
    if encoding in ("rgb8", "bgr8"):
        channels = 3
        row_pixels = message.step // channels
        image = data.reshape(message.height, row_pixels, channels)[:, : message.width]
        image = np.ascontiguousarray(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if encoding == "rgb8" else image.copy()
    if encoding in ("rgba8", "bgra8"):
        channels = 4
        row_pixels = message.step // channels
        image = data.reshape(message.height, row_pixels, channels)[:, : message.width]
        image = np.ascontiguousarray(image)
        code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(image, code)
    if encoding in ("mono8", "8uc1"):
        image = data.reshape(message.height, message.step)[:, : message.width]
        return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported color image encoding: {message.encoding}")


def read_nearest_color_messages_sqlite(args, target_stamps):
    """Read only indexed SQLite rows near the requested timestamps.

    Raw color bags are often many gigabytes.  rosbag2's sequential reader still
    streams every image even with a topic filter, so opening a 20-frame click
    session can otherwise take minutes.
    """
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    bag_path = Path(args.bag_path)
    database_paths = (
        [bag_path]
        if bag_path.is_file() and bag_path.suffix == ".db3"
        else sorted(bag_path.glob("*.db3"))
    )
    if not database_paths:
        raise ValueError(f"No SQLite .db3 files found in {bag_path}")

    best = [(10**30, None, None, None) for _ in target_stamps]
    search_radius_ns = int(2.0e9)
    for database_path in database_paths:
        connection = sqlite3.connect(
            f"file:{database_path}?mode=ro", uri=True
        )
        try:
            topic_row = connection.execute(
                "SELECT id, type FROM topics WHERE name = ?",
                (args.image_topic,),
            ).fetchone()
            if topic_row is None:
                continue
            topic_id, type_name = topic_row
            message_type = get_message(type_name)
            for index, target in enumerate(target_stamps):
                rows = connection.execute(
                    "SELECT timestamp, data FROM messages INDEXED BY timestamp_idx "
                    "WHERE topic_id = ? AND timestamp BETWEEN ? AND ? "
                    "ORDER BY ABS(timestamp - ?) LIMIT 3",
                    (
                        int(topic_id),
                        int(target) - search_radius_ns,
                        int(target) + search_radius_ns,
                        int(target),
                    ),
                ).fetchall()
                for bag_stamp, data in rows:
                    message = deserialize_message(bytes(data), message_type)
                    image_stamp = stamp_to_ns(message.header.stamp) or int(bag_stamp)
                    delta = abs(int(target) - image_stamp)
                    if delta < best[index][0]:
                        best[index] = (
                            delta,
                            message,
                            int(image_stamp),
                            int(bag_stamp),
                        )
        finally:
            connection.close()

    output = []
    for delta, message, image_stamp, bag_stamp in best:
        if message is None:
            raise ValueError("Could not find a color frame for every stable pose")
        output.append((delta, message, image_stamp, bag_stamp))
    return output


def read_nearest_color_messages_sequential(args, target_stamps):
    """Portable fallback for rosbag storage backends other than SQLite."""
    from rclpy.serialization import deserialize_message

    reader, message_types = open_bag_reader(
        args.bag_path, [args.image_topic], args.storage_id
    )
    best = [(10**30, None, None, None) for _ in target_stamps]
    while reader.has_next():
        topic, data, bag_stamp = reader.read_next()
        message = deserialize_message(data, message_types[topic])
        image_stamp = stamp_to_ns(message.header.stamp) or int(bag_stamp)
        index = nearest_index(target_stamps, image_stamp)
        delta = abs(target_stamps[index] - image_stamp)
        if delta < best[index][0]:
            best[index] = (delta, message, int(image_stamp), int(bag_stamp))
    output = []
    for delta, message, image_stamp, bag_stamp in best:
        if message is None:
            raise ValueError("Could not find a color frame for every stable pose")
        output.append((delta, message, image_stamp, bag_stamp))
    return output


def read_nearest_color_messages(args, target_stamps):
    """Read the color image nearest every selected stationary timestamp."""
    if args.storage_id == "sqlite3":
        return read_nearest_color_messages_sqlite(args, target_stamps)
    return read_nearest_color_messages_sequential(args, target_stamps)


def camera_model(camera_info):
    """Return OpenCV camera matrix, distortion coefficients, and model name."""
    matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera_info.d, dtype=np.float64).reshape(-1)
    model = str(camera_info.distortion_model or "plumb_bob")
    if model not in ("plumb_bob", "rational_polynomial", "equidistant"):
        if np.any(np.abs(distortion) > 1e-12):
            raise ValueError(f"Unsupported nonzero distortion model: {model}")
        model = "plumb_bob"
    return matrix, distortion, model


def project_camera_points(points_camera, matrix, distortion, model="plumb_bob"):
    """Project camera-frame points using the recorded raw-image camera model."""
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    if model == "equidistant":
        projected, _ = cv2.fisheye.projectPoints(
            points.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            matrix,
            distortion[:4],
        )
    else:
        projected, _ = cv2.projectPoints(
            points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            matrix,
            distortion,
        )
    return projected.reshape(-1, 2)


def tcp_points_in_base(base_from_tcp_matrices, feature_offset_tcp):
    """Transform one TCP-frame feature point into base for every pose."""
    transforms = np.asarray(base_from_tcp_matrices, dtype=np.float64)
    local = np.asarray([*feature_offset_tcp, 1.0], dtype=np.float64)
    return np.asarray([(transform @ local)[:3] for transform in transforms])


def predicted_pixels(
    correction,
    initial_base_from_camera,
    base_from_tcp_matrices,
    feature_offset_tcp,
    matrix,
    distortion,
    distortion_model,
):
    """Project the selected TCP feature for all robot poses."""
    base_from_camera = candidate_base_from_camera(
        initial_base_from_camera, np.asarray(correction, dtype=np.float64)
    )
    camera_from_base = invert_transform(base_from_camera)
    points_base = tcp_points_in_base(base_from_tcp_matrices, feature_offset_tcp)
    points_camera = (
        camera_from_base[:3, :3] @ points_base.T
    ).T + camera_from_base[:3, 3]
    pixels = project_camera_points(
        points_camera, matrix, distortion, distortion_model
    )
    return pixels, points_camera[:, 2]


def reprojection_metrics(predicted, observed):
    """Summarize Euclidean pixel error."""
    errors = np.linalg.norm(
        np.asarray(predicted, dtype=np.float64) - np.asarray(observed, dtype=np.float64),
        axis=1,
    )
    return {
        "count": int(len(errors)),
        "median_px": float(np.median(errors)),
        "p90_px": float(np.percentile(errors, 90.0)),
        "mean_px": float(np.mean(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors * errors))),
        "max_px": float(np.max(errors)),
        "per_pose_px": errors.tolist(),
    }


def optimize_reprojection(
    args,
    initial_base_from_camera,
    base_from_tcp_matrices,
    observed_pixels,
    matrix,
    distortion,
    distortion_model,
    indices,
    seed,
):
    """Fit camera correction and, optionally, an unknown rigid feature offset."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A NumPy version.*")
        from scipy.optimize import least_squares

    transforms = np.asarray(base_from_tcp_matrices, dtype=np.float64)[indices]
    observed = np.asarray(observed_pixels, dtype=np.float64)[indices]
    fixed_offset = np.asarray(args.feature_offset_tcp, dtype=np.float64)
    camera_translation_bound = float(args.translation_bound_m)
    camera_rotation_bound = math.radians(args.rotation_bound_deg)
    camera_lower = np.asarray(
        [-camera_translation_bound] * 3 + [-camera_rotation_bound] * 3,
        dtype=np.float64,
    )
    camera_upper = -camera_lower
    if args.estimate_feature_offset:
        feature_bound = float(args.feature_offset_bound_m)
        lower = np.r_[camera_lower, fixed_offset - feature_bound]
        upper = np.r_[camera_upper, fixed_offset + feature_bound]
        start = np.r_[np.zeros(6, dtype=np.float64), fixed_offset]
    else:
        lower, upper = camera_lower, camera_upper
        start = np.zeros(6, dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    results = []

    def residual(parameters):
        correction = parameters[:6]
        feature = parameters[6:9] if args.estimate_feature_offset else fixed_offset
        predicted, depth = predicted_pixels(
            correction,
            initial_base_from_camera,
            transforms,
            feature,
            matrix,
            distortion,
            distortion_model,
        )
        values = (predicted - observed).reshape(-1)
        if np.any(depth <= 0.05):
            invalid = np.repeat(depth <= 0.05, 2)
            values[invalid] += np.sign(values[invalid] + 1e-9) * 1000.0
        return values

    for restart in range(args.restarts):
        initial = start.copy()
        if restart:
            jitter = rng.normal(0.0, 0.12, size=len(initial)) * (upper - lower)
            initial = np.clip(initial + jitter, lower * 0.95, upper * 0.95)
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=float(args.robust_loss_px),
            x_scale="jac",
            max_nfev=int(args.max_function_evaluations),
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        results.append(result)
    return min(results, key=lambda item: float(item.cost)), results


def read_initial_output_transform(path, recorded_base_from_output):
    """Load a previous candidate matrix, or use the transform recorded in the bag."""
    if not path:
        return np.asarray(recorded_base_from_output, dtype=np.float64), "recorded_tf"
    calibration_path = Path(path).expanduser().resolve()
    with calibration_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    for key in ("candidate_base_T_output_camera", "candidate_base_T_camera"):
        if key in data:
            matrix = np.asarray(data[key], dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"{key} in {calibration_path} is not a 4x4 matrix")
            return matrix, str(calibration_path)
    if "launch_arguments" in data:
        values = data["launch_arguments"]
        from rmp_camera.charuco_common import transform_from_xyz_rpy

        matrix = transform_from_xyz_rpy(
            [values["x"], values["y"], values["z"]],
            [values["roll"], values["pitch"], values["yaw"]],
        )
        return matrix, str(calibration_path)
    raise ValueError(
        f"{calibration_path} has no candidate camera matrix or launch_arguments"
    )


def load_click_file(path, expected_bag_path):
    """Load resumable clicks keyed by selected-pose timestamp."""
    click_path = Path(path).expanduser().resolve()
    if not click_path.exists():
        return {}
    with click_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    saved_bag = data.get("bag_path")
    if saved_bag and str(Path(saved_bag).resolve()) != str(Path(expected_bag_path).resolve()):
        raise ValueError(
            f"Click file belongs to a different bag: {saved_bag} != {expected_bag_path}"
        )
    return {
        int(item["target_stamp_ns"]): np.asarray(item["pixel_uv"], dtype=np.float64)
        for item in data.get("samples", [])
    }


def write_click_file(args, frames, clicks, camera_frame):
    """Atomically save all accepted clicks so collection can be resumed."""
    output = {
        "schema_version": 1,
        "bag_path": args.bag_path,
        "image_topic": args.image_topic,
        "camera_info_topic": args.camera_info_topic,
        "camera_frame": camera_frame,
        "tcp_frame": args.tcp_frame,
        "feature_offset_tcp_m": [float(value) for value in args.feature_offset_tcp],
        "samples": [],
    }
    for frame in frames:
        if frame.target_stamp_ns not in clicks:
            continue
        output["samples"].append(
            {
                "pose_index": int(frame.pose_index),
                "target_stamp_ns": int(frame.target_stamp_ns),
                "image_stamp_ns": int(frame.image_stamp_ns),
                "image_delta_ms": float(frame.image_delta_ms),
                "pixel_uv": [float(value) for value in clicks[frame.target_stamp_ns]],
                "base_T_tcp": list_from_matrix(frame.base_from_tcp),
            }
        )
    path = Path(args.clicks).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    temporary.replace(path)


def draw_cross(image, pixel, color, size=13, thickness=2):
    """Draw a crosshair that leaves the exact center pixel visible."""
    u, v = np.rint(pixel).astype(int)
    cv2.line(image, (u - size, v), (u - 3, v), color, thickness)
    cv2.line(image, (u + 3, v), (u + size, v), color, thickness)
    cv2.line(image, (u, v - size), (u, v - 3), color, thickness)
    cv2.line(image, (u, v + 3), (u, v + size), color, thickness)


def collect_clicks(
    args,
    frames,
    clicks,
    initial_base_from_camera,
    matrix,
    distortion,
    distortion_model,
    camera_frame,
):
    """Run the resumable OpenCV point-click GUI."""
    window = "TCP click calibration"
    zoom_window = "TCP click zoom"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(zoom_window, cv2.WINDOW_AUTOSIZE)
    state = {
        "selected": None,
        "cursor": None,
        "zoom_origin": np.zeros(2, dtype=np.float64),
        "zoom_scale": 10.0,
    }

    def mouse(event, x, y, _flags, _userdata):
        state["cursor"] = np.asarray([x, y], dtype=np.float64)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["selected"] = state["cursor"].copy()

    def zoom_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            display_point = np.asarray([x, y], dtype=np.float64)
            state["selected"] = (
                state["zoom_origin"]
                + (display_point + 0.5) / state["zoom_scale"]
                - 0.5
            )
            state["cursor"] = state["selected"].copy()

    cv2.setMouseCallback(window, mouse)
    cv2.setMouseCallback(zoom_window, zoom_mouse)
    pending = [
        index
        for index, frame in enumerate(frames)
        if args.review_all or frame.target_stamp_ns not in clicks
    ]
    if not pending:
        cv2.destroyWindow(window)
        cv2.destroyWindow(zoom_window)
        return True

    cursor = 0
    while cursor < len(pending):
        frame_index = pending[cursor]
        frame = frames[frame_index]
        existing = clicks.get(frame.target_stamp_ns)
        state["selected"] = None if existing is None else existing.copy()
        base_from_tcp = np.asarray([frame.base_from_tcp])
        predicted, _ = predicted_pixels(
            np.zeros(6),
            initial_base_from_camera,
            base_from_tcp,
            args.feature_offset_tcp,
            matrix,
            distortion,
            distortion_model,
        )
        while True:
            display = frame.image_bgr.copy()
            draw_cross(display, predicted[0], (0, 0, 255), size=16, thickness=2)
            if state["selected"] is not None:
                draw_cross(display, state["selected"], (0, 255, 255), size=18, thickness=2)
            cv2.rectangle(display, (0, 0), (display.shape[1], 72), (0, 0, 0), -1)
            cv2.putText(
                display,
                f"Pose {frame_index + 1}/{len(frames)}  accepted {len(clicks)}/{len(frames)}",
                (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                "RED=predicted; click TCP then refine in ZOOM; Enter=accept, X=skip, B=back, Q=quit",
                (14, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
            )
            focus = state["selected"] if state["selected"] is not None else state["cursor"]
            if focus is None:
                focus = predicted[0]
            radius = 24
            center = np.rint(focus).astype(int)
            x0 = int(np.clip(center[0] - radius, 0, max(0, display.shape[1] - 2 * radius)))
            y0 = int(np.clip(center[1] - radius, 0, max(0, display.shape[0] - 2 * radius)))
            x1 = min(display.shape[1], x0 + 2 * radius)
            y1 = min(display.shape[0], y0 + 2 * radius)
            crop = frame.image_bgr[y0:y1, x0:x1]
            state["zoom_origin"] = np.asarray([x0, y0], dtype=np.float64)
            zoom = cv2.resize(
                crop,
                None,
                fx=state["zoom_scale"],
                fy=state["zoom_scale"],
                interpolation=cv2.INTER_CUBIC,
            )
            if state["selected"] is not None:
                zoom_point = (
                    state["selected"] - state["zoom_origin"] + 0.5
                ) * state["zoom_scale"] - 0.5
                draw_cross(zoom, zoom_point, (0, 255, 255), size=22, thickness=2)
            cv2.putText(
                zoom,
                "click here to refine",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window, display)
            cv2.imshow(zoom_window, zoom)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10, 32, ord("n")) and state["selected"] is not None:
                clicks[frame.target_stamp_ns] = state["selected"].copy()
                write_click_file(args, frames, clicks, camera_frame)
                cursor += 1
                break
            if key in (ord("x"), ord("s")):
                cursor += 1
                break
            if key in (ord("b"), 8, 127):
                if cursor > 0:
                    cursor -= 1
                    previous = frames[pending[cursor]]
                    clicks.pop(previous.target_stamp_ns, None)
                    write_click_file(args, frames, clicks, camera_frame)
                break
            if key in (ord("q"), 27):
                write_click_file(args, frames, clicks, camera_frame)
                cv2.destroyWindow(window)
                cv2.destroyWindow(zoom_window)
                return False
    cv2.destroyWindow(window)
    cv2.destroyWindow(zoom_window)
    write_click_file(args, frames, clicks, camera_frame)
    return True


def write_debug_images(
    args,
    frames,
    clicked_frames,
    observed,
    initial_pixels,
    candidate_pixels,
):
    """Save click overlays for visual verification."""
    if not args.debug_dir:
        return []
    output_dir = Path(args.debug_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for local_index, frame in enumerate(clicked_frames):
        image = frame.image_bgr.copy()
        draw_cross(image, initial_pixels[local_index], (0, 0, 255), 18, 2)
        draw_cross(image, candidate_pixels[local_index], (0, 255, 0), 14, 2)
        draw_cross(image, observed[local_index], (0, 255, 255), 22, 2)
        cv2.putText(
            image,
            "red=initial  green=candidate  yellow=manual click",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        path = output_dir / f"pose_{frame.pose_index:02d}_click_overlay.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write {path}")
        outputs.append(str(path))
    return outputs


def pose_diversity(base_from_tcp_matrices, feature_offset_tcp):
    """Report base-frame spread and singular values of clicked 3-D points."""
    points = tcp_points_in_base(base_from_tcp_matrices, feature_offset_tcp)
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return {
        "base_xyz_span_m": np.ptp(points, axis=0).tolist(),
        "centered_singular_values_m": singular.tolist(),
        "minimum_pair_distance_m": float(
            min(
                np.linalg.norm(points[a] - points[b])
                for a in range(len(points))
                for b in range(a + 1, len(points))
            )
        ),
    }


def solve_clicks(
    args,
    frames,
    clicks,
    initial_base_from_camera,
    initial_base_from_output,
    output_from_observation,
    matrix,
    distortion,
    distortion_model,
    initial_source,
):
    """Optimize, validate on alternating poses, and write candidate reports."""
    clicked_frames = [frame for frame in frames if frame.target_stamp_ns in clicks]
    if len(clicked_frames) < args.min_clicks:
        raise ValueError(
            f"Only {len(clicked_frames)} clicks saved; need at least {args.min_clicks}"
        )
    transforms = np.asarray([frame.base_from_tcp for frame in clicked_frames])
    observed = np.asarray([clicks[frame.target_stamp_ns] for frame in clicked_frames])
    all_indices = np.arange(len(clicked_frames), dtype=int)
    train_indices = all_indices[::2]
    validation_indices = all_indices[1::2]
    if len(validation_indices) < 4:
        validation_indices = train_indices

    validation_fit, _ = optimize_reprojection(
        args,
        initial_base_from_camera,
        transforms,
        observed,
        matrix,
        distortion,
        distortion_model,
        train_indices,
        args.seed + 1009,
    )
    validation_feature = (
        validation_fit.x[6:9]
        if args.estimate_feature_offset
        else np.asarray(args.feature_offset_tcp, dtype=np.float64)
    )
    validation_predicted, _ = predicted_pixels(
        validation_fit.x[:6],
        initial_base_from_camera,
        transforms[validation_indices],
        validation_feature,
        matrix,
        distortion,
        distortion_model,
    )
    validation_metrics = reprojection_metrics(
        validation_predicted, observed[validation_indices]
    )

    final_fit, restart_results = optimize_reprojection(
        args,
        initial_base_from_camera,
        transforms,
        observed,
        matrix,
        distortion,
        distortion_model,
        all_indices,
        args.seed,
    )
    correction = np.asarray(final_fit.x[:6], dtype=np.float64)
    feature_offset = (
        np.asarray(final_fit.x[6:9], dtype=np.float64)
        if args.estimate_feature_offset
        else np.asarray(args.feature_offset_tcp, dtype=np.float64)
    )
    candidate_observation = candidate_base_from_camera(
        initial_base_from_camera, correction
    )
    candidate_base_from_output = candidate_observation @ invert_transform(
        output_from_observation
    )

    initial_pixels, _ = predicted_pixels(
        np.zeros(6),
        initial_base_from_camera,
        transforms,
        args.feature_offset_tcp,
        matrix,
        distortion,
        distortion_model,
    )
    candidate_pixels, depths = predicted_pixels(
        correction,
        initial_base_from_camera,
        transforms,
        feature_offset,
        matrix,
        distortion,
        distortion_model,
    )
    initial_metrics = reprojection_metrics(initial_pixels, observed)
    candidate_metrics = reprojection_metrics(candidate_pixels, observed)
    diversity = pose_diversity(transforms, feature_offset)
    debug_images = write_debug_images(
        args,
        frames,
        clicked_frames,
        observed,
        initial_pixels,
        candidate_pixels,
    )
    launch = transform_launch_arguments(candidate_base_from_output)
    restart_camera_parameters = np.asarray([result.x[:6] for result in restart_results])
    restart_translation_spread = float(
        max(
            np.linalg.norm(first[:3] - second[:3])
            for first in restart_camera_parameters
            for second in restart_camera_parameters
        )
    )
    restart_rotation_spread_deg = float(
        math.degrees(
            max(
                np.linalg.norm(first[3:] - second[3:])
                for first in restart_camera_parameters
                for second in restart_camera_parameters
            )
        )
    )
    translation_bound_ratio = float(
        np.max(np.abs(correction[:3])) / max(args.translation_bound_m, 1e-12)
    )
    rotation_bound_ratio = float(
        np.max(np.abs(correction[3:]))
        / max(math.radians(args.rotation_bound_deg), 1e-12)
    )
    touches_optimizer_bound = bool(
        translation_bound_ratio >= 0.995 or rotation_bound_ratio >= 0.995
    )
    passes_gate = bool(
        candidate_metrics["median_px"] <= args.gate_median_px
        and candidate_metrics["p90_px"] <= args.gate_p90_px
        and validation_metrics["median_px"] <= args.gate_validation_median_px
        and np.min(depths) > 0.05
        and restart_translation_spread <= 0.01
        and restart_rotation_spread_deg <= 0.5
        and not touches_optimizer_bound
    )
    result = {
        "status": "candidate_only",
        "calibration_method": "manual_tcp_pixel_reprojection",
        "passes_offline_gate": passes_gate,
        "requires_visual_validation": True,
        "bag_path": args.bag_path,
        "click_file": str(Path(args.clicks).expanduser().resolve()),
        "initial_calibration_source": initial_source,
        "base_frame": args.base_frame,
        "tcp_frame": args.tcp_frame,
        "observation_camera_frame": args.camera_frame,
        "output_camera_frame": args.output_camera_frame,
        "click_count": int(len(clicked_frames)),
        "train_pose_indices": train_indices.tolist(),
        "validation_pose_indices": validation_indices.tolist(),
        "feature_offset_tcp_m": feature_offset.tolist(),
        "feature_offset_was_estimated": bool(args.estimate_feature_offset),
        "initial_base_T_observation_camera": list_from_matrix(initial_base_from_camera),
        "candidate_base_T_observation_camera": list_from_matrix(
            candidate_observation
        ),
        "initial_base_T_output_camera": list_from_matrix(initial_base_from_output),
        "candidate_base_T_output_camera": list_from_matrix(
            candidate_base_from_output
        ),
        "optical_correction": {
            "translation_xyz_m": correction[:3].tolist(),
            "rotation_rpy_rad": correction[3:].tolist(),
            "rotation_rpy_deg": np.degrees(correction[3:]).tolist(),
        },
        "launch_arguments": launch,
        "launch_arguments_text": (
            f"x:={launch['x']:.9f} y:={launch['y']:.9f} z:={launch['z']:.9f} "
            f"roll:={launch['roll']:.9f} pitch:={launch['pitch']:.9f} "
            f"yaw:={launch['yaw']:.9f}"
        ),
        "initial_reprojection": initial_metrics,
        "candidate_reprojection": candidate_metrics,
        "held_out_validation_reprojection": validation_metrics,
        "pose_diversity": diversity,
        "restart_translation_spread_m": restart_translation_spread,
        "restart_rotation_spread_deg": restart_rotation_spread_deg,
        "translation_bound_usage_ratio": translation_bound_ratio,
        "rotation_bound_usage_ratio": rotation_bound_ratio,
        "touches_optimizer_bound": touches_optimizer_bound,
        "debug_images": debug_images,
        "pose_inputs": [
            {
                "pose_index": int(frame.pose_index),
                "target_stamp_ns": int(frame.target_stamp_ns),
                "image_delta_ms": float(frame.image_delta_ms),
                "clicked_pixel_uv": observed[index].tolist(),
                "initial_pixel_uv": initial_pixels[index].tolist(),
                "candidate_pixel_uv": candidate_pixels[index].tolist(),
            }
            for index, frame in enumerate(clicked_frames)
        ],
        "application_note": (
            "After inspecting every click overlay, copy launch_arguments into the "
            "experiment static_tf values. Keep robot_sphere_marker_correction at zero."
        ),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(result, stream, sort_keys=False)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate result: {output_path}")
    print(f"JSON result: {json_path}")
    print(
        f"Reprojection median: {initial_metrics['median_px']:.2f} -> "
        f"{candidate_metrics['median_px']:.2f} px; held-out="
        f"{validation_metrics['median_px']:.2f} px"
    )
    if touches_optimizer_bound:
        print(
            "Warning: camera correction reached an optimizer bound; "
            "independent depth validation is required."
        )
    print(f"Offline gate: {'PASS' if passes_gate else 'FAIL'}")
    print(f"Launch args: {result['launch_arguments_text']}")
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Collect manual TCP clicks from stationary bag poses and refine the "
            "base-to-camera transform with robust 2-D reprojection optimization."
        )
    )
    parser.add_argument("bag_path", help="ROS 2 bag directory")
    parser.add_argument("--clicks", default="/tmp/rmp_camera_tcp_clicks.yaml")
    parser.add_argument("--output", default="/tmp/rmp_camera_tcp_click_result.yaml")
    parser.add_argument("--debug-dir", default="/tmp/rmp_camera_tcp_click_debug")
    parser.add_argument("--initial-calibration", default="")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--solve-only", action="store_true")
    parser.add_argument("--review-all", action="store_true")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--tcp-frame", default="tcp")
    parser.add_argument("--output-camera-frame", default="camera0_link")
    parser.add_argument("--image-topic", default="/camera0/camera/color/image_raw")
    parser.add_argument(
        "--camera-info-topic", default="/camera0/camera/color/camera_info"
    )
    parser.add_argument("--joint-topic", default="/rmp_camera/joint_states_urdf")
    parser.add_argument("--joint-names", default=",".join(DEFAULT_JOINTS))
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument("--min-poses", type=int, default=10)
    parser.add_argument("--max-poses", type=int, default=20)
    parser.add_argument("--min-clicks", type=int, default=10)
    parser.add_argument("--settling-time-s", type=float, default=5.0)
    parser.add_argument("--stable-speed-deg-s", type=float, default=0.5)
    parser.add_argument("--min-stable-duration-s", type=float, default=0.4)
    parser.add_argument("--min-pose-separation-deg", type=float, default=1.0)
    parser.add_argument(
        "--feature-offset-tcp",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Clicked feature coordinates in the tcp frame; default is TCP origin.",
    )
    parser.add_argument(
        "--estimate-feature-offset",
        action="store_true",
        help="Jointly estimate an unknown rigid clicked point; requires varied orientations.",
    )
    parser.add_argument("--feature-offset-bound-m", type=float, default=0.20)
    parser.add_argument("--translation-bound-m", type=float, default=0.06)
    parser.add_argument("--rotation-bound-deg", type=float, default=3.0)
    parser.add_argument("--robust-loss-px", type=float, default=3.0)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--max-function-evaluations", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gate-median-px", type=float, default=3.0)
    parser.add_argument("--gate-p90-px", type=float, default=7.0)
    parser.add_argument("--gate-validation-median-px", type=float, default=6.0)
    args = parser.parse_args(argv)
    args.bag_path = str(Path(args.bag_path).expanduser().resolve())
    args.clicks = str(Path(args.clicks).expanduser().resolve())
    args.output = str(Path(args.output).expanduser().resolve())
    args.feature_offset_tcp = tuple(float(value) for value in args.feature_offset_tcp)
    if args.collect_only and args.solve_only:
        parser.error("--collect-only and --solve-only cannot be used together")
    if args.min_poses < 6 or args.max_poses < args.min_poses:
        parser.error("pose limits must satisfy 6 <= min_poses <= max_poses")
    if args.min_clicks < 6 or args.min_clicks > args.max_poses:
        parser.error("min-clicks must be between 6 and max-poses")
    if args.restarts < 1:
        parser.error("--restarts must be positive")
    return args


def run(args):
    """Extract stable frames, collect/resume clicks, and optionally solve."""
    camera_info = read_first_camera_info(
        args.bag_path, args.camera_info_topic, args.storage_id
    )
    args.camera_frame = str(camera_info.header.frame_id)
    if not args.camera_frame:
        raise ValueError("Color CameraInfo has an empty frame_id")
    matrix, distortion, distortion_model = camera_model(camera_info)
    print("Selecting stationary, distinct robot poses...", flush=True)
    pose_targets, stable_count, _ = select_stable_pose_targets(args)
    stamps = [item[0] for item in pose_targets]
    print(
        f"Selected {len(pose_targets)} poses from {stable_count} stable segments.",
        flush=True,
    )
    snapshots, _ = load_tf_snapshots(
        args, stamps, [args.tcp_frame], args.camera_frame
    )
    print(f"Loading {len(stamps)} indexed color frames...", flush=True)
    color_messages = read_nearest_color_messages(args, stamps)
    frames = []
    for index, (target, snapshot, image_item) in enumerate(
        zip(stamps, snapshots, color_messages)
    ):
        delta, message, image_stamp, _ = image_item
        frames.append(
            ClickFrame(
                pose_index=index,
                target_stamp_ns=int(target),
                image_stamp_ns=int(image_stamp),
                image_delta_ms=float(delta) / 1e6,
                image_bgr=decode_color_image(message),
                base_from_tcp=snapshot.base_from_links[args.tcp_frame],
            )
        )
    recorded_base_from_camera = snapshots[0].base_from_observation
    recorded_base_from_output = snapshots[0].base_from_output
    output_from_observation = (
        invert_transform(recorded_base_from_output) @ recorded_base_from_camera
    )
    initial_base_from_output, initial_source = read_initial_output_transform(
        args.initial_calibration, recorded_base_from_output
    )
    initial_base_from_camera = initial_base_from_output @ output_from_observation
    clicks = load_click_file(args.clicks, args.bag_path)
    if not args.solve_only:
        print("Opening TCP click and zoom windows...", flush=True)
        completed = collect_clicks(
            args,
            frames,
            clicks,
            initial_base_from_camera,
            matrix,
            distortion,
            distortion_model,
            args.camera_frame,
        )
        print(f"Saved {len(clicks)} clicks to {args.clicks}")
        if args.collect_only or not completed:
            return None
    return solve_clicks(
        args,
        frames,
        clicks,
        initial_base_from_camera,
        initial_base_from_output,
        output_from_observation,
        matrix,
        distortion,
        distortion_model,
        initial_source,
    )


def main(argv=None):
    args = parse_arguments(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
