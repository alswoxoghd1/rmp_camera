"""Validate depth-aware robot self filtering directly from a ROS 2 bag."""

import argparse
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import yaml

from rmp_camera.robot_depth_filter_core import (
    predict_sphere_surface_depth,
    surface_depth_removal_mask,
)
from rmp_camera.vision_geometry import transform_to_matrix


@dataclass(frozen=True)
class SphereSpec:
    """Collision sphere expressed in its URDF link frame."""

    link: str
    offset: np.ndarray
    radius: float


@dataclass(frozen=True)
class RobotState:
    """Camera-frame collision spheres at one recorded TF timestamp."""

    stamp_ns: int
    centers_camera: np.ndarray


def stamp_to_ns(stamp):
    """Convert a ROS builtin time message to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def resolve_transform(tree, target_frame, root_frame):
    """Return root-from-target for a child-to-parent transform tree."""
    if target_frame == root_frame:
        return np.eye(4)

    matrices = []
    current = target_frame
    visited = set()
    while current != root_frame:
        if current in visited:
            raise KeyError(f"TF cycle while resolving {root_frame} -> {target_frame}")
        visited.add(current)
        if current not in tree:
            raise KeyError(
                f"No TF path {root_frame} -> {target_frame}; stopped at {current}"
            )
        parent, parent_from_child = tree[current]
        matrices.append(parent_from_child)
        current = parent

    root_from_target = np.eye(4)
    for parent_from_child in reversed(matrices):
        root_from_target = root_from_target @ parent_from_child
    return root_from_target


def nearest_state_index(sorted_stamps, target_stamp):
    """Return the index of the timestamp nearest to target_stamp."""
    if not sorted_stamps:
        raise ValueError("at least one robot state timestamp is required")
    index = bisect_left(sorted_stamps, target_stamp)
    if index <= 0:
        return 0
    if index >= len(sorted_stamps):
        return len(sorted_stamps) - 1
    before = index - 1
    if target_stamp - sorted_stamps[before] <= sorted_stamps[index] - target_stamp:
        return before
    return index


def sphere_centers_in_camera(tree, spheres, base_frame, camera_frame):
    """Transform configured link-frame sphere centers into the camera frame."""
    camera_from_base = np.linalg.inv(
        resolve_transform(tree, camera_frame, base_frame)
    )
    centers = []
    for sphere in spheres:
        base_from_link = resolve_transform(tree, sphere.link, base_frame)
        local_point = np.asarray([*sphere.offset, 1.0], dtype=np.float64)
        centers.append((camera_from_base @ base_from_link @ local_point)[:3])
    return np.asarray(centers, dtype=np.float64)


def depth_image_to_meters(msg):
    """Decode a 16UC1/mono16/32FC1 ROS image including padded rows."""
    endian = ">" if msg.is_bigendian else "<"
    if msg.encoding in ("16UC1", "mono16"):
        dtype = np.dtype(endian + "u2")
        scale_m = 0.001
    elif msg.encoding == "32FC1":
        dtype = np.dtype(endian + "f4")
        scale_m = 1.0
    else:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

    if msg.step < msg.width * dtype.itemsize:
        raise ValueError(
            f"Invalid image step={msg.step} for width={msg.width}, "
            f"encoding={msg.encoding}"
        )
    row_items = msg.step // dtype.itemsize
    expected_items = row_items * msg.height
    values = np.frombuffer(msg.data, dtype=dtype, count=expected_items)
    image = values.reshape(msg.height, row_items)[:, :msg.width]
    return image.astype(np.float32) * scale_m


def camera_intrinsics(camera_info):
    """Return rectified fx, fy, cx, cy from CameraInfo."""
    if camera_info.p[0] != 0.0 and camera_info.p[5] != 0.0:
        return (
            float(camera_info.p[0]),
            float(camera_info.p[5]),
            float(camera_info.p[2]),
            float(camera_info.p[6]),
        )
    return (
        float(camera_info.k[0]),
        float(camera_info.k[4]),
        float(camera_info.k[2]),
        float(camera_info.k[5]),
    )


def load_sphere_config(path):
    """Load collision spheres and their base frame from YAML."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    spheres = [
        SphereSpec(
            link=str(item["link"]),
            offset=np.asarray(item["offset"], dtype=np.float64),
            radius=float(item["radius"]),
        )
        for item in config.get("collision_spheres", [])
    ]
    if not spheres:
        raise ValueError(f"No collision spheres found in {path}")
    return str(config.get("base_frame", "base_link")), spheres


def open_bag_reader(bag_path, topics, storage_id):
    """Open a filtered rosbag2 reader and resolve selected message types."""
    import rosbag2_py
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    available = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    missing = [topic for topic in topics if topic not in available]
    if missing:
        raise ValueError(
            f"Bag {bag_path} is missing required topics: {', '.join(missing)}"
        )
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    message_types = {topic: get_message(available[topic]) for topic in topics}
    return reader, message_types


def read_first_camera_info(bag_path, topic, storage_id):
    """Read the first CameraInfo message from a bag."""
    from rclpy.serialization import deserialize_message

    reader, message_types = open_bag_reader(bag_path, [topic], storage_id)
    if not reader.has_next():
        raise ValueError(f"CameraInfo topic {topic} is empty in {bag_path}")
    _, data, _ = reader.read_next()
    return deserialize_message(data, message_types[topic])


def load_robot_states(
    bag_path,
    spheres,
    base_frame,
    camera_frame,
    tf_topic,
    tf_static_topic,
    storage_id,
):
    """Build camera-frame collision sphere states from recorded TF messages."""
    from rclpy.serialization import deserialize_message

    reader, message_types = open_bag_reader(
        bag_path,
        [tf_topic, tf_static_topic],
        storage_id,
    )
    static_tree = {}
    dynamic_messages = []
    while reader.has_next():
        topic, data, bag_stamp = reader.read_next()
        msg = deserialize_message(data, message_types[topic])
        edges = {}
        stamps = []
        for item in msg.transforms:
            edges[item.child_frame_id] = (
                item.header.frame_id,
                transform_to_matrix(item.transform),
            )
            item_stamp = stamp_to_ns(item.header.stamp)
            if item_stamp:
                stamps.append(item_stamp)
        if topic == tf_static_topic:
            static_tree.update(edges)
        elif edges:
            state_stamp = int(np.median(stamps)) if stamps else int(bag_stamp)
            dynamic_messages.append((state_stamp, edges))

    required_links = {sphere.link for sphere in spheres}
    rolling_dynamic_tree = {}
    states = []
    incomplete_messages = 0
    for state_stamp, edges in dynamic_messages:
        rolling_dynamic_tree.update(edges)
        if not required_links.intersection(edges):
            continue
        tree = dict(static_tree)
        tree.update(rolling_dynamic_tree)
        try:
            centers = sphere_centers_in_camera(
                tree,
                spheres,
                base_frame,
                camera_frame,
            )
        except KeyError:
            incomplete_messages += 1
            continue
        states.append(RobotState(state_stamp, centers))

    states.sort(key=lambda state: state.stamp_ns)
    if not states:
        links = ", ".join(sorted(required_links))
        raise ValueError(
            f"Could not resolve recorded TF states for collision sphere links: {links}"
        )
    return states, incomplete_messages


def volume_overlap_count(points_camera, centers_camera, radii):
    """Count points inside at least one collision sphere."""
    if len(points_camera) == 0:
        return 0
    inside_any = np.zeros(len(points_camera), dtype=bool)
    for center, radius in zip(centers_camera, radii):
        inside_any |= (
            np.sum((points_camera - center) ** 2, axis=1) < radius * radius
        )
    return int(np.count_nonzero(inside_any))


def colorize_depth(depth_m, max_visual_depth_m):
    """Create a BGR debug visualization without changing source depth."""
    import cv2

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        depth_m[valid] / max(max_visual_depth_m, 1e-6) * 255.0,
        1.0,
        255.0,
    ).astype(np.uint8)
    color = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def write_debug_image(
    output_path,
    measured_depth,
    predicted_depth,
    removal_mask,
    foreground_mask,
    behind_mask,
    max_visual_depth_m,
):
    """Write a raw/predicted/classification/masked diagnostic contact sheet."""
    import cv2

    raw_color = colorize_depth(measured_depth, max_visual_depth_m)
    predicted_color = colorize_depth(predicted_depth, max_visual_depth_m)
    classification = raw_color.copy()
    classification[behind_mask] = (255, 0, 0)
    classification[foreground_mask] = (0, 255, 0)
    classification[removal_mask] = (0, 0, 255)
    masked = measured_depth.copy()
    masked[removal_mask] = 0.0
    masked_color = colorize_depth(masked, max_visual_depth_m)

    panels = [raw_color, predicted_color, classification, masked_color]
    labels = [
        "measured depth",
        "predicted robot depth",
        "red=removed green=foreground blue=behind",
        "robot-masked depth",
    ]
    for panel, label in zip(panels, labels):
        cv2.putText(
            panel,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    sheet = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Failed to write debug image: {output_path}")


def analyze_bag(args, bag_path, base_frame, spheres):
    """Analyze one bag and return JSON-serializable metrics."""
    from rclpy.serialization import deserialize_message

    camera_info = read_first_camera_info(
        bag_path,
        args.camera_info_topic,
        args.storage_id,
    )
    camera_frame = args.camera_frame or camera_info.header.frame_id
    if not camera_frame:
        raise ValueError("Camera frame is empty; pass --camera-frame explicitly")
    fx, fy, cx, cy = camera_intrinsics(camera_info)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid CameraInfo focal lengths in {bag_path}")

    states, incomplete_tf_messages = load_robot_states(
        bag_path,
        spheres,
        base_frame,
        camera_frame,
        args.tf_topic,
        args.tf_static_topic,
        args.storage_id,
    )
    state_stamps = [state.stamp_ns for state in states]
    radii = np.asarray([sphere.radius for sphere in spheres], dtype=np.float64)
    render_radii = radii + max(0.0, args.surface_sphere_padding_m)

    reader, message_types = open_bag_reader(
        bag_path,
        [args.depth_topic],
        args.storage_id,
    )
    totals = defaultdict(int)
    tf_delta_ms = []
    depth_frames_seen = 0
    sampled_frames = 0
    processed_frames = 0
    time_miss_frames = 0
    debug_written = 0
    debug_dir = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir) / Path(bag_path).name
        debug_dir.mkdir(parents=True, exist_ok=True)

    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, message_types[args.depth_topic])
        frame_index = depth_frames_seen
        depth_frames_seen += 1
        if frame_index % args.sample_every:
            continue
        if args.max_frames > 0 and sampled_frames >= args.max_frames:
            break
        sampled_frames += 1

        depth_stamp = stamp_to_ns(msg.header.stamp)
        state_index = nearest_state_index(state_stamps, depth_stamp)
        state = states[state_index]
        delta_ms = abs(state.stamp_ns - depth_stamp) / 1e6
        tf_delta_ms.append(delta_ms)
        if args.max_tf_delta_s > 0.0 and delta_ms > args.max_tf_delta_s * 1000.0:
            time_miss_frames += 1
            continue

        measured_depth = depth_image_to_meters(msg)
        predicted_depth = predict_sphere_surface_depth(
            measured_depth.shape,
            state.centers_camera,
            render_radii,
            fx,
            fy,
            cx,
            cy,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
        )
        removal_mask = surface_depth_removal_mask(
            measured_depth,
            predicted_depth,
            args.front_tolerance_m,
            args.back_tolerance_m,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            mask_shadow_behind_robot=args.mask_shadow_behind_robot,
        )

        valid_measured = (
            np.isfinite(measured_depth)
            & (measured_depth >= args.min_depth_m)
            & (measured_depth <= args.max_depth_m)
        )
        has_robot_surface = np.isfinite(predicted_depth)
        delta = measured_depth - predicted_depth
        valid_robot_ray = valid_measured & has_robot_surface
        foreground = (
            valid_robot_ray & (delta < -max(0.0, args.front_tolerance_m))
        )
        behind = valid_robot_ray & (
            delta > max(0.0, args.back_tolerance_m)
        )
        surface_band = valid_robot_ray & ~foreground & ~behind

        totals["predicted_robot_ray_pixels"] += int(
            np.count_nonzero(has_robot_surface)
        )
        totals["valid_robot_ray_pixels"] += int(np.count_nonzero(valid_robot_ray))
        totals["surface_band_pixels"] += int(np.count_nonzero(surface_band))
        totals["removed_pixels"] += int(np.count_nonzero(removal_mask))
        totals["foreground_kept_pixels"] += int(
            np.count_nonzero(foreground & ~removal_mask)
        )
        totals["foreground_removed_pixels"] += int(
            np.count_nonzero(foreground & removal_mask)
        )
        totals["behind_pixels"] += int(np.count_nonzero(behind))
        totals["invalid_robot_ray_pixels"] += int(
            np.count_nonzero(has_robot_surface & ~valid_measured)
        )

        if args.compare_volume_margin_m >= 0.0 and np.any(foreground):
            vs, us = np.nonzero(foreground)
            z = measured_depth[foreground]
            foreground_points = np.column_stack(
                ((us - cx) * z / fx, (vs - cy) * z / fy, z)
            )
            totals["foreground_volume_would_remove_pixels"] += volume_overlap_count(
                foreground_points,
                state.centers_camera,
                radii + args.compare_volume_margin_m,
            )

        if (
            debug_dir is not None
            and debug_written < args.max_debug_images
            and processed_frames % args.debug_every == 0
        ):
            output_path = debug_dir / f"frame_{frame_index:06d}.png"
            write_debug_image(
                output_path,
                measured_depth,
                predicted_depth,
                removal_mask,
                foreground,
                behind,
                args.debug_max_depth_m,
            )
            debug_written += 1

        processed_frames += 1
        if args.progress_every > 0 and processed_frames % args.progress_every == 0:
            print(
                f"[{Path(bag_path).name}] processed {processed_frames} sampled frames",
                flush=True,
            )

    if processed_frames == 0:
        raise ValueError(f"No depth frames were processed from {bag_path}")

    valid_robot_pixels = max(totals["valid_robot_ray_pixels"], 1)
    result = {
        "bag_path": str(Path(bag_path).resolve()),
        "camera_frame": camera_frame,
        "image_width": int(camera_info.width),
        "image_height": int(camera_info.height),
        "collision_spheres": len(spheres),
        "robot_tf_states": len(states),
        "incomplete_tf_messages": incomplete_tf_messages,
        "depth_frames_seen": depth_frames_seen,
        "sample_every": args.sample_every,
        "sampled_frames": sampled_frames,
        "processed_frames": processed_frames,
        "time_miss_frames": time_miss_frames,
        "tf_delta_ms_mean": float(np.mean(tf_delta_ms)),
        "tf_delta_ms_max": float(np.max(tf_delta_ms)),
        "front_tolerance_m": args.front_tolerance_m,
        "back_tolerance_m": args.back_tolerance_m,
        "surface_sphere_padding_m": args.surface_sphere_padding_m,
        "mask_shadow_behind_robot": args.mask_shadow_behind_robot,
        "debug_directory": str(debug_dir) if debug_dir else None,
        **dict(totals),
        "surface_band_ratio": totals["surface_band_pixels"] / valid_robot_pixels,
        "foreground_kept_ratio": totals["foreground_kept_pixels"] / valid_robot_pixels,
        "behind_ratio": totals["behind_pixels"] / valid_robot_pixels,
        "foreground_preservation_passed": (
            totals["foreground_removed_pixels"] == 0
        ),
    }
    return result


def print_report(report):
    """Print a concise human-readable validation report."""
    valid = max(report["valid_robot_ray_pixels"], 1)
    per_frame = max(report["processed_frames"], 1)
    print(f"\nBag: {report['bag_path']}")
    print(
        f"  frames: processed={report['processed_frames']} "
        f"sampled={report['sampled_frames']} seen={report['depth_frames_seen']}"
    )
    print(
        f"  robot TF: states={report['robot_tf_states']} "
        f"nearest_delta_mean={report['tf_delta_ms_mean']:.3f} ms "
        f"max={report['tf_delta_ms_max']:.3f} ms "
        f"time_misses={report['time_miss_frames']}"
    )
    print(
        f"  valid predicted robot rays/frame: "
        f"{report['valid_robot_ray_pixels'] / per_frame:.1f}"
    )
    for key, label in (
        ("surface_band_pixels", "surface removed"),
        ("foreground_kept_pixels", "foreground kept"),
        ("behind_pixels", "behind kept/classified"),
    ):
        print(
            f"  {label}: {report[key]} "
            f"({100.0 * report[key] / valid:.2f}% of valid robot rays)"
        )
    print(
        f"  foreground incorrectly removed: "
        f"{report['foreground_removed_pixels']} "
        f"[{'PASS' if report['foreground_preservation_passed'] else 'FAIL'}]"
    )
    if "foreground_volume_would_remove_pixels" in report:
        print(
            f"  foreground legacy volume filter would remove: "
            f"{report['foreground_volume_would_remove_pixels']}"
        )
    if report["debug_directory"]:
        print(f"  debug images: {report['debug_directory']}")


def default_sphere_config_path():
    """Locate collision_spheres.yaml in a source or installed package."""
    source_candidate = (
        Path(__file__).resolve().parents[1] / "config" / "collision_spheres.yaml"
    )
    if source_candidate.exists():
        return source_candidate
    from ament_index_python.packages import get_package_share_directory

    return (
        Path(get_package_share_directory("rmp_camera"))
        / "config"
        / "collision_spheres.yaml"
    )


def parse_arguments(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate surface-depth robot self filtering offline using raw depth, "
            "CameraInfo, and recorded link TF from ROS 2 bags."
        )
    )
    parser.add_argument("bag_paths", nargs="+", help="ROS 2 bag directories")
    parser.add_argument(
        "--sphere-config",
        default=str(default_sphere_config_path()),
        help="collision_spheres.yaml path",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera0/camera/depth/image_rect_raw",
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/camera0/camera/depth/camera_info",
    )
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument("--camera-frame", default="")
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--front-tolerance-m", type=float, default=0.02)
    parser.add_argument("--back-tolerance-m", type=float, default=0.03)
    parser.add_argument("--surface-sphere-padding-m", type=float, default=0.0)
    parser.add_argument("--mask-shadow-behind-robot", action="store_true")
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    parser.add_argument("--max-tf-delta-s", type=float, default=0.15)
    parser.add_argument(
        "--compare-volume-margin-m",
        type=float,
        default=0.06,
        help="Legacy volume margin to compare; set negative to disable",
    )
    parser.add_argument("--debug-dir", default="")
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument("--max-debug-images", type=int, default=12)
    parser.add_argument("--debug-max-depth-m", type=float, default=3.0)
    parser.add_argument("--json-report", default="")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args(argv)
    if args.sample_every < 1:
        parser.error("--sample-every must be at least 1")
    if args.debug_every < 1:
        parser.error("--debug-every must be at least 1")
    if args.max_debug_images < 0:
        parser.error("--max-debug-images must be non-negative")
    return args


def main(argv=None):
    """Run offline validation for one or more bags."""
    args = parse_arguments(argv)
    base_frame, spheres = load_sphere_config(args.sphere_config)
    reports = []
    for bag_path in args.bag_paths:
        report = analyze_bag(args, bag_path, base_frame, spheres)
        reports.append(report)
        print_report(report)

    if args.json_report:
        output_path = Path(args.json_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(reports, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report: {output_path}")

    return 0 if all(item["foreground_preservation_passed"] for item in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
