"""Estimate a fixed eye-to-hand transform from robot-only RGB-D bag data.

The calibrator treats the robot CAD meshes as the calibration target.  It
selects stationary, geometrically distinct robot poses, crops measured depth
near the current CAD prediction, and optimizes one rigid correction shared by
all poses.  The result is deliberately written as a candidate report; this
tool never edits launch or experiment configuration files.
"""

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import warnings
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml

from rmp_camera.charuco_common import (
    invert_transform,
    list_from_matrix,
    rpy_from_matrix,
    transform_from_xyz_rpy,
)
from rmp_camera.robot_self_filter_bag_validator import (
    camera_intrinsics,
    colorize_depth,
    depth_image_to_meters,
    open_bag_reader,
    read_first_camera_info,
    resolve_transform,
    stamp_to_ns,
)
from rmp_camera.vision_geometry import transform_to_matrix


DEFAULT_LINKS = ("link2", "link3", "link4", "link5", "link6")
DEFAULT_JOINTS = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")


@dataclass
class TfSnapshot:
    """Robot and camera transforms at one recorded timestamp."""

    stamp_ns: int
    base_from_links: dict
    base_from_observation: np.ndarray
    base_from_output: np.ndarray


@dataclass
class CalibrationFrame:
    """One stationary depth observation and its matching robot CAD cloud."""

    stamp_ns: int
    depth_delta_ms: float
    depth_m: np.ndarray
    observed_camera: np.ndarray
    model_base: np.ndarray
    model_tree: object
    initial_crop_median_m: float


def parse_numbers(text, expected, label):
    """Parse a whitespace-separated vector with an exact length."""
    values = [float(value) for value in (text or "").split()]
    if len(values) != expected:
        raise ValueError(f"{label} must contain {expected} values, got {text!r}")
    return values


def nearest_index(sorted_values, target):
    """Return the closest index in a non-empty sorted sequence."""
    if not sorted_values:
        raise ValueError("nearest_index requires at least one value")
    index = bisect_left(sorted_values, target)
    if index <= 0:
        return 0
    if index >= len(sorted_values):
        return len(sorted_values) - 1
    before = index - 1
    if target - sorted_values[before] <= sorted_values[index] - target:
        return before
    return index


def read_binary_stl(path):
    """Return binary STL triangles as an ``N x 3 x 3`` float64 array."""
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"STL file is too short: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(payload) != expected_size:
        raise ValueError(
            f"Only binary STL is supported: {path} has {len(payload)} bytes, "
            f"expected {expected_size}"
        )
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    records = np.frombuffer(payload, dtype=dtype, offset=84, count=triangle_count)
    return records["vertices"].astype(np.float64)


def sample_triangle_surface(triangles, sample_count, seed):
    """Area-weighted deterministic sampling of triangle surfaces."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (N, 3, 3)")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 1e-12)
    triangles = triangles[valid]
    areas = areas[valid]
    if not len(triangles):
        raise ValueError("mesh contains no non-degenerate triangles")
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(triangles), size=int(sample_count), replace=True, p=areas / areas.sum()
    )
    selected = triangles[indices]
    first = rng.random(sample_count)
    second = rng.random(sample_count)
    flip = first + second > 1.0
    first[flip] = 1.0 - first[flip]
    second[flip] = 1.0 - second[flip]
    return (
        selected[:, 0]
        + first[:, None] * (selected[:, 1] - selected[:, 0])
        + second[:, None] * (selected[:, 2] - selected[:, 0])
    )


def resolve_package_mesh(uri, urdf_path):
    """Resolve this package's ``package://`` mesh URI from source/install trees."""
    prefix = "package://rmp_camera/"
    if not uri.startswith(prefix):
        raise ValueError(f"Unsupported mesh URI: {uri}")
    relative = uri[len(prefix):]
    source_candidate = Path(urdf_path).resolve().parents[1] / relative
    if source_candidate.exists():
        return source_candidate
    from ament_index_python.packages import get_package_share_directory

    installed = Path(get_package_share_directory("rmp_camera")) / relative
    if not installed.exists():
        raise FileNotFoundError(f"Could not resolve mesh {uri}")
    return installed


def load_link_mesh_samples(urdf_path, links, samples_per_link, seed):
    """Load the first collision mesh for each selected URDF link."""
    root = ET.parse(urdf_path).getroot()
    by_name = {item.attrib.get("name"): item for item in root.findall("link")}
    samples = {}
    for index, link in enumerate(links):
        if link not in by_name:
            raise ValueError(f"URDF has no link named {link}")
        collision_mesh = None
        collision_origin = None
        for collision in by_name[link].findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is not None:
                collision_mesh = mesh
                collision_origin = collision.find("origin")
                break
        if collision_mesh is None:
            raise ValueError(f"URDF link {link} has no collision mesh")
        mesh_path = resolve_package_mesh(collision_mesh.attrib["filename"], urdf_path)
        triangles = read_binary_stl(mesh_path)
        scale = parse_numbers(collision_mesh.attrib.get("scale", "1 1 1"), 3, "mesh scale")
        triangles = triangles * np.asarray(scale, dtype=np.float64)[None, None, :]
        points = sample_triangle_surface(
            triangles,
            int(samples_per_link),
            int(seed) + index,
        )
        if collision_origin is not None:
            xyz = parse_numbers(collision_origin.attrib.get("xyz", "0 0 0"), 3, "origin xyz")
            rpy = parse_numbers(collision_origin.attrib.get("rpy", "0 0 0"), 3, "origin rpy")
            link_from_mesh = transform_from_xyz_rpy(xyz, rpy)
            points = (
                link_from_mesh[:3, :3] @ points.T
            ).T + link_from_mesh[:3, 3]
        samples[link] = points
    return samples


def select_stable_pose_targets(args):
    """Select stationary joint-state segments with diverse configurations."""
    from rclpy.serialization import deserialize_message

    reader, message_types = open_bag_reader(
        args.bag_path, [args.joint_topic], args.storage_id
    )
    messages = []
    while reader.has_next():
        topic, data, bag_stamp = reader.read_next()
        msg = deserialize_message(data, message_types[topic])
        stamp = stamp_to_ns(msg.header.stamp) or int(bag_stamp)
        messages.append((stamp, list(msg.name), np.asarray(msg.position, dtype=np.float64)))
    if len(messages) < 2:
        raise ValueError(f"Joint topic {args.joint_topic} has fewer than two messages")

    joint_names = [item.strip() for item in args.joint_names.split(",") if item.strip()]
    first_names = messages[0][1]
    missing = [name for name in joint_names if name not in first_names]
    if missing:
        raise ValueError(f"Joint topic is missing names: {', '.join(missing)}")
    indices = [first_names.index(name) for name in joint_names]
    stamps = np.asarray([item[0] for item in messages], dtype=np.int64)
    positions = np.asarray([item[2][indices] for item in messages], dtype=np.float64)
    delta_t = np.diff(stamps) * 1e-9
    delta_q = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = np.divide(
        delta_q,
        delta_t,
        out=np.full(delta_q.shape, np.inf, dtype=np.float64),
        where=delta_t > 1e-6,
    )
    stable = np.r_[False, speed < math.radians(args.stable_speed_deg_s)]

    segments = []
    start = None
    for index, is_stable in enumerate(stable):
        if is_stable and start is None:
            start = index
        is_last = index == len(stable) - 1
        if (not is_stable or is_last) and start is not None:
            end = index if not is_stable else index + 1
            duration = (stamps[end - 1] - stamps[start]) * 1e-9
            if duration >= args.min_stable_duration_s:
                segments.append((start, end, duration))
            start = None

    selected = []
    settling_cutoff = stamps[0] + int(args.settling_time_s * 1e9)
    for start, end, duration in segments:
        target = int((stamps[start] + stamps[end - 1]) // 2)
        if target < settling_cutoff:
            continue
        pose = positions[start:end].mean(axis=0)
        if selected:
            rms_degrees = min(
                np.linalg.norm(np.degrees(pose - item[1])) / math.sqrt(len(pose))
                for item in selected
            )
            if rms_degrees < args.min_pose_separation_deg:
                continue
        selected.append((target, pose, duration))

    if len(selected) > args.max_poses:
        keep = np.linspace(0, len(selected) - 1, args.max_poses, dtype=int)
        selected = [selected[index] for index in keep]
    if len(selected) < args.min_poses:
        raise ValueError(
            f"Only {len(selected)} distinct stable poses found; need {args.min_poses}"
        )
    return selected, len(segments), joint_names


def load_tf_snapshots(args, target_stamps, links, observation_frame):
    """Read robot-link and fixed camera transforms nearest selected poses."""
    from rclpy.serialization import deserialize_message

    topics = [args.tf_topic, args.tf_static_topic]
    reader, message_types = open_bag_reader(args.bag_path, topics, args.storage_id)
    static_tree = {}
    rolling_dynamic_tree = {}
    snapshots = []
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
            stamps.append(stamp_to_ns(item.header.stamp) or int(bag_stamp))
        if topic == args.tf_static_topic:
            static_tree.update(edges)
            continue
        rolling_dynamic_tree.update(edges)
        tree = dict(static_tree)
        tree.update(rolling_dynamic_tree)
        try:
            base_from_links = {
                link: resolve_transform(tree, link, args.base_frame) for link in links
            }
            base_from_observation = resolve_transform(
                tree, observation_frame, args.base_frame
            )
            base_from_output = resolve_transform(
                tree, args.output_camera_frame, args.base_frame
            )
        except KeyError:
            continue
        snapshots.append(
            TfSnapshot(
                stamp_ns=int(np.median(stamps)) if stamps else int(bag_stamp),
                base_from_links=base_from_links,
                base_from_observation=base_from_observation,
                base_from_output=base_from_output,
            )
        )
    if not snapshots:
        raise ValueError("No complete robot/camera TF snapshots could be reconstructed")
    snapshot_stamps = [item.stamp_ns for item in snapshots]
    selected = [
        snapshots[nearest_index(snapshot_stamps, target)] for target in target_stamps
    ]
    return selected, len(snapshots)


def read_nearest_depth_messages(args, target_stamps):
    """Read the depth message nearest each selected stationary timestamp."""
    from rclpy.serialization import deserialize_message

    reader, message_types = open_bag_reader(
        args.bag_path, [args.depth_topic], args.storage_id
    )
    best = [(10**30, None, None, None) for _ in target_stamps]
    while reader.has_next():
        topic, data, bag_stamp = reader.read_next()
        index = nearest_index(target_stamps, int(bag_stamp))
        delta = abs(target_stamps[index] - int(bag_stamp))
        if delta < best[index][0]:
            best[index] = (delta, data, message_types[topic], int(bag_stamp))
    messages = []
    for delta, data, message_type, bag_stamp in best:
        if data is None:
            raise ValueError("Could not find a depth frame for every stable pose")
        messages.append((delta, deserialize_message(data, message_type), bag_stamp))
    return messages


def build_calibration_frames(args, pose_targets, snapshots, mesh_samples, intrinsics):
    """Build model clouds and observed depth crops for optimization."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A NumPy version.*")
        from scipy.spatial import cKDTree

    fx, fy, cx, cy = intrinsics
    target_stamps = [item[0] for item in pose_targets]
    depth_messages = read_nearest_depth_messages(args, target_stamps)
    frames = []
    for target, snapshot, depth_item in zip(target_stamps, snapshots, depth_messages):
        delta, message, _ = depth_item
        depth_m = depth_image_to_meters(message)
        model_parts = []
        for link, local_points in mesh_samples.items():
            base_from_link = snapshot.base_from_links[link]
            model_parts.append(
                (base_from_link[:3, :3] @ local_points.T).T
                + base_from_link[:3, 3]
            )
        model_base = np.concatenate(model_parts, axis=0)
        model_tree = cKDTree(model_base)

        rows, columns = np.mgrid[
            0 : depth_m.shape[0] : args.depth_stride,
            0 : depth_m.shape[1] : args.depth_stride,
        ]
        sampled_depth = depth_m[:: args.depth_stride, :: args.depth_stride]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= args.min_depth_m)
            & (sampled_depth <= args.max_depth_m)
        )
        z = sampled_depth[valid]
        u = columns[valid]
        v = rows[valid]
        observed_camera = np.column_stack(
            ((u - cx) * z / fx, (v - cy) * z / fy, z)
        )
        initial = snapshot.base_from_observation
        observed_base = (
            initial[:3, :3] @ observed_camera.T
        ).T + initial[:3, 3]
        initial_distances = model_tree.query(observed_base, k=1, workers=-1)[0]
        keep = initial_distances < args.crop_radius_m
        observed_camera = observed_camera[keep]
        kept_distances = initial_distances[keep]
        if len(observed_camera) < args.min_observed_points:
            raise ValueError(
                f"Pose at {target} has only {len(observed_camera)} cropped depth points; "
                f"need {args.min_observed_points}"
            )
        if len(observed_camera) > args.max_observed_points:
            indices = np.linspace(
                0, len(observed_camera) - 1, args.max_observed_points, dtype=int
            )
            observed_camera = observed_camera[indices]
            kept_distances = kept_distances[indices]
        frames.append(
            CalibrationFrame(
                stamp_ns=target,
                depth_delta_ms=float(delta) / 1e6,
                depth_m=depth_m,
                observed_camera=observed_camera,
                model_base=model_base,
                model_tree=model_tree,
                initial_crop_median_m=float(np.median(kept_distances)),
            )
        )
    return frames


def candidate_base_from_camera(initial_base_from_camera, correction_parameters):
    """Apply an optical-frame correction and return the candidate camera pose."""
    correction_parameters = np.asarray(correction_parameters, dtype=np.float64)
    if correction_parameters.shape != (6,):
        raise ValueError("correction_parameters must have shape (6,)")
    correction = transform_from_xyz_rpy(
        correction_parameters[:3], correction_parameters[3:]
    )
    return initial_base_from_camera @ invert_transform(correction)


def trimmed_frame_distances(
    correction_parameters,
    initial_base_from_camera,
    frames,
    trim_fraction,
):
    """Return robust nearest-mesh distances for every selected frame."""
    candidate = candidate_base_from_camera(
        initial_base_from_camera, correction_parameters
    )
    distances = []
    for frame in frames:
        observed_base = (
            candidate[:3, :3] @ frame.observed_camera.T
        ).T + candidate[:3, 3]
        values = frame.model_tree.query(observed_base, k=1, workers=-1)[0]
        values.sort()
        keep = max(20, int(len(values) * float(trim_fraction)))
        distances.append(values[:keep])
    return distances


def distance_metrics(distances, loss_cap_m):
    """Summarize robust distance arrays for reports and optimization."""
    merged = np.concatenate(distances)
    clipped = np.minimum(merged, float(loss_cap_m))
    return {
        "point_count": int(len(merged)),
        "median_m": float(np.median(merged)),
        "p90_m": float(np.percentile(merged, 90.0)),
        "mean_m": float(np.mean(merged)),
        "rmse_clipped_m": float(np.sqrt(np.mean(clipped * clipped))),
        "inlier_20mm_ratio": float(np.mean(merged <= 0.02)),
        "inlier_30mm_ratio": float(np.mean(merged <= 0.03)),
        "inlier_50mm_ratio": float(np.mean(merged <= 0.05)),
    }


def optimize_correction(args, initial_base_from_camera, frames, train_indices, seed):
    """Optimize one six-DoF optical correction on selected training poses."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="A NumPy version.*")
        from scipy.optimize import differential_evolution

    training_frames = [frames[index] for index in train_indices]
    cache = {}

    def objective(parameters):
        key = tuple(np.round(parameters, 8))
        if key not in cache:
            distances = trimmed_frame_distances(
                parameters,
                initial_base_from_camera,
                training_frames,
                args.trim_fraction,
            )
            metrics = distance_metrics(distances, args.loss_cap_m)
            cache[key] = metrics["rmse_clipped_m"] ** 2
        return cache[key]

    rotation_bound = math.radians(args.rotation_bound_deg)
    bounds = [(-args.translation_bound_m, args.translation_bound_m)] * 3 + [
        (-rotation_bound, rotation_bound)
    ] * 3
    return differential_evolution(
        objective,
        bounds,
        popsize=args.population_size,
        maxiter=args.max_iterations,
        tol=args.optimizer_tolerance,
        seed=int(seed),
        polish=True,
        workers=1,
        updating="immediate",
    )


def metrics_for_indices(args, parameters, initial, frames, indices):
    """Compute robust metrics on an explicit frame subset."""
    selected = [frames[index] for index in indices]
    distances = trimmed_frame_distances(
        parameters, initial, selected, args.trim_fraction
    )
    return distance_metrics(distances, args.loss_cap_m)


def transform_launch_arguments(matrix):
    """Return xyz/rpy launch values from a homogeneous transform."""
    roll, pitch, yaw = rpy_from_matrix(matrix[:3, :3])
    x, y, z = matrix[:3, 3]
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def write_debug_overlays(args, frames, initial, candidate):
    """Write measured-depth images with initial/candidate mesh projections."""
    if not args.debug_dir:
        return []
    output_dir = Path(args.debug_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_info = read_first_camera_info(
        args.bag_path, args.camera_info_topic, args.storage_id
    )
    fx, fy, cx, cy = camera_intrinsics(camera_info)
    outputs = []
    for index, frame in enumerate(frames):
        image = colorize_depth(frame.depth_m, args.debug_max_depth_m)
        for transform, color in ((initial, (0, 0, 255)), (candidate, (0, 255, 0))):
            camera_from_base = invert_transform(transform)
            points = (
                camera_from_base[:3, :3] @ frame.model_base.T
            ).T + camera_from_base[:3, 3]
            valid = points[:, 2] > args.min_depth_m
            points = points[valid]
            columns = np.rint(fx * points[:, 0] / points[:, 2] + cx).astype(int)
            rows = np.rint(fy * points[:, 1] / points[:, 2] + cy).astype(int)
            inside = (
                (columns >= 0)
                & (columns < image.shape[1])
                & (rows >= 0)
                & (rows < image.shape[0])
            )
            for column, row in zip(columns[inside], rows[inside]):
                cv2.circle(image, (int(column), int(row)), 1, color, -1)
        cv2.putText(
            image,
            "red=initial CAD  green=candidate CAD",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        path = output_dir / f"pose_{index:02d}_overlay.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to write debug image {path}")
        outputs.append(str(path))
    return outputs


def default_package_path(relative):
    """Resolve a source-tree file first, then an installed package file."""
    source = Path(__file__).resolve().parents[1] / relative
    if source.exists():
        return source
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("rmp_camera")) / relative


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Estimate base-to-camera extrinsics from robot CAD and a robot-only "
            "RGB-D ROS 2 bag. The tool writes a candidate report and never edits config."
        )
    )
    parser.add_argument("bag_path", help="ROS 2 bag directory")
    parser.add_argument("--output", default="/tmp/rmp_camera_markerless_result.yaml")
    parser.add_argument("--debug-dir", default="/tmp/rmp_camera_markerless_debug")
    parser.add_argument("--urdf", default=str(default_package_path("urdf/rb10_1300e.urdf")))
    parser.add_argument("--links", default=",".join(DEFAULT_LINKS))
    parser.add_argument("--joint-names", default=",".join(DEFAULT_JOINTS))
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--output-camera-frame", default="camera0_link")
    parser.add_argument("--depth-topic", default="/camera0/camera/depth/image_rect_raw")
    parser.add_argument("--camera-info-topic", default="/camera0/camera/depth/camera_info")
    parser.add_argument("--joint-topic", default="/rmp_camera/joint_states_urdf")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-static-topic", default="/tf_static")
    parser.add_argument("--storage-id", default="sqlite3")
    parser.add_argument("--min-poses", type=int, default=8)
    parser.add_argument("--max-poses", type=int, default=16)
    parser.add_argument("--settling-time-s", type=float, default=5.0)
    parser.add_argument("--stable-speed-deg-s", type=float, default=0.5)
    parser.add_argument("--min-stable-duration-s", type=float, default=0.5)
    parser.add_argument("--min-pose-separation-deg", type=float, default=3.0)
    parser.add_argument("--samples-per-link", type=int, default=1200)
    parser.add_argument("--depth-stride", type=int, default=3)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--crop-radius-m", type=float, default=0.10)
    parser.add_argument("--min-observed-points", type=int, default=250)
    parser.add_argument("--max-observed-points", type=int, default=3500)
    parser.add_argument("--trim-fraction", type=float, default=0.75)
    parser.add_argument("--loss-cap-m", type=float, default=0.08)
    parser.add_argument("--translation-bound-m", type=float, default=0.10)
    parser.add_argument("--rotation-bound-deg", type=float, default=3.5)
    parser.add_argument("--population-size", type=int, default=7)
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument("--optimizer-tolerance", type=float, default=0.003)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--debug-max-depth-m", type=float, default=3.0)
    args = parser.parse_args(argv)
    args.bag_path = str(Path(args.bag_path).expanduser().resolve())
    args.urdf = str(Path(args.urdf).expanduser().resolve())
    if not 0.0 < args.trim_fraction <= 1.0:
        parser.error("--trim-fraction must be in (0, 1]")
    if args.min_poses < 4 or args.max_poses < args.min_poses:
        parser.error("pose limits must satisfy 4 <= min_poses <= max_poses")
    if args.depth_stride < 1 or args.samples_per_link < 100:
        parser.error("depth stride must be positive and samples-per-link >= 100")
    if args.restarts < 1:
        parser.error("--restarts must be positive")
    return args


def run_calibration(args):
    """Run extraction, optimization, held-out validation, and report writing."""
    links = [item.strip() for item in args.links.split(",") if item.strip()]
    camera_info = read_first_camera_info(
        args.bag_path, args.camera_info_topic, args.storage_id
    )
    observation_frame = camera_info.header.frame_id
    if not observation_frame:
        raise ValueError("Depth CameraInfo has an empty frame_id")
    intrinsics = camera_intrinsics(camera_info)

    print("Selecting stationary, distinct robot poses...", flush=True)
    pose_targets, stable_segment_count, joint_names = select_stable_pose_targets(args)
    target_stamps = [item[0] for item in pose_targets]
    print(
        f"Selected {len(pose_targets)} poses from {stable_segment_count} stable segments.",
        flush=True,
    )
    snapshots, tf_snapshot_count = load_tf_snapshots(
        args, target_stamps, links, observation_frame
    )
    mesh_samples = load_link_mesh_samples(
        args.urdf, links, args.samples_per_link, args.seed
    )
    frames = build_calibration_frames(
        args, pose_targets, snapshots, mesh_samples, intrinsics
    )
    for index, frame in enumerate(frames):
        print(
            f"Pose {index:02d}: points={len(frame.observed_camera)}, "
            f"depth_delta={frame.depth_delta_ms:.1f} ms, "
            f"initial_crop_median={frame.initial_crop_median_m * 1000.0:.1f} mm",
            flush=True,
        )

    initial_observation = snapshots[0].base_from_observation
    initial_output = snapshots[0].base_from_output
    output_from_observation = invert_transform(initial_output) @ initial_observation
    train_indices = list(range(0, len(frames), 2))
    validation_indices = list(range(1, len(frames), 2))
    if not validation_indices:
        validation_indices = train_indices
    zero = np.zeros(6, dtype=np.float64)
    baseline_train = metrics_for_indices(
        args, zero, initial_observation, frames, train_indices
    )
    baseline_validation = metrics_for_indices(
        args, zero, initial_observation, frames, validation_indices
    )
    print(
        "Baseline: "
        f"train median={baseline_train['median_m'] * 1000.0:.1f} mm, "
        f"validation median={baseline_validation['median_m'] * 1000.0:.1f} mm",
        flush=True,
    )

    results = []
    for restart in range(args.restarts):
        print(
            f"Optimizing restart {restart + 1}/{args.restarts}...",
            flush=True,
        )
        result = optimize_correction(
            args,
            initial_observation,
            frames,
            train_indices,
            args.seed + restart * 997,
        )
        validation = metrics_for_indices(
            args, result.x, initial_observation, frames, validation_indices
        )
        train = metrics_for_indices(
            args, result.x, initial_observation, frames, train_indices
        )
        results.append((result, train, validation))
        print(
            f"Restart {restart + 1}: train={train['median_m'] * 1000.0:.1f} mm, "
            f"validation={validation['median_m'] * 1000.0:.1f} mm, "
            f"validation_p90={validation['p90_m'] * 1000.0:.1f} mm",
            flush=True,
        )
    best_result, candidate_train, candidate_validation = min(
        results, key=lambda item: item[2]["rmse_clipped_m"]
    )
    parameters = np.asarray(best_result.x, dtype=np.float64)
    candidate_observation = candidate_base_from_camera(
        initial_observation, parameters
    )
    candidate_output = candidate_observation @ invert_transform(output_from_observation)
    all_indices = list(range(len(frames)))
    baseline_all = metrics_for_indices(
        args, zero, initial_observation, frames, all_indices
    )
    candidate_all = metrics_for_indices(
        args, parameters, initial_observation, frames, all_indices
    )

    restart_parameter_spread = 0.0
    restart_rotation_spread_deg = 0.0
    if len(results) > 1:
        translations = np.asarray([item[0].x[:3] for item in results])
        rotations = np.asarray([item[0].x[3:] for item in results])
        restart_parameter_spread = float(
            max(np.linalg.norm(a - b) for a in translations for b in translations)
        )
        restart_rotation_spread_deg = float(
            math.degrees(
                max(np.linalg.norm(a - b) for a in rotations for b in rotations)
            )
        )

    validation_improvement = 1.0 - (
        candidate_validation["median_m"] / max(baseline_validation["median_m"], 1e-12)
    )
    passes_offline_gate = bool(
        validation_improvement >= 0.25
        and candidate_validation["p90_m"] <= 0.05
        and restart_parameter_spread <= 0.03
        and restart_rotation_spread_deg <= 1.0
    )
    debug_images = write_debug_overlays(
        args, frames, initial_observation, candidate_observation
    )
    launch = transform_launch_arguments(candidate_output)
    correction_rpy = parameters[3:].tolist()
    output = {
        "status": "candidate_only",
        "passes_offline_gate": passes_offline_gate,
        "requires_visual_validation": True,
        "bag_path": args.bag_path,
        "base_frame": args.base_frame,
        "observation_camera_frame": observation_frame,
        "output_camera_frame": args.output_camera_frame,
        "selected_pose_count": len(frames),
        "stable_segment_count": stable_segment_count,
        "tf_snapshot_count": tf_snapshot_count,
        "joint_names": joint_names,
        "mesh_links": links,
        "train_pose_indices": train_indices,
        "validation_pose_indices": validation_indices,
        "initial_base_T_observation_camera": list_from_matrix(initial_observation),
        "candidate_base_T_observation_camera": list_from_matrix(candidate_observation),
        "initial_base_T_output_camera": list_from_matrix(initial_output),
        "candidate_base_T_output_camera": list_from_matrix(candidate_output),
        "optical_correction": {
            "translation_xyz_m": parameters[:3].tolist(),
            "rotation_rpy_rad": correction_rpy,
            "rotation_rpy_deg": np.degrees(parameters[3:]).tolist(),
        },
        "launch_arguments": launch,
        "launch_arguments_text": (
            f"x:={launch['x']:.9f} y:={launch['y']:.9f} z:={launch['z']:.9f} "
            f"roll:={launch['roll']:.9f} pitch:={launch['pitch']:.9f} "
            f"yaw:={launch['yaw']:.9f}"
        ),
        "baseline_metrics": {
            "train": baseline_train,
            "validation": baseline_validation,
            "all": baseline_all,
        },
        "candidate_metrics": {
            "train": candidate_train,
            "validation": candidate_validation,
            "all": candidate_all,
        },
        "validation_median_improvement_ratio": float(validation_improvement),
        "restart_translation_spread_m": restart_parameter_spread,
        "restart_rotation_spread_deg": restart_rotation_spread_deg,
        "optimizer": {
            "success": bool(best_result.success),
            "message": str(best_result.message),
            "function_evaluations": int(best_result.nfev),
            "restarts": int(args.restarts),
            "translation_bound_m": float(args.translation_bound_m),
            "rotation_bound_deg": float(args.rotation_bound_deg),
        },
        "pose_inputs": [
            {
                "index": index,
                "stamp_ns": int(frame.stamp_ns),
                "depth_delta_ms": frame.depth_delta_ms,
                "observed_point_count": int(len(frame.observed_camera)),
                "initial_crop_median_m": frame.initial_crop_median_m,
            }
            for index, frame in enumerate(frames)
        ],
        "debug_images": debug_images,
        "application_note": (
            "After visual validation, copy launch_arguments into experiment "
            "static_tf_* and set robot_sphere_marker_correction translations/rotations "
            "to zero. Do not apply optical_correction and static_tf together."
        ),
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate result: {output_path}")
    print(f"JSON result: {json_path}")
    print(
        f"Validation median: {baseline_validation['median_m'] * 1000.0:.1f} -> "
        f"{candidate_validation['median_m'] * 1000.0:.1f} mm; "
        f"p90={candidate_validation['p90_m'] * 1000.0:.1f} mm"
    )
    print(f"Offline gate: {'PASS' if passes_offline_gate else 'FAIL'}")
    print(f"Launch args: {output['launch_arguments_text']}")
    return output


def main(argv=None):
    args = parse_arguments(argv)
    run_calibration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
