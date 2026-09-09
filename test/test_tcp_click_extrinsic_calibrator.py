from types import SimpleNamespace

import cv2
import numpy as np

from rmp_camera.charuco_common import transform_from_xyz_rpy
from rmp_camera.robot_depth_extrinsic_calibrator import candidate_base_from_camera
from rmp_camera.tcp_click_extrinsic_calibrator import (
    camera_model,
    decode_color_image,
    optimize_reprojection,
    predicted_pixels,
    reprojection_metrics,
)


def test_decode_rgb8_honors_padded_rows():
    # Two RGB pixels plus one padding pixel per row.
    message = SimpleNamespace(
        encoding="rgb8",
        width=2,
        height=1,
        step=9,
        data=bytes([255, 0, 0, 0, 255, 0, 9, 9, 9]),
    )
    image = decode_color_image(message)
    assert image.shape == (1, 2, 3)
    assert image[0, 0].tolist() == [0, 0, 255]
    assert image[0, 1].tolist() == [0, 255, 0]


def test_camera_model_uses_raw_image_k_and_distortion():
    info = SimpleNamespace(
        k=[600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0],
        d=[0.1, -0.2, 0.001, 0.002, 0.0],
        distortion_model="plumb_bob",
    )
    matrix, distortion, model = camera_model(info)
    assert matrix[0, 0] == 600.0
    assert matrix[1, 1] == 601.0
    assert np.allclose(distortion[:4], [0.1, -0.2, 0.001, 0.002])
    assert model == "plumb_bob"


def test_reprojection_metrics_are_euclidean_pixels():
    metrics = reprojection_metrics([[0.0, 0.0], [4.0, 6.0]], [[3.0, 4.0], [4.0, 6.0]])
    assert metrics["per_pose_px"] == [5.0, 0.0]
    assert metrics["median_px"] == 2.5


def test_tcp_click_optimizer_recovers_camera_from_diverse_tcp_positions():
    rng = np.random.default_rng(4)
    matrix = np.asarray(
        [[910.0, 0.0, 640.0], [0.0, 908.0, 360.0], [0.0, 0.0, 1.0]]
    )
    initial = transform_from_xyz_rpy(
        [1.36, -1.74, 0.99], [-0.016, 0.281, 2.006]
    )
    truth_correction = np.asarray(
        [0.012, -0.009, 0.007, 0.004, -0.006, 0.003], dtype=np.float64
    )
    transforms = []
    truth_camera = candidate_base_from_camera(initial, truth_correction)
    # Generate a non-coplanar set that is guaranteed to lie in front of and
    # inside the synthetic camera image.
    for x in np.linspace(-0.35, 0.35, 4):
        for y in np.linspace(-0.22, 0.22, 3):
            camera_point = np.asarray([x, y, 1.3 + 0.7 * rng.random(), 1.0])
            base_point = (truth_camera @ camera_point)[:3]
            transforms.append(
                transform_from_xyz_rpy(base_point, [0.0, 0.0, 0.0])
            )
    observed, _ = predicted_pixels(
        truth_correction,
        initial,
        transforms,
        [0.0, 0.0, 0.0],
        matrix,
        np.zeros(5),
        "plumb_bob",
    )
    observed += rng.normal(0.0, 0.15, observed.shape)
    args = SimpleNamespace(
        feature_offset_tcp=(0.0, 0.0, 0.0),
        estimate_feature_offset=False,
        feature_offset_bound_m=0.2,
        translation_bound_m=0.04,
        rotation_bound_deg=2.0,
        robust_loss_px=2.0,
        restarts=2,
        max_function_evaluations=1500,
    )
    result, _ = optimize_reprojection(
        args,
        initial,
        np.asarray(transforms),
        observed,
        matrix,
        np.zeros(5),
        "plumb_bob",
        np.arange(len(transforms)),
        seed=3,
    )
    estimated = candidate_base_from_camera(initial, result.x[:6])
    truth = candidate_base_from_camera(initial, truth_correction)
    translation_error = np.linalg.norm(estimated[:3, 3] - truth[:3, 3])
    rotation_delta = estimated[:3, :3].T @ truth[:3, :3]
    rotation_error = np.linalg.norm(cv2.Rodrigues(rotation_delta)[0])
    assert translation_error < 0.004
    assert rotation_error < np.deg2rad(0.25)
