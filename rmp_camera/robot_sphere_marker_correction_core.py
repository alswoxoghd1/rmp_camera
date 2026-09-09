"""Rigid correction helpers for recorded robot collision sphere markers."""

import math

import numpy as np


def rpy_correction_matrix(translation_xyz_m, rotation_rpy_rad):
    """Return a homogeneous transform applied in the correction frame."""
    translation = np.asarray(translation_xyz_m, dtype=np.float64)
    rpy = np.asarray(rotation_rpy_rad, dtype=np.float64)
    if translation.shape != (3,) or rpy.shape != (3,):
        raise ValueError("translation and rotation must each contain three values")

    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def correction_in_marker_frame(marker_to_correction, correction):
    """Express a correction defined in another frame in the marker frame."""
    marker_to_correction = np.asarray(marker_to_correction, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if marker_to_correction.shape != (4, 4) or correction.shape != (4, 4):
        raise ValueError("transforms must be 4x4 matrices")
    return np.linalg.inv(marker_to_correction) @ correction @ marker_to_correction


def transform_pose_matrix(pose_matrix, marker_frame_correction):
    """Apply the marker-frame correction to a pose matrix."""
    pose_matrix = np.asarray(pose_matrix, dtype=np.float64)
    marker_frame_correction = np.asarray(
        marker_frame_correction, dtype=np.float64)
    if pose_matrix.shape != (4, 4) or marker_frame_correction.shape != (4, 4):
        raise ValueError("transforms must be 4x4 matrices")
    return marker_frame_correction @ pose_matrix


def matrix_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm
