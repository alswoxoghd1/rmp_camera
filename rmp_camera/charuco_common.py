import math

import cv2
import numpy as np


def aruco_dictionary(name):
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def make_charuco_board(squares_x, squares_y, square_length_m, marker_length_m, dictionary_name):
    dictionary = aruco_dictionary(dictionary_name)
    return cv2.aruco.CharucoBoard_create(
        int(squares_x),
        int(squares_y),
        float(square_length_m),
        float(marker_length_m),
        dictionary,
    )


def transform_from_rt(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def transform_from_rvec_tvec(rvec, tvec):
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return transform_from_rt(rotation, tvec)


def transform_from_rotvec_xyz(rotvec, xyz):
    rotation, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return transform_from_rt(rotation, xyz)


def rotvec_from_matrix(rotation):
    rotvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    return rotvec.reshape(3)


def invert_transform(transform):
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def matrix_from_quaternion(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(rotation):
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
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
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def matrix_from_rpy(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rotation_z @ rotation_y @ rotation_x


def rpy_from_matrix(rotation):
    matrix = np.asarray(rotation, dtype=np.float64)
    sy = math.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0])
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def transform_from_xyz_rpy(xyz, rpy):
    return transform_from_rt(matrix_from_rpy(*rpy), xyz)


def list_from_matrix(transform):
    return np.asarray(transform, dtype=np.float64).tolist()
