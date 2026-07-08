import math

import numpy as np


def quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_matrix(transform.rotation)
    matrix[:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return matrix


def transform_points(points, transform_matrix):
    if len(points) == 0:
        return points.reshape(0, 3)
    return (transform_matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]


def marker_spheres(marker_array):
    centers = []
    radii = []
    ids = []
    for marker in marker_array.markers:
        if marker.type != marker.SPHERE or marker.action != marker.ADD:
            continue
        centers.append([marker.pose.position.x, marker.pose.position.y, marker.pose.position.z])
        radii.append(max(marker.scale.x, marker.scale.y, marker.scale.z) * 0.5)
        ids.append(marker.id)
    return np.asarray(centers, dtype=np.float64), np.asarray(radii, dtype=np.float64), ids


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]
