"""Compact exact-generation support packets for coverage-preserving fusion."""
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def support_key(sphere, index):
    return sphere.track_id if sphere.track_id >= 0 else -index - 1


def current_support(spheres, components, tolerance, fresh_ids=None):
    if sum(len(c.voxel_centers) for c in components) * len(spheres) > 200000:
        return {}
    groups = {c.component_id: c.voxel_centers for c in components}
    result = {}
    for index, sphere in enumerate(spheres):
        if fresh_ids is not None and sphere.track_id not in fresh_ids:
            continue
        points = groups.get(sphere.component_id)
        if points is not None:
            mask = np.sum((points - sphere.center) ** 2, axis=1) <= (sphere.raw_radius + tolerance) ** 2
            if mask.any():
                result[support_key(sphere, index)] = points[mask]
    return result


def support_cloud(header, spheres, groups, evidence_stamp):
    selections = [(index, groups[support_key(s, index)]) for index, s in enumerate(spheres)
                  if support_key(s, index) in groups]
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('sphere_index', '<i4'), ('evidence_stamp', '<f8')])
    data = np.empty(sum(len(points) for _, points in selections), dtype=dtype)
    offset = 0
    for index, points in selections:
        end = offset + len(points)
        for axis, name in enumerate(('x', 'y', 'z')):
            data[name][offset:end] = points[:, axis]
        data['sphere_index'][offset:end] = index
        offset = end
    data['evidence_stamp'] = evidence_stamp
    fields = [PointField(name=name, offset=dtype.fields[name][1], count=1,
              datatype=PointField.INT32 if name == 'sphere_index' else
                       PointField.FLOAT64 if name == 'evidence_stamp' else PointField.FLOAT32)
              for name in dtype.names]
    return PointCloud2(header=header, height=1, width=len(data), fields=fields,
        is_bigendian=False, point_step=dtype.itemsize, row_step=len(data) * dtype.itemsize,
        is_dense=True, data=data.tobytes())
