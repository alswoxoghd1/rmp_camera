from collections import deque
from dataclasses import replace
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import TransformException

from rmp_camera.sphere_depth_evidence import SphereDepthEvidence, SphereDepthBuffer
from rmp_camera.local_width_sphere_cover import VoxelSupportGuard
from rmp_camera.dynamic_obstacle_sphere_core import DynamicCandidateValidator, DynamicSphereParameters


def geometry():
    points = (np.indices((8, 8, 1)).reshape(3, -1).T + .5) * .05 + [-.2, -.2, 1.]
    guard = VoxelSupportGuard(points, [0, 0, 0], .05, .7)
    evidence = SphereDepthEvidence(np.full((100, 100), 1.025), (100., 100., 50., 50.), np.eye(4))
    return points, guard, evidence


def test_current_coherent_surface_supported_but_known_free_rejected():
    points, guard, evidence = geometry()
    assert evidence.validate(points, guard, monotonic()+1) is True
    assert evidence.validate(points-[0, 0, .2], guard, monotonic()+1) is False


@pytest.mark.parametrize('kind', ['zero', 'nan', 'discontinuous', 'occluded', 'foreign', 'deadline'])
def test_missing_or_occluded_rays_are_not_free_space(kind):
    points, guard, evidence = geometry()
    if kind == 'zero':
        evidence.depth[:] = 0.
    elif kind == 'nan':
        evidence.depth[:] = np.nan
    elif kind == 'discontinuous':
        evidence.depth[:, ::2] = .5
    elif kind == 'occluded':
        points = points + [0, 0, .3]
    elif kind == 'foreign':
        guard = VoxelSupportGuard(points+[1., 0, 0], [0, 0, 0], .05, .7)
    assert evidence.validate(points, guard, monotonic()+(-1 if kind == 'deadline' else 1)) is None


def test_audit_keeps_density_decision_and_missing_evidence_falls_back():
    points, guard, evidence = geometry()
    params = DynamicSphereParameters(min_x_m=-1, min_y_m=-1, min_z_m=0,
        observation_guard_mode='audit', dynamic_merge_enable_empty_space_guard=True)
    center, radius = points.mean(axis=0), .15
    expected = DynamicCandidateValidator(points, replace(params, observation_guard_mode='off'))(
        center, radius, monotonic()+1)
    assert DynamicCandidateValidator(points, params, evidence)(center, radius, monotonic()+1) == expected
    assert DynamicCandidateValidator(points, replace(params, observation_guard_mode='observed'))(
        center, radius, monotonic()+1) == expected
    assert evidence.stats['ray_candidates'] == 1


def test_observed_mode_can_accept_supported_surface_without_filling_map():
    points, guard, evidence = geometry()
    original = points.copy()
    params = DynamicSphereParameters(min_x_m=-1, min_y_m=-1, min_z_m=0,
        dynamic_merge_enable_empty_space_guard=True, dynamic_merge_max_empty_fraction=.7)
    center, radius = points.mean(axis=0), .15
    assert not DynamicCandidateValidator(points, params)(center, radius, monotonic()+1)
    assert DynamicCandidateValidator(points, replace(params, observation_guard_mode='observed'),
                                     evidence)(center, radius, monotonic()+1)
    np.testing.assert_array_equal(points, original)


def test_observed_mode_rejects_old_cloud_contradicting_fresh_free_rays():
    points = (np.indices((4, 4, 4)).reshape(3, -1).T+.5)*.05 + [0, 0, 1.]
    params = DynamicSphereParameters(dynamic_merge_enable_empty_space_guard=True)
    evidence = SphereDepthEvidence(np.full((100, 100), 2.), (100., 100., 50., 50.), np.eye(4))
    center, radius = points.mean(axis=0), .08
    assert DynamicCandidateValidator(points, params)(center, radius, monotonic()+1)
    assert not DynamicCandidateValidator(points, replace(params, observation_guard_mode='observed'),
                                         evidence)(center, radius, monotonic()+1)


def fake_buffer(encoding='16UC1', bigendian=False):
    now = SimpleNamespace(nanoseconds=1_000_000_000)
    transform = TransformStamped()
    transform.transform.rotation.w = 1.
    def lookup(*args, **kwargs):
        assert kwargs['timeout'].nanoseconds == 0
        return transform
    node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: now),
        target_frame='base', tf_buffer=SimpleNamespace(lookup_transform=lookup))
    obj = SphereDepthBuffer.__new__(SphereDepthBuffer)
    obj.node, obj.frames, obj.last_clock = node, deque(maxlen=8), None
    info = CameraInfo()
    info.width, info.height, info.header.frame_id = 4, 3, 'depth'
    info.k = [100., 0., 2., 0., 100., 1., 0., 0., 1.]
    obj.info = info
    image = Image()
    image.width, image.height, image.header.frame_id = 4, 3, 'depth'
    image.header.stamp.sec = 1
    image.encoding, image.is_bigendian = encoding, int(bigendian)
    dtype = np.dtype(('>' if bigendian else '<') + ('f4' if encoding == '32FC1' else 'u2'))
    # Include row padding to exercise step-aware decoding.
    array = np.full((3, 6), 1. if encoding == '32FC1' else 1000, dtype=dtype)
    image.step, image.data = 6*dtype.itemsize, array.tobytes()
    obj.add(image)
    return obj, image, now


@pytest.mark.parametrize('encoding,bigendian', [('16UC1', False), ('16UC1', True), ('32FC1', False), ('32FC1', True)])
def test_snapshot_decodes_padded_rows_without_waiting_for_tf(encoding, bigendian):
    obj, image, now = fake_buffer(encoding, bigendian)
    snap = obj.snapshot(now.nanoseconds)
    assert snap is not None
    np.testing.assert_array_equal(snap.depth, np.ones((3, 4)))


@pytest.mark.parametrize('kind', ['age', 'skew', 'wall_age', 'rewind', 'tf', 'payload', 'distortion', 'frame', 'intrinsics'])
def test_invalid_or_stale_depth_falls_back_without_blocking(kind):
    obj, image, now = fake_buffer()
    source = now.nanoseconds
    if kind == 'age':
        now.nanoseconds += 300_000_000
    elif kind == 'skew':
        source += 60_000_000
    elif kind == 'wall_age':
        obj.frames = deque([(image, monotonic()-1.)])
    elif kind == 'rewind':
        now.nanoseconds -= 1
    elif kind == 'tf':
        def missing(*args, **kwargs):
            raise TransformException('missing')
        obj.node.tf_buffer.lookup_transform = missing
    elif kind == 'payload':
        image.data = b'\x00'
    elif kind == 'distortion':
        obj.info.d = [.01]
    elif kind == 'frame':
        obj.info.header.frame_id = 'wrong'
    elif kind == 'intrinsics':
        obj.info.k = [0.]*9
    assert obj.snapshot(source) is None
