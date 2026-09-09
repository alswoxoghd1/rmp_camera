from array import array
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import Image
from rmp_camera.depth_edge_filter_core import DepthEdgeFilterConfig
from rmp_camera.depth_edge_filter_node import depth_image_view, filter_depth_message


def make_image(encoding, bigendian=False, padding=0):
    floating = encoding == '32FC1'
    dtype = np.dtype(('>' if bigendian else '<') + ('f4' if floating else 'u2'))
    msg = Image()
    msg.header.frame_id = 'depth_optical'
    msg.header.stamp.sec, msg.header.stamp.nanosec = 1788415697, 123456789
    msg.width, msg.height = 45, 37
    msg.encoding, msg.is_bigendian = encoding, int(bigendian)
    msg.step = msg.width * dtype.itemsize + padding
    msg.data = array('B', [173]) * (msg.step * msg.height)
    view = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.step, dtype.itemsize))
    view[:] = 1.4 if floating else 1400
    view[:, 23:] = 2.3 if floating else 2300
    view[:, 22] = 1.8 if floating else 1800
    return msg


@pytest.mark.parametrize('encoding', ['16UC1', 'mono16', '32FC1'])
@pytest.mark.parametrize('bigendian', [False, True])
@pytest.mark.parametrize('padding', [0, 1, 8])
def test_preserves_timestamp_encoding_endian_retained_pixels_and_padding(encoding, bigendian, padding):
    original = make_image(encoding, bigendian, padding)
    before = bytes(original.data)
    output, result = filter_depth_message(original, DepthEdgeFilterConfig())
    assert output is not original
    assert output.header == original.header
    for name in ('step', 'encoding', 'is_bigendian', 'height', 'width'):
        assert getattr(output, name) == getattr(original, name)
    assert bytes(original.data) == before
    input_depth, _ = depth_image_view(original)
    filtered_depth, _ = depth_image_view(output)
    assert np.all(filtered_depth[result.rejected] == 0)
    np.testing.assert_array_equal(filtered_depth[~result.rejected], input_depth[~result.rejected])
    if padding:
        raw = np.asarray(output.data).reshape(output.height, output.step)
        assert np.all(raw[:, -padding:] == 173)


def test_valid_noop_republishes_without_copy():
    msg = make_image('16UC1')
    pixels, _ = depth_image_view(msg)
    pixels[:] = 1500
    output, result = filter_depth_message(msg, DepthEdgeFilterConfig())
    assert output is msg
    assert not result.rejected.any()


def test_nonfinite_depth_is_not_filled_or_used_as_background():
    msg = make_image('32FC1')
    pixels, _ = depth_image_view(msg)
    pixels[:, 23:] = np.nan
    output, result = filter_depth_message(msg, DepthEdgeFilterConfig())
    assert not result.rejected.any()
    assert bytes(output.data) == bytes(msg.data)


@pytest.mark.parametrize('fault', ['encoding', 'step', 'truncated', 'empty'])
def test_malformed_input_raises_explicit_error(fault):
    msg = make_image('16UC1')
    if fault == 'encoding':
        msg.encoding = 'rgb8'
    elif fault == 'step':
        msg.step = 1
    elif fault == 'truncated':
        msg.data = msg.data[:-1]
    else:
        msg.width = 0
    with pytest.raises(ValueError):
        filter_depth_message(msg, DepthEdgeFilterConfig())


def launch_module():
    path = Path(__file__).resolve().parents[1] / 'launch/d435_nvblox_dynamic_spheres.launch.py'
    spec = importlib.util.spec_from_file_location('edge_filter_launch_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('edge,robot', [(False, False), (True, False), (False, True), (True, True)])
def test_all_filter_toggle_routings(edge, robot):
    validated, nvblox = launch_module()._depth_processing_topics('/raw', edge, robot, '/robot')
    assert validated == ('/rmp_camera/edge_filtered_depth/image_rect_raw' if edge else '/raw')
    assert nvblox == ('/robot' if robot else validated)


def test_rejects_topic_feedback_loops():
    module = launch_module()
    with pytest.raises(RuntimeError):
        module._depth_processing_topics('/rmp_camera/edge_filtered_depth/image_rect_raw', True, False, '/r')
    with pytest.raises(RuntimeError):
        module._depth_processing_topics('/raw', False, True, '/raw')
