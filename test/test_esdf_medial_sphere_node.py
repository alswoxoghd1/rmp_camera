from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from visualization_msgs.msg import Marker

from rmp_camera.esdf_medial_sphere_node import EsdfMedialSphereNode


def test_static_markers_clear_stale_state_and_expire():
    published = []
    fake_node = SimpleNamespace(
        target_frame="base_link",
        clear_markers_on_next_publish=True,
        previous_marker_keys=set(),
        marker_lifetime_s=3.0,
        marker_pub=SimpleNamespace(publish=published.append),
        _header=lambda stamp: Header(stamp=stamp, frame_id="base_link"),
        _component_color=lambda _component_id: (0.1, 0.8, 0.2),
    )
    sphere = SimpleNamespace(
        component_id=4,
        center=np.asarray((0.1, 0.2, 0.3)),
        output_radius=0.25,
    )
    stamp = Time(sec=1, nanosec=0)

    EsdfMedialSphereNode.publish_markers(fake_node, stamp, [sphere])

    first = published[-1].markers
    assert first[0].action == Marker.DELETEALL
    assert first[1].action == Marker.ADD
    assert first[1].ns == "esdf_medial_component_4"
    assert first[1].lifetime.sec == 3

    EsdfMedialSphereNode.publish_markers(fake_node, stamp, [])

    second = published[-1].markers
    assert len(second) == 1
    assert second[0].action == Marker.DELETE
    assert second[0].ns == "esdf_medial_component_4"
