"""Current, registered depth evidence for local-sphere candidate validation.

This never creates/deletes points or writes map free space. Invalid/occluded
rays are NOT evidence of free space. The default use is an audit comparison.
"""
from collections import deque
from dataclasses import dataclass, field
from time import monotonic
import numpy as np


@dataclass
class SphereDepthEvidence:
    depth: np.ndarray
    intrinsics: tuple
    camera_from_target: np.ndarray
    free_margin: float = .04
    stats: dict = field(default_factory=dict)

    def __post_init__(self):
        self.depth = np.asarray(self.depth)
        self.camera_from_target = np.asarray(self.camera_from_target)
        if (self.depth.ndim != 2 or self.camera_from_target.shape != (4,4)
                or not np.isfinite(self.camera_from_target).all()
                or len(self.intrinsics) != 4 or not np.isfinite(self.intrinsics).all()
                or min(self.intrinsics[:2]) <= 0
                or not np.isfinite(self.free_margin) or self.free_margin < 0
                or not np.allclose(self.camera_from_target[3], [0, 0, 0, 1])):
            raise ValueError('invalid depth observation geometry')
        self.target_from_camera = np.linalg.inv(self.camera_from_target)

    def validate(self, xyz, support, deadline):
        """None means insufficient evidence; False means reliable contradiction.

        Bounded voxel samples have to project through valid, locally coherent
        depth. An alternative to volumetric density requires their ray endpoints
        to belong predominantly to THIS component, with bounded occlusion.
        """
        self.stats['ray_candidates'] = self.stats.get('ray_candidates',0)+1
        if monotonic() >= deadline or len(xyz) < 6:
            return None
        camera = xyz @ self.camera_from_target[:3,:3].T + self.camera_from_target[:3,3]
        fx,fy,cx,cy = self.intrinsics
        z = camera[:,2]
        front = z > .1
        safe_z = np.maximum(z,.1)
        u = np.floor(fx*camera[:,0]/safe_z+cx).astype(int)
        v = np.floor(fy*camera[:,1]/safe_z+cy).astype(int)
        h,w = self.depth.shape
        valid = front & (u>=0) & (v>=0) & (u<w-1) & (v<h-1)
        ii = np.flatnonzero(valid)
        if len(ii) < .9*len(xyz):
            return None
        uu,vv = u[ii],v[ii]
        neighborhood = np.stack((self.depth[vv,uu],self.depth[vv,uu+1],
                                 self.depth[vv+1,uu],self.depth[vv+1,uu+1]),axis=1)
        coherent = (np.isfinite(neighborhood).all(axis=1)
            & (neighborhood.min(axis=1)>.1) & (neighborhood.max(axis=1)<5.)
            & (np.ptp(neighborhood,axis=1)<=.08))
        ii,uu,vv,neighborhood = ii[coherent],uu[coherent],vv[coherent],neighborhood[coherent]
        if len(ii) < .9*len(xyz) or monotonic()>=deadline:
            return None
        depth = np.median(neighborhood,axis=1)
        free = z[ii] < neighborhood.min(axis=1)-self.free_margin
        occluded = z[ii] > neighborhood.max(axis=1)+self.free_margin
        endpoints = np.column_stack(((uu+.5-cx)*depth/fx,(vv+.5-cy)*depth/fy,depth))
        endpoints = endpoints @ self.target_from_camera[:3,:3].T + self.target_from_camera[:3,3]
        endpoint_support = support.support_at_points(endpoints)
        self.stats['ray_max_free_fraction'] = max(self.stats.get('ray_max_free_fraction',0.),float(free.mean()))
        # Even a very dense old cloud cannot authorize contradicting fresh rays.
        if free.mean() > support.max_empty + 1e-12:
            self.stats['ray_free_rejected'] = self.stats.get('ray_free_rejected',0)+1
            return False
        if (endpoint_support.mean()<.7 or occluded.mean()>.6
                or np.count_nonzero(support.support_at_points(xyz))<6
                or monotonic()>=deadline):
            return None
        self.stats['ray_supported'] = self.stats.get('ray_supported',0)+1
        return True


class SphereDepthBuffer:
    """Raw bounded image history; decode ONLY for a selected cloud observation."""
    def __init__(self,node,topic,info_topic):
        from sensor_msgs.msg import Image, CameraInfo
        from rclpy.qos import qos_profile_sensor_data
        self.node = node
        self.frames = deque(maxlen=8)
        self.info = None
        self.last_clock = None
        self.subscriptions = [node.create_subscription(Image,topic,self.add,qos_profile_sensor_data),
            node.create_subscription(CameraInfo,info_topic,self.set_info,qos_profile_sensor_data)]

    def set_info(self,message):
        self.info=message

    def add(self,message):
        now=self.node.get_clock().now().nanoseconds
        if self.last_clock is not None and now<self.last_clock:
            self.frames.clear()
        self.last_clock=now
        self.frames.append((message,monotonic()))

    def snapshot(self,source_ns):
        from rclpy.time import Time
        from rclpy.duration import Duration
        from tf2_ros import TransformException
        from rmp_camera.vision_geometry import transform_to_matrix
        info=self.info
        now_ns=self.node.get_clock().now().nanoseconds
        if self.last_clock is not None and now_ns<self.last_clock:
            self.frames.clear()
        self.last_clock=now_ns
        if info is None or not self.frames:
            return None
        candidates=[]
        for message,received in self.frames:
            ns=message.header.stamp.sec*10**9+message.header.stamp.nanosec
            if (abs(ns-source_ns)<=50_000_000 and -.05e9<=now_ns-ns<=.25e9
                    and monotonic()-received<=.25):
                candidates.append((abs(ns-source_ns),-ns,message))
        if not candidates:
            return None
        message=min(candidates,key=lambda x:x[:2])[2]
        if (message.header.frame_id!=info.header.frame_id or message.width!=info.width
                or message.height!=info.height or np.any(np.abs(info.d)>1e-9)):
            return None  # Unrectified input needs a distortion-aware projection.
        if message.encoding in ('16UC1','mono16'):
            dtype=np.dtype('>u2' if message.is_bigendian else '<u2');scale=.001
        elif message.encoding=='32FC1':
            dtype=np.dtype('>f4' if message.is_bigendian else '<f4');scale=1.
        else:
            return None
        if message.step<message.width*dtype.itemsize or len(message.data)<message.step*message.height:
            return None
        try:
            transform=self.node.tf_buffer.lookup_transform(message.header.frame_id,self.node.target_frame,
                Time.from_msg(message.header.stamp),timeout=Duration(seconds=0.))
            depth=np.ndarray((message.height,message.width),dtype=dtype,buffer=message.data,
                strides=(message.step,dtype.itemsize)).astype(np.float32)*scale
            return SphereDepthEvidence(depth,(info.k[0],info.k[4],info.k[2],info.k[5]),
                transform_to_matrix(transform.transform))
        except (TransformException,ValueError,np.linalg.LinAlgError):
            return None
