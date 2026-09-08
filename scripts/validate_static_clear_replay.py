"""Headless raw-input replay with fixed bag-time metrics; never starts hardware."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import tempfile

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path, help='New output directory (must not exist)')
    parser.add_argument('--fast-empty', choices=['true', 'false'], default='true')
    parser.add_argument('--empty-check-rate', type=float, default=5.0)
    parser.add_argument('--human-refit', choices=['true', 'false'], default='true')
    parser.add_argument('--depth-edge-filter', choices=['true', 'false'], default='false',
                        help='Explicit A/B toggle; historical replay baseline is filter OFF')
    parser.add_argument('--capture-geometry', action='store_true')
    parser.add_argument('--capture-raw-dynamic', action='store_true',
                        help='Save diagnostic raw point arrays after replay; no pipeline parameter changes')
    parser.add_argument('--capture-human-points', action='store_true')
    parser.add_argument('--capture-static-support', action='store_true')
    parser.add_argument('--human-occlusion-hold', choices=['true', 'false'], default='false')
    parser.add_argument('--human-low-latency', choices=['true', 'false'], default='false')
    parser.add_argument('--async-refinement', choices=['true', 'false'], default='false')
    parser.add_argument('--latest-input', choices=['true', 'false'], default='false')
    parser.add_argument('--component-tracking', choices=['true', 'false'], default='false')
    parser.add_argument('--coverage-guard', choices=['true', 'false'], default='false')
    parser.add_argument('--local-width-cover', choices=['true', 'false'], default='false')
    parser.add_argument('--integrated-cover', choices=['true', 'false'], default='false')
    parser.add_argument('--observation-guard', choices=['off','audit','observed'], default='off')
    parser.add_argument('--domain-id', type=int, default=77)
    args = parser.parse_args()
    # Never compare two concurrently running pipelines on the same DDS domain.
    domain_lock = open(Path(tempfile.gettempdir()) /
                       f'rmp_camera_replay_domain_{args.domain_id}.lock', 'a')
    try:
        fcntl.flock(domain_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f'Another validation replay is using ROS domain {args.domain_id}')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = yaml.safe_load((args.bag / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    t0 = metadata['starting_time']['nanoseconds_since_epoch'] * 1e-9
    duration = metadata['duration']['nanoseconds'] * 1e-9
    os.environ['ROS_DOMAIN_ID'] = str(args.domain_id)

    class Monitor(Node):
        def __init__(self):
            super().__init__('static_clear_validation_monitor')
            self.clock_sec = None
            self.records = []
            self.frames = set()
            self.raw_clouds = []
            self.raw_stamps = []
            self.human_clouds = []
            self.human_stamps = []
            self.static_support = []
            self.static_support_stamps = []
            self.create_subscription(Clock, '/clock', self.clock_cb, qos_profile_sensor_data)
            for label, topic in (
                ('static', '/rmp_camera/esdf_medial_sphere_cloud'),
                ('human', '/rmp_camera/human_sphere_cloud'),
                ('human_points', '/rmp_camera/human_points'),
                ('dynamic_points', '/nvblox_node/dynamic_points'),
                ('dynamic', '/rmp_camera/dynamic_obstacle_sphere_cloud'),
                ('fusion', '/rmp_camera/combined_obstacle_sphere_cloud')):
                self.create_subscription(PointCloud2, topic,
                    lambda m, label=label: self.cloud(m, label), qos_profile_sensor_data)
            self.create_subscription(DiagnosticArray, '/rmp_camera/static_sphere_result_status',
                                     self.status, 10)
            self.create_subscription(DiagnosticArray, '/rmp_camera/depth_edge_filter/status',
                                     lambda m: self.status(m, 'depth_edge_filter'), 10)
            for label, topic in (
                ('dynamic_status', '/rmp_camera/dynamic_obstacle_sphere_cloud/status'),
                ('human_status', '/rmp_camera/human_sphere_cloud/status')):
                self.create_subscription(DiagnosticArray, topic,
                    lambda m, label=label: self.status(m, label), 10)
            self.create_subscription(DiagnosticArray, '/rmp_camera/combined_obstacle_sphere_cloud/status',
                                     lambda m: self.status(m, 'fusion_status'), 10)
            self.create_subscription(Image, '/camera0/camera/color/image_raw',
                                     self.rgb, qos_profile_sensor_data)
            self.create_subscription(Image, '/camera0/segmentation/people_mask',
                                     self.mask_timing, qos_profile_sensor_data)
            if args.capture_static_support:
                self.create_subscription(PointCloud2, '/rmp_camera/static_sphere_support',
                                         self.support, qos_profile_sensor_data)

        def support(self, message):
            row = self.base_row(message, 'static_support')
            if row is None:
                return
            points = np.asarray(point_cloud2.read_points_list(message,
                field_names=['x', 'y', 'z', 'sphere_index']), dtype=float).reshape(-1, 4)
            self.static_support.append(points)
            self.static_support_stamps.append(self.stamp(message))

        @staticmethod
        def stamp(message):
            return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

        def clock_cb(self, message):
            self.clock_sec = message.clock.sec + message.clock.nanosec * 1e-9

        def base_row(self, message, label):
            if self.clock_sec is None:
                return None
            stamp = self.stamp(message)
            if not t0 - .1 <= stamp <= t0 + duration + 1:
                return None
            return dict(t=round(self.clock_sec - t0, 6),
                        stamp=round(stamp - t0, 6), label=label)

        def cloud(self, message, label):
            row = self.base_row(message, label)
            if row is None:
                return
            row['count'] = message.width * message.height
            if label == 'fusion':
                types = [p.source_type for p in point_cloud2.read_points_list(
                    message, field_names=['source_type'])]
                row.update(static=types.count(0), dynamic=types.count(1), human=types.count(2))
            if args.capture_geometry and label in ('static', 'dynamic', 'human', 'fusion'):
                fields = ['x', 'y', 'z', 'raw_radius', 'output_radius']
                if label == 'fusion':
                    fields.append('source_type')
                row['geometry'] = [[float(v) for v in p] for p in point_cloud2.read_points_list(
                    message, field_names=fields)]
            if args.capture_raw_dynamic and label == 'dynamic_points':
                self.raw_clouds.append(point_cloud2.read_points_numpy(
                    message, field_names=['x', 'y', 'z'], skip_nans=True).copy())
                self.raw_stamps.append(self.stamp(message))
            if args.capture_human_points and label == 'human_points':
                self.human_clouds.append(point_cloud2.read_points_numpy(
                    message, field_names=['x', 'y', 'z'], skip_nans=True).copy())
                self.human_stamps.append(self.stamp(message))
            self.records.append(row)

        def mask_timing(self, message):
            row = self.base_row(message, 'human_mask')
            if row is not None:
                self.records.append(row)

        def status(self, message, label='status'):
            row = self.base_row(message, label)
            if row is not None and message.status:
                row.update(state=message.status[0].message,
                           details={v.key: v.value for v in message.status[0].values})
                self.records.append(row)

        def rgb(self, message):
            second = int(self.stamp(message) - t0)
            if second in self.frames or not 24 <= second <= 29:
                return
            if message.encoding not in ('rgb8', 'bgr8'):
                return
            frame = np.ndarray((message.height, message.width, 3), dtype=np.uint8,
                buffer=message.data, strides=(message.step, 3, 1))
            if message.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(args.output / f'rgb_{second}.png'), frame)
            self.frames.add(second)

    def stop(process):
        if process is None or process.poll() is not None:
            return
        for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
            try:
                if sig == signal.SIGINT:
                    # ros2 launch forwards SIGINT to its children itself. A
                    # group signal would hit each child twice during cleanup.
                    process.send_signal(sig)
                else:
                    os.killpg(process.pid, sig)
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                return

    launch = player = None
    rclpy.init()
    node = Monitor()
    try:
        with (args.output / 'launch.txt').open('w') as log, (args.output / 'bag.txt').open('w') as bag_log:
            command = ['ros2', 'launch', 'rmp_camera', 'd435_nvblox_dynamic_spheres.launch.py',
                'source:=camera', 'use_sim_time:=true', 'run_realsense:=false',
                'run_rb10_joint_state_source:=false', 'run_robot_collision_spheres:=false',
                'run_robot_self_filter:=false', 'run_robot_obstacle_filters:=true',
                'run_people_segmentation:=true', 'run_human_obstacle_spheres:=true',
                'run_esdf_medial_spheres:=true', 'run_dynamic_obstacle_spheres:=true',
                'run_obstacle_sphere_fusion:=true', 'esdf_trim_z_min_cm:=10',
                'run_rviz:=false', 'record_debug_data:=false', 'run_human_static_filter:=true',
                'static_fast_empty_clear:=' + args.fast_empty,
                'human_static_refit_enabled:=' + args.human_refit,
                'human_static_occlusion_hold_enabled:=' + args.human_occlusion_hold,
                'run_depth_edge_filter:=' + args.depth_edge_filter,
                'human_low_latency:=' + args.human_low_latency,
                'async_sphere_refinement:=' + args.async_refinement,
                'latest_sphere_input:=' + args.latest_input,
                'component_sphere_tracking:=' + args.component_tracking,
                'fusion_coverage_guard:=' + args.coverage_guard,
                'local_width_sphere_cover:=' + args.local_width_cover,
                'integrated_sphere_cover:=' + args.integrated_cover,
                'sphere_observation_guard:=' + args.observation_guard,
                'static_empty_check_rate_hz:=' + str(args.empty_check_rate)]
            launch = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            started = time.monotonic()
            while time.monotonic() - started < 5:
                rclpy.spin_once(node, timeout_sec=.1)
            player = subprocess.Popen(['ros2', 'bag', 'play', str(args.bag), '--clock',
                '--rate', '1.0', '--topics', '/camera0/realsense_splitter_node/output/depth',
                '/camera0/camera/depth/camera_info', '/camera0/camera/color/image_raw',
                '/camera0/camera/color/camera_info', '/rmp_camera/robot_collision_sphere_markers',
                '/tf_static'], stdout=bag_log, stderr=subprocess.STDOUT, start_new_session=True)
            ended = None
            while time.monotonic() - started < duration + 15:
                rclpy.spin_once(node, timeout_sec=.1)
                if player.poll() is not None:
                    ended = ended or time.monotonic()
                    if time.monotonic() - ended > 1:
                        break
            (args.output / 'records.json').write_text(json.dumps(node.records))
            if args.capture_raw_dynamic:
                offsets = np.cumsum([0] + [len(p) for p in node.raw_clouds])
                points = (np.concatenate(node.raw_clouds) if node.raw_clouds
                          else np.empty((0, 3), dtype=np.float32))
                np.savez_compressed(args.output / 'dynamic_points.npz',
                    stamps=np.asarray(node.raw_stamps), offsets=offsets, points=points)
            if args.capture_human_points:
                offsets = np.cumsum([0] + [len(p) for p in node.human_clouds])
                points = (np.concatenate(node.human_clouds) if node.human_clouds
                          else np.empty((0, 3), dtype=np.float32))
                np.savez_compressed(args.output / 'human_points.npz',
                    stamps=np.asarray(node.human_stamps), offsets=offsets, points=points)
            if args.capture_static_support:
                offsets = np.cumsum([0] + [len(p) for p in node.static_support])
                points = (np.concatenate(node.static_support) if node.static_support
                          else np.empty((0, 4), dtype=float))
                np.savez_compressed(args.output / 'static_support.npz',
                    stamps=np.asarray(node.static_support_stamps), offsets=offsets, points=points)
            result = {'bag_start': t0, 'fast_empty': args.fast_empty,
                      'human_low_latency': args.human_low_latency,
                      'async_refinement': args.async_refinement,
                      'latest_input': args.latest_input,
                      'component_tracking': args.component_tracking,
                      'coverage_guard': args.coverage_guard,
                      'local_width_cover': args.local_width_cover,
                      'integrated_cover': args.integrated_cover,
                      'observation_guard': args.observation_guard,
                      'depth_edge_filter': args.depth_edge_filter,
                      'human_refit': args.human_refit,
                      'human_occlusion_hold': args.human_occlusion_hold,
                      'empty_check_rate': args.empty_check_rate}
            for label in ('human_points', 'human', 'static', 'dynamic_points', 'dynamic', 'fusion'):
                rows = [r for r in node.records if r['label'] == label]
                occupied_indices = [i for i, r in enumerate(rows) if r['count']]
                last_index = occupied_indices[-1] if occupied_indices else None
                last = rows[last_index]['t'] if last_index is not None else None
                cleared = ([r for r in rows[last_index + 1:] if not r['count']]
                           if last_index is not None else [])
                result[label] = dict(messages=len(rows), last_nonempty=last,
                                     final_clear=cleared[0]['t'] if cleared else None)
            (args.output / 'summary.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result, indent=2), flush=True)
            stop(player)
            stop(launch)
    finally:
        stop(player)
        stop(launch)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
