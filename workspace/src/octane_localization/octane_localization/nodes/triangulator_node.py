#!/usr/bin/env python3
"""AprilTag triangulator node.

Subscribes to tag detections from far_camera_receiver_node, triangulates
rover position from tag distances, and derives heading from per-camera
bearing angles.

Algorithm:
  1. Maintain latest {tag_id: (distance, camera, angle_rad, timestamp)}.
  2. For every pair of tags with fresh observations, run law-of-cosines
     triangulation (both geometric solutions).
  3. Pick the median (x, y) across all candidate solutions.
  4. For each observation, compute:
       bearing_world = atan2(tag_y - rover_y, tag_x - rover_x)
       rover_heading  = bearing_world - mount_angle - camera_angle
  5. Circular-mean all heading estimates.
  6. Publish geometry_msgs/Pose2D on localization/pose.

Subscribed topics:
    localization/far_tags  — std_msgs/String (JSON from far_camera_receiver_node)

Published topics:
    localization/pose      — geometry_msgs/Pose2D  (x metres, y metres, theta radians)

Parameters:
    config_file    (str)   — path to apriltags.yaml (defaults to package share/config)
    publish_rate   (float) — Hz (default 10.0)
    obs_timeout    (float) — seconds before a detection is considered stale (default 2.0)
    min_tags       (int)   — minimum distinct tags needed to publish (default 2)
"""

import json
import math
import os
import threading

import yaml
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from std_msgs.msg import String

_DEFAULT_TIMEOUT = 2.0


def _triangulate(p1, p2, dist_p1, dist_p2, flip=False):
    """Law-of-cosines triangulation. Returns (x, y) or raises ValueError.

    p1, p2   — world positions of the two anchor tags
    dist_p1  — distance from rover to p1
    dist_p2  — distance from rover to p2
    flip     — selects the second geometric solution
    """
    x1, y1 = p1
    x2, y2 = p2
    C = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if C == 0:
        raise ValueError('anchor points are identical')
    cos_alpha = (dist_p1 ** 2 + C ** 2 - dist_p2 ** 2) / (2 * dist_p1 * C)
    if not (-1.0 <= cos_alpha <= 1.0):
        raise ValueError('distances cannot form a triangle')
    alpha = math.acos(cos_alpha)
    base  = math.atan2(y2 - y1, x2 - x1)
    gamma = base + alpha if flip else base - alpha
    return x1 + dist_p1 * math.cos(gamma), y1 + dist_p1 * math.sin(gamma)


def _circular_mean(angles_rad: list) -> float:
    sx = sum(math.cos(a) for a in angles_rad)
    sy = sum(math.sin(a) for a in angles_rad)
    return math.atan2(sy, sx)


class TriangulatorNode(Node):

    def __init__(self):
        super().__init__('triangulator_node')

        self.declare_parameter('config_file',  '')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('obs_timeout',  5.0)
        self.declare_parameter('min_tags',     2)

        config_file  = self.get_parameter('config_file').value
        publish_rate = float(self.get_parameter('publish_rate').value)
        self._timeout  = float(self.get_parameter('obs_timeout').value)
        self._min_tags = int(self.get_parameter('min_tags').value)

        if not config_file:
            pkg = get_package_share_directory('octane')
            config_file = os.path.join(pkg, 'config', 'apriltags.yaml')

        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        self._tag_map: dict[int, tuple] = {
            t['id']: (float(t['x']), float(t['y']))
            for t in cfg['apriltag_map']
        }
        self._cam_mount: dict[str, float] = {
            cam: math.radians(float(info['mount_angle_deg']))
            for cam, info in cfg['cameras'].items()
        }
        self.get_logger().info(
            f'Loaded {len(self._tag_map)} tags, '
            f'{len(self._cam_mount)} cameras from {config_file}'
        )

        # {tag_id: (dist_m, cam_name, angle_rad, ros_timestamp_sec)}
        self._obs: dict[int, tuple] = {}
        self._lock = threading.Lock()

        self._pub = self.create_publisher(Pose2D, 'localization/pose', 10)
        self.create_subscription(String, 'localization/far_tags', self._tag_cb, 10)
        self.create_timer(1.0 / publish_rate, self._run)

    def _tag_cb(self, msg: String):
        data = json.loads(msg.data)
        cam  = data.get('cam', '')
        now  = self.get_clock().now().nanoseconds / 1e9  # use arrival time, not Pi ts
        with self._lock:
            for t in data.get('tags', []):
                tid = int(t['id'])
                if tid in self._tag_map:
                    self._obs[tid] = (
                        float(t['dist']),
                        cam,
                        math.radians(float(t['angle_deg'])),
                        now,
                    )

    def _run(self):
        now = self.get_clock().now().nanoseconds / 1e9
        with self._lock:
            obs = {
                tid: v for tid, v in self._obs.items()
                if now - v[3] < self._timeout
            }

        if len(obs) < self._min_tags:
            self.get_logger().warn(
                f'triangulator: only {len(obs)}/{self._min_tags} fresh tag obs — skipping',
                throttle_duration_sec=5.0,
            )
            return

        tag_ids = list(obs.keys())

        # Collect all triangulation candidates from every tag pair
        candidates: list[tuple] = []
        for i in range(len(tag_ids)):
            for j in range(i + 1, len(tag_ids)):
                id1, id2 = tag_ids[i], tag_ids[j]
                p1, p2   = self._tag_map[id1], self._tag_map[id2]
                d1, d2   = obs[id1][0], obs[id2][0]
                for flip in (False, True):
                    try:
                        candidates.append(_triangulate(p1, p2, d1, d2, flip))
                    except ValueError:
                        pass

        if not candidates:
            self.get_logger().warn(
                f'triangulator: all {len(tag_ids)} tag pairs failed triangle inequality — '
                f'check distances vs tag positions in apriltags.yaml',
                throttle_duration_sec=5.0,
            )
            return

        # Median position across all candidates (robust to one bad solution)
        xs = sorted(c[0] for c in candidates)
        ys = sorted(c[1] for c in candidates)
        x  = xs[len(xs) // 2]
        y  = ys[len(ys) // 2]

        # Heading: bearing from rover to each tag minus camera mount and measured angle
        headings = []
        for tid, (dist, cam, angle_rad, _) in obs.items():
            tx, ty   = self._tag_map[tid]
            bearing  = math.atan2(ty - y, tx - x)
            mount    = self._cam_mount.get(cam, 0.0)
            headings.append(bearing - mount - angle_rad)

        theta = _circular_mean(headings)

        pose       = Pose2D()
        pose.x     = x
        pose.y     = y
        pose.theta = theta
        self._pub.publish(pose)

        self.get_logger().info(
            f'pose ({x:.3f}, {y:.3f}) heading {math.degrees(theta):.1f} deg'
            f'  [{len(tag_ids)} tags  {len(candidates)} candidates]',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = TriangulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
