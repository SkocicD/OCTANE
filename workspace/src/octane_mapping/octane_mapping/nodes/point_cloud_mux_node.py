#!/usr/bin/env python3
"""Point cloud mux — merges all per-camera point clouds into one combined cloud.

Subscribes to mapping/<cam>/points for every camera defined in cameras.yaml,
transforms each cloud into base_link, concatenates the point buffers, and
publishes the result to /mapping/point_cloud/combined at a configurable rate.

Intended as input to a neural network — no GUI or network streaming here.
"""

import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException, TransformListener


def _quat_to_rotation(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _transform_stamped_to_matrix(tf_stamped) -> np.ndarray:
    t = tf_stamped.transform.translation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_to_rotation(tf_stamped.transform.rotation)
    mat[:3, 3]  = [t.x, t.y, t.z]
    return mat


def _apply_matrix(xyz: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Apply a 4×4 transform matrix to (N,3) xyz array, return (N,3)."""
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    pts  = np.hstack([xyz, ones]).T        # (4, N)
    return (mat @ pts).T[:, :3].astype(np.float32)


def _cloud_to_xyz_rgb(cloud: PointCloud2):
    """Return (N,3) float32 xyz and (N,) uint32 rgb from a PointCloud2."""
    offsets = {f.name: f.offset for f in cloud.fields}
    step    = cloud.point_step
    raw     = np.frombuffer(bytes(cloud.data), dtype=np.uint8)
    n       = cloud.width * cloud.height
    raw     = raw[:n * step].reshape(n, step)

    xyz = np.stack([
        np.frombuffer(raw[:, offsets['x']:offsets['x']+4].tobytes(), dtype=np.float32),
        np.frombuffer(raw[:, offsets['y']:offsets['y']+4].tobytes(), dtype=np.float32),
        np.frombuffer(raw[:, offsets['z']:offsets['z']+4].tobytes(), dtype=np.float32),
    ], axis=1)

    rgb_off = offsets.get('rgb', offsets.get('rgba', None))
    if rgb_off is not None:
        rgb = np.frombuffer(raw[:, rgb_off:rgb_off+4].tobytes(), dtype=np.uint32)
    else:
        rgb = np.zeros(n, dtype=np.uint32)

    # Filter NaN/Inf points
    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], rgb[valid]


def _build_cloud(xyz: np.ndarray, rgb: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Pack (N,3) xyz and (N,) rgb into an unorganised PointCloud2."""
    n = xyz.shape[0]

    fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    # Interleave xyz + rgb into 16-byte records
    rgb_f = rgb.view(np.float32)
    data  = np.zeros((n, 4), dtype=np.float32)
    data[:, :3] = xyz
    data[:, 3]  = rgb_f
    raw   = data.tobytes()

    msg              = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height       = 1
    msg.width        = n
    msg.fields       = fields
    msg.is_bigendian  = False
    msg.point_step   = 16
    msg.row_step     = 16 * n
    msg.data         = list(raw)
    msg.is_dense     = True
    return msg


class PointCloudMuxNode(Node):

    TARGET_FRAME = 'base_link'

    def __init__(self):
        super().__init__('point_cloud_mux_node')

        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('config_file',  '')

        rate        = self.get_parameter('publish_rate').value
        config_file = self.get_parameter('config_file').value

        if not config_file:
            config_file = os.path.join(
                get_package_share_directory('octane'), 'config', 'cameras.yaml'
            )
        with open(config_file) as f:
            cfg = yaml.safe_load(f)

        self._cam_names: list = list(cfg['cameras'].keys())
        self._latest:   dict  = {name: None for name in self._cam_names}
        self._tf_cache: dict  = {}

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        for cam in self._cam_names:
            self.create_subscription(
                PointCloud2,
                f'mapping/{cam}/points',
                lambda msg, c=cam: self._cloud_cb(c, msg),
                qos,
            )

        self._pub = self.create_publisher(
            PointCloud2, '/mapping/point_cloud/combined', 10
        )
        self.create_timer(1.0 / rate, self._merge_and_publish)

        self.get_logger().info(
            f'PointCloudMux ready  cameras={self._cam_names}  '
            f'rate={rate}Hz  target_frame={self.TARGET_FRAME}'
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _cloud_cb(self, cam_name: str, msg: PointCloud2):
        self._latest[cam_name] = msg

    # ── merge ─────────────────────────────────────────────────────────────────

    def _merge_and_publish(self):
        all_xyz = []
        all_rgb = []
        stamp   = self.get_clock().now().to_msg()

        for cam in self._cam_names:
            cloud = self._latest[cam]
            if cloud is None or cloud.width == 0:
                continue

            mat = self._get_transform(cloud.header.frame_id)
            if mat is None:
                continue

            xyz, rgb = _cloud_to_xyz_rgb(cloud)
            if xyz.shape[0] == 0:
                continue

            all_xyz.append(_apply_matrix(xyz, mat))
            all_rgb.append(rgb)

        if not all_xyz:
            return

        merged_xyz = np.vstack(all_xyz)
        merged_rgb = np.concatenate(all_rgb)
        self._pub.publish(_build_cloud(merged_xyz, merged_rgb, self.TARGET_FRAME, stamp))

    # ── TF ────────────────────────────────────────────────────────────────────

    def _get_transform(self, source_frame: str):
        if source_frame in self._tf_cache:
            return self._tf_cache[source_frame]
        try:
            tf = self._tf_buffer.lookup_transform(
                self.TARGET_FRAME, source_frame, rclpy.time.Time()
            )
            mat = _transform_stamped_to_matrix(tf)
            # Cache static transforms permanently
            self._tf_cache[source_frame] = mat
            return mat
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
