#!/usr/bin/env python3
"""Splits an octane_msgs/CameraFrame into the standard topics nvblox expects.

For one input CameraFrame topic, this node publishes:
  <out_ns>/image            sensor_msgs/Image
  <out_ns>/camera_info      sensor_msgs/CameraInfo

It also broadcasts a static TF transform from base_link → <frame_id>
based on the camera's mounting offset (loaded from cameras.yaml).
"""

import math
import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from octane_msgs.msg import CameraFrame


def _euler_deg_to_quat(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy,  # w
    )


class CameraFrameSplitter(Node):

    def __init__(self):
        super().__init__('camera_frame_splitter')

        self.declare_parameter('camera_name', '')
        self.declare_parameter('input_topic', '')
        self.declare_parameter('output_namespace', '')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('parent_frame', 'base_link')
        self.declare_parameter('config_file', '')

        camera_name = self.get_parameter('camera_name').value
        input_topic = self.get_parameter('input_topic').value
        out_ns = self.get_parameter('output_namespace').value
        frame_id = self.get_parameter('frame_id').value or f'{camera_name}_frame'
        parent_frame = self.get_parameter('parent_frame').value
        config_file = self.get_parameter('config_file').value

        if not camera_name or not input_topic or not out_ns:
            raise RuntimeError('camera_name, input_topic, and output_namespace are required')

        # Load camera config to find this camera's mounting offset
        if not config_file:
            config_file = os.path.join(
                get_package_share_directory('octane'), 'config', 'cameras.yaml'
            )
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        cam_cfg = cfg['cameras'].get(camera_name)
        if cam_cfg is None:
            raise RuntimeError(f'camera "{camera_name}" not found in {config_file}')

        self.frame_id = frame_id

        # Publishers — standard topic names that nvblox subscribes to
        self.image_pub = self.create_publisher(Image, f'{out_ns}/image', 10)
        self.info_pub  = self.create_publisher(CameraInfo, f'{out_ns}/camera_info', 10)

        # Subscribe to the bundled CameraFrame topic
        self.create_subscription(CameraFrame, input_topic, self._frame_cb, 10)

        # Broadcast static TF for this camera's mounting offset
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        image_roll_deg = float(cam_cfg.get('image_roll_deg', 90))
        self._publish_static_tf(parent_frame, frame_id, cam_cfg['offset'], image_roll_deg)

        self.get_logger().info(
            f'Splitter started: {input_topic} → {out_ns}/{{image,camera_info}}  '
            f'(tf {parent_frame}→{frame_id})'
        )

    def _frame_cb(self, msg: CameraFrame):
        img = msg.image
        img.header.frame_id = self.frame_id
        if img.encoding == 'bgr8':
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, 3)
            img.data = arr[:, :, ::-1].tobytes()
            img.encoding = 'rgb8'
        msg.info.header = img.header
        self.image_pub.publish(img)
        self.info_pub.publish(msg.info)

    def _publish_static_tf(self, parent: str, child: str, offset: dict, image_roll_deg: float = 90.0):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(offset['pos']['x'])
        t.transform.translation.y = float(offset['pos']['y'])
        t.transform.translation.z = float(offset['pos']['z'])
        qx, qy, qz, qw = _euler_deg_to_quat(
            offset['rot']['x'], offset['rot']['y'], offset['rot']['z']
        )
        # Correct image roll caused by physical mounting rotation.
        # Per-camera value in cameras.yaml (image_roll_deg); defaults to +90°.
        roll_rad = math.radians(image_roll_deg) * 0.5
        cz, cw = math.sin(roll_rad), math.cos(roll_rad)
        qx, qy, qz, qw = (
            qx * cw + qy * cz,
            qy * cw - qx * cz,
            qw * cz + qz * cw,
            qw * cw - qz * cz,
        )
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CameraFrameSplitter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
