#!/usr/bin/env python3
"""Far-camera ethernet receiver node.

Accepts TCP connections from the Raspberry Pi running the 4 far ESP32 cameras.
Each camera stream is a persistent connection sending binary-framed messages:

    [4B magic 'OCTF'][4B payload_length][2B json_length][json_bytes][jpeg_bytes]

JSON header per frame:
    {"cam": "far_front", "ts": 1234567.89,
     "tags": [{"id": 1, "dist": 2.5, "angle_deg": 15.3}, ...]}

Published topics (per camera):
    perception/camera/far/front/frame  — sensor_msgs/Image (bgr8)
    perception/camera/far/right/frame
    perception/camera/far/back/frame
    perception/camera/far/left/frame

Internal topic:
    localization/far_tags  — std_msgs/String (JSON, one message per frame with detections)

Parameters:
    port        (int)   — TCP listen port (default 5010)
    frame_id_prefix (str) — TF frame prefix (default 'far_')
"""

import json
import socket
import struct
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

MAGIC = b'OCTF'

_CAM_TOPICS = {
    'far_front': 'perception/camera/far/front/frame',
    'far_right':  'perception/camera/far/right/frame',
    'far_back':   'perception/camera/far/back/frame',
    'far_left':   'perception/camera/far/left/frame',
}


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('connection closed')
        buf += chunk
    return bytes(buf)


class FarCameraReceiverNode(Node):

    def __init__(self):
        super().__init__('far_camera_receiver_node')
        self.declare_parameter('port', 5010)
        self.declare_parameter('frame_id_prefix', 'far_')

        port   = self.get_parameter('port').value
        prefix = self.get_parameter('frame_id_prefix').value

        self._prefix = prefix
        self.bridge  = CvBridge()

        self._img_pubs = {
            cam: self.create_publisher(Image, topic, 10)
            for cam, topic in _CAM_TOPICS.items()
        }
        self._tag_pub = self.create_publisher(String, 'localization/far_tags', 10)

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('', port))
        self._server.listen(8)
        self.get_logger().info(f'Far camera receiver listening on TCP port {port}')

        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while rclpy.ok():
            try:
                conn, addr = self._server.accept()
                self.get_logger().info(f'Pi connected from {addr[0]}:{addr[1]}')
                threading.Thread(
                    target=self._handle_conn, args=(conn, addr), daemon=True
                ).start()
            except Exception as e:
                self.get_logger().error(f'Accept error: {e}')

    def _handle_conn(self, conn: socket.socket, addr):
        try:
            with conn:
                while rclpy.ok():
                    # 4B magic + 4B payload length
                    hdr = _recv_exact(conn, 8)
                    if hdr[:4] != MAGIC:
                        self.get_logger().warn(
                            f'Bad magic from {addr[0]} — dropping connection'
                        )
                        break

                    payload_len = struct.unpack('>I', hdr[4:])[0]
                    payload     = _recv_exact(conn, payload_len)

                    # 2B json length, then json, then jpeg
                    json_len = struct.unpack('>H', payload[:2])[0]
                    meta     = json.loads(payload[2:2 + json_len])
                    jpeg     = payload[2 + json_len:]

                    cam  = meta.get('cam', '')
                    tags = meta.get('tags', [])
                    ts   = meta.get('ts', 0.0)

                    # Publish image
                    if jpeg and cam in self._img_pubs:
                        arr = np.frombuffer(jpeg, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            ros_img = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                            ros_img.header.stamp    = self.get_clock().now().to_msg()
                            ros_img.header.frame_id = f'{self._prefix}{cam}_frame'
                            self._img_pubs[cam].publish(ros_img)

                    # Publish tag detections
                    if tags:
                        tag_msg = String()
                        tag_msg.data = json.dumps({'cam': cam, 'ts': ts, 'tags': tags})
                        self._tag_pub.publish(tag_msg)

        except Exception as e:
            self.get_logger().warn(f'Connection from {addr[0]} closed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FarCameraReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
