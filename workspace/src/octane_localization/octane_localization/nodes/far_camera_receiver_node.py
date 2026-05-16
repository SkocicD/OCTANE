#!/usr/bin/env python3
"""Far-camera ethernet receiver node.

Accepts one TCP connection from the Raspberry Pi running the 4 far ESP32
cameras. All 4 cameras share a single socket (protected by a lock on the Pi
side), so packets from different cameras arrive sequentially on one connection.

Wire format (little-endian, no outer framing — read fields in order):

    [4B LE uint32 : name_len  ]
    [name_len     : camera name  e.g. "esp32cam_front"]
    [4B LE uint32 : jpeg_len  ]
    [jpeg_len     : JPEG bytes]
    [4B LE uint32 : tag_count ]
    for each tag:
        [4B LE int32  : tag_id  ]
        [4B LE float32: distance (metres)]
        [4B LE float32: angle   (degrees, positive = left of boresight)]

Camera name → ROS topic mapping:
    esp32cam_front → perception/camera/far/front/frame
    esp32cam_right → perception/camera/far/right/frame
    esp32cam_back  → perception/camera/far/back/frame
    esp32cam_left  → perception/camera/far/left/frame

Published topics:
    perception/camera/far/{front,right,back,left}/frame  — sensor_msgs/Image (bgr8)
    localization/far_tags  — std_msgs/String (JSON for triangulator_node)

Parameters:
    port  (int) — TCP listen port (default 5051, must match Pi SERVER_PORT)
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

# Pi camera name → ROS topic suffix
_CAM_MAP = {
    'esp32cam_front': 'far_front',
    'esp32cam_right': 'far_right',
    'esp32cam_back':  'far_back',
    'esp32cam_left':  'far_left',
}

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


def _read_packet(sock: socket.socket):
    """Read one complete camera packet from the Pi. Returns (cam_name, jpeg, tags).

    tags: list of {'id': int, 'dist': float, 'angle_deg': float}
    Raises ConnectionError on disconnect, ValueError on bad data.
    """
    # Camera name
    name_len = struct.unpack('<I', _recv_exact(sock, 4))[0]
    if name_len == 0 or name_len > 64:
        raise ValueError(f'bad name_len {name_len}')
    cam_name = _recv_exact(sock, name_len).decode(errors='ignore')

    # JPEG
    jpeg_len = struct.unpack('<I', _recv_exact(sock, 4))[0]
    if jpeg_len == 0 or jpeg_len > 500_000:
        raise ValueError(f'bad jpeg_len {jpeg_len}')
    jpeg = _recv_exact(sock, jpeg_len)

    # Tags
    tag_count = struct.unpack('<I', _recv_exact(sock, 4))[0]
    if tag_count > 64:
        raise ValueError(f'bad tag_count {tag_count}')
    tags = []
    for _ in range(tag_count):
        raw = _recv_exact(sock, 12)  # int32 + float32 + float32
        tag_id, dist, angle = struct.unpack('<iff', raw)
        tags.append({'id': int(tag_id), 'dist': float(dist), 'angle_deg': float(angle)})

    return cam_name, jpeg, tags


class FarCameraReceiverNode(Node):

    def __init__(self):
        super().__init__('far_camera_receiver_node')
        self.declare_parameter('port', 5051)
        port = self.get_parameter('port').value

        self.bridge = CvBridge()

        self._img_pubs = {
            key: self.create_publisher(Image, topic, 10)
            for key, topic in _CAM_TOPICS.items()
        }
        self._tag_pub = self.create_publisher(String, 'localization/far_tags', 10)

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('', port))
        self._server.listen(4)
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
                    try:
                        pi_name, jpeg, tags = _read_packet(conn)
                    except ValueError as e:
                        self.get_logger().warn(f'Bad packet from {addr[0]}: {e}')
                        continue

                    cam_key = _CAM_MAP.get(pi_name)
                    if cam_key is None:
                        self.get_logger().warn(
                            f'Unknown camera name "{pi_name}" — add to _CAM_MAP',
                            throttle_duration_sec=5.0,
                        )
                        continue

                    stamp = self.get_clock().now().to_msg()

                    # Publish image
                    arr = np.frombuffer(jpeg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        ros_img = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
                        ros_img.header.stamp    = stamp
                        ros_img.header.frame_id = f'{cam_key}_frame'
                        self._img_pubs[cam_key].publish(ros_img)

                    # Publish tag detections for triangulator
                    if tags:
                        msg = String()
                        msg.data = json.dumps({
                            'cam':  cam_key,
                            'ts':   stamp.sec + stamp.nanosec * 1e-9,
                            'tags': tags,
                        })
                        self._tag_pub.publish(msg)

        except ConnectionError:
            self.get_logger().warn(f'Pi at {addr[0]} disconnected')
        except Exception as e:
            self.get_logger().error(f'Connection error from {addr[0]}: {e}')


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
