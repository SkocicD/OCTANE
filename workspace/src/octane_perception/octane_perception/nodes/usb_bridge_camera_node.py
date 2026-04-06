#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from octane_msgs.msg import CameraFrame, CameraParam
import socket
import struct
import threading


class USBBridgeCameraNode(Node):

    def __init__(self):
        super().__init__('usb_bridge_camera_node')

        # Declare parameters
        self.declare_parameter('bridge_host', 'host.docker.internal')
        self.declare_parameter('bridge_port', 5555)
        self.declare_parameter('serial', '')
        self.declare_parameter('topic', 'camera/frame')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('encoding', 'bgr8')

        # Get parameters
        self.bridge_host = self.get_parameter('bridge_host').value
        self.bridge_port = self.get_parameter('bridge_port').value
        self.serial = self.get_parameter('serial').value
        topic = self.get_parameter('topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.encoding = self.get_parameter('encoding').value

        # Single bundled publisher
        self.frame_pub = self.create_publisher(CameraFrame, topic, 10)

        # Socket connection state
        self.socket = None
        self.running = True
        self.connected = False

        # Build static camera info + param (update once calibrated)
        fx = fy = 570.0
        cx = self.width / 2.0
        cy = self.height / 2.0
        self.camera_info = self._build_camera_info(self.width, self.height, fx, fy, cx, cy)
        self.camera_param = self._build_camera_param(self.width, self.height, fx, fy, cx, cy)

        # Start connection thread
        self.connection_thread = threading.Thread(target=self.connection_loop, daemon=True)
        self.connection_thread.start()

        self.get_logger().info(
            f'USB Bridge Camera node started (port={self.bridge_port}, serial={self.serial}, topic={topic})'
        )

    def connect_to_bridge(self):
        try:
            self.get_logger().info(f'Connecting to USB bridge at {self.bridge_host}:{self.bridge_port}')
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.bridge_host, self.bridge_port))
            self.connected = True
            self.get_logger().info('Connected to USB bridge')
            return True
        except Exception as e:
            self.get_logger().warn(f'Failed to connect: {e}')
            return False

    def receive_frame(self):
        try:
            header = self.socket.recv(4)
            if len(header) < 4:
                return None
            data_length = struct.unpack('!I', header)[0]

            data = b''
            while len(data) < data_length:
                chunk = self.socket.recv(min(4096, data_length - len(data)))
                if not chunk:
                    return None
                data += chunk
            return data
        except Exception as e:
            self.get_logger().error(f'Error receiving frame: {e}')
            return None

    def publish_frame(self, data):
        stamp = self.get_clock().now().to_msg()

        image_msg = Image()
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id
        image_msg.height = self.height
        image_msg.width = self.width
        image_msg.encoding = self.encoding
        image_msg.step = len(data) // self.height
        image_msg.data = list(data)

        self.camera_info.header.stamp = stamp
        self.camera_info.header.frame_id = self.frame_id

        msg = CameraFrame()
        msg.serial = self.serial
        msg.image = image_msg
        msg.info = self.camera_info
        msg.param = self.camera_param
        self.frame_pub.publish(msg)

        self.get_logger().info('Published frame', once=True)

    def connection_loop(self):
        import time
        while self.running:
            if not self.connected:
                if self.connect_to_bridge():
                    while self.running and self.connected:
                        data = self.receive_frame()
                        if data:
                            self.publish_frame(data)
                        else:
                            self.connected = False
                            self.get_logger().warn('Connection lost, reconnecting...')
                            if self.socket:
                                self.socket.close()
                            break
                else:
                    time.sleep(2)
            else:
                time.sleep(0.1)

    @staticmethod
    def _build_camera_info(w, h, fx, fy, cx, cy):
        info = CameraInfo()
        info.width = w
        info.height = h
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    @staticmethod
    def _build_camera_param(w, h, fx, fy, cx, cy):
        param = CameraParam()
        param.fx = fx
        param.fy = fy
        param.cx = cx
        param.cy = cy
        param.dist = [0.0, 0.0, 0.0, 0.0, 0.0]
        param.width = w
        param.height = h
        return param

    def destroy_node(self):
        self.running = False
        self.connected = False
        if self.socket:
            self.socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = USBBridgeCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
