#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import socket
import struct
import threading
from octane_msgs.msg import IDImage


class USBBridgeCameraNode(Node):

    def __init__(self):
        super().__init__('usb_bridge_camera_node')

        # Declare parameters for configurable topics and connection
        self.declare_parameter('bridge_host', 'host.docker.internal')
        self.declare_parameter('bridge_port', 5555)
        self.declare_parameter(
            'image_topic', '/perception/camera/far/image_raw')
        self.declare_parameter('camera_info_topic', 'camera/camera_info')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('encoding', 'bgr8')

        # Get parameters
        self.bridge_host = self.get_parameter('bridge_host').value
        self.bridge_port = self.get_parameter('bridge_port').value
        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.encoding = self.get_parameter('encoding').value

        # Create publishers
        self.image_pub = self.create_publisher(IDImage, image_topic, 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, camera_info_topic, 10)

        # Socket connection
        self.socket = None
        self.running = True
        self.connected = False

        # Create camera info message
        self.camera_info = self.create_camera_info()

        # Start connection thread
        self.connection_thread = threading.Thread(target=self.connection_loop)
        self.connection_thread.daemon = True
        self.connection_thread.start()

        self.get_logger().info(
            f'USB Bridge Camera node started (port={self.bridge_port}, topic={image_topic})')

    def create_camera_info(self):
        camera_info = CameraInfo()
        camera_info.width = self.width
        camera_info.height = self.height

        # Typical focal lengths (adjust as needed)
        fx = fy = 570.0
        cx = self.width / 2.0
        cy = self.height / 2.0

        camera_info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.distortion_model = "plumb_bob"

        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        camera_info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        return camera_info

    def connect_to_bridge(self):
        try:
            self.get_logger().info(
                f'Connecting to USB bridge at {self.bridge_host}:{self.bridge_port}')
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
            # Read header: data_length (4 bytes)
            header = self.socket.recv(4)
            if len(header) < 4:
                return None

            data_length = struct.unpack('!I', header)[0]

            # Read frame data
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
        # Create Image message
        imgmsg = Image()
        imgmsg.header.stamp = self.get_clock().now().to_msg()
        imgmsg.header.frame_id = self.frame_id
        imgmsg.height = self.height
        imgmsg.width = self.width
        imgmsg.encoding = self.encoding
        imgmsg.step = len(data) // self.height
        imgmsg.data = list(data)
        msg = IDImage()
        msg.img = imgmsg
        msg.camera_id = 'camera'

        # Publish image and camera info
        self.image_pub.publish(msg)
        self.camera_info.header.stamp = imgmsg.header.stamp
        self.camera_info.header.frame_id = imgmsg.header.frame_id
        self.camera_info_pub.publish(self.camera_info)

        self.get_logger().info('Published frame', once=True)

    def connection_loop(self):
        while self.running:
            if not self.connected:
                if self.connect_to_bridge():
                    # Connected, start receiving
                    while self.running and self.connected:
                        data = self.receive_frame()
                        if data:
                            self.publish_frame(data)
                        else:
                            # Connection lost
                            self.connected = False
                            self.get_logger().warn('Connection lost, reconnecting...')
                            if self.socket:
                                self.socket.close()
                            break
                else:
                    # Failed to connect, retry after delay
                    import time
                    time.sleep(2)
            else:
                import time
                time.sleep(0.1)

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
