import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from octane_msgs.msg import CameraFrame, CameraParam
from cv_bridge import CvBridge
import cv2


class RGBCameraNode(Node):

    def __init__(self):
        super().__init__('rgb_camera_node')

        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('device_path', '')
        self.declare_parameter('serial', '')
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        # Parameter values
        device_path = self.get_parameter('device_path').value
        camera_id = device_path if device_path else self.get_parameter('camera_id').value
        self.serial = self.get_parameter('serial').value
        frame_rate = self.get_parameter('frame_rate').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value

        # Publisher — single bundled CameraFrame topic
        self.frame_pub = self.create_publisher(CameraFrame, 'camera/frame', 10)

        # CV bridge init
        self.bridge = CvBridge()

        # Open camera
        self.capture = cv2.VideoCapture(camera_id)
        if not self.capture.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_id}')
            return

        # Set camera properties
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Build static camera info + param (update these once calibrated)
        fx = fy = float(width)
        cx = width / 2.0
        cy = height / 2.0
        self.camera_info = self._build_camera_info(width, height, fx, fy, cx, cy)
        self.camera_param = self._build_camera_param(width, height, fx, fy, cx, cy)

        # Capture timer
        self.timer = self.create_timer(1.0 / frame_rate, self.capture_frame)
        self.get_logger().info(
            f'RGB Camera node started (id={camera_id}, serial={self.serial}, '
            f'{width}x{height} @ {frame_rate}Hz)'
        )

    def capture_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        stamp = self.get_clock().now().to_msg()

        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = 'camera_link'

        self.camera_info.header.stamp = stamp
        self.camera_info.header.frame_id = 'camera_link'

        msg = CameraFrame()
        msg.serial = self.serial
        msg.image = image_msg
        msg.info = self.camera_info
        msg.param = self.camera_param
        self.frame_pub.publish(msg)

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
        if self.capture.isOpened():
            self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RGBCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
