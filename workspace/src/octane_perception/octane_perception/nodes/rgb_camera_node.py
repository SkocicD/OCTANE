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
        self.declare_parameter('debug_images', False)

        # Parameter values
        device_path = self.get_parameter('device_path').value
        camera_id = device_path if device_path else self.get_parameter('camera_id').value
        self.serial = self.get_parameter('serial').value
        frame_rate = self.get_parameter('frame_rate').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        self.debug_images = self.get_parameter('debug_images').value

        # Publisher — single bundled CameraFrame topic
        self.frame_pub = self.create_publisher(CameraFrame, 'camera/frame', 10)
        # Debug image publisher — raw sensor_msgs/Image for RViz/rqt (opt-in)
        self.image_pub = self.create_publisher(Image, 'camera/image', 10) if self.debug_images else None

        # CV bridge init
        self.bridge = CvBridge()

        # Force V4L2 — Jetson OpenCV defaults to GStreamer which can't handle /dev paths
        self.capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            self.get_logger().error(f'Failed to open camera {camera_id}')
            return

        # MJPEG must be set before resolution — required for 5× USB2 cameras at 30fps.
        # YUY2 at 640×480×5 cameras saturates USB2 bandwidth; MJPEG keeps it ~5× lower.
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, float(frame_rate))

        # Log what the driver actually negotiated (may differ from requested)
        actual_w   = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        actual_fmt = ''.join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Build static camera info + param (update these once calibrated)
        fx = fy = float(actual_w)
        cx = actual_w / 2.0
        cy = actual_h / 2.0
        self.camera_info = self._build_camera_info(actual_w, actual_h, fx, fy, cx, cy)
        self.camera_param = self._build_camera_param(actual_w, actual_h, fx, fy, cx, cy)

        # Capture timer
        self.timer = self.create_timer(1.0 / frame_rate, self.capture_frame)
        self.get_logger().info(
            f'RGB Camera node started  id={camera_id}  serial={self.serial}  '
            f'{actual_w}x{actual_h} @ {actual_fps:.0f}Hz  fmt={actual_fmt}'
        )

    def capture_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        frame = cv2.rotate(frame, cv2.ROTATE_180)
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

        if self.image_pub:
            self.image_pub.publish(image_msg)

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
