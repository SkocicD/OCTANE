import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2


class RGBCameraNode(Node):

    def __init__(self):
        super().__init__('rgb_camera_node')

        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('device_path', '')
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        # Parameter values
        device_path = self.get_parameter('device_path').value
        camera_id = device_path if device_path else self.get_parameter('camera_id').value
        frame_rate = self.get_parameter('frame_rate').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value

        # Create publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

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

        # Create camera info message
        self.camera_info = self.create_camera_info(width, height)

        # Create timer to capture frames
        timer_period = 1.0 / frame_rate
        self.timer = self.create_timer(timer_period, self.capture_frame)

        self.get_logger().info(f'RGB Camera node started (camera_id={camera_id}, {width}x{height} @ {frame_rate}Hz)')

    def capture_frame(self):
        ret, frame = self.capture.read()

        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'

        self.camera_info.header.stamp = msg.header.stamp
        self.camera_info.header.frame_id = msg.header.frame_id
        
        # Publish image and camera info
        self.image_pub.publish(msg)
        self.camera_info_pub.publish(self.camera_info)

    def create_camera_info(self, width, height):
        camera_info = CameraInfo()
        camera_info.width = width
        camera_info.height = height

        # Camera matrix (typical webcam approximation)
        fx = fy = width  # Focal length approximation
        cx = width / 2.0
        cy = height / 2.0

        camera_info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        # No distortion (simplification)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.distortion_model = 'plumb_bob'

        # Rectification matrix (identity)
        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        # Projection matrix
        camera_info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        return camera_info

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
