import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, CameraInfo

from octane_msgs.msg import CameraFrame, CameraParam, CameraOffset


class AstraDepthNode(Node):

    def __init__(self):
        super().__init__('astra_depth_node')

        self.depth_pub = self.create_publisher(CameraFrame, 'depth_camera/depth', 10)
        self.color_pub = self.create_publisher(CameraFrame, 'depth_camera/color', 10)

        depth_img_sub  = Subscriber(self, Image,      '/camera/depth/image_raw')
        depth_info_sub = Subscriber(self, CameraInfo, '/camera/depth/camera_info')
        color_img_sub  = Subscriber(self, Image,      '/camera/color/image_raw')
        color_info_sub = Subscriber(self, CameraInfo, '/camera/color/camera_info')

        self._depth_sync = ApproximateTimeSynchronizer(
            [depth_img_sub, depth_info_sub], queue_size=10, slop=0.1)
        self._depth_sync.registerCallback(self._depth_cb)

        self._color_sync = ApproximateTimeSynchronizer(
            [color_img_sub, color_info_sub], queue_size=10, slop=0.1)
        self._color_sync.registerCallback(self._color_cb)

        self.get_logger().info('AstraDepthNode started', once=True)

    def _build_frame(self, img: Image, info: CameraInfo) -> CameraFrame:
        frame = CameraFrame()
        frame.serial = 'orbbec_001'
        frame.image  = img
        frame.info   = info
        frame.param.fx     = info.k[0]
        frame.param.fy     = info.k[4]
        frame.param.cx     = info.k[2]
        frame.param.cy     = info.k[5]
        frame.param.width  = info.width
        frame.param.height = info.height
        if len(info.d) >= 5:
            frame.param.dist = list(info.d[:5])
        return frame

    def _depth_cb(self, img: Image, info: CameraInfo):
        self.depth_pub.publish(self._build_frame(img, info))

    def _color_cb(self, img: Image, info: CameraInfo):
        self.color_pub.publish(self._build_frame(img, info))


def main(args=None):
    rclpy.init(args=args)
    node = AstraDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
