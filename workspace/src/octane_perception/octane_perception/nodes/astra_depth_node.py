import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Publish depth and image data from Astra Pro
class AstraDepthNode(Node):

    def __init__(self):
        super().__init__('astra_depth_node')
        self.get_logger().info('Depth Camera node started')

        # Depth and Color Publishers
        self.depth_publisher = self.create_publisher(Image, 'depth_camera/depth', 10)
        self.color_publisher = self.create_publisher(Image, 'depth_camera/color', 10)

        # Subscribe to Orbbec Driver Topics (raw sensor data)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_callback,
            10
        )
        
    # Callbacks for processing incoming messages
    def depth_callback(self, msg):
        # Process and republish depth images
        self.depth_publisher.publish(msg)
        self.get_logger().info('Published depth image', once=True)

    def color_callback(self, msg):
        # Process and republish color images
        self.color_publisher.publish(msg)
        self.get_logger().info('Published color image', once=True)

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