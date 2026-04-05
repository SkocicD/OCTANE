import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from pupil_apriltags import Detector
from itertools import pairwise

from octane_msgs.msg import AprilTagDetection
from octane_msgs.msg import IDImage


class april_tag_detector(Node):

    TAG_SIZE = 0.165
    WHITE = 255

    def __init__(self):
        super().__init__('apriltag_detector')

        self.bridge = CvBridge()

        # Camera params example
        self.camera_params = [
            1395.1495929116268,
            1397.2438500585477,
            953.230777216243,
            533.6658327173194]

        self.detector = Detector(families='tag36h11')

        # Publishers
        self.image_pub = self.create_publisher(
            Image,
            '/localization/apriltag_outline',
            10
        )
        self.distance_pub = self.create_publisher(
            AprilTagDetection,
            '/localization/april_tag_distance',
            10
        )

        # Subscribers
        self.image_sub = self.create_subscription(
            IDImage,
            '/perception/camera/far/image_raw',
            self.detect_april_tags,
            10
        )

    def detect_april_tags(self, msg):
        self.get_logger().info("detecting april tags")

        # Convert ROS image -> OpenCV
        grayscale_photo = self.bridge.imgmsg_to_cv2(
            msg.img,
            desired_encoding='mono8'
        )

        tags = self.detector.detect(
            grayscale_photo,
            estimate_tag_pose=True,
            tag_size=self.TAG_SIZE,
            camera_params=self.camera_params
        )

        ids = []
        distances = []

        for tag in tags:

            corners = list(tag.corners.astype(int))
            corners.append(corners[0])

            segments = list(pairwise(corners))

            for pt1, pt2 in segments:
                cv2.line(
                    grayscale_photo,
                    pt1,
                    pt2,
                    self.WHITE,
                    4
                )

            center = tag.center.astype(int)

            distance = np.linalg.norm(tag.pose_t)

            ids.append(int(tag.tag_id))
            distances.append(float(distance))

            # center dot
            cv2.circle(
                grayscale_photo,
                center,
                5,
                self.WHITE,
                3
            )

            # ID text
            cv2.putText(
                grayscale_photo,
                f'ID: {tag.tag_id}',
                (center[0] - 10, center[1] - 30),
                cv2.FONT_HERSHEY_COMPLEX,
                0.75,
                self.WHITE,
                3
            )

            # distance text
            cv2.putText(
                grayscale_photo,
                f'Dist: {distance:.2f} m',
                (center[0] - 10, center[1] - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.75,
                self.WHITE,
                3
            )

        # Publish outlined image
        img_msg = self.bridge.cv2_to_imgmsg(grayscale_photo, encoding='mono8')
        self.image_pub.publish(img_msg)

        for id, distance in zip(ids, distances):
            if id < -128 or id > 127:
                continue
            self.get_logger().info(f"{id}")
            april_tag_msg = AprilTagDetection()
            april_tag_msg.id = id
            april_tag_msg.distance = distance
            self.distance_pub.publish(april_tag_msg)


def main(args=None):

    rclpy.init(args=args)

    node = april_tag_detector()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
