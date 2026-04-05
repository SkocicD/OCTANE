import rclpy
from rclpy.node import Node
import time
import math

from octane_msgs.msg import AprilTagDetection
from geometry_msgs.msg import Point


class April_Tag:
    def __init__(self, id, distance):
        self.id = id
        self.distance = distance
        self.last_update = time.time()

    def update(self, distance):
        self.distance = distance
        self.last_update = time.time()


class triangulator(Node):
    TAG_POSITIONS = {
        4: (0, 4),
        5: (4, 0)
    }

    def __init__(self):
        super().__init__('apriltag_detector')
        # Publishers
        self.pos_pub = self.create_publisher(
            Point,
            '/localization/position',
            10
        )

        # Subscribers
        self.distance_sub = self.create_subscription(
            AprilTagDetection,
            '/localization/april_tag_distance',
            self.triangulate,
            10
        )

        self.tags = []

    def triangulate(self, msg):
        # check to see if we already have the tag
        for tag in self.tags:
            if tag.id == msg.id:
                tag.update(msg.distance)
                break
        else:
            self.tags.append(April_Tag(msg.id, msg.distance))

        # sort in order of update time
        self.tags = sorted(self.tags, key=lambda tag: tag.last_update)

        # remove any april tag that was updated more than 5 seconds ago
        t = time.time()
        for i, tag in enumerate(self.tags):
            if t - tag.last_update < 5:
                break
        self.tags = self.tags[i:]

        # check to see that we have two april tags detected
        if len(self.tags) < 2:
            self.get_logger().info("Not enough april tags to triangulate."
                                   f" have {len(self.tags)} need 2")

        # get the two most recently updated
        tag1, tag2 = self.tags[-2:]

        # triangulate
        A = tag1.distance
        B = tag2.distance

        x1, y1 = self.TAG_POSITIONS[tag1.id]
        x2, y2 = self.TAG_POSITIONS[tag2.id]
        # choose x2, y2 to be the point down and to the right.
        if x2 < x1 or y2 > y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
            A, B = B, A

        u = math.atan(abs((y1-y2)/(x1-x2)))

        # C is the distance between the tags
        C = ((x1-x2)**2 + (y1-y2)**2)**.5

        alpha = math.acos(
            (B**2+C**2-A**2)
            /
            (2*B*C)
        )
        G = math.pi - u - alpha
        x = x2 + B*math.cos(G)
        y = y2 + B*math.sin(G)
        return (x, y)


def main(args=None):

    rclpy.init(args=args)

    node = triangulator()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
