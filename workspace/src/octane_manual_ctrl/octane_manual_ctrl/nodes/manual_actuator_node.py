#!/usr/bin/env python3
"""Manual actuator node for OCTANE rover.

Subscribes to raw key state bitfield from the ground station GUI and
publishes discrete actuator commands for arm and bucket relay control.

Key bitfield (from GUI ManualControl):
  bit 4: ↑ arrow  (arm up)
  bit 5: ↓ arrow  (arm down)
  bit 6: ← arrow  (bucket one direction)
  bit 7: → arrow  (bucket other direction)

Output values per axis: -1 (reverse), 0 (stop), 1 (forward)
The GPIO hardware node maps these to relay on/off signals.

Topics:
  /manual_ctrl/key_state (std_msgs/UInt8)       - raw key bitfield (sub)
  /actuator/command (octane_msgs/ActuatorCommand) - arm + bucket (pub)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8
from octane_msgs.msg import ActuatorCommand

# Bit positions matching GUI GetKeyBitfield
BIT_UP = 4     # arm up
BIT_DOWN = 5   # arm down
BIT_LEFT = 6   # bucket direction A
BIT_RIGHT = 7  # bucket direction B


def bit(bitfield: int, pos: int) -> bool:
    return bool(bitfield & (1 << pos))


class ManualActuatorNode(Node):

    def __init__(self):
        super().__init__('manual_actuator_node')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.sub = self.create_subscription(
            UInt8, '/manual_ctrl/key_state', self.on_key_state, qos)
        self.pub = self.create_publisher(ActuatorCommand, '/actuator/command', qos)

        self.get_logger().info('Manual actuator node ready')

    def on_key_state(self, msg: UInt8):
        bf = msg.data

        arm_up = bit(bf, BIT_UP)
        arm_down = bit(bf, BIT_DOWN)
        bucket_left = bit(bf, BIT_LEFT)
        bucket_right = bit(bf, BIT_RIGHT)

        # Resolve conflicting inputs: both pressed = stop
        arm = 0
        if arm_up and not arm_down:
            arm = 1
        elif arm_down and not arm_up:
            arm = -1

        bucket = 0
        if bucket_left and not bucket_right:
            bucket = -1
        elif bucket_right and not bucket_left:
            bucket = 1

        cmd = ActuatorCommand()
        cmd.arm = arm
        cmd.bucket = bucket
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ManualActuatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
