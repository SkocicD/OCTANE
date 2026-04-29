#!/usr/bin/env python3
"""Manual drive node for OCTANE rover.

Subscribes to raw key state bitfield from the ground station GUI and
publishes normalized left/right velocity commands for tank drive.

Key bitfield (from GUI ManualControl):
  bit 0: W  (forward)
  bit 1: A  (turn left)
  bit 2: S  (backward)
  bit 3: D  (turn right)
  bit 4-7: arrow keys (handled by actuator node)

Tank drive mixing:
  Forward/back: W/S sets base throttle on both sides
  Turn: A/D adds differential (reduces one side, increases other)

Topics:
  /manual_ctrl/key_state (std_msgs/UInt8) - raw key bitfield (sub)
  /drive/command (octane_msgs/DriveCommand)  - left/right velocity (pub)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8, Bool
from octane_msgs.msg import DriveCommand

# Bit positions matching GUI GetKeyBitfield
BIT_W = 0  # forward
BIT_A = 1  # turn left
BIT_S = 2  # backward
BIT_D = 3  # turn right


def bit(bitfield: int, pos: int) -> bool:
    return bool(bitfield & (1 << pos))


class ManualDriveNode(Node):

    def __init__(self):
        super().__init__('manual_drive_node')

        self.declare_parameter('throttle_scale', 1.0)
        self.declare_parameter('turn_scale', 0.6)

        self.throttle_scale = self.get_parameter('throttle_scale').value
        self.turn_scale = self.get_parameter('turn_scale').value

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.manual_enabled = False

        self.create_subscription(UInt8, '/manual_ctrl/key_state', self.on_key_state, qos)
        self.create_subscription(Bool, '/supervisor/manual_enabled', self.on_manual_enabled, qos)
        self.pub = self.create_publisher(DriveCommand, '/drive/command', qos)

        self.get_logger().info(
            f'Manual drive node ready — throttle={self.throttle_scale}, '
            f'turn={self.turn_scale}')

    def on_manual_enabled(self, msg: Bool):
        self.manual_enabled = msg.data
        if not msg.data:
            # Supervisor left manual mode — publish stop immediately
            stop = DriveCommand()
            stop.left_velocity = 0.0
            stop.right_velocity = 0.0
            self.pub.publish(stop)

    def on_key_state(self, msg: UInt8):
        if not self.manual_enabled:
            return

        bf = msg.data

        forward = bit(bf, BIT_W)
        backward = bit(bf, BIT_S)
        left = bit(bf, BIT_A)
        right = bit(bf, BIT_D)

        # Base throttle: +1 forward, -1 backward, 0 if both or neither
        throttle = 0.0
        if forward and not backward:
            throttle = self.throttle_scale
        elif backward and not forward:
            throttle = -self.throttle_scale

        # Turn differential: reduce the inside wheel
        turn = 0.0
        if left and not right:
            turn = -self.turn_scale
        elif right and not left:
            turn = self.turn_scale

        left_vel = max(-1.0, min(1.0, throttle - turn))
        right_vel = max(-1.0, min(1.0, throttle + turn))

        cmd = DriveCommand()
        cmd.left_velocity = float(left_vel)
        cmd.right_velocity = float(right_vel)
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ManualDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
