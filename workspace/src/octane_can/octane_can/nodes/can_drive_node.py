#!/usr/bin/env python3
"""CAN bus hardware interface for OCTANE drive motors.

Subscribes to normalized drive commands and sends CAN frames to
6 motors: front/mid/rear on left and right sides (tank drive).

Motor layout:
  Left:  front_left, mid_left, rear_left
  Right: front_right, mid_right, rear_right

Topics:
  /drive/command (octane_msgs/DriveCommand) - left/right velocity -1..1 (sub)

TODO: fill in CAN interface once motor controller protocol is confirmed.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from octane_msgs.msg import DriveCommand

# TODO: import CAN library once chosen, e.g.:
# import can


# TODO: define CAN IDs for each motor once confirmed
# MOTOR_IDS = {
#     'front_left':  0x01,
#     'mid_left':    0x02,
#     'rear_left':   0x03,
#     'front_right': 0x04,
#     'mid_right':   0x05,
#     'rear_right':  0x06,
# }


class CanDriveNode(Node):

    def __init__(self):
        super().__init__('can_drive_node')

        # TODO: declare CAN interface parameters
        # self.declare_parameter('can_interface', 'can0')
        # self.declare_parameter('bitrate', 500000)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.sub = self.create_subscription(
            DriveCommand, '/drive/command', self.on_drive_command, qos)

        # TODO: initialize CAN bus
        # interface = self.get_parameter('can_interface').value
        # self.bus = can.interface.Bus(interface, bustype='socketcan')

        self.get_logger().info('CAN drive node ready (hardware not yet implemented)')

    def on_drive_command(self, msg: DriveCommand):
        left = msg.left_velocity    # -1.0 to 1.0
        right = msg.right_velocity  # -1.0 to 1.0

        # TODO: convert normalized velocity to motor controller units
        # e.g. scale to RPM, PWM counts, or whatever the controller expects
        # left_cmd = int(left * MAX_RPM)
        # right_cmd = int(right * MAX_RPM)

        # TODO: send CAN frame to each motor
        # self._send_velocity('front_left',  left_cmd)
        # self._send_velocity('mid_left',    left_cmd)
        # self._send_velocity('rear_left',   left_cmd)
        # self._send_velocity('front_right', right_cmd)
        # self._send_velocity('mid_right',   right_cmd)
        # self._send_velocity('rear_right',  right_cmd)

        self.get_logger().debug(f'Drive command: left={left:.2f}, right={right:.2f}')

    # TODO: implement once CAN protocol is known
    # def _send_velocity(self, motor_name: str, value: int):
    #     motor_id = MOTOR_IDS[motor_name]
    #     data = [value & 0xFF, (value >> 8) & 0xFF]  # example: 2-byte little-endian
    #     frame = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
    #     self.bus.send(frame)

    def destroy_node(self):
        # TODO: shut down CAN bus cleanly
        # if hasattr(self, 'bus'):
        #     self.bus.shutdown()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CanDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
