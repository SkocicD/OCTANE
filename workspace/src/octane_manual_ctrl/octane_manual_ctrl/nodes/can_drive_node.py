#!/usr/bin/env python3
"""CAN drive node — translates DriveCommand into CANOpen PDO motor commands.

Subscribes to /drive/command and /supervisor/state.
Only drives motors when state == MANUAL.

Motor layout (node IDs 0-5):
  Left  side: 0=front-left, 1=mid-left,  2=back-left
  Right side: 3=back-right, 4=mid-right, 5=front-right  (polarity flipped)
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from octane_msgs.msg import DriveCommand

from octane_manual_ctrl.can.can_transceiver import CANTransceiver
from octane_manual_ctrl.can.motor_controller import MotorController

LEFT_IDS  = [0, 1, 2]
RIGHT_IDS = [3, 4, 5]
ALL_IDS   = LEFT_IDS + RIGHT_IDS


class CANDriveNode(Node):

    def __init__(self):
        super().__init__('can_drive_node')
        self.declare_parameter('bitrate', 1_000_000)

        self._manual = False
        self._motors: list[MotorController] = []

        # Latched status so can_debug_node sees it even if it starts late
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(String, '/manual_ctrl/can_status', status_qos)

        try:
            tx = CANTransceiver(bitrate=self.get_parameter('bitrate').value)
            for nid in ALL_IDS:
                m = MotorController(nid, tx)
                m.turn_on()
                time.sleep(0.05)   # allow motor to complete NMT boot before SDO config
                m.set_mode()
                time.sleep(0.05)
                self._motors.append(m)
            self._publish_status(f'OK — {len(self._motors)} motors initialised on nodes {ALL_IDS}')
            self.get_logger().info(f'CAN drive ready, motors {ALL_IDS}')
        except Exception as e:
            self._publish_status(f'ERROR: {e}')
            self.get_logger().error(f'CAN init failed: {e}')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String,       '/supervisor/state', self._on_state,    qos)
        self.create_subscription(DriveCommand, '/drive/command',    self._on_drive,    qos)

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _on_state(self, msg: String):
        was = self._manual
        self._manual = msg.data == 'MANUAL'
        if was and not self._manual:
            self._stop_all()

    def _on_drive(self, msg: DriveCommand):
        if not self._manual or not self._motors:
            return

        l = max(-1.0, min(1.0, msg.left_velocity))
        r = max(-1.0, min(1.0, msg.right_velocity))

        for m in self._motors[:3]:   # left side
            m.move(abs(l), reverse=(l < 0))
        for m in self._motors[3:]:   # right side
            m.move(abs(r), reverse=(r < 0))

    def _stop_all(self):
        for m in self._motors:
            try:
                m.stop()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = CANDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_all()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
