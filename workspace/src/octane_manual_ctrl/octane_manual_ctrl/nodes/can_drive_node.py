#!/usr/bin/env python3
"""CAN drive node — translates DriveCommand into CANOpen PDO motor commands.

Subscribes to /drive/command and /supervisor/state.
Only drives motors when state == MANUAL.

Motor layout (node IDs 0-5):
  Left  side: 0=front-left, 1=mid-left,  2=back-left
  Right side: 3=back-right, 4=mid-right, 5=front-right  (polarity flipped)

Velocity ramping: a 20 Hz control loop steps current velocity toward the
target at RAMP_RATE units/sec so motors never see a step change.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from octane_msgs.msg import DriveCommand

from octane_manual_ctrl.can.can_transceiver import CANTransceiver
from octane_manual_ctrl.can.motor_controller import MotorController

LEFT_IDS  = [1, 2, 3]
RIGHT_IDS = [4, 5, 6]
ALL_IDS   = LEFT_IDS + RIGHT_IDS

CONTROL_HZ      = 20   # ramp update rate (Hz)
PDO_WATCHDOG_HZ = 2    # always re-send at least this often even when steady


class CANDriveNode(Node):

    def __init__(self):
        super().__init__('can_drive_node')
        self.declare_parameter('bitrate',    1_000_000)
        self.declare_parameter('ramp_rate',  3.0)
        self.declare_parameter('dead_band',  0.02)

        self._manual = False
        self._motors: list[MotorController] = []

        self._target_l  = 0.0
        self._target_r  = 0.0
        self._current_l = 0.0
        self._current_r = 0.0
        self._sent_l    = None   # last value actually written to CAN (None = force first send)
        self._sent_r    = None
        self._ramp_rate      = self.get_parameter('ramp_rate').value
        self._dead_band      = self.get_parameter('dead_band').value
        self._watchdog_ticks = 0
        self._watchdog_every = max(1, int(CONTROL_HZ / PDO_WATCHDOG_HZ))

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
                time.sleep(0.05)
                m.set_mode()
                time.sleep(0.05)
                self._motors.append(m)
            self._publish_status(f'OK — {len(self._motors)} motors initialised on nodes {ALL_IDS}')
            self.get_logger().info(f'CAN drive ready, motors {ALL_IDS}')
        except Exception as e:
            self._publish_status(f'ERROR: {e}')
            self.get_logger().error(f'CAN init failed: {e}')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._tx_pub = self.create_publisher(String, '/manual_ctrl/can_tx', qos)
        self.create_subscription(String,       '/supervisor/state', self._on_state, qos)
        self.create_subscription(DriveCommand, '/drive/command',    self._on_drive, qos)

        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _on_state(self, msg: String):
        was = self._manual
        self._manual = msg.data == 'MANUAL'
        if was and not self._manual:
            self._target_l = 0.0
            self._target_r = 0.0

    def _on_drive(self, msg: DriveCommand):
        if not self._manual:
            status = String()
            status.data = f'GATED:STANDBY  L={msg.left_velocity:+.2f} R={msg.right_velocity:+.2f}'
            self._tx_pub.publish(status)
            return
        self._target_l = max(-1.0, min(1.0, msg.left_velocity))
        self._target_r = max(-1.0, min(1.0, msg.right_velocity))

    def _control_loop(self):
        step = self._ramp_rate / CONTROL_HZ

        def ramp(current, target):
            diff = target - current
            if abs(diff) <= step:
                return target
            return current + step * (1 if diff > 0 else -1)

        self._current_l = ramp(self._current_l, self._target_l)
        self._current_r = ramp(self._current_r, self._target_r)

        if not self._manual:
            # Ramp down to zero even after leaving MANUAL, then stop
            if self._current_l == 0.0 and self._current_r == 0.0:
                return
            self._target_l = 0.0
            self._target_r = 0.0

        if not self._motors:
            return

        # Dead-band: only push a new PDO when speed changed meaningfully, or
        # on the watchdog tick so motors know we're still alive.
        self._watchdog_ticks += 1
        force = self._watchdog_ticks >= self._watchdog_every
        if force:
            self._watchdog_ticks = 0

        send_l = force or self._sent_l is None or abs(self._current_l - self._sent_l) > self._dead_band
        send_r = force or self._sent_r is None or abs(self._current_r - self._sent_r) > self._dead_band

        if send_l:
            for m in self._motors[:3]:
                m.move(abs(self._current_l), reverse=(self._current_l < 0))
            self._sent_l = self._current_l

        if send_r:
            for m in self._motors[3:]:
                m.move(abs(self._current_r), reverse=(self._current_r < 0))
            self._sent_r = self._current_r

        if send_l or send_r:
            status = String()
            status.data = f'TX  L={self._current_l:+.3f} R={self._current_r:+.3f}'
            self._tx_pub.publish(status)

    def _stop_all(self):
        self._target_l  = 0.0
        self._target_r  = 0.0
        self._current_l = 0.0
        self._current_r = 0.0
        self._sent_l    = None
        self._sent_r    = None
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
