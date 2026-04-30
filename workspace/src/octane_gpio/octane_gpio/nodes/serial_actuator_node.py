#!/usr/bin/env python3
"""Serial actuator node for OCTANE.

Subscribes to /actuator/command and emits a single byte over UART whenever the
relay state changes. The byte's low 4 bits encode the four relay channels:

    bit 3 (0x08)  arm UP
    bit 2 (0x04)  arm DOWN
    bit 1 (0x02)  bucket A
    bit 0 (0x01)  bucket B

High 4 bits are always zero. Byte is sent only on state change.

If both bits in a pair are set (which manual_actuator_node already prevents,
but enforce here as well), the pair is forced to 0,0 — the receiver should
never see both relays of one module energised.

Default port is /dev/ttyTHS1 (40-pin header pins 8 TX / 10 RX on AGX Orin).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from octane_msgs.msg import ActuatorCommand

import serial


def encode(arm: int, bucket: int) -> int:
    arm_up    = 1 if arm    ==  1 else 0
    arm_down  = 1 if arm    == -1 else 0
    bucket_a  = 1 if bucket ==  1 else 0
    bucket_b  = 1 if bucket == -1 else 0

    # Defensive: never allow both channels of one relay simultaneously
    if arm_up and arm_down:
        arm_up = arm_down = 0
    if bucket_a and bucket_b:
        bucket_a = bucket_b = 0

    return (arm_up << 3) | (arm_down << 2) | (bucket_a << 1) | bucket_b


class SerialActuatorNode(Node):

    def __init__(self):
        super().__init__('serial_actuator_node')

        self.declare_parameter('port', '/dev/ttyTHS1')
        self.declare_parameter('baud', 9600)
        port = self.get_parameter('port').value
        baud = int(self.get_parameter('baud').value)

        self._last_byte: int | None = None

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'Serial actuator ready on {port} @ {baud}')
        except serial.SerialException as e:
            self.ser = None
            self.get_logger().error(f'Failed to open {port}: {e}')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            ActuatorCommand, '/actuator/command', self._on_command, qos)

    def _on_command(self, msg: ActuatorCommand):
        b = encode(msg.arm, msg.bucket)
        if b == self._last_byte:
            return
        self._last_byte = b
        if self.ser is not None:
            try:
                self.ser.write(bytes([b]))
            except serial.SerialException as e:
                self.get_logger().warn(f'Serial write failed: {e}')
        self.get_logger().info(f'sent 0x{b:02X}  ({b:04b})  arm={msg.arm} bucket={msg.bucket}')

    def destroy_node(self):
        # Send all-off on shutdown so the Arduino releases everything.
        if self.ser is not None:
            try:
                self.ser.write(bytes([0x00]))
                self.ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialActuatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
