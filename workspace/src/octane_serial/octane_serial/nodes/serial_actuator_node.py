#!/usr/bin/env python3
"""Serial actuator node for OCTANE.

Subscribes to /actuator/command and emits a single byte over UART whenever the
relay state changes. The byte's low 4 bits encode the four relay channels;
the high 4 bits are always zero (padding):

    bit 7..4  always 0
    bit 3  arm UP
    bit 2  arm DOWN
    bit 1  bucket A
    bit 0  bucket B

So the wire byte looks like:  0000ABCD  where ABCD are the four relay bits
(LSB on the right, as usual).

If both bits in a pair are set (which manual_actuator_node already prevents,
but enforce here as well), the pair is forced to 0,0 — the receiver should
never see both relays of one module energised.

Default port is the udev symlink for the Arduino Nano, pinned by physical USB
port (KERNELS=="1-4.2.1", external USB hub). The RS485 adapter is also CH340
(1a86:7523) and is pinned to KERNELS=="1-4.3" (Jetson direct port) via
99-rs485-drive.rules so the two never collide.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
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

        self.declare_parameter(
            'port',
            '/dev/arduino_nano',
        )
        self.declare_parameter('baud', 9600)
        port = self.get_parameter('port').value
        baud = int(self.get_parameter('baud').value)

        self._last_byte: int | None = None

        # Latched status topic so late subscribers (the debug terminal) get
        # the connection result even if they start after this node.
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, '/serial_actuator/status', status_qos)

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            status = f'ready: {port}'
            self.get_logger().info(f'Serial actuator ready on {port} @ {baud}')
        except serial.SerialException as e:
            self.ser = None
            status = f'FAIL: {e}'
            self.get_logger().error(f'Failed to open {port}: {e}')
        self._status_pub.publish(String(data=status))

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            ActuatorCommand, '/actuator/command', self._on_command, qos)

    def _on_command(self, msg: ActuatorCommand):
        b = encode(msg.arm, msg.bucket)
        if b == self._last_byte:
            return
        self._last_byte = b
        if self.ser is None:
            self.get_logger().warn(
                f'would send {b:08b} but port is not open  arm={msg.arm} bucket={msg.bucket}')
            return
        try:
            self.ser.write(bytes([b]))
        except serial.SerialException as e:
            self.get_logger().warn(f'Serial write failed: {e}')
            return
        self.get_logger().info(f'sent {b:08b}  arm={msg.arm} bucket={msg.bucket}')

    def destroy_node(self):
        # Send all-off on shutdown so the Arduino releases everything.
        if self.ser is not None:
            try:
                self.ser.write(bytes([0b00000000]))
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
