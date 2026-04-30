#!/usr/bin/env python3
"""GPIO hardware interface for OCTANE actuator relays.

Subscribes to actuator commands and drives Jetson GPIO pins to control
two-channel relay modules for arm (up/down) and bucket (two directions).

Wiring (Jetson AGX Orin 40-pin header, BOARD numbering):
  Pin 11 -> IN1 -> Arm UP relay    (NO1/COM1 closes when arm=1)
  Pin 13 -> IN2 -> Arm DOWN relay  (NO2/COM2 closes when arm=-1)
  Pin 15 -> IN3 -> Bucket dir-A    (second relay module)
  Pin 18 -> IN4 -> Bucket dir-B    (second relay module)
  Pin 6  -> DC- (shared GND between Jetson and relay module)

Command values: -1 (reverse), 0 (stop), 1 (forward)
  arm=1  -> PIN_ARM_UP HIGH,   PIN_ARM_DOWN LOW
  arm=-1 -> PIN_ARM_UP LOW,    PIN_ARM_DOWN HIGH
  arm=0  -> both LOW (stop)

Topics:
  /actuator/command (octane_msgs/ActuatorCommand) - arm + bucket int8 (sub)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from octane_msgs.msg import ActuatorCommand

try:
    import Jetson.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

# Jetson AGX Orin 40-pin header, BOARD (physical) numbering
# All four pins use tegra234-gpio (not AON) — no permission issues
PIN_ARM_UP   = 11
PIN_ARM_DOWN = 13
PIN_BUCKET_A = 15   # enabled via jetson-io overlay (see resources/jetson_gpio_setup.md)
PIN_BUCKET_B = 18   # enabled via jetson-io overlay (see resources/jetson_gpio_setup.md)

_ARM_PINS    = (PIN_ARM_UP, PIN_ARM_DOWN)
_BUCKET_PINS = (PIN_BUCKET_A, PIN_BUCKET_B)
_ALL_PINS    = _ARM_PINS + _BUCKET_PINS


class GpioActuatorNode(Node):

    def __init__(self):
        super().__init__('gpio_actuator_node')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.sub = self.create_subscription(
            ActuatorCommand, '/actuator/command', self.on_actuator_command, qos)

        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BOARD)
            for pin in _ALL_PINS:
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            self.get_logger().info(
                f'GPIO actuator node ready — pins ARM({PIN_ARM_UP},{PIN_ARM_DOWN}) '
                f'BUCKET({PIN_BUCKET_A},{PIN_BUCKET_B})'
            )
        else:
            self.get_logger().warn(
                'Jetson.GPIO not found — running in simulation mode (no hardware output)'
            )

    def on_actuator_command(self, msg: ActuatorCommand):
        self._set_arm(msg.arm)
        self._set_bucket(msg.bucket)

    def _set_arm(self, value: int):
        if _GPIO_AVAILABLE:
            GPIO.output(PIN_ARM_UP,   GPIO.HIGH if value == 1  else GPIO.LOW)
            GPIO.output(PIN_ARM_DOWN, GPIO.HIGH if value == -1 else GPIO.LOW)
        if value != 0:
            self.get_logger().info(f'ARM gpio: pin {PIN_ARM_UP if value==1 else PIN_ARM_DOWN} HIGH')

    def _set_bucket(self, value: int):
        if _GPIO_AVAILABLE:
            GPIO.output(PIN_BUCKET_A, GPIO.HIGH if value == 1  else GPIO.LOW)
            GPIO.output(PIN_BUCKET_B, GPIO.HIGH if value == -1 else GPIO.LOW)
        if value != 0:
            self.get_logger().info(f'BUCKET gpio: pin {PIN_BUCKET_A if value==1 else PIN_BUCKET_B} HIGH')

    def _all_off(self):
        if _GPIO_AVAILABLE:
            for pin in _ALL_PINS:
                GPIO.output(pin, GPIO.LOW)

    def destroy_node(self):
        self._all_off()
        if _GPIO_AVAILABLE:
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GpioActuatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
