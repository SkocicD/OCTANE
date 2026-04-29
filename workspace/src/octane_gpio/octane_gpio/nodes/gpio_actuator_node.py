#!/usr/bin/env python3
"""GPIO hardware interface for OCTANE actuator relays.

Subscribes to actuator commands and drives Jetson GPIO pins to control
two-channel relays for arm (up/down) and bucket (two directions).

Relay wiring (two channels each):
  Arm:    one pin = up relay,   one pin = down relay
  Bucket: one pin = dir-A relay, one pin = dir-B relay

Command values: -1 (reverse), 0 (stop), 1 (forward)
  arm=1  -> arm_up ON,   arm_down OFF
  arm=-1 -> arm_up OFF,  arm_down ON
  arm=0  -> both OFF

Topics:
  /actuator/command (octane_msgs/ActuatorCommand) - arm + bucket int8 (sub)

TODO: fill in GPIO pin numbers and library calls once wiring is confirmed.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from octane_msgs.msg import ActuatorCommand

# TODO: import Jetson GPIO once pin mapping confirmed
# import Jetson.GPIO as GPIO

# TODO: set actual GPIO pin numbers (BCM numbering)
# PIN_ARM_UP    = None
# PIN_ARM_DOWN  = None
# PIN_BUCKET_A  = None
# PIN_BUCKET_B  = None


class GpioActuatorNode(Node):

    def __init__(self):
        super().__init__('gpio_actuator_node')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.sub = self.create_subscription(
            ActuatorCommand, '/actuator/command', self.on_actuator_command, qos)

        # TODO: configure GPIO pins
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup(PIN_ARM_UP,   GPIO.OUT, initial=GPIO.LOW)
        # GPIO.setup(PIN_ARM_DOWN, GPIO.OUT, initial=GPIO.LOW)
        # GPIO.setup(PIN_BUCKET_A, GPIO.OUT, initial=GPIO.LOW)
        # GPIO.setup(PIN_BUCKET_B, GPIO.OUT, initial=GPIO.LOW)

        self.get_logger().info('GPIO actuator node ready (hardware not yet implemented)')

    def on_actuator_command(self, msg: ActuatorCommand):
        self._set_arm(msg.arm)
        self._set_bucket(msg.bucket)

    def _set_arm(self, value: int):
        # value: -1=down, 0=stop, 1=up
        # TODO: replace with actual GPIO calls
        # GPIO.output(PIN_ARM_UP,   GPIO.HIGH if value == 1  else GPIO.LOW)
        # GPIO.output(PIN_ARM_DOWN, GPIO.HIGH if value == -1 else GPIO.LOW)
        self.get_logger().debug(f'Arm: {value}')

    def _set_bucket(self, value: int):
        # value: -1=dir-A, 0=stop, 1=dir-B
        # TODO: replace with actual GPIO calls
        # GPIO.output(PIN_BUCKET_A, GPIO.HIGH if value == -1 else GPIO.LOW)
        # GPIO.output(PIN_BUCKET_B, GPIO.HIGH if value == 1  else GPIO.LOW)
        self.get_logger().debug(f'Bucket: {value}')

    def destroy_node(self):
        # TODO: ensure all relays are de-energised on shutdown
        # GPIO.output(PIN_ARM_UP,   GPIO.LOW)
        # GPIO.output(PIN_ARM_DOWN, GPIO.LOW)
        # GPIO.output(PIN_BUCKET_A, GPIO.LOW)
        # GPIO.output(PIN_BUCKET_B, GPIO.LOW)
        # GPIO.cleanup()
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
