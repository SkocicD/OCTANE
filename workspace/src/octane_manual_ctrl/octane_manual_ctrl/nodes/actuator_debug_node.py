#!/usr/bin/env python3
"""Sky-blue actuator debug terminal — arrow key state + live arm/bucket relay status."""

import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import UInt8, String
from octane_msgs.msg import ActuatorCommand


def encode_byte(arm: int, bucket: int) -> int:
    """Mirror of serial_actuator_node.encode — keep these in sync."""
    arm_up    = 1 if arm    ==  1 else 0
    arm_down  = 1 if arm    == -1 else 0
    bucket_a  = 1 if bucket ==  1 else 0
    bucket_b  = 1 if bucket == -1 else 0
    if arm_up and arm_down:
        arm_up = arm_down = 0
    if bucket_a and bucket_b:
        bucket_a = bucket_b = 0
    return (arm_up << 3) | (arm_down << 2) | (bucket_a << 1) | bucket_b

CYAN   = '\033[96m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
RESET  = '\033[0m'

STATE_COLORS = {'MANUAL': YELLOW, 'AUTONOMOUS': GREEN, 'FAULT': RED}


class ActuatorDebugNode(Node):

    def __init__(self):
        super().__init__('actuator_debug_node')

        self._state    = '---'
        self._keys     = 0
        self._arm      = 0
        self._bucket   = 0
        self._last_byte    = 0
        self._serial_status = '(no /serial_actuator/status yet)'
        self._last_key_t = 0.0
        self._last_cmd_t = 0.0
        self._log: deque = deque(maxlen=24)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(String,          '/supervisor/state',      self._on_state,   qos)
        self.create_subscription(UInt8,           '/manual_ctrl/key_state', self._on_keys,    qos)
        self.create_subscription(ActuatorCommand, '/actuator/command',      self._on_actuator, qos)

        # Latched status from serial_actuator_node — match its TRANSIENT_LOCAL QoS
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, '/serial_actuator/status', self._on_serial_status, status_qos)

        self.create_timer(0.1, self._render)
        self._render()

    def _on_state(self, msg: String):
        self._state = msg.data

    def _on_keys(self, msg: UInt8):
        self._keys = msg.data
        self._last_key_t = time.time()

    def _on_actuator(self, msg: ActuatorCommand):
        self._arm    = msg.arm
        self._bucket = msg.bucket
        self._last_byte = encode_byte(msg.arm, msg.bucket)
        self._last_cmd_t = time.time()
        ts = time.strftime('%H:%M:%S')
        arm_str    = {1: '▲ UP', -1: '▼ DOWN', 0: '■ STOP'}[msg.arm]
        bucket_str = {1: '→ RIGHT', -1: '← LEFT', 0: '■ STOP'}[msg.bucket]
        self._log.append(
            f'{DIM}[{ts}]{RESET}  arm={GREEN if msg.arm != 0 else DIM}{arm_str}{RESET}'
            f'  bucket={GREEN if msg.bucket != 0 else DIM}{bucket_str}{RESET}'
            f'  {DIM}byte={self._last_byte:08b}{RESET}'
        )

    def _on_serial_status(self, msg: String):
        self._serial_status = msg.data

    def _relay_bar(self, value: int, pos_label: str, neg_label: str) -> str:
        if value == 1:
            return f'{GREEN}{BOLD}{pos_label}{RESET}'
        elif value == -1:
            return f'{RED}{BOLD}{neg_label}{RESET}'
        else:
            return f'{DIM}■ STOP{RESET}'

    def _render(self):
        # Arrow key bits: 4=↑ 5=↓ 6=← 7=→
        arrow_map  = {4: '↑', 5: '↓', 6: '←', 7: '→'}
        held       = [arrow_map[i] for i in range(4, 8) if self._keys & (1 << i)]
        key_str    = f'{GREEN}{BOLD}{" ".join(held)}{RESET}' if held else f'{DIM}(none){RESET}'
        key_age    = f'{time.time() - self._last_key_t:.1f}s ago' if self._last_key_t else 'no data'
        cmd_age    = f'{time.time() - self._last_cmd_t:.1f}s ago' if self._last_cmd_t else 'no data yet'
        sc         = STATE_COLORS.get(self._state, DIM)

        arm_display    = self._relay_bar(self._arm,    '▲ UP',    '▼ DOWN')
        bucket_display = self._relay_bar(self._bucket, '→ RIGHT', '← LEFT')

        print('\033[2J\033[H', end='')
        print(f'{BOLD}{CYAN}{"=" * 58}{RESET}')
        print(f'{BOLD}{CYAN}    OCTANE | ACTUATOR DEBUG{RESET}')
        print(f'{BOLD}{CYAN}{"=" * 58}{RESET}')
        print(f'  State        : {sc}{BOLD}{self._state}{RESET}')
        print()
        print(f'  Arrow keys   : {key_str}  {DIM}({key_age}){RESET}')
        print()
        print(f'  Arm (↑/↓)    : {arm_display}')
        print(f'  Bucket (←/→) : {bucket_display}')
        print(f'  {DIM}Last /actuator/command: {cmd_age}{RESET}')
        print()
        # Wire byte (4 bits sent to Arduino over USB serial)
        if self._serial_status.startswith('ready'):
            serial_color = GREEN
        elif self._serial_status.startswith('FAIL'):
            serial_color = RED
        else:
            serial_color = DIM
        print(f'  Serial       : {serial_color}{self._serial_status}{RESET}')
        print(f'  Wire byte    : {GREEN}{BOLD}{self._last_byte:08b}{RESET}'
              f'  {DIM}(arm-UP arm-DOWN bucket-A bucket-B in low nibble){RESET}')
        print()
        print(f'{BOLD}  ACTUATOR COMMAND LOG{RESET}')
        print(f'  {DIM}{"-" * 50}{RESET}')
        if not self._log:
            print(f'  {DIM}Waiting for /actuator/command...{RESET}')
        else:
            for entry in self._log:
                print(f'  {entry}')
        print(f'\n{BOLD}{CYAN}{"=" * 58}{RESET}')
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
