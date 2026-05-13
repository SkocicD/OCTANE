#!/usr/bin/env python3
"""RS485 debug terminal — live view for the BLD-510B replacement motor."""

import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from octane_msgs.msg import DriveCommand

KEY_NAMES = ['W', 'A', 'S', 'D', '↑', '↓', '←', '→']

VIOLET = '\033[95m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
RESET  = '\033[0m'

STATE_COLORS = {'MANUAL': YELLOW, 'AUTONOMOUS': GREEN, 'FAULT': RED}

FAULT_NAMES = {
    0x01: 'Locked rotor',
    0x02: 'Over-current',
    0x04: 'Hall abnormal',
    0x08: 'Bus voltage low',
    0x10: 'Bus voltage high',
    0x20: 'Current peak',
}


class RS485DebugNode(Node):

    def __init__(self):
        super().__init__('rs485_debug_node')

        self._state      = '---'
        self._rs485_status = f'{DIM}waiting for rs485_drive_node...{RESET}'
        self._keys       = 0
        self._left_vel   = 0.0   # target  (from /drive/command)
        self._sent       = 0.0   # actual sent (ramped, from TX log)
        self._last_key_t = 0.0
        self._last_cmd_t = 0.0
        self._log: deque  = deque(maxlen=10)
        self._tx_log: deque = deque(maxlen=4)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(String,       '/supervisor/state',         self._on_state,        qos)
        self.create_subscription(String,       '/manual_ctrl/key_state_str',self._on_keys_str,     qos)
        self.create_subscription(DriveCommand, '/drive/command',            self._on_drive,        qos)
        self.create_subscription(String,       '/manual_ctrl/rs485_status', self._on_rs485_status, status_qos)
        self.create_subscription(String,       '/manual_ctrl/rs485_tx',     self._on_rs485_tx,     qos)

        from std_msgs.msg import UInt8
        self.create_subscription(UInt8, '/manual_ctrl/key_state', self._on_keys, qos)

        self.create_timer(0.1, self._render)
        self._render()

    def _on_state(self, msg: String):
        self._state = msg.data

    def _on_rs485_status(self, msg: String):
        if msg.data.startswith('OK'):
            self._rs485_status = f'{GREEN}{BOLD}{msg.data}{RESET}'
        else:
            self._rs485_status = f'{RED}{BOLD}{msg.data}{RESET}'

    def _on_rs485_tx(self, msg: String):
        ts = time.strftime('%H:%M:%S')
        if msg.data.startswith('GATED'):
            entry = f'{DIM}[{ts}]{RESET}  {RED}{msg.data}{RESET}'
        else:
            entry = f'{DIM}[{ts}]{RESET}  {GREEN}{msg.data}{RESET}'
            for part in msg.data.split():
                if '=' in part:
                    k, v = part.split('=', 1)
                    try:
                        if k == 'L': self._sent = float(v)
                    except ValueError:
                        pass
        self._tx_log.append(entry)

    def _on_keys(self, msg):
        self._keys = msg.data
        self._last_key_t = time.time()

    def _on_keys_str(self, msg: String):
        pass

    def _on_drive(self, msg: DriveCommand):
        self._left_vel   = msg.left_velocity
        self._last_cmd_t = time.time()
        self._log.append(
            f'{DIM}[{time.strftime("%H:%M:%S")}]{RESET}  '
            f'L={self._left_vel:+.2f}'
        )

    def _vel_bar(self, v: float) -> str:
        filled = int(abs(v) * 10)
        bar    = ('█' * filled).ljust(10)
        color  = GREEN if v > 0 else (RED if v < 0 else DIM)
        sign   = '+' if v >= 0 else '-'
        return f'{color}{sign}[{bar}]{RESET}'

    def _render(self):
        held    = [KEY_NAMES[i] for i in range(8) if self._keys & (1 << i)]
        key_str = f'{GREEN}{BOLD}{" ".join(held)}{RESET}' if held else f'{DIM}(none){RESET}'
        key_age = f'{time.time() - self._last_key_t:.1f}s ago' if self._last_key_t else 'no data'
        cmd_age = f'{time.time() - self._last_cmd_t:.1f}s ago' if self._last_cmd_t else 'no data yet'
        sc      = STATE_COLORS.get(self._state, DIM)

        print('\033[2J\033[H', end='')
        print(f'{BOLD}{VIOLET}{"=" * 58}{RESET}')
        print(f'{BOLD}{VIOLET}    OCTANE | RS485 DEBUG  (motor #2 — mid-left){RESET}')
        print(f'{BOLD}{VIOLET}{"=" * 58}{RESET}')
        print(f'  Transceiver : {self._rs485_status}')
        print(f'  State       : {sc}{BOLD}{self._state}{RESET}')
        print()
        print(f'  Keys held   : {key_str}  {DIM}({key_age}){RESET}')
        print()
        print(f'  Target  L   : {self._vel_bar(self._left_vel)}  {self._left_vel:+.3f}')
        print(f'  Sent    L   : {self._vel_bar(self._sent)}  {self._sent:+.3f}')
        print(f'  {DIM}Last /drive/command: {cmd_age}{RESET}')
        print()
        print(f'{BOLD}  DRIVE COMMAND LOG{RESET}')
        print(f'  {DIM}{"-" * 50}{RESET}')
        if not self._log:
            print(f'  {DIM}Waiting for /drive/command...{RESET}')
        else:
            for entry in self._log:
                print(f'  {entry}')
        print()
        print(f'{BOLD}  RS485 TX LOG  {DIM}(green=sent, red=gated){RESET}')
        print(f'  {DIM}{"-" * 50}{RESET}')
        if not self._tx_log:
            print(f'  {DIM}No RS485 TX yet — is state MANUAL?{RESET}')
        else:
            for entry in self._tx_log:
                print(f'  {entry}')
        print(f'\n{BOLD}{VIOLET}{"=" * 58}{RESET}')
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = RS485DebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
