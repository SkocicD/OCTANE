#!/usr/bin/env python3
"""CAN raw debug terminal — live byte-level view of every RX and TX frame."""

import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

CYAN   = '\033[96m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
DIM    = '\033[2m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

# Known CANOpen frame ID prefixes for labelling
_ID_LABELS = {
    0x000: 'NMT    ',
    0x080: 'SYNC   ',
    0x100: 'TIME   ',
    0x180: 'TPDO1  ',
    0x280: 'TPDO2  ',
    0x380: 'TPDO3  ',
    0x480: 'TPDO4  ',
    0x200: 'RPDO1  ',
    0x300: 'RPDO2  ',
    0x400: 'RPDO3  ',
    0x500: 'RPDO4  ',
    0x580: 'SDO-TX ',
    0x600: 'SDO-RX ',
    0x700: 'HB     ',
}


def _label(can_id: int) -> str:
    base = can_id & 0x780
    return _ID_LABELS.get(base, '       ')


class CANRawDebugNode(Node):

    def __init__(self):
        super().__init__('can_raw_debug_node')

        self._log: deque = deque(maxlen=30)

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/can/rx',     self._on_rx, qos)
        self.create_subscription(String, '/can/tx_raw', self._on_tx, qos)

        self.create_timer(0.1, self._render)
        self._render()

    def _on_rx(self, msg: String):
        ts = time.strftime('%H:%M:%S')
        # msg.data format: "RX  0x181  [8]  01 00 ..."
        parts = msg.data.split(None, 3)
        can_id = int(parts[1], 16) if len(parts) > 1 else 0
        rest   = '  '.join(parts[1:]) if len(parts) > 1 else msg.data
        label  = _label(can_id)
        self._log.append(
            f'{DIM}[{ts}]{RESET}  {YELLOW}RX{RESET}  {DIM}{label}{RESET}{YELLOW}{rest}{RESET}'
        )

    def _on_tx(self, msg: String):
        ts = time.strftime('%H:%M:%S')
        parts = msg.data.split(None, 3)
        can_id = int(parts[1], 16) if len(parts) > 1 else 0
        rest   = '  '.join(parts[1:]) if len(parts) > 1 else msg.data
        label  = _label(can_id)
        self._log.append(
            f'{DIM}[{ts}]{RESET}  {GREEN}TX{RESET}  {DIM}{label}{RESET}{GREEN}{rest}{RESET}'
        )

    def _render(self):
        print('\033[2J\033[H', end='')
        print(f'{BOLD}{CYAN}{"=" * 70}{RESET}')
        print(f'{BOLD}{CYAN}    OCTANE | CAN RAW DEBUG{RESET}')
        print(f'{BOLD}{CYAN}{"=" * 70}{RESET}')
        print(f'  {DIM}{"direction":<4}  {"type":<7}  {"id":<6}  {"dlc":<4}  data bytes{RESET}')
        print(f'  {DIM}{"-" * 62}{RESET}')
        if not self._log:
            print(f'  {DIM}Waiting for CAN frames — is can_worker running?{RESET}')
        else:
            for entry in self._log:
                print(f'  {entry}')
        print(f'\n{BOLD}{CYAN}{"=" * 70}{RESET}')
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = CANRawDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
