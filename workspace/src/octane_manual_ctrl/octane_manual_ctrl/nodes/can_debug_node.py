#!/usr/bin/env python3
"""Purple CAN debug terminal — transceiver status, live keys, velocity bars, drive log.

Press SPACE to toggle between status view and raw CAN frame view (RX/TX hex).
"""

import select
import sys
import termios
import threading
import time
import tty
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import UInt8, String
from octane_msgs.msg import DriveCommand

KEY_NAMES = ['W', 'A', 'S', 'D', '↑', '↓', '←', '→']

PURPLE = '\033[95m'
BOLD   = '\033[1m'
DIM    = '\033[2m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
RESET  = '\033[0m'

STATE_COLORS = {'MANUAL': YELLOW, 'AUTONOMOUS': GREEN, 'FAULT': RED}

VIEW_STATUS = 'status'
VIEW_RAW    = 'raw'


class CANDebugNode(Node):

    def __init__(self):
        super().__init__('can_debug_node')

        self._state      = '---'
        self._can_status = f'{DIM}waiting for can_drive_node...{RESET}'
        self._keys       = 0
        self._left_vel   = 0.0
        self._right_vel  = 0.0
        self._sent_l     = 0.0
        self._sent_r     = 0.0
        self._speed_modifier = 100
        self._last_key_t = 0.0
        self._last_cmd_t = 0.0
        self._log: deque    = deque(maxlen=10)
        self._tx_log: deque = deque(maxlen=4)
        self._raw_log: deque = deque(maxlen=30)  # raw RX/TX hex frames

        # SPACE toggles between status view and raw CAN frame view
        self._view = VIEW_STATUS

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(String,       '/supervisor/state',       self._on_state,      qos)
        self.create_subscription(UInt8,        '/manual_ctrl/key_state',  self._on_keys,       qos)
        self.create_subscription(DriveCommand, '/drive/command',          self._on_drive,      qos)
        self.create_subscription(String,       '/manual_ctrl/can_status', self._on_can_status, status_qos)
        self.create_subscription(String,       '/manual_ctrl/can_tx',     self._on_can_tx,     qos)
        self.create_subscription(String,       '/manual_ctrl/can_raw',    self._on_can_raw,    qos)

        # Background thread reads SPACE from the xterm without blocking the ROS executor
        self._input_thread = threading.Thread(target=self._read_keys, daemon=True)
        self._input_thread.start()

        self.create_timer(0.1, self._render)
        self._render()

    # ── keyboard input ──────────────────────────────────────────────────────

    def _read_keys(self):
        """Non-blocking stdin reader — toggles view on SPACE."""
        if not sys.stdin.isatty():
            return
        old = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch == ' ':
                        self._view = VIEW_RAW if self._view == VIEW_STATUS else VIEW_STATUS
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)

    # ── subscribers ─────────────────────────────────────────────────────────

    def _on_state(self, msg: String):
        self._state = msg.data

    def _on_can_status(self, msg: String):
        if msg.data.startswith('OK'):
            self._can_status = f'{GREEN}{BOLD}{msg.data}{RESET}'
        else:
            self._can_status = f'{RED}{BOLD}{msg.data}{RESET}'

    def _on_can_tx(self, msg: String):
        ts = time.strftime('%H:%M:%S')
        if msg.data.startswith('GATED'):
            entry = f'{DIM}[{ts}]{RESET}  {RED}{msg.data}{RESET}'
        else:
            entry = f'{DIM}[{ts}]{RESET}  {GREEN}{msg.data}{RESET}'
            for part in msg.data.split():
                if '=' in part:
                    k, v = part.split('=', 1)
                    try:
                        if k == 'L': self._sent_l = float(v)
                        elif k == 'R': self._sent_r = float(v)
                    except ValueError:
                        pass
        self._tx_log.append(entry)

    def _on_can_raw(self, msg: String):
        # Format: "TX 0x001 08 01 00 00 00 00 00 00 00"
        ts = time.strftime('%H:%M:%S.') + f'{int(time.time() * 1000) % 1000:03d}'
        parts = msg.data.split(None, 3)
        if not parts:
            return
        direction = parts[0]
        color = GREEN if direction == 'TX' else CYAN
        self._raw_log.append(
            f'{DIM}[{ts}]{RESET}  {color}{BOLD}{direction}{RESET}  {" ".join(parts[1:])}'
        )

    def _on_keys(self, msg: UInt8):
        self._keys = msg.data
        self._last_key_t = time.time()

    def _on_drive(self, msg: DriveCommand):
        self._left_vel       = msg.left_velocity
        self._right_vel      = msg.right_velocity
        self._speed_modifier = msg.speed_modifier
        self._last_cmd_t     = time.time()
        self._log.append(
            f'{DIM}[{time.strftime("%H:%M:%S")}]{RESET}  '
            f'L={self._left_vel:+.2f}  R={self._right_vel:+.2f}  spd={self._speed_modifier}%'
        )

    # ── rendering ───────────────────────────────────────────────────────────

    def _vel_bar(self, v: float) -> str:
        filled = int(abs(v) * 10)
        bar    = ('█' * filled).ljust(10)
        color  = GREEN if v > 0 else (RED if v < 0 else DIM)
        sign   = '+' if v >= 0 else '-'
        return f'{color}{sign}[{bar}]{RESET}'

    def _speed_bar(self, pct: int) -> str:
        filled = min(10, pct // 50)
        bar    = ('█' * filled).ljust(10)
        color  = GREEN if pct <= 100 else (YELLOW if pct <= 300 else RED)
        return f'{color}[{bar}]{RESET}  {pct}%'

    def _render(self):
        print('\033[2J\033[H', end='')
        if self._view == VIEW_RAW:
            self._render_raw()
        else:
            self._render_status()
        sys.stdout.flush()

    def _render_status(self):
        held    = [KEY_NAMES[i] for i in range(8) if self._keys & (1 << i)]
        key_str = f'{GREEN}{BOLD}{" ".join(held)}{RESET}' if held else f'{DIM}(none){RESET}'
        key_age = f'{time.time() - self._last_key_t:.1f}s ago' if self._last_key_t else 'no data'
        cmd_age = f'{time.time() - self._last_cmd_t:.1f}s ago' if self._last_cmd_t else 'no data yet'
        sc      = STATE_COLORS.get(self._state, DIM)

        print(f'{BOLD}{PURPLE}{"=" * 58}{RESET}')
        print(f'{BOLD}{PURPLE}    OCTANE | CAN DEBUG{RESET}  {DIM}[SPACE = raw frames]{RESET}')
        print(f'{BOLD}{PURPLE}{"=" * 58}{RESET}')
        print(f'  Transceiver : {self._can_status}')
        print(f'  State       : {sc}{BOLD}{self._state}{RESET}')
        print()
        print(f'  Keys held   : {key_str}  {DIM}({key_age}){RESET}')
        print()
        print(f'  Target  L   : {self._vel_bar(self._left_vel)}  {self._left_vel:+.3f}   R : {self._vel_bar(self._right_vel)}  {self._right_vel:+.3f}')
        print(f'  Sent    L   : {self._vel_bar(self._sent_l)}  {self._sent_l:+.3f}   R : {self._vel_bar(self._sent_r)}  {self._sent_r:+.3f}')
        print(f'  Speed mod   : {self._speed_bar(self._speed_modifier)}')
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
        print(f'{BOLD}  CAN TX LOG  {DIM}(green=sent, red=gated){RESET}')
        print(f'  {DIM}{"-" * 50}{RESET}')
        if not self._tx_log:
            print(f'  {DIM}No CAN TX yet — is state MANUAL?{RESET}')
        else:
            for entry in self._tx_log:
                print(f'  {entry}')
        print(f'\n{BOLD}{PURPLE}{"=" * 58}{RESET}')

    def _render_raw(self):
        sc = STATE_COLORS.get(self._state, DIM)
        print(f'{BOLD}{PURPLE}{"=" * 58}{RESET}')
        print(f'{BOLD}{PURPLE}    OCTANE | CAN RAW FRAMES{RESET}  {DIM}[SPACE = status]{RESET}')
        print(f'{BOLD}{PURPLE}{"=" * 58}{RESET}')
        print(f'  Transceiver : {self._can_status}')
        print(f'  State       : {sc}{BOLD}{self._state}{RESET}')
        print()
        print(f'  {GREEN}{BOLD}TX{RESET} = sent to motors   {CYAN}{BOLD}RX{RESET} = replies from motors')
        print(f'  Format: [timestamp]  DIR  ID  DLC  DATA...')
        print(f'  {DIM}{"-" * 54}{RESET}')
        print()
        if not self._raw_log:
            print(f'  {DIM}No CAN frames yet — waiting for motor traffic...{RESET}')
        else:
            for entry in self._raw_log:
                print(f'  {entry}')
        print(f'\n{BOLD}{PURPLE}{"=" * 58}{RESET}')


def main(args=None):
    rclpy.init(args=args)
    node = CANDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
