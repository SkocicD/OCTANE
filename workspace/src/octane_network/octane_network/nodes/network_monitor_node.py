#!/usr/bin/env python3
"""Network monitor node — live display of GUI commands received over TCP.

Subscribes to the decoded command topics published by network_comm_node and
renders a human-readable rolling log so operators can confirm what the GUI
sent without reading raw ROS logs.
"""

import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Empty


MODE_DESCRIPTIONS = {
    'STANDBY':     'Disable all control — return to idle',
    'MANUAL':      'Enable joystick / remote control',
    'AUTONOMOUS':  'Enable RL policy navigation',
    'FAULT_RESET': 'Clear faults and return to standby',
}

MODE_COLORS = {
    'STANDBY':     '\033[96m',   # cyan
    'MANUAL':      '\033[93m',   # yellow
    'AUTONOMOUS':  '\033[92m',   # green
    'FAULT_RESET': '\033[94m',   # blue
}

STATE_COLORS = {
    'STANDBY':    '\033[96m',
    'MANUAL':     '\033[93m',
    'AUTONOMOUS': '\033[92m',
    'FAULT':      '\033[91m',
}


class NetworkMonitorNode(Node):

    BLUE  = '\033[94m'
    BOLD  = '\033[1m'
    DIM   = '\033[2m'
    RESET = '\033[0m'

    def __init__(self):
        super().__init__('network_monitor_node')

        self.current_state  = '---'
        self.last_cmd_time: float = 0.0
        self.last_hb_status = ''
        self.last_hb_time:  float = 0.0
        self.log: deque = deque(maxlen=16)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(String, '/supervisor/mode_command',  self._on_mode_command, qos)
        self.create_subscription(Empty,  '/supervisor/fault_reset',   self._on_fault_reset,  qos)
        self.create_subscription(String, '/supervisor/state',         self._on_state,        qos)
        self.create_subscription(String, '/network/heartbeat_tx',     self._on_heartbeat,    qos)

        self.create_timer(1.0, self._refresh)
        self._render()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_mode_command(self, msg: String):
        mode = msg.data.upper()
        desc  = MODE_DESCRIPTIONS.get(mode, 'Unknown command')
        color = MODE_COLORS.get(mode, self.RESET)
        self._append_log(f'{color}{self.BOLD}CMD  {mode:<12}{self.RESET}  {self.DIM}{desc}{self.RESET}')
        self.last_cmd_time = time.time()

    def _on_fault_reset(self, msg: Empty):
        desc = MODE_DESCRIPTIONS['FAULT_RESET']
        self._append_log(f'{self.BLUE}{self.BOLD}CMD  FAULT_RESET  {self.RESET}  {self.DIM}{desc}{self.RESET}')
        self.last_cmd_time = time.time()

    def _on_state(self, msg: String):
        self.current_state = msg.data

    def _on_heartbeat(self, msg: String):
        self.last_hb_status = msg.data
        self.last_hb_time   = time.time()

    # ── display ───────────────────────────────────────────────────────────────

    def _append_log(self, body: str):
        ts = time.strftime('%H:%M:%S')
        self.log.append(f'{self.DIM}[{ts}]{self.RESET}  {body}')

    def _refresh(self):
        self._render()

    def _render(self):
        state_color = STATE_COLORS.get(self.current_state, self.RESET)

        idle_s = time.time() - self.last_cmd_time if self.last_cmd_time else None
        if idle_s is None:
            idle_str = f'{self.DIM}no commands yet{self.RESET}'
        elif idle_s < 5:
            idle_str = f'\033[92m{self.BOLD}active{self.RESET}'
        else:
            idle_str = f'{self.DIM}idle ({idle_s:.0f}s){self.RESET}'

        print('\033[2J\033[H', end='')
        print(f'{self.BOLD}{self.BLUE}{"=" * 60}{self.RESET}')
        print(f'{self.BOLD}{self.BLUE}    OCTANE NETWORK MONITOR{self.RESET}')
        print(f'{self.BOLD}{self.BLUE}{"=" * 60}{self.RESET}')
        if self.last_hb_time:
            hb_age = time.time() - self.last_hb_time
            if self.last_hb_status.startswith('ERR'):
                hb_str = f'\033[91m{self.BOLD}{self.last_hb_status}{self.RESET}'
            else:
                hb_str = f'\033[92m{self.last_hb_status}{self.RESET}  {self.DIM}({hb_age:.0f}s ago){self.RESET}'
        else:
            hb_str = f'{self.DIM}waiting...{self.RESET}'

        print(f'    Rover state : {state_color}{self.BOLD}{self.current_state}{self.RESET}')
        print(f'    GUI link    : {idle_str}')
        print(f'    Heartbeat   : {hb_str}')
        print(f'    TCP port    : 5000\n')

        print(f'{self.BOLD}    COMMAND LOG{self.RESET}')
        print(f'    {self.DIM}{"-" * 54}{self.RESET}')

        if not self.log:
            print(f'    {self.DIM}Waiting for commands from GUI...{self.RESET}')
        else:
            for entry in self.log:
                print(f'    {entry}')

        print(f'\n{self.BOLD}{self.BLUE}{"=" * 60}{self.RESET}')
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = NetworkMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
