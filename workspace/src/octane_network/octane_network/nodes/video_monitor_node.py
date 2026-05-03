#!/usr/bin/env python3
"""Video stream monitor — live display of camera streaming activity.

Shows current stream state (camera, variant, quality, fps) and a rolling
log of every stream switch, start, and stop event.

Launched as an orange xterm by network.launch.py.
"""

import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

SOURCE_NAMES = {
    0:   'orbbec_depth',
    1:   'near_rgb_left_side',
    2:   'near_rgb_left_front',
    3:   'near_rgb_right_side',
    4:   'near_rgb_right_front',
    5:   'near_rgb_back_rear',
    6:   'mosaic (all 6)',
    7:   'nvblox map',
    255: '--- STOP ---',
}


class VideoMonitorNode(Node):

    ORG   = '\033[38;5;214m'   # orange
    BOLD  = '\033[1m'
    DIM   = '\033[2m'
    GREEN = '\033[92m'
    RED   = '\033[91m'
    RESET = '\033[0m'

    def __init__(self):
        super().__init__('video_monitor_node')

        self._gui_ip         = None
        self._active_source  = None
        self._active_variant = None
        self._active_quality = None
        self._active_fps     = None
        self._stream_start:  float = 0.0
        self._last_switch:   float = 0.0
        self.log: deque = deque(maxlen=18)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(String, '/network/client_ip',     self._on_client_ip, latched_qos)
        self.create_subscription(String, '/network/stream_request', self._on_request,   qos)

        self.create_timer(1.0, self._render)
        self._render()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _on_client_ip(self, msg: String):
        ip = msg.data.strip()
        if ip:
            self._gui_ip = ip
            self._append_log(f'{self.GREEN}{self.BOLD}CONNECT  {self.RESET}{self.DIM}{ip}{self.RESET}')
        else:
            self._gui_ip = None
            self._active_source = None
            self._append_log(f'{self.RED}{self.BOLD}DISCONNECT{self.RESET}  stream stopped')

    def _on_request(self, msg: String):
        try:
            parts     = msg.data.strip().split(',')
            source_id = int(parts[0])
            variant   = parts[1].upper() if len(parts) > 1 else 'R'
            quality   = int(parts[2])    if len(parts) > 2 else 0
            fps       = int(parts[3])    if len(parts) > 3 else 0
        except (ValueError, IndexError):
            return

        if source_id == 255:
            self._active_source = None
            self._append_log(f'{self.RED}{self.BOLD}STOP     {self.RESET}all streams off')
            return

        name     = SOURCE_NAMES.get(source_id, f'src_{source_id}')
        vname    = 'RGB' if variant == 'R' else 'DEPTH'
        v_color  = self.ORG if variant == 'R' else '\033[38;5;51m'  # cyan for depth

        self._active_source  = source_id
        self._active_variant = variant
        self._active_quality = quality
        self._active_fps     = fps
        self._last_switch    = time.time()
        if not self._stream_start:
            self._stream_start = time.time()

        self._append_log(
            f'{self.ORG}{self.BOLD}STREAM   {self.RESET}'
            f'{self.BOLD}{name:<24}{self.RESET}  '
            f'{v_color}{vname:<5}{self.RESET}  '
            f'q={self.BOLD}{quality}{self.RESET}  '
            f'{fps}fps'
        )

    # ── display ───────────────────────────────────────────────────────────────

    def _append_log(self, body: str):
        ts = time.strftime('%H:%M:%S')
        self.log.append(f'{self.DIM}[{ts}]{self.RESET}  {body}')
        self._render()

    def _render(self):
        gui_str = (
            f'{self.GREEN}{self.BOLD}{self._gui_ip}{self.RESET}'
            if self._gui_ip else
            f'{self.DIM}not connected{self.RESET}'
        )

        if self._active_source is not None:
            name    = SOURCE_NAMES.get(self._active_source, f'src_{self._active_source}')
            vname   = 'RGB' if self._active_variant == 'R' else 'DEPTH'
            v_color = self.ORG if self._active_variant == 'R' else '\033[38;5;51m'
            age     = time.time() - self._last_switch
            src_str = (
                f'{self.BOLD}{name}{self.RESET}  '
                f'{v_color}{self.BOLD}{vname}{self.RESET}  '
                f'q={self.BOLD}{self._active_quality}{self.RESET}  '
                f'{self._active_fps}fps  '
                f'{self.DIM}({age:.0f}s){self.RESET}'
            )
        else:
            src_str = f'{self.DIM}idle — no active stream{self.RESET}'

        print('\033[2J\033[H', end='')
        print(f'{self.BOLD}{self.ORG}{"=" * 62}{self.RESET}')
        print(f'{self.BOLD}{self.ORG}    OCTANE VIDEO STREAM MONITOR{self.RESET}')
        print(f'{self.BOLD}{self.ORG}{"=" * 62}{self.RESET}')
        print(f'    GUI           :  {gui_str}')
        print(f'    UDP port      :  5002')
        print(f'    Now streaming :  {src_str}')
        print()

        print(f'{self.BOLD}    STREAM LOG{self.RESET}')
        print(f'    {self.DIM}{"-" * 56}{self.RESET}')

        if not self.log:
            print(f'    {self.DIM}Waiting for stream requests...{self.RESET}')
        else:
            for entry in self.log:
                print(f'    {entry}')

        print(f'\n{self.BOLD}{self.ORG}{"=" * 62}{self.RESET}')
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = VideoMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
