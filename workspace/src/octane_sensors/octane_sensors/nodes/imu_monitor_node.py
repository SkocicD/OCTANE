#!/usr/bin/env python3
"""IMU monitor node — live display of ADXL345 accelerometer readings."""

import sys
import time
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu

G = 9.80665  # m/s²

# Lime-yellow-green palette (used in the xterm window itself)
LM   = '\033[38;2;170;255;0m'    # #AAFF00 lime
BOLD = '\033[1m'
DIM  = '\033[2m'
RST  = '\033[0m'
RED  = '\033[91m'
YLW  = '\033[93m'
WHT  = '\033[97m'


def _bar(val: float, max_val: float = 20.0, width: int = 20) -> str:
    """ASCII bar: green for low, yellow for mid, red for high."""
    ratio = min(abs(val) / max_val, 1.0)
    filled = round(ratio * width)
    sign = '+' if val >= 0 else '-'
    if ratio < 0.4:
        color = LM
    elif ratio < 0.75:
        color = YLW
    else:
        color = RED
    bar = '█' * filled + '░' * (width - filled)
    return f'{color}{sign}{bar}{RST}'


class ImuMonitorNode(Node):

    def __init__(self):
        super().__init__('imu_monitor_node')

        self._latest: Imu | None = None
        self._last_stamp: float = 0.0
        self._rate_hist: deque = deque(maxlen=20)
        self._prev_stamp: float | None = None
        self._total_rx = 0

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Imu, 'sensors/imu/accel', self._cb, qos)
        self.create_timer(0.2, self._render)  # 5 Hz display
        self._render()

    def _cb(self, msg: Imu):
        self._latest = msg
        self._last_stamp = time.time()
        self._total_rx += 1
        if self._prev_stamp is not None:
            dt = self._last_stamp - self._prev_stamp
            if dt > 0:
                self._rate_hist.append(1.0 / dt)
        self._prev_stamp = self._last_stamp

    def _render(self):
        print('\033[2J\033[H', end='')

        print(f'{LM}{BOLD}{"═" * 52}{RST}')
        print(f'{LM}{BOLD}    OCTANE  ·  IMU ACCELEROMETER  (ADXL345){RST}')
        print(f'{LM}{BOLD}{"═" * 52}{RST}')

        if self._latest is None:
            print(f'\n    {DIM}Waiting for data on sensors/imu/accel ...{RST}\n')
            print(f'{LM}{"═" * 52}{RST}')
            sys.stdout.flush()
            return

        a = self._latest.linear_acceleration
        x, y, z = a.x, a.y, a.z
        mag = math.sqrt(x*x + y*y + z*z)
        age = time.time() - self._last_stamp
        avg_rate = (sum(self._rate_hist) / len(self._rate_hist)) if self._rate_hist else 0.0

        print(f'\n    {DIM}topic  : sensors/imu/accel{RST}')
        print(f'    {DIM}rx     : {self._total_rx} msgs   rate ≈ {avg_rate:.1f} Hz{RST}')
        print(f'    {DIM}age    : {age*1000:.0f} ms{RST}\n')

        print(f'    {BOLD}{WHT}{"Axis":<6}  {"m/s²":>8}   {"g":>7}   {"bar (±20 m/s²)"}{RST}')
        print(f'    {DIM}{"─"*48}{RST}')

        for label, val in (('X', x), ('Y', y), ('Z', z)):
            gval = val / G
            bar = _bar(val)
            print(f'    {LM}{BOLD}{label:<6}{RST}  {val:>+8.3f}   {gval:>+7.4f}   {bar}')

        print()
        print(f'    {DIM}{"─"*48}{RST}')
        print(f'    {WHT}magnitude : {mag:>8.3f} m/s²   ({mag/G:.4f} g){RST}')

        # Tilt indicator (gravity vector vs Z axis)
        tilt_deg = math.degrees(math.acos(min(abs(z) / max(mag, 0.001), 1.0)))
        if tilt_deg < 10:
            tilt_str = f'{LM}{BOLD}level ({tilt_deg:.1f}°){RST}'
        elif tilt_deg < 30:
            tilt_str = f'{YLW}{tilt_deg:.1f}° tilt{RST}'
        else:
            tilt_str = f'{RED}{BOLD}{tilt_deg:.1f}° tilt{RST}'
        print(f'    {DIM}tilt      :{RST} {tilt_str}')

        print(f'\n{LM}{"═" * 52}{RST}')
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = ImuMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
