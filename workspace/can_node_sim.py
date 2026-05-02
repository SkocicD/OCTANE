#!/usr/bin/env python3
"""Standalone simulation of can_drive_node logic.
Mirrors the exact init + PDO sequence the ROS node runs.
Runs without sudo (requires udev rule from setup_can.sh).

Usage:
  python3 can_node_sim.py          # forward 2s, stop 1s, reverse 2s
  python3 can_node_sim.py --init   # init only, no movement (check if frames go out)
"""

import sys
import time
from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame

# ── protocol constants (mirror can_open_handler.py) ──────────────────────────
NMT_TX = 0x000
SDO_TX = 0x600
PDO_TX = 0x200

LEFT_IDS  = [0, 1, 2]
RIGHT_IDS = [3, 4, 5]
ALL_IDS   = LEFT_IDS + RIGHT_IDS

MOTOR_MODE_INDEX    = b'\x61\x00'
MOTOR_MODE_SUBINDEX = b'\x02'
MOTOR_MODE_VALUE    = b'\x0f'

MIN_RAW = 100
MAX_RAW = 2000


def pad(data: bytes) -> bytes:
    return data + b'\x00' * (8 - len(data))

def hex_str(data) -> str:
    return ' '.join(f'{b:02x}' for b in data)

def send(dev, can_id: int, data: bytes, label: str):
    frame = GsUsbFrame(can_id=can_id, data=data)
    dev.send(frame)
    print(f'  TX [{label}]  id=0x{can_id:03X}  data=[{hex_str(data)}]')

def nmt_start(dev, node_id: int):
    data = pad(b'\x01' + node_id.to_bytes(1, 'little'))
    send(dev, NMT_TX, data, f'NMT start node={node_id}')

def sdo_set_mode(dev, node_id: int):
    # specifier 0x2f = 1-byte SDO write
    data = pad(b'\x2f' + MOTOR_MODE_INDEX[::-1] + MOTOR_MODE_SUBINDEX + MOTOR_MODE_VALUE[::-1])
    send(dev, SDO_TX | node_id, data, f'SDO mode node={node_id}')

def pdo_drive(dev, node_id: int, speed: float, reverse: bool):
    right_side = node_id > 2
    if right_side:
        reverse = not reverse
    raw = int(MAX_RAW * speed)
    if raw < MIN_RAW:
        pdo_stop(dev, node_id)
        return
    ctrl = (0x01 | (0x02 if reverse else 0x00)).to_bytes(1, 'little')
    spd  = raw.to_bytes(2, 'little')
    data = pad(ctrl + spd)
    dir_str = 'REV' if reverse else 'FWD'
    send(dev, PDO_TX | node_id, data, f'PDO drive node={node_id} spd={raw} {dir_str}')

def pdo_stop(dev, node_id: int):
    data = pad(b'')
    send(dev, PDO_TX | node_id, data, f'PDO stop  node={node_id}')


def main():
    init_only = '--init' in sys.argv

    print('Opening CAN adapter (no sudo — requires udev rule)...')
    devs = GsUsb.scan()
    if not devs:
        print('ERROR: no gs_usb adapter found')
        sys.exit(1)

    dev = devs[0]
    try:
        dev.gs_usb.detach_kernel_driver(0)
    except Exception:
        pass
    if not dev.set_bitrate(1_000_000):
        print('ERROR: could not set bitrate')
        sys.exit(1)
    dev.start()
    print('Adapter open at 1 Mbit/s\n')

    # ── Init sequence (mirrors can_drive_node.__init__) ───────────────────────
    print('=== INIT ===')
    for nid in ALL_IDS:
        nmt_start(dev, nid)
        time.sleep(0.05)
        sdo_set_mode(dev, nid)
        time.sleep(0.05)
    print('Init done.\n')

    if init_only:
        print('--init flag set, stopping here.')
        return

    # ── Drive sequence ────────────────────────────────────────────────────────
    SPEED = 0.4

    print(f'=== FORWARD speed={SPEED} for 2s ===')
    for nid in ALL_IDS:
        pdo_drive(dev, nid, SPEED, reverse=False)
    time.sleep(2.0)

    print('\n=== STOP for 1s ===')
    for nid in ALL_IDS:
        pdo_stop(dev, nid)
    time.sleep(1.0)

    print('\n=== REVERSE speed={SPEED} for 2s ===')
    for nid in ALL_IDS:
        pdo_drive(dev, nid, SPEED, reverse=True)
    time.sleep(2.0)

    print('\n=== STOP ===')
    for nid in ALL_IDS:
        pdo_stop(dev, nid)

    print('\nDone.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted')
    except PermissionError as e:
        print(f'\nPERMISSION ERROR: {e}')
        print('Run setup_can.sh then unplug/replug the adapter, or use sudo.')
