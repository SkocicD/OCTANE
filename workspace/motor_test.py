#!/usr/bin/env python3
"""Standalone motor test — no ROS. Sweeps all 6 motors: forward → stop → reverse.
Node IDs 0-5: 0=front-left, 1=mid-left, 2=back-left, 3=back-right, 4=mid-right, 5=front-right
Right side (3,4,5) polarity is flipped because motors mount mirrored.
"""

import time
from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame

# CANOpen constants
NMT_TX = 0x000
SDO_TX = 0x600
PDO_TX = 0x200

NODE_IDS     = [0, 1, 2, 3, 4, 5]
RIGHT_NODES  = {3, 4, 5}   # polarity flipped

SPEED = 0.4   # 0.0 - 1.0


# ── frame helpers ────────────────────────────────────────────────────────────

def pad(data: bytes) -> bytes:
    return data + b'\x00' * (8 - len(data))

def nmt_start(node_id: int) -> GsUsbFrame:
    return GsUsbFrame(can_id=NMT_TX, data=pad(b'\x01' + node_id.to_bytes(1, 'little')))

def sdo_write(node_id: int, index: bytes, subindex: bytes, value: bytes) -> GsUsbFrame:
    specifier = {1: b'\x2f', 2: b'\x2B', 3: b'\x27', 4: b'\x23'}[len(value)]
    data = specifier + index[::-1] + subindex + value[::-1]
    return GsUsbFrame(can_id=SDO_TX | node_id, data=pad(data))

def pdo_drive(node_id: int, speed: float, reverse: bool) -> GsUsbFrame:
    if node_id in RIGHT_NODES:
        reverse = not reverse
    ctrl = (0x01 | (0x02 if reverse else 0x00)).to_bytes(1, 'little')
    spd  = int(2000 * speed).to_bytes(2, 'little')
    return GsUsbFrame(can_id=PDO_TX | node_id, data=pad(ctrl + spd))

def pdo_stop(node_id: int) -> GsUsbFrame:
    return GsUsbFrame(can_id=PDO_TX | node_id, data=pad(b''))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    devs = GsUsb.scan()
    if not devs:
        print("ERROR: No gs_usb CAN adapter found")
        return

    dev = devs[0]
    try:
        dev.gs_usb.detach_kernel_driver(0)
    except Exception:
        pass  # no kernel driver attached — fine
    if not dev.set_bitrate(1_000_000):
        print("ERROR: Could not set bitrate")
        return
    dev.start()
    print("CAN adapter open at 1 Mbit/s")

    def send(frame):
        dev.send(frame)

    # Init all motors
    print("Initialising motors...")
    for nid in NODE_IDS:
        send(nmt_start(nid))
        time.sleep(0.05)
        # mode: control=1, speed_reg=1, sensory=1, open_loop=1 → 0x0F
        send(sdo_write(nid, b'\x61\x00', b'\x02', b'\x0f'))
        time.sleep(0.05)
    print("Motors initialised")

    print(f"\nForward at speed={SPEED} for 2s...")
    for nid in NODE_IDS:
        send(pdo_drive(nid, SPEED, reverse=False))
    time.sleep(2.0)

    print("Stop for 1s...")
    for nid in NODE_IDS:
        send(pdo_stop(nid))
    time.sleep(1.0)

    print(f"Reverse at speed={SPEED} for 2s...")
    for nid in NODE_IDS:
        send(pdo_drive(nid, SPEED, reverse=True))
    time.sleep(2.0)

    print("Stop.")
    for nid in NODE_IDS:
        send(pdo_stop(nid))

    print("\nDone.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — stopping motors")
