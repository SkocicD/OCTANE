#!/usr/bin/env python3
"""CAN bus discovery — listen for any frames, then poke each node."""

import time
from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame

NMT_TX = 0x000
PDO_TX = 0x200
SDO_TX = 0x600


def pad(data: bytes) -> bytes:
    return data + b'\x00' * (8 - len(data))


def drain(dev, duration=1.0, label=''):
    deadline = time.time() + duration
    frame = GsUsbFrame()
    count = 0
    while time.time() < deadline:
        if dev.read(frame, 1):
            print(f'  [{label}] RX: {frame}')
            count += 1
    if count == 0:
        print(f'  [{label}] (silence)')


def main():
    devs = GsUsb.scan()
    if not devs:
        print("ERROR: no adapter"); return

    dev = devs[0]
    try:
        dev.gs_usb.detach_kernel_driver(0)
    except Exception:
        pass
    assert dev.set_bitrate(1_000_000), "bitrate failed"
    dev.start()
    print("Adapter open.\n")

    print("=== 1. Passive listen 2s ===")
    drain(dev, 2.0, 'passive')

    print("\n=== 2. NMT broadcast start-all + listen ===")
    dev.send(GsUsbFrame(can_id=NMT_TX, data=pad(b'\x01\x00')))
    drain(dev, 1.0, 'nmt-bcast')

    print("\n=== 3. Init + move node 2 (known working) then listen ===")
    dev.send(GsUsbFrame(can_id=NMT_TX, data=pad(b'\x01\x02')))
    time.sleep(0.05)
    # set mode
    dev.send(GsUsbFrame(can_id=SDO_TX | 2, data=pad(b'\x2f\x00\x61\x02\x0f')))
    time.sleep(0.05)
    # forward
    dev.send(GsUsbFrame(can_id=PDO_TX | 2, data=pad(b'\x01\xa0\x00')))  # ctrl=1, speed=160
    drain(dev, 1.0, 'node2-move')
    # stop
    dev.send(GsUsbFrame(can_id=PDO_TX | 2, data=pad(b'')))

    print("\n=== 4. Probe nodes 0-10 with NMT start + SDO read ===")
    for nid in range(11):
        dev.send(GsUsbFrame(can_id=NMT_TX, data=pad(b'\x01' + nid.to_bytes(1, 'little'))))
        time.sleep(0.02)
        dev.send(GsUsbFrame(can_id=SDO_TX | nid, data=pad(b'\x40\x00\x10\x00')))
        drain(dev, 0.1, f'node{nid}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
