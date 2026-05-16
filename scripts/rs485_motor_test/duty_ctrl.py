#!/usr/bin/env python3
"""Interactive sensorless duty control — type a value, motor runs at that duty.
Usage: python3 duty_ctrl.py [/dev/ttyUSB1]
"""
import struct, serial, time, sys

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB1'
BAUD = 9600
ADDR = 1
PP   = 4

def crc16(d):
    crc = 0xFFFF
    for b in d:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc

def wr(reg, val):
    p = struct.pack('>BBHH', ADDR, 0x06, reg, val)
    f = p + struct.pack('<H', crc16(p))
    port.write(f); r = port.read(8); port.reset_input_buffer()
    return r == f

port = serial.Serial(PORT, BAUD, timeout=0.2)
time.sleep(0.1); port.reset_input_buffer()
wr(0x8004, 0xAA10)   # sensorless
wr(0x8003, 0x0000)   # zero accel
wr(0x8000, (0x08 << 8) | PP)  # stop
time.sleep(0.2)

print("Interactive duty control  (0-255)  |  0 = stop  |  q = quit\n")
current = 0

while True:
    try:
        line = input(f"duty [{current}]> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if line.lower() == 'q':
        break
    if not line:
        continue
    try:
        duty = max(0, min(255, int(line)))
    except ValueError:
        print("  integers only")
        continue
    if duty == 0:
        wr(0x8000, (0x08 << 8) | PP)
        print("  STOPPED")
    else:
        wr(0x8005, duty)
        wr(0x8000, (0x09 << 8) | PP)
        print(f"  running  duty={duty}/255  ({duty/255*100:.1f}%)")
    current = duty

wr(0x8000, (0x08 << 8) | PP)
print("Stopped.")
port.close()
