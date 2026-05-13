#!/usr/bin/env python3
"""BLD-510B direct Modbus RTU test — speed ramp + register dump.

Expects:
  - Motor to spin up smoothly from 0 → MAX_RPM over RAMP_STEPS
  - Actual speed readback (0x8018) to track commanded speed within ~10%
  - Accel/decel register (0x8003) printed — if non-zero the driver adds its
    own ramp ON TOP of any software ramp, causing sluggish response.
    Script zeros it so driver response is immediate.

Usage:
  python3 bld510b_test.py [/dev/rs485_drive]
"""

import struct
import sys
import time
import serial

PORT       = sys.argv[1] if len(sys.argv) > 1 else '/dev/rs485_drive'
BAUD       = 9600
MB_ADDR    = 1
POLE_PAIRS = 4

# Registers
REG_CONTROL  = 0x8000
REG_ACCEL    = 0x8003   # high byte = accel time (0.1s units), low byte = decel time
REG_SPEED    = 0x8005
REG_ACTUAL   = 0x8018
REG_FAULT    = 0x801B

# Control high-byte values (NW=1 = RS485 mode)
CTRL_FWD  = 0x09   # NW EN
CTRL_STOP = 0x08   # NW only

MAX_RPM     = 500    # conservative test ceiling
RAMP_STEPS  = 10
HOLD_SECS   = 1.5


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def write_frame(addr, reg, value):
    p = struct.pack('>BBHH', addr, 0x06, reg, value)
    return p + struct.pack('<H', crc16(p))


def read_frame(addr, reg, count=1):
    p = struct.pack('>BBHH', addr, 0x03, reg, count)
    return p + struct.pack('<H', crc16(p))


def write_reg(port, reg, value):
    frame = write_frame(MB_ADDR, reg, value)
    port.write(frame)
    resp = port.read(8)
    ok = resp == frame
    if not ok:
        port.reset_input_buffer()
    return ok


def read_reg(port, reg):
    port.write(read_frame(MB_ADDR, reg))
    resp = port.read(7)
    if len(resp) == 7 and resp[1] == 0x03:
        return struct.unpack('>H', resp[3:5])[0]
    return None


def actual_rpm(raw):
    # Manual: actual RPM = raw * 20 / pole_pairs
    if raw is None:
        return None
    return raw * 20 / POLE_PAIRS


def stop(port):
    write_reg(port, REG_CONTROL, (CTRL_STOP << 8) | POLE_PAIRS)


print(f'Connecting to {PORT} @ {BAUD} baud...')
port = serial.Serial(PORT, BAUD, timeout=0.2)
time.sleep(0.1)
port.reset_input_buffer()

# ── Read current state ──────────────────────────────────────────────────────
print('\n=== CURRENT DRIVER REGISTERS ===')
accel_raw = read_reg(port, REG_ACCEL)
if accel_raw is not None:
    accel_hi = (accel_raw >> 8) & 0xFF
    accel_lo = accel_raw & 0xFF
    print(f'  0x8003  accel_time={accel_hi * 0.1:.1f}s  decel_time={accel_lo * 0.1:.1f}s', end='')
    if accel_hi or accel_lo:
        print('  ← NON-ZERO: driver ramps internally, doubles with software ramp!')
    else:
        print('  (OK: instant response)')
else:
    print('  0x8003  read failed')

fault_raw = read_reg(port, REG_FAULT)
print(f'  0x801B  fault=0x{fault_raw or 0:04X}  {"OK" if not fault_raw else "FAULT!"}')

actual_raw = read_reg(port, REG_ACTUAL)
print(f'  0x8018  actual_speed={actual_rpm(actual_raw):.0f} RPM' if actual_raw is not None else '  0x8018  read failed')

# ── Zero accel/decel so driver responds instantly ───────────────────────────
print('\n=== ZEROING ACCEL/DECEL (0x8003) ===')
ok = write_reg(port, REG_ACCEL, 0x0000)
print(f'  Write 0x8003 = 0x0000 → {"OK" if ok else "ACK MISMATCH (check wiring)"}')

# ── Send stop first ──────────────────────────────────────────────────────────
print('\n=== SENDING STOP ===')
stop(port)
time.sleep(0.5)

# ── Speed ramp up ────────────────────────────────────────────────────────────
print(f'\n=== RAMP UP  (0 → {MAX_RPM} RPM in {RAMP_STEPS} steps) ===')
print(f'  {"CMD RPM":>8}  {"ACK":>5}  {"ACTUAL RPM":>12}  {"MATCH":>6}')
print(f'  {"-"*8}  {"-"*5}  {"-"*12}  {"-"*6}')

for i in range(1, RAMP_STEPS + 1):
    rpm = int(MAX_RPM * i / RAMP_STEPS)
    ok = write_reg(port, REG_SPEED, rpm)
    write_reg(port, REG_CONTROL, (CTRL_FWD << 8) | POLE_PAIRS)
    time.sleep(0.3)
    actual_raw = read_reg(port, REG_ACTUAL)
    arpm = actual_rpm(actual_raw)
    if arpm is not None:
        match = '✓' if abs(arpm - rpm) / max(rpm, 1) < 0.2 else '✗'
        print(f'  {rpm:>8}  {"OK" if ok else "ERR":>5}  {arpm:>12.0f}  {match:>6}')
    else:
        print(f'  {rpm:>8}  {"OK" if ok else "ERR":>5}  {"read fail":>12}')

# ── Hold at max ───────────────────────────────────────────────────────────────
print(f'\n=== HOLD at {MAX_RPM} RPM for {HOLD_SECS}s ===')
t0 = time.time()
while time.time() - t0 < HOLD_SECS:
    actual_raw = read_reg(port, REG_ACTUAL)
    arpm = actual_rpm(actual_raw)
    elapsed = time.time() - t0
    print(f'  t={elapsed:.1f}s  actual={arpm:.0f} RPM' if arpm is not None else f'  t={elapsed:.1f}s  read fail')
    time.sleep(0.3)

# ── Ramp down ─────────────────────────────────────────────────────────────────
print(f'\n=== RAMP DOWN  ({MAX_RPM} → 0 RPM) ===')
for i in range(RAMP_STEPS - 1, -1, -1):
    rpm = int(MAX_RPM * i / RAMP_STEPS)
    if rpm == 0:
        stop(port)
    else:
        write_reg(port, REG_SPEED, rpm)
        write_reg(port, REG_CONTROL, (CTRL_FWD << 8) | POLE_PAIRS)
    time.sleep(0.3)
    actual_raw = read_reg(port, REG_ACTUAL)
    arpm = actual_rpm(actual_raw)
    print(f'  cmd={rpm:>5}  actual={arpm:.0f} RPM' if arpm is not None else f'  cmd={rpm:>5}  read fail')

stop(port)
port.close()
print('\n=== DONE — motor stopped ===')
print("""
WHAT TO EXPECT:
  - Actual RPM tracks commanded RPM within ~10-20% (closed-loop Hall control)
  - If actual RPM is ~150 for all low commands → driver minimum speed floor active
  - If actual RPM is always near MAX regardless of command → open-loop mode (jumpers removed)
  - If ACK MISMATCH on every write → wiring issue or wrong baud rate
  - If 0x8003 was non-zero → that was causing double-ramping / slow acceleration
""")
