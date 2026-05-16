#!/usr/bin/env python3
"""BLD-510B direct Modbus RTU test — sensorless duty-cycle ramp.

Hall sensors not connected → sensorless mode (0x8004=0xAA10).
REG_SPEED (0x8005) accepts 0-255 duty cycle in sensorless/open-loop mode.

Usage:
  python3 bld510b_test.py [/dev/rs485_drive]
"""

import struct
import sys
import time
import serial

PORT      = sys.argv[1] if len(sys.argv) > 1 else '/dev/rs485_drive'
BAUD      = 9600
MB_ADDR   = 1
POLE_PAIRS = 4

REG_CONTROL = 0x8000
REG_ACCEL   = 0x8003
REG_MODEL   = 0x8004
REG_SPEED   = 0x8005
REG_ACTUAL  = 0x8018
REG_FAULT   = 0x801B

CTRL_FWD  = 0x09   # NW=1 EN=1 FR=0
CTRL_STOP = 0x08   # NW=1 EN=0

# Sensorless: REG_SPEED is 0-255 duty cycle.
# 60/255 ≈ 24% duty — gentle test ceiling; raise once confirmed working.
MAX_DUTY   = 60
RAMP_STEPS = 6
HOLD_SECS  = 2.0

MODEL_SENSORED   = 0x0F
MODEL_SENSORLESS = 0x10

FAULT_NAMES = {0x01: 'Locked rotor', 0x02: 'Over-current', 0x04: 'Hall abnormal',
               0x08: 'Bus voltage low', 0x10: 'Bus voltage high', 0x20: 'Current peak'}


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


def fault_str(v):
    if v is None:
        return 'READ FAIL'
    bits = [n for k, n in FAULT_NAMES.items() if v & k]
    return f'0x{v:04X}  {", ".join(bits)}' if bits else f'0x{v:04X}  CLEAR'


def stop(port):
    write_reg(port, REG_CONTROL, (CTRL_STOP << 8) | POLE_PAIRS)


print(f'Connecting to {PORT} @ {BAUD} baud...')
port = serial.Serial(PORT, BAUD, timeout=0.2)
time.sleep(0.1)
port.reset_input_buffer()

# ── Enable RS485 mode (NW=1) ─────────────────────────────────────────────────
print('\n=== ENABLE RS485 MODE ===')
ok = write_reg(port, REG_CONTROL, (CTRL_STOP << 8) | POLE_PAIRS)
print(f'  NW=1 write → {"OK" if ok else "no echo (may still work)"}')
time.sleep(0.2)

# ── Read state ────────────────────────────────────────────────────────────────
print('\n=== DRIVER STATE ===')
print(f'  0x801B  fault  = {fault_str(read_reg(port, REG_FAULT))}')
model_raw = read_reg(port, REG_MODEL)
if model_raw is not None:
    mode = model_raw & 0xFF
    mode_str = 'SENSORED' if mode == MODEL_SENSORED else 'SENSORLESS' if mode == MODEL_SENSORLESS else hex(mode)
    print(f'  0x8004  model  = 0x{model_raw:04X}  ({mode_str})')
accel_raw = read_reg(port, REG_ACCEL)
if accel_raw is not None:
    print(f'  0x8003  accel  = {((accel_raw >> 8) & 0xFF) * 0.1:.1f}s  decel={(accel_raw & 0xFF) * 0.1:.1f}s')
print(f'  0x8018  actual = {read_reg(port, REG_ACTUAL)}')

# ── Switch to sensorless ──────────────────────────────────────────────────────
print('\n=== SENSORLESS MODE (Hall sensors not connected) ===')
ok = write_reg(port, REG_MODEL, (0xAA << 8) | MODEL_SENSORLESS)
print(f'  0x8004 = 0xAA10 → {"OK" if ok else "FAILED"}')

# ── Clear fault: drop NW=0 first, then restore NW=1 ──────────────────────────
# NW=0 fully releases EN (equivalent to physically disconnecting EN pin).
# This is the only software path to clear a sticky locked-rotor fault.
print('\n=== FAULT CLEAR (NW=0 → NW=1) ===')
write_reg(port, REG_CONTROL, 0x0000)   # NW=0: external IO mode, EN released
time.sleep(0.8)
write_reg(port, REG_CONTROL, (CTRL_STOP << 8) | POLE_PAIRS)  # NW=1 EN=0
time.sleep(0.2)
fault_raw = read_reg(port, REG_FAULT)
print(f'  fault after clear: {fault_str(fault_raw)}')

# Locked rotor or overcurrent after fault clear = need physical power cycle
if fault_raw and (fault_raw & 0x03):
    print('  !! Locked rotor / overcurrent persists — power-cycle the BLD-510B !!')
    stop(port)
    port.close()
    raise SystemExit(1)

# ── Zero accel so driver responds instantly ───────────────────────────────────
print('\n=== ZERO ACCEL/DECEL ===')
ok = write_reg(port, REG_ACCEL, 0x0000)
print(f'  0x8003 = 0x0000 → {"OK" if ok else "no echo"}')

stop(port)
time.sleep(0.3)

# ── Duty ramp ─────────────────────────────────────────────────────────────────
# In sensorless mode 0x8005 is duty cycle 0-255 (NOT RPM).
# actual_raw (0x8018) is back-EMF estimation — noisy but directionally useful.
print(f'\n=== RAMP UP duty 0→{MAX_DUTY}/255 ===')
for i in range(1, RAMP_STEPS + 1):
    duty = int(MAX_DUTY * i / RAMP_STEPS)
    ok_s = write_reg(port, REG_SPEED,   duty)
    ok_c = write_reg(port, REG_CONTROL, (CTRL_FWD << 8) | POLE_PAIRS)
    time.sleep(0.5)
    raw   = read_reg(port, REG_ACTUAL)
    fault = read_reg(port, REG_FAULT)
    print(f'  duty={duty:>3}/255 ({duty/255*100:>4.1f}%)  spd={"OK" if ok_s else "ERR"}'
          f'  ctrl={"OK" if ok_c else "ERR"}  raw={raw}  fault={fault_str(fault)}')
    if fault and (fault & 0x03):
        print('  !! Fault — aborting ramp')
        break

# ── Hold ──────────────────────────────────────────────────────────────────────
print(f'\n=== HOLD duty={MAX_DUTY} for {HOLD_SECS}s ===')
t0 = time.time()
while time.time() - t0 < HOLD_SECS:
    raw   = read_reg(port, REG_ACTUAL)
    fault = read_reg(port, REG_FAULT)
    print(f'  t={time.time()-t0:.1f}s  raw={raw}  fault={fault_str(fault)}')
    time.sleep(0.4)

# ── Ramp down ─────────────────────────────────────────────────────────────────
print(f'\n=== RAMP DOWN duty {MAX_DUTY}→0 ===')
for i in range(RAMP_STEPS - 1, -1, -1):
    duty = int(MAX_DUTY * i / RAMP_STEPS)
    if duty == 0:
        stop(port)
    else:
        write_reg(port, REG_SPEED,   duty)
        write_reg(port, REG_CONTROL, (CTRL_FWD << 8) | POLE_PAIRS)
    time.sleep(0.3)
    raw = read_reg(port, REG_ACTUAL)
    print(f'  duty={duty:>3}/255  raw={raw}')

stop(port)
port.close()
print('\n=== DONE ===')
