#!/usr/bin/env python3
"""Lean TCP protocol for OCTANE ground station comm.

Frame format (minimal):
  [MAGIC:1B][TYPE:1B][LEN:1B][PAYLOAD:N][CRC:1B]
  Total overhead: 4 bytes + variable payload

Types:
  T = Telemetry (0x54)
  C = Command (0x43)
  A = Ack (0x41)
  F = Fault (0x46)

Modes (single char):
  0 = Standby
  1 = Manual
  2 = Autonomous
  3 = Fault Reset

States (single char):
  0 = STANDBY
  1 = MANUAL
  2 = AUTONOMOUS
  3 = FAULT

Fault severities:
  0 = Info
  1 = Warning
  2 = Critical
"""

import struct
import zlib
from enum import IntEnum
from typing import Optional, Dict, Any

# Constants
MAGIC = 0x4F  # 'O' for OCTANE
HEADER_SIZE = 3  # magic + type + length
CRC_SIZE = 1  # 8-bit CRC (good enough for short messages)

# Message types (ASCII for debugging)
TYPE_TELEMETRY = ord('T')  # 0x54
TYPE_COMMAND = ord('C')    # 0x43
TYPE_ACK = ord('A')        # 0x41
TYPE_FAULT = ord('F')      # 0x46

# Mode/state codes
MODE_STANDBY = b'0'
MODE_MANUAL = b'1'
MODE_AUTONOMOUS = b'2'
MODE_FAULT_RESET = b'3'

# Severity codes
SEV_INFO = b'0'
SEV_WARNING = b'1'
SEV_CRITICAL = b'2'


def crc8(data: bytes) -> int:
    """Simple CRC-8 for small messages."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_telemetry(state: str, fault: Optional[str] = None,
                     battery: Optional[float] = None) -> bytes:
    """Encode telemetry: T + state_char + optional fault/battery.

    Example: T1 = Manual, T2 = Autonomous
    With fault: T1|battery_fault| (fault char appended)

    Wire format: [O][T][n][state][optional data][crc]
    """
    # Map state string to single char
    state_map = {'STANDBY': b'0', 'MANUAL': b'1',
                 'AUTONOMOUS': b'2', 'FAULT': b'3'}
    state_byte = state_map.get(state, b'0')

    # Build payload
    payload = state_byte
    if battery is not None:
        # Pack battery as float (4 bytes) with marker
        payload += b'B' + struct.pack('<f', battery)
    if fault:
        # Truncate fault name to single char for now (can expand later)
        payload += b'F' + fault[:1].encode()

    # Header + payload
    header = struct.pack('!BBB', MAGIC, TYPE_TELEMETRY, len(payload))
    frame = header + payload

    # Add CRC
    crc = crc8(frame)
    return frame + bytes([crc])


def encode_command(mode: str, estop: bool = False) -> bytes:
    """Encode command: C + mode_char + estop_flag.

    Wire format: [O][C][3][mode][estop][crc]
    Always 5 bytes on wire.
    """
    mode_map = {'standby': MODE_STANDBY, 'manual': MODE_MANUAL,
                'autonomous': MODE_AUTONOMOUS, 'fault_reset': MODE_FAULT_RESET}
    mode_byte = mode_map.get(mode.lower(), MODE_STANDBY)

    estop_byte = b'1' if estop else b'0'
    payload = mode_byte + estop_byte

    header = struct.pack('!BBB', MAGIC, TYPE_COMMAND, len(payload))
    frame = header + payload

    crc = crc8(frame)
    return frame + bytes([crc])


def encode_ack(success: bool) -> bytes:
    """Encode ACK: A + success_flag.

    Wire format: [O][A][1][0/1][crc]
    Always 4 bytes on wire.
    """
    success_byte = b'1' if success else b'0'
    payload = success_byte

    header = struct.pack('!BBB', MAGIC, TYPE_ACK, len(payload))
    frame = header + payload

    crc = crc8(frame)
    return frame + bytes([crc])


def encode_fault(fault_type: str, severity: str) -> bytes:
    """Encode fault alert: F + severity + fault_char.

    Wire format: [O][F][3][severity][fault_char][crc]
    Always 5 bytes on wire.
    """
    sev_map = {'info': SEV_INFO, 'warning': SEV_WARNING, 'critical': SEV_CRITICAL}
    sev_byte = sev_map.get(severity.lower(), SEV_INFO)

    fault_byte = fault_type[:1].encode()  # First char of fault name

    payload = sev_byte + fault_byte
    header = struct.pack('!BBB', MAGIC, TYPE_FAULT, len(payload))
    frame = header + payload

    crc = crc8(frame)
    return frame + bytes([crc])


def decode_message(data: bytes) -> Optional[Dict[str, Any]]:
    """Decode message frame.

    Returns dict with keys: type, success, or fault data
    Returns None if incomplete or invalid CRC.
    """
    if len(data) < HEADER_SIZE:
        return None

    # Parse header
    magic, msg_type, payload_len = struct.unpack('!BBB', data[:HEADER_SIZE])

    if magic != MAGIC:
        return None

    expected_len = HEADER_SIZE + payload_len + CRC_SIZE
    if len(data) < expected_len:
        return None

    # Extract and verify CRC
    frame = data[:expected_len - CRC_SIZE]
    received_crc = data[expected_len - 1]
    calc_crc = crc8(frame)

    if received_crc != calc_crc:
        return None  # CRC failure

    # Extract payload
    payload = data[HEADER_SIZE:HEADER_SIZE + payload_len]

    # Decode based on type
    if msg_type == TYPE_TELEMETRY:
        state_map = {b'0': 'STANDBY', b'1': 'MANUAL',
                     b'2': 'AUTONOMOUS', b'3': 'FAULT'}
        state_byte = payload[0:1]
        state = state_map.get(state_byte, 'STANDBY')

        result = {'type': 'telemetry', 'state': state}

        # Parse optional fields
        idx = 1
        while idx < len(payload):
            marker = payload[idx:idx+1]
            if marker == b'B' and idx + 5 <= len(payload):
                battery = struct.unpack('<f', payload[idx+1:idx+5])[0]
                result['battery'] = battery
                idx += 5
            elif marker == b'F' and idx + 1 < len(payload):
                fault_char = payload[idx+1:idx+2].decode()
                result['fault'] = fault_char
                idx += 2
            else:
                idx += 1

        return result

    elif msg_type == TYPE_COMMAND:
        if len(payload) >= 2:
            mode_map = {MODE_STANDBY: 'standby', MODE_MANUAL: 'manual',
                       MODE_AUTONOMOUS: 'autonomous', MODE_FAULT_RESET: 'fault_reset'}
            mode_byte = payload[0:1]
            mode = mode_map.get(mode_byte, 'standby')
            estop = payload[1:2] == b'1'

            return {'type': 'command', 'mode': mode, 'estop': estop}

    elif msg_type == TYPE_ACK:
        if len(payload) >= 1:
            return {'type': 'ack', 'success': payload[0:1] == b'1'}

    elif msg_type == TYPE_FAULT:
        if len(payload) >= 2:
            sev_map = {SEV_INFO: 'info', SEV_WARNING: 'warning',
                      SEV_CRITICAL: 'critical'}
            sev_byte = payload[0:1]
            severity = sev_map.get(sev_byte, 'info')
            fault_char = payload[1:2].decode()

            return {'type': 'fault', 'severity': severity, 'fault': fault_char}

    return None
