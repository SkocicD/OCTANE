#!/usr/bin/env python3
"""Lean TCP protocol for OCTANE ground station comm.

Frame format (minimal):
  [MAGIC:1B][TYPE:1B][LEN:1B][PAYLOAD:N][CRC:1B]
  Total overhead: 4 bytes + variable payload

Types:
  T = Telemetry     (0x54)
  C = Command       (0x43)
  A = Ack           (0x41)
  F = Fault         (0x46)
  H = Heartbeat     (0x48)
  V = VideoRequest  (0x56)

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
TYPE_TELEMETRY   = ord('T')   # 0x54
TYPE_COMMAND     = ord('C')   # 0x43
TYPE_ACK         = ord('A')   # 0x41
TYPE_FAULT       = ord('F')   # 0x46
TYPE_MANIPULATOR    = ord('M')   # 0x4D
TYPE_HEARTBEAT      = ord('H')   # 0x48
TYPE_VIDEO_REQUEST  = ord('V')   # 0x56

# Video source IDs (match cameras.yaml order)
VIDEO_SRC_ORBBEC       = 0
VIDEO_SRC_LEFT_SIDE    = 1
VIDEO_SRC_LEFT_FRONT   = 2
VIDEO_SRC_RIGHT_SIDE   = 3
VIDEO_SRC_RIGHT_FRONT  = 4
VIDEO_SRC_BACK_REAR    = 5
VIDEO_SRC_MOSAIC       = 6   # all 6 cameras tiled
VIDEO_SRC_MAP          = 7   # nvblox ESDF slice
VIDEO_SRC_FAR_FRONT    = 8   # ESP32 localization camera
VIDEO_SRC_FAR_RIGHT    = 9
VIDEO_SRC_FAR_BACK     = 10
VIDEO_SRC_FAR_LEFT     = 11
VIDEO_SRC_STOP         = 0xFF

# Video variant codes
VIDEO_VARIANT_RGB   = ord('R')   # 0x52
VIDEO_VARIANT_DEPTH = ord('D')   # 0x44

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
                     battery: Optional[float] = None,
                     accel: Optional[tuple] = None,
                     pose: Optional[tuple] = None,
                     tags: Optional[list] = None) -> bytes:
    """Encode telemetry: T + state_char + optional fields.

    Wire format: [O][T][n][state][optional fields][crc]

    Optional field markers:
      B + float32LE        — battery voltage (volts)
      F + char             — active fault code
      I + 3×float32LE     — accelerometer x, y, z (m/s²)
      L + 3×float32LE     — localization pose: x (m), y (m), theta (rad)
      G + count + count×(uint8 tag_id + 2×float32LE dist_m, angle_deg)
                           — AprilTag observations (up to 3)
    """
    state_map = {'STANDBY': b'0', 'MANUAL': b'1',
                 'AUTONOMOUS': b'2', 'FAULT': b'3'}
    state_byte = state_map.get(state, b'0')

    payload = state_byte
    if battery is not None:
        payload += b'B' + struct.pack('<f', battery)
    if fault:
        payload += b'F' + fault[:1].encode()
    if accel is not None:
        ax, ay, az = accel
        payload += b'I' + struct.pack('<fff', ax, ay, az)
    if pose is not None:
        x, y, theta = pose
        payload += b'L' + struct.pack('<fff', x, y, theta)
    if tags:
        capped = tags[:3]
        payload += b'G' + bytes([len(capped)])
        for t in capped:
            payload += struct.pack('<Bff', t['id'] & 0xFF, t['dist'], t['angle_deg'])

    header = struct.pack('!BBB', MAGIC, TYPE_TELEMETRY, len(payload))
    frame = header + payload
    return frame + bytes([crc8(frame)])


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


def encode_heartbeat(state: str, seq: int) -> bytes:
    """Encode heartbeat: H + state_char + seq_hi + seq_lo.

    Wire format: [O][H][3][state][seq_hi][seq_lo][crc]  — 7 bytes total.
    """
    state_map = {'STANDBY': b'0', 'MANUAL': b'1', 'AUTONOMOUS': b'2', 'FAULT': b'3'}
    state_byte = state_map.get(state, b'0')
    seq_hi = (seq >> 8) & 0xFF
    seq_lo = seq & 0xFF
    payload = state_byte + bytes([seq_hi, seq_lo])
    header = struct.pack('!BBB', MAGIC, TYPE_HEARTBEAT, len(payload))
    frame = header + payload
    return frame + bytes([crc8(frame)])


def encode_manipulator(bitfield: int, speed_modifier: int = 100) -> bytes:
    """Encode a manipulator (key state + speed dial) frame.

    Wire format: [O][M][3][bitfield][speed_hi][speed_lo][CRC]  — 7 bytes total.

    bitfield:      8-bit key state (W=bit0, A=bit1, S=bit2, D=bit3, arrows=bits4-7)
    speed_modifier: 0-500 integer percentage (100 = 1.0x, default; 500 = 5.0x max)
    """
    speed_modifier = max(0, min(500, speed_modifier))
    payload = bytes([bitfield & 0xFF, (speed_modifier >> 8) & 0xFF, speed_modifier & 0xFF])
    header = struct.pack('!BBB', MAGIC, TYPE_MANIPULATOR, len(payload))
    frame = header + payload
    return frame + bytes([crc8(frame)])


def encode_video_request(source_id: int, variant: str, quality: int, fps: int) -> bytes:
    """Encode a video stream request.

    Wire format: [O][V][4][source_id][variant][quality][fps][CRC]  — 8 bytes total.

    source_id: 0-5 = individual camera (cameras.yaml order), 6 = mosaic,
               7 = map, 0xFF = stop all streams
    variant:   'R' = RGB, 'D' = depth heatmap
    quality:   1-100 (JPEG encode quality); 0 = use server default
    fps:       1-30 (target frame rate)
    """
    variant_byte = ord(variant.upper()[0]) if variant else VIDEO_VARIANT_RGB
    payload = bytes([
        source_id & 0xFF,
        variant_byte,
        max(0, min(100, quality)),
        max(1, min(30, fps)),
    ])
    header = struct.pack('!BBB', MAGIC, TYPE_VIDEO_REQUEST, len(payload))
    frame = header + payload
    return frame + bytes([crc8(frame)])


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
            elif marker == b'F' and idx + 2 <= len(payload):
                fault_char = payload[idx+1:idx+2].decode()
                result['fault'] = fault_char
                idx += 2
            elif marker == b'I' and idx + 13 <= len(payload):
                ax, ay, az = struct.unpack('<fff', payload[idx+1:idx+13])
                result['accel'] = (ax, ay, az)
                idx += 13
            elif marker == b'L' and idx + 13 <= len(payload):
                x, y, theta = struct.unpack('<fff', payload[idx+1:idx+13])
                result['pose'] = (x, y, theta)
                idx += 13
            elif marker == b'G' and idx + 1 <= len(payload):
                count = payload[idx+1]
                needed = idx + 2 + count * 9
                if needed <= len(payload):
                    obs = []
                    for i in range(count):
                        base = idx + 2 + i * 9
                        tag_id, dist, angle_deg = struct.unpack('<Bff', payload[base:base+9])
                        obs.append({'id': tag_id, 'dist': dist, 'angle_deg': angle_deg})
                    result['tags'] = obs
                    idx = needed
                else:
                    idx += 1
            else:
                idx += 1

        return result

    elif msg_type == TYPE_MANIPULATOR:
        if len(payload) >= 1:
            bitfield = payload[0]
            speed_modifier = 100  # default: 1.0x — preserves old single-byte frames
            if len(payload) >= 3:
                speed_modifier = (payload[1] << 8) | payload[2]
                speed_modifier = max(0, min(500, speed_modifier))
            return {'type': 'manipulator', 'bitfield': bitfield, 'speed_modifier': speed_modifier}

    elif msg_type == TYPE_COMMAND:
        if len(payload) >= 2:
            mode_map = {MODE_STANDBY: 'standby', MODE_MANUAL: 'manual',
                       MODE_AUTONOMOUS: 'autonomous', MODE_FAULT_RESET: 'fault_reset'}
            mode_byte = payload[0:1]
            mode = mode_map.get(mode_byte, 'standby')
            estop = payload[1:2] == b'1'

            return {'type': 'command', 'mode': mode, 'estop': estop}

    elif msg_type == TYPE_VIDEO_REQUEST:
        if len(payload) >= 4:
            source_id    = payload[0]
            variant_byte = payload[1]
            quality      = payload[2]  # 1-100 JPEG quality; 0 = use server default
            fps          = payload[3]
            variant      = 'R' if variant_byte == VIDEO_VARIANT_RGB else 'D'
            return {
                'type':      'video_request',
                'source_id': source_id,
                'variant':   variant,
                'quality':   quality,
                'fps':       fps,
            }

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
