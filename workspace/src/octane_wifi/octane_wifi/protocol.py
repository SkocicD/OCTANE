#!/usr/bin/env python3
"""TCP protocol encoder/decoder for OCTANE ground station communication.

Message frame format:
  [MAGIC:2B][TYPE:1B][SEQ:2B][LENGTH:4B][PAYLOAD:N bytes][CRC32:4B]

Types:
  0x01 = Telemetry (rover -> ground)
  0x02 = Command (ground -> rover)
  0x03 = Command ACK (rover -> ground)
  0x04 = Fault Alert (rover -> ground, high priority)
"""

import struct
import json
import zlib
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import IntEnum


class MessageType(IntEnum):
    """TCP message types."""
    TELEMETRY = 0x01
    COMMAND = 0x02
    COMMAND_ACK = 0x03
    FAULT_ALERT = 0x04


# Magic bytes for OCTANE protocol
MAGIC = b'OT'
HEADER_SIZE = 9  # 2 (magic) + 1 (type) + 2 (seq) + 4 (length)
CRC_SIZE = 4


@dataclass
class OctaneMessage:
    """Parsed OCTANE protocol message."""
    msg_type: MessageType
    seq: int
    payload: Dict[str, Any]
    crc_valid: bool = True


def encode_message(msg_type: MessageType, payload: Dict[str, Any], seq: int = 0) -> bytes:
    """Encode a message to wire format.

    Args:
        msg_type: Message type (Telemetry, Command, etc.)
        payload: Dictionary to serialize as JSON
        seq: Sequence number (auto-increment if 0)

    Returns:
        Complete message frame ready for TCP socket
    """
    # Serialize payload to compact JSON
    json_str = json.dumps(payload, separators=(',', ':'))
    payload_bytes = json_str.encode('utf-8')

    # Build header
    header = struct.pack(
        '!2sBHI',
        MAGIC,
        msg_type.value,
        seq,
        len(payload_bytes)
    )

    # Calculate CRC32
    crc = zlib.crc32(header + payload_bytes) & 0xFFFFFFFF

    return header + payload_bytes + struct.pack('!I', crc)


def decode_message(data: bytes) -> Optional[OctaneMessage]:
    """Decode a message from wire format.

    Args:
        data: Raw bytes from TCP socket (may contain partial messages)

    Returns:
        OctaneMessage if complete valid message found, None otherwise
    """
    if len(data) < HEADER_SIZE:
        return None

    # Parse header
    magic, msg_type, seq, payload_len = struct.unpack('!2sBHI', data[:HEADER_SIZE])

    if magic != MAGIC:
        return None

    if len(data) < HEADER_SIZE + payload_len + CRC_SIZE:
        return None  # Need more data

    # Extract payload and CRC
    payload_bytes = data[HEADER_SIZE:HEADER_SIZE + payload_len]
    received_crc = struct.unpack('!I', data[HEADER_SIZE + payload_len:HEADER_SIZE + payload_len + CRC_SIZE])[0]

    # Verify CRC
    calculated_crc = zlib.crc32(data[:HEADER_SIZE + payload_len]) & 0xFFFFFFFF
    crc_valid = (received_crc == calculated_crc)

    # Parse JSON payload
    try:
        payload = json.loads(payload_bytes.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    return OctaneMessage(
        msg_type=MessageType(msg_type),
        seq=seq,
        payload=payload,
        crc_valid=crc_valid
    )


def create_telemetry_packet(state: str, fault: Optional[str] = None,
                           battery: Optional[float] = None, seq: int = 0) -> bytes:
    """Create a telemetry packet.

    Args:
        state: Current state (STANDBY, MANUAL, AUTONOMOUS, FAULT)
        fault: Active fault type if any
        battery: Battery voltage if available
        seq: Sequence number

    Returns:
        Encoded message frame
    """
    payload = {'t': state}
    if fault:
        payload['f'] = fault
    if battery is not None:
        payload['b'] = battery
    return encode_message(MessageType.TELEMETRY, payload, seq)


def create_command_packet(mode: str, estop: bool = False, seq: int = 0) -> bytes:
    """Create a command packet from ground station.

    Args:
        mode: Target mode (standby, manual, autonomous, fault_reset)
        estop: Emergency stop flag
        seq: Sequence number

    Returns:
        Encoded message frame
    """
    payload = {'m': mode, 'e': estop}
    return encode_message(MessageType.COMMAND, payload, seq)


def create_command_ack(success: bool, seq: int = 0) -> bytes:
    """Create a command acknowledgment.

    Args:
        success: Whether command was executed successfully
        seq: Original command sequence number

    Returns:
        Encoded message frame
    """
    payload = {'s': success, 'seq': seq}
    return encode_message(MessageType.COMMAND_ACK, payload, 0)


def create_fault_alert(fault_type: str, severity: str) -> bytes:
    """Create a high-priority fault alert.

    Args:
        fault_type: Type of fault (e.g., "battery_undervoltage")
        severity: Fault severity (critical, warning, info)

    Returns:
        Encoded message frame
    """
    payload = {'f': fault_type, 's': severity}
    return encode_message(MessageType.FAULT_ALERT, payload, 0)
