#!/usr/bin/env python3
"""Heartbeat sender for OCTANE network communication.

Sends periodic UDP heartbeat packets to indicate rover is alive.
Used by GUI to detect connection loss without TCP dependency.

Topics:
  /supervisor/state (subscription) - To include current state in heartbeat

Publishes:
  UDP multicast/unicast on configured port with minimal heartbeat frame
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import socket
import struct
from typing import Optional

# Minimal heartbeat frame format:
# [MAGIC:1B][STATE:1B][SEQ:2B][CRC:1B] = 5 bytes total
# Magic: 0x4F ('O')
# State: '0'=STANDBY, '1'=MANUAL, '2'=AUTONOMOUS, '3'=FAULT
# Seq: 16-bit sequence number (wraps at 65535)
# CRC: 8-bit CRC of first 4 bytes

MAGIC = 0x4F
DEFAULT_HOST = "255.255.255.255"  # Broadcast by default
DEFAULT_PORT = 5001  # Separate port from TCP command port
DEFAULT_RATE_HZ = 0.3333  # 3 second interval between heartbeats


def calc_crc8(data: bytes) -> int:
    """Calculate CRC-8 using polynomial 0x07."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def encode_heartbeat(state_char: str, seq_num: int) -> bytes:
    """Encode minimal heartbeat frame.

    Format: [MAGIC][STATE][SEQ_HI][SEQ_LO][CRC]
    Total: 5 bytes
    """
    magic = MAGIC
    state = ord(state_char)
    seq_hi = (seq_num >> 8) & 0xFF
    seq_lo = seq_num & 0xFF

    # Calculate CRC over first 4 bytes
    data = bytes([magic, state, seq_hi, seq_lo])
    crc = calc_crc8(data)

    return data + bytes([crc])


class HeartbeatSenderNode(Node):
    """Node that sends periodic UDP heartbeat packets."""

    def __init__(self):
        super().__init__('heartbeat_sender')

        # Declare parameters
        self.declare_parameter('host', DEFAULT_HOST)
        self.declare_parameter('port', DEFAULT_PORT)
        self.declare_parameter('rate_hz', DEFAULT_RATE_HZ)

        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value
        self.rate_hz = self.get_parameter('rate_hz').value

        # Validate timeout vs rate
        if self.rate_hz > 0:
            interval_sec = 1.0 / self.rate_hz
            self.get_logger().info(
                f"Heartbeat interval: {interval_sec:.1f}s, "
                f"GUI timeout should be >= {interval_sec * 2:.1f}s (2x interval)"
            )

        # Current state
        self.current_state = '0'  # STANDBY by default ('0' char)

        # Sequence number
        self.seq_num = 0

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.settimeout(0.1)  # Short timeout for non-blocking send

        # Timer
        interval_sec = 1.0 / self.rate_hz  # 3 seconds
        self.timer = self.create_timer(interval_sec, self.send_heartbeat)

        self.get_logger().info(
            f"Heartbeat sender initialized: {self.host}:{self.port} "
            f"every {interval_sec:.1f}s"
        )
        self.timer = self.create_timer(period, self.send_heartbeat)

        # Subscribe to supervisor state
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            String,
            '/supervisor/state',
            self.state_callback,
            qos
        )

        self.get_logger().info(
            f"Heartbeat sender initialized: {self.host}:{self.port} "
            f"at {self.rate_hz:.2f}Hz"
        )

    def state_callback(self, msg: String):
        """Update current state from supervisor."""
        state_map = {
            'STANDBY': '0',
            'MANUAL': '1',
            'AUTONOMOUS': '2',
            'FAULT': '3'
        }
        self.current_state = state_map.get(msg.data, '0')

    def send_heartbeat(self):
        """Send heartbeat packet."""
        try:
            # Encode heartbeat
            frame = encode_heartbeat(self.current_state, self.seq_num)

            # Send via UDP
            self.sock.sendto(frame, (self.host, self.port))

            # Increment sequence
            self.seq_num = (self.seq_num + 1) % 65536

            # Debug logging every 30th packet
            if self.seq_num % 30 == 0:
                self.get_logger().debug(
                    f"Heartbeat sent: state={self.current_state}, seq={self.seq_num}"
                )

        except Exception as e:
            self.get_logger().warn(f"Heartbeat send failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = HeartbeatSenderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
