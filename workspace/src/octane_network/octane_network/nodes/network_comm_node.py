#!/usr/bin/env python3
"""Network communication node for OCTANE rover ground station link.

This node provides TCP communication between the ground station GUI and ROS2.
It bridges mode commands, state updates, and fault alerts using the lean binary protocol.

Protocol: workspace/src/octane_network/resource/messages.md
Implementation: workspace/src/octane_network/octane_network/protocol.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Empty, UInt8
import socket
import threading
from typing import Optional, List

from octane_network.classes.protocol import (
    encode_command, encode_telemetry, encode_ack, encode_fault,
    decode_message, TYPE_TELEMETRY, TYPE_COMMAND, TYPE_ACK, TYPE_FAULT, TYPE_MANIPULATOR
)


class NetworkCommNode(Node):
    """TCP server for ground station communication using lean binary protocol."""

    def __init__(self):
        super().__init__('network_comm_node')

        # Declare parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 5000)
        self.declare_parameter('telemetry_rate', 10.0)

        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value
        self.telemetry_rate = self.get_parameter('telemetry_rate').value

        # QoS
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # ROS2 subscribers
        self.supervisor_state_sub = self.create_subscription(
            String, '/supervisor/state', self.state_callback, qos
        )
        self.supervisor_fault_sub = self.create_subscription(
            String, '/supervisor/fault_signal', self.fault_callback, qos
        )

        # ROS2 publishers
        self.mode_command_pub = self.create_publisher(String, '/supervisor/mode_command', qos)
        self.fault_reset_pub = self.create_publisher(Empty, '/supervisor/fault_reset', qos)
        self.key_state_pub = self.create_publisher(UInt8, '/manual_ctrl/key_state', qos)

        # State
        self.current_state = 'STANDBY'
        self.current_fault = None
        self.client_socket = None
        self.connected = False
        self.running = False

        # Binary buffer for partial frames
        self.buffer: bytes = b''
        self.buffer_lock = threading.Lock()

        # Server
        self.server_socket = None
        self.accept_thread = None
        self.recv_thread = None

        # Timer
        period = 1.0 / self.telemetry_rate
        self.timer = self.create_timer(period, self.send_telemetry)

        # Start server
        self.start_server()
        self.get_logger().info(f'Network node listening on {self.host}:{self.port}')

    def start_server(self):
        """Start TCP server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)  # Allow only one connection (GUI)
        self.server_socket.settimeout(1.0)
        self.running = True

        self.accept_thread = threading.Thread(target=self.accept_loop, daemon=True)
        self.accept_thread.start()

    def accept_loop(self):
        """Accept incoming connections."""
        while self.running:
            try:
                client, addr = self.server_socket.accept()
                self.get_logger().info(f'Ground station connected: {addr}')

                # Close existing connection if any
                if self.client_socket:
                    self.get_logger().warn('New connection, closing old one')
                    self.client_socket.close()

                self.client_socket = client
                self.connected = True

                # Start receive thread
                self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
                self.recv_thread.start()

            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f'Accept error: {e}')
                if not self.running:
                    break

    def recv_loop(self):
        """Receive binary frames and decode messages."""
        try:
            self.client_socket.settimeout(1.0)

            while self.running and self.client_socket:
                try:
                    data = self.client_socket.recv(1024)
                    if not data:
                        break

                    # Add to binary buffer
                    with self.buffer_lock:
                        self.buffer += data

                    # Process complete frames
                    self.process_buffer()

                except socket.timeout:
                    continue

        except Exception as e:
            self.get_logger().error(f'Receive error: {e}')
        finally:
            self.get_logger().warn('Disconnected from ground station')
            self.connected = False
            with self.buffer_lock:
                self.buffer = b''
            if self.client_socket:
                self.client_socket.close()
            self.client_socket = None

    def process_buffer(self):
        """Process complete frames from binary buffer."""
        while True:
            # Need at least header (3 bytes) to determine frame length
            with self.buffer_lock:
                if len(self.buffer) < 3:
                    return

                # Parse header
                magic = self.buffer[0]
                msg_type = self.buffer[1]
                payload_len = self.buffer[2]

                # Calculate total frame size (header + payload + CRC)
                expected_len = 3 + payload_len + 1  # 3 byte header + payload + 1 CRC

                if len(self.buffer) < expected_len:
                    return  # Wait for more data

                # Extract complete frame
                frame = self.buffer[:expected_len]
                self.buffer = self.buffer[expected_len:]

            # Decode frame
            msg = decode_message(frame)
            if msg:
                self.handle_message(msg)
            # If decode fails (bad CRC/magic), silently discard

    def handle_message(self, msg: dict):
        """Process decoded binary message."""
        msg_type = msg.get('type')

        if msg_type == 'manipulator':
            key_msg = UInt8()
            key_msg.data = msg.get('bitfield', 0)
            self.key_state_pub.publish(key_msg)

        elif msg_type == 'command':
            mode = msg.get('mode', '')
            estop = msg.get('estop', False)

            self.get_logger().info(f'Received command: mode={mode}, estop={estop}')

            # Publish mode command to ROS2
            mode_msg = String()
            mode_msg.data = mode.upper()
            self.mode_command_pub.publish(mode_msg)

            # Send ACK if connected
            if self.client_socket:
                try:
                    # Encode ACK using binary protocol
                    ack_frame = encode_ack(success=True)
                    self.client_socket.sendall(ack_frame)
                except Exception as e:
                    self.get_logger().error(f'ACK send failed: {e}')

        else:
            self.get_logger().warn(f'Unknown message type: {msg_type}')

    def state_callback(self, msg: String):
        """Handle supervisor state change."""
        old_state = self.current_state
        self.current_state = msg.data

        if old_state != self.current_state:
            self.get_logger().info(f'State changed: {old_state} -> {self.current_state}')

    def fault_callback(self, msg: String):
        """Handle fault signal."""
        self.current_fault = msg.data
        self.get_logger().error(f'Fault detected: {self.current_fault}')

        # Send immediate fault alert to GUI using binary protocol
        if self.client_socket and self.connected:
            try:
                # Map fault name to single-char code
                fault_map = {
                    'battery_undervoltage': 'b',
                    'battery_overcurrent': 'b',
                    'motor_overcurrent': 'm',
                    'motor_driver_failure': 'm',
                    'can_bus_down': 'c',
                    'can_dead': 'c',
                    'wifi_connection_loss': 'w',
                    'ml_inference_crash': 's',
                    'april_tag_detection_failure': 'l',
                    'depth_camera_error': 'd',
                    'actuator_unresponsive': 'a',
                }
                fault_char = fault_map.get(self.current_fault, 's')  # default to 'software'

                # Encode fault alert as binary frame
                fault_frame = encode_fault(fault_char, 'critical')
                self.client_socket.sendall(fault_frame)
            except Exception as e:
                self.get_logger().error(f'Fault alert send failed: {e}')

    def send_telemetry(self):
        """Send periodic telemetry to GUI using binary protocol."""
        if not self.connected or not self.client_socket:
            return

        try:
            # Encode telemetry as binary frame
            telemetry_frame = encode_telemetry(self.current_state, fault=self.current_fault)
            self.client_socket.sendall(telemetry_frame)

        except Exception as e:
            self.get_logger().error(f'Telemetry send failed: {e}')
            self.connected = False

    def destroy_node(self):
        """Clean up resources."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.client_socket:
            self.client_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NetworkCommNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
