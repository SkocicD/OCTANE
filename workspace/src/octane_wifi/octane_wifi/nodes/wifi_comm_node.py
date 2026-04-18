#!/usr/bin/env python3
"""WiFi communication node for OCTANE rover ground station link.

TCP server that bridges ROS2 topics to ground station:
  - Sends: supervisor state, fault status, telemetry
  - Receives: mode commands, fault reset commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Empty
import socket
import threading
import json
import time
from typing import Optional, Dict, List
import struct

from octane_wifi.protocol import (
    OctaneMessage, MessageType,
    decode_message, create_telemetry_packet,
    create_command_ack, create_fault_alert
)


class WifiCommNode(Node):
    """TCP server node for ground station communication."""

    def __init__(self):
        super().__init__('wifi_comm_node')

        # Declare parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 5000)
        self.declare_parameter('telemetry_rate', 10.0)
        self.declare_parameter('timeout_seconds', 2.0)

        # Get parameters
        self.host = self.get_parameter('host').value
        self.port = self.get_parameter('port').value
        self.telemetry_rate = self.get_parameter('telemetry_rate').value
        self.timeout_seconds = self.get_parameter('timeout_seconds').value

        # QoS for reliable delivery
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # ROS2 subscribers
        self.supervisor_state_sub = self.create_subscription(
            String, '/supervisor/state', self.supervisor_state_callback, qos
        )
        self.supervisor_fault_sub = self.create_subscription(
            String, '/supervisor/fault_signal', self.fault_signal_callback, qos
        )

        # ROS2 publishers
        self.mode_command_pub = self.create_publisher(String, '/supervisor/mode_command', qos)
        self.fault_reset_pub = self.create_publisher(Empty, '/supervisor/fault_reset', qos)

        # State tracking
        self.current_state = 'STANDBY'
        self.current_fault: Optional[str] = None
        self.sequence_number = 0

        # TCP server
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.client_address: Optional[tuple] = None
        self.running = True
        self.connected = False

        # Buffer for partial messages
        self.rx_buffer = b''

        # Telemetry timer
        telemetry_period = 1.0 / self.telemetry_rate
        self.timer = self.create_timer(telemetry_period, self.send_telemetry)

        # Accept thread
        self.accept_thread = threading.Thread(target=self.accept_loop, daemon=True)
        self.accept_thread.start()

        self.get_logger().info(f'WiFi comm node started, listening on {self.host}:{self.port}')

    def accept_loop(self):
        """Accept incoming TCP connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    self.get_logger().info(f'Ground station connected from {address}')

                    # Close previous connection if any
                    if self.client_socket:
                        self.client_socket.close()

                    self.client_socket = client_socket
                    self.client_address = address
                    self.connected = True
                    self.rx_buffer = b''

                    # Start receive thread
                    recv_thread = threading.Thread(target=self.receive_loop, daemon=True)
                    recv_thread.start()

                except socket.timeout:
                    continue

        except Exception as e:
            self.get_logger().error(f'Server error: {e}')
        finally:
            if self.server_socket:
                self.server_socket.close()
            self.get_logger().info('WiFi comm server stopped')

    def receive_loop(self):
        """Receive and process incoming commands."""
        try:
            while self.running and self.client_socket:
                try:
                    data = self.client_socket.recv(4096)
                    if not data:
                        break

                    self.rx_buffer += data

                    # Process all complete messages in buffer
                    while self.rx_buffer:
                        msg = decode_message(self.rx_buffer)
                        if msg is None:
                            # Need more data
                            break

                        # Remove processed data from buffer
                        json_len = struct.unpack('!I', msg.payload.get('length', b'\x00\x00\x00\x00'))[0] if 'length' in msg.payload else 0
                        # Simple approach for now - just clear buffer after processing
                        # TODO: Properly track byte positions
                        self.rx_buffer = b''

                        # Handle message
                        self.handle_message(msg)

                except socket.timeout:
                    continue

        except Exception as e:
            self.get_logger().error(f'Receive error: {e}')
        finally:
            self.get_logger().warn(f'Disconnected from ground station')
            self.connected = False
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None

    def handle_message(self, msg: OctaneMessage):
        """Process incoming message from ground station."""
        if msg.msg_type == MessageType.COMMAND:
            self.handle_command(msg)
        elif msg.msg_type == MessageType.COMMAND_ACK:
            self.get_logger().debug(f'Command ACK: success={msg.payload.get("s")}, seq={msg.payload.get("seq")}')

    def handle_command(self, msg: OctaneMessage):
        """Handle command message from ground station."""
        mode = msg.payload.get('m', '')
        estop = msg.payload.get('e', False)

        if estop:
            self.get_logger().error('Emergency stop requested!')
            # TODO: Send e-suggestion command to rover

        if mode in ['standby', 'manual', 'autonomous', 'fault_reset']:
            # Publish mode command
            mode_msg = String()
            mode_msg.data = mode
            self.mode_command_pub.publish(mode_msg)
            self.get_logger().info(f'Mode command received: {mode}')

            # Send acknowledgment
            ack = create_command_ack(True, msg.seq)
            if self.client_socket:
                self.client_socket.sendall(ack)
        else:
            self.get_logger().warn(f'Invalid mode command: {mode}')
            ack = create_command_ack(False, msg.seq)
            if self.client_socket:
                self.client_socket.sendall(ack)

    def supervisor_state_callback(self, msg: String):
        """Handle supervisor state change."""
        self.current_state = msg.data
        # Telemetry will be sent on next timer callback

    def fault_signal_callback(self, msg: String):
        """Handle fault signal."""
        self.current_fault = msg.data
        self.get_logger().error(f'Fault triggered: {msg.data}')

        # Immediately send fault alert to ground station
        if self.client_socket:
            alert = create_fault_alert(msg.data, 'critical')
            try:
                self.client_socket.sendall(alert)
            except Exception as e:
                self.get_logger().error(f'Failed to send fault alert: {e}')

    def send_telemetry(self):
        """Send periodic telemetry to ground station."""
        if not self.connected or not self.client_socket:
            return

        try:
            packet = create_telemetry_packet(
                state=self.current_state,
                fault=self.current_fault
            )
            self.client_socket.sendall(packet)
        except Exception as e:
            self.get_logger().error(f'Failed to send telemetry: {e}')
            self.connected = False

    def destroy_node(self):
        """Clean up on shutdown."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.client_socket:
            self.client_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WifiCommNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
