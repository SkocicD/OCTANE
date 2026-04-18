#!/usr/bin/env python3
"""WiFi communication node for OCTANE rover ground station link."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Empty
import socket
import threading
from typing import Optional
import struct

from octane_wifi.protocol import (
    encode_telemetry, encode_command, encode_ack, encode_fault,
    decode_message, HEADER_SIZE, CRC_SIZE
)


class WifiCommNode(Node):
    """TCP server for ground station communication."""

    def __init__(self):
        super().__init__('wifi_comm_node')

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

        # State
        self.current_state = 'STANDBY'
        self.current_fault: Optional[str] = None
        self.client_socket: Optional[socket.socket] = None
        self.connected = False
        self.running = False

        # Buffer for partial messages
        self.buffer = b''

        # Server
        self.server_socket: Optional[socket.socket] = None
        self.accept_thread: Optional[threading.Thread] = None
        self.recv_thread: Optional[threading.Thread] = None

        # Timer
        period = 1.0 / self.telemetry_rate
        self.timer = self.create_timer(period, self.send_telemetry)

        # Start server
        self.start_server()
        self.get_logger().info(f'WiFi node listening on {self.host}:{self.port}')

    def start_server(self):
        """Start TCP server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
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

                if self.client_socket:
                    self.client_socket.close()

                self.client_socket = client
                self.connected = True
                self.buffer = b''

                self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
                self.recv_thread.start()

            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f'Accept error: {e}')
                break

    def recv_loop(self):
        """Receive and decode commands."""
        try:
            while self.running and self.client_socket:
                data = self.client_socket.recv(256)
                if not data:
                    break

                self.buffer += data

                # Process complete messages
                while len(self.buffer) >= HEADER_SIZE:
                    # Peek at header to get length
                    _, _, payload_len = struct.unpack('!BBB', self.buffer[:HEADER_SIZE])
                    expected = HEADER_SIZE + payload_len + CRC_SIZE

                    if len(self.buffer) < expected:
                        break  # Wait for more data

                    # Extract complete frame
                    frame = self.buffer[:expected]
                    self.buffer = self.buffer[expected:]

                    # Decode
                    msg = decode_message(frame)
                    if msg:
                        self.handle_message(msg)

        except Exception as e:
            self.get_logger().error(f'Receive error: {e}')
        finally:
            self.get_logger().warn('Disconnected from ground station')
            self.connected = False
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None

    def handle_message(self, msg: dict):
        """Process decoded message."""
        if msg['type'] == 'command':
            mode = msg['mode']
            estop = msg['estop']

            if estop:
                self.get_logger().error('E-STOP requested!')
                # TODO: Send emergency stop to rover

            # Publish mode command
            mode_msg = String()
            mode_msg.data = mode
            self.mode_command_pub.publish(mode_msg)
            self.get_logger().info(f'Command: {mode} (estop={estop})')

            # Send ACK
            if self.client_socket:
                ack = encode_ack(True)
                self.client_socket.sendall(ack)

    def state_callback(self, msg: String):
        """Handle state change."""
        self.current_state = msg.data

    def fault_callback(self, msg: String):
        """Handle fault signal."""
        self.current_fault = msg.data
        self.get_logger().error(f'Fault: {msg.data}')

        # Send immediate fault alert
        if self.client_socket:
            alert = encode_fault(msg.data, 'critical')
            try:
                self.client_socket.sendall(alert)
            except Exception as e:
                self.get_logger().error(f'Failed to send alert: {e}')

    def send_telemetry(self):
        """Send periodic telemetry."""
        if not self.connected or not self.client_socket:
            return

        try:
            packet = encode_telemetry(self.current_state, self.current_fault)
            self.client_socket.sendall(packet)
        except Exception as e:
            self.get_logger().error(f'Telemetry send failed: {e}')
            self.connected = False

    def destroy_node(self):
        """Clean up."""
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
