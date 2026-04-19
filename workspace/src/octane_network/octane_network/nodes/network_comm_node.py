#!/usr/bin/env python3
"""Network communication node for OCTANE rover ground station link.

This node provides TCP communication between theground station GUI and ROS2.
It bridges mode commands, state updates, and fault alerts.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Empty
import socket
import threading
import json
from typing import Optional


class NetworkCommNode(Node):
    """TCP server for ground station communication."""

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

        # State
        self.current_state = 'STANDBY'
        self.current_fault = None
        self.client_socket = None
        self.connected = False
        self.running = False

        # Buffer for partial messages
        self.buffer = ''
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
        """Receive and decode JSON messages."""
        try:
            self.client_socket.settimeout(1.0)

            while self.running and self.client_socket:
                try:
                    data = self.client_socket.recv(1024)
                    if not data:
                        break

                    # Convert to string and add to buffer
                    with self.buffer_lock:
                        self.buffer += data.decode('utf-8')

                    # Process complete messages (newline-delimited)
                    self.process_buffer()

                except socket.timeout:
                    continue

        except Exception as e:
            self.get_logger().error(f'Receive error: {e}')
        finally:
            self.get_logger().warn('Disconnected from ground station')
            self.connected = False
            with self.buffer_lock:
                self.buffer = ''
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None

    def process_buffer(self):
        """Process complete messages from buffer."""
        while '\n' in self.buffer:
            # Extract complete message
            msg_end = self.buffer.index('\n')
            msg_str = self.buffer[:msg_end].strip()
            self.buffer = self.buffer[msg_end + 1:]

            if not msg_str:
                continue

            try:
                msg = json.loads(msg_str)
                self.handle_message(msg)
            except json.JSONDecodeError as e:
                self.get_logger().error(f'Invalid JSON: {e}')

    def handle_message(self, msg: dict):
        """Process decoded JSON message."""
        if msg.get('type') != 'mode_command':
            self.get_logger().warn(f'Unknown message type: {msg.get("type")}')
            return

        mode = msg.get('mode', '').upper()
        timestamp = msg.get('timestamp', 0)

        # Validate mode
        valid_modes = ['STANDBY', 'MANUAL', 'AUTONOMOUS', 'FAULT_RESET']
        if mode not in valid_modes:
            self.get_logger().warn(f'Invalid mode: {mode}')
            return

        self.get_logger().info(f'Received mode command: {mode} (timestamp: {timestamp})')

        # Publish mode command to ROS2
        mode_msg = String()
        mode_msg.data = mode
        self.mode_command_pub.publish(mode_msg)

        # Send ACK to GUI
        if self.client_socket:
            try:
                ack = json.dumps({
                    'type': 'ack',
                    'success': True,
                    'timestamp': self.get_clock().now().nanoseconds / 1e9
                }) + '\n'
                self.client_socket.sendall(ack.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f'ACK send failed: {e}')

    def state_callback(self, msg: String):
        """Handle supervisor state change."""
        old_state = self.current_state
        self.current_state = msg.data

        if old_state != self.current_state:
            self.get_logger().info(f'State changed: {old_state} → {self.current_state}')

    def fault_callback(self, msg: String):
        """Handle fault signal."""
        self.current_fault = msg.data
        self.get_logger().error(f'Fault detected: {msg.data}')

        # Send immediate fault alert to GUI
        if self.client_socket and self.connected:
            try:
                alert = json.dumps({
                    'type': 'fault_alert',
                    'fault': self.current_fault,
                    'severity': 'critical',
                    'timestamp': self.get_clock().now().nanoseconds / 1e9
                }) + '\n'
                self.client_socket.sendall(alert.encode('utf-8'))
            except Exception as e:
                self.get_logger().error(f'Fault alert send failed: {e}')

    def send_telemetry(self):
        """Send periodic telemetry to GUI."""
        if not self.connected or not self.client_socket:
            return

        try:
            telemetry = json.dumps({
                'type': 'state_update',
                'mode': self.current_state,
                'fault': self.current_fault,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            }) + '\n'

            self.client_socket.sendall(telemetry.encode('utf-8'))

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
