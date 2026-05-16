#!/usr/bin/env python3
"""Network communication node for OCTANE rover ground station link.

This node provides TCP communication between the ground station GUI and ROS2.
It bridges mode commands, state updates, and fault alerts using the lean binary protocol.

Protocol: workspace/src/octane_network/resource/messages.md
Implementation: workspace/src/octane_network/octane_network/protocol.py
"""

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Empty, UInt8, UInt16
import socket
import threading
from typing import Optional, List

from octane_network.classes.protocol import (
    encode_command, encode_telemetry, encode_ack, encode_fault, encode_heartbeat,
    encode_manipulator, decode_message, TYPE_TELEMETRY, TYPE_COMMAND, TYPE_ACK,
    TYPE_FAULT, TYPE_MANIPULATOR, TYPE_VIDEO_REQUEST,
)
from octane_network.classes.network_guard import NetworkGuard


class NetworkCommNode(Node):
    """TCP server for ground station communication using lean binary protocol."""

    def __init__(self):
        super().__init__('network_comm_node')

        # Declare parameters
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 5000)
        self.declare_parameter('telemetry_rate', 10.0)
        self.declare_parameter('heartbeat_rate', 0.3333)

        self.host           = self.get_parameter('host').value
        self.port           = self.get_parameter('port').value
        self.telemetry_rate = self.get_parameter('telemetry_rate').value
        self.heartbeat_rate = self.get_parameter('heartbeat_rate').value

        # QoS — standard reliable for most topics
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        # Latched QoS for client_ip: late-joining subscribers (video_stream_node)
        # get the cached value even if they start after the GUI connects.
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ROS2 subscribers
        self.supervisor_state_sub = self.create_subscription(
            String, '/supervisor/state', self.state_callback, qos
        )
        self.supervisor_fault_sub = self.create_subscription(
            String, '/supervisor/fault_signal', self.fault_callback, qos
        )
        self.create_subscription(
            Imu, 'sensors/imu/accel', self._imu_cb,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(Pose2D, 'localization/pose',     self._pose_cb,     qos)
        self.create_subscription(String, 'localization/far_tags', self._far_tags_cb, qos)

        # ROS2 publishers
        self.mode_command_pub    = self.create_publisher(String, '/supervisor/mode_command',  qos)
        self.fault_reset_pub     = self.create_publisher(Empty,  '/supervisor/fault_reset',   qos)
        self.key_state_pub       = self.create_publisher(UInt8,   '/manual_ctrl/key_state',    qos)
        self.speed_modifier_pub  = self.create_publisher(UInt16,  '/manual_ctrl/speed_modifier', qos)
        self._hb_status_pub      = self.create_publisher(String, '/network/heartbeat_tx',     qos)
        self._client_ip_pub      = self.create_publisher(String, '/network/client_ip',        latched_qos)
        self._stream_request_pub = self.create_publisher(String, '/network/stream_request',   qos)

        # State
        self.current_state = 'STANDBY'
        self.current_fault = None
        self._latest_accel: tuple | None = None
        self._latest_pose:  tuple | None = None   # (x, y, theta)
        self._tag_obs:      dict  = {}            # tag_id → {id, dist, angle_deg, ts}
        self._tag_timeout   = 2.0                 # seconds before a tag obs goes stale
        self.client_socket = None
        self.connected = False
        self.running = False
        self._hb_seq = 0

        # Binary buffer for partial frames
        self.buffer: bytes = b''
        self.buffer_lock = threading.Lock()

        # Connection watchdog — fires handlers after 3 consecutive HB failures
        self._guard = NetworkGuard(fail_threshold=3)
        self._guard.register(self._kill_video_feed)

        # Server
        self.server_socket = None
        self.accept_thread = None
        self.recv_thread = None

        self.create_timer(1.0 / self.telemetry_rate,  self.send_telemetry)
        self.create_timer(1.0 / self.heartbeat_rate, self._send_heartbeat)

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

                # Tell video_stream_node where to send UDP frames
                ip_msg = String()
                ip_msg.data = addr[0]
                self._client_ip_pub.publish(ip_msg)

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
            # Stop any active video stream
            stop_msg = String()
            stop_msg.data = '255,R,0,10'
            self._stream_request_pub.publish(stop_msg)
            # Clear GUI IP so video_stream_node stops sending
            ip_msg = String()
            ip_msg.data = ''
            self._client_ip_pub.publish(ip_msg)

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

            spd_msg = UInt16()
            spd_msg.data = msg.get('speed_modifier', 100)
            self.speed_modifier_pub.publish(spd_msg)

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

        elif msg_type == 'video_request':
            source_id = msg.get('source_id', 0xFF)
            variant   = msg.get('variant', 'R')
            quality   = msg.get('quality', 0)
            fps       = msg.get('fps', 10)
            req_msg = String()
            req_msg.data = f'{source_id},{variant},{quality},{fps}'
            self._stream_request_pub.publish(req_msg)
            self.get_logger().info(
                f'Video request: src={source_id} variant={variant} quality={quality} fps={fps}'
            )

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

    def _send_heartbeat(self):
        if not self.connected or not self.client_socket:
            return
        try:
            frame = encode_heartbeat(self.current_state, self._hb_seq)
            self.client_socket.sendall(frame)
            self._guard.heartbeat_ok()
            status = f'OK  seq={self._hb_seq}  state={self.current_state}'
            self._hb_seq = (self._hb_seq + 1) % 65536
        except Exception as e:
            self._guard.heartbeat_fail()
            status = f'ERR {e}'
        pub_msg = String()
        pub_msg.data = status
        self._hb_status_pub.publish(pub_msg)

    def _kill_video_feed(self):
        stop_msg = String()
        stop_msg.data = '255,R,0,10'
        self._stream_request_pub.publish(stop_msg)
        ip_msg = String()
        ip_msg.data = ''
        self._client_ip_pub.publish(ip_msg)
        self.get_logger().warn('NetworkGuard: heartbeat lost — video feed killed')

    def _imu_cb(self, msg: Imu):
        a = msg.linear_acceleration
        self._latest_accel = (a.x, a.y, a.z)

    def _pose_cb(self, msg: Pose2D):
        self._latest_pose = (msg.x, msg.y, msg.theta)

    def _far_tags_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            now = time.monotonic()
            for t in data.get('tags', []):
                tid = int(t['id'])
                self._tag_obs[tid] = {
                    'id': tid,
                    'dist': float(t['dist']),
                    'angle_deg': float(t['angle_deg']),
                    'ts': now,
                }
        except Exception:
            pass

    def send_telemetry(self):
        """Send periodic telemetry to GUI using binary protocol."""
        if not self.connected or not self.client_socket:
            return

        # Collect fresh tag observations (drop stale)
        now = time.monotonic()
        fresh_tags = [
            v for v in self._tag_obs.values()
            if now - v['ts'] < self._tag_timeout
        ]
        self._tag_obs = {v['id']: v for v in fresh_tags}

        try:
            telemetry_frame = encode_telemetry(
                self.current_state,
                fault=self.current_fault,
                accel=self._latest_accel,
                pose=self._latest_pose,
                tags=fresh_tags if fresh_tags else None,
            )
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
