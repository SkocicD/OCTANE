#!/usr/bin/env python3
"""RS485 drive node — Modbus RTU commands to a BLD-510B BLDC driver.

Replaces the broken CAN motor #2 (middle-left wheel).
Subscribes to /drive/command and mirrors the left_velocity channel.
The existing CAN drive node still runs unchanged; this node silently
fills the gap for motor #2 over RS485.

BLD-510B Modbus RTU (8N1, default 9600 baud, address 1):
  Reg 0x8000  control (high byte) + pole pairs (low byte)
              high byte bits: EN=0, FR=1, BK=2, NW=3
              NW=1 enables RS485 speed/start/stop control
  Reg 0x8005  target speed in RPM (0-65535)
  Reg 0x8018  actual speed (read-only)
  Reg 0x801B  fault state (read-only)

USB-RS485 adapter is located by serial number (parameter: serial_number).
Falls back to the first available USB serial port if not set.
"""

import struct
import time

import serial
import serial.tools.list_ports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from octane_msgs.msg import DriveCommand

REG_CONTROL     = 0x8000
REG_SPEED       = 0x8005
REG_ACTUAL_SPD  = 0x8018
REG_FAULT       = 0x801B

# High-byte control values (NW=1 = RS485 mode active)
CTRL_FORWARD = 0x09   # NW=1 EN=1 FR=0 BK=0
CTRL_REVERSE = 0x0B   # NW=1 EN=1 FR=1 BK=0
CTRL_STOP    = 0x08   # NW=1 EN=0 FR=0 BK=0

CONTROL_HZ      = 20
PDO_WATCHDOG_HZ = 2

FAULT_NAMES = {
    0x01: 'Locked rotor',
    0x02: 'Over-current',
    0x04: 'Hall abnormal',
    0x08: 'Bus voltage low',
    0x10: 'Bus voltage high',
    0x20: 'Current peak',
}


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _write_reg_frame(addr: int, reg: int, value: int) -> bytes:
    payload = struct.pack('>BBHH', addr, 0x06, reg, value)
    return payload + struct.pack('<H', _crc16(payload))


def _read_regs_frame(addr: int, reg: int, count: int) -> bytes:
    payload = struct.pack('>BBHH', addr, 0x03, reg, count)
    return payload + struct.pack('<H', _crc16(payload))


class RS485DriveNode(Node):

    def __init__(self):
        super().__init__('rs485_drive_node')

        self.declare_parameter('port',           '/dev/rs485_drive')
        self.declare_parameter('baud_rate',      9600)
        self.declare_parameter('modbus_address', 1)
        self.declare_parameter('max_rpm',        3000)
        self.declare_parameter('pole_pairs',     4)    # factory default for BLD-510B
        self.declare_parameter('reverse',        False) # flip direction if motor wired backwards
        self.declare_parameter('ramp_time_up',   0.33)
        self.declare_parameter('ramp_time_down', 0.33)
        self.declare_parameter('dead_band',      0.02)

        self._mb_addr    = self.get_parameter('modbus_address').value
        self._max_rpm    = self.get_parameter('max_rpm').value
        self._pole_pairs = self.get_parameter('pole_pairs').value
        self._reverse    = self.get_parameter('reverse').value
        self._ramp_time_up   = self.get_parameter('ramp_time_up').value
        self._ramp_time_down = self.get_parameter('ramp_time_down').value
        self._dead_band      = self.get_parameter('dead_band').value

        self._manual  = False
        self._target  = 0.0
        self._current = 0.0
        self._sent    = None
        self._watchdog_ticks  = 0
        self._watchdog_every  = max(1, int(CONTROL_HZ / PDO_WATCHDOG_HZ))
        self._port: serial.Serial | None = None
        self._fault_str = ''

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(String, '/manual_ctrl/rs485_status', status_qos)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._tx_pub = self.create_publisher(String, '/manual_ctrl/rs485_tx', qos)
        self.create_subscription(String,       '/supervisor/state', self._on_state, qos)
        self.create_subscription(DriveCommand, '/drive/command',    self._on_drive, qos)

        self._open_port()
        self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

    # ── port setup ──────────────────────────────────────────────────────────

    def _open_port(self):
        dev  = self.get_parameter('port').value
        baud = self.get_parameter('baud_rate').value
        try:
            self._port = serial.Serial(dev, baud, timeout=0.1)
            self._write_reg(REG_CONTROL, (CTRL_STOP << 8) | self._pole_pairs)
            self._publish_status(f'OK — {dev} @ {baud} baud, Modbus addr {self._mb_addr}')
            self.get_logger().info(f'RS485 drive ready on {dev}')
        except Exception as e:
            self._publish_status(f'ERROR: {e}')
            self.get_logger().error(f'RS485 serial open failed: {e}')

    # ── Modbus helpers ───────────────────────────────────────────────────────

    def _write_reg(self, reg: int, value: int):
        if self._port is None:
            return
        frame = _write_reg_frame(self._mb_addr, reg, value)
        try:
            self._port.write(frame)
            self._port.read(8)   # consume ACK echo
        except serial.SerialException as e:
            self.get_logger().warn(f'RS485 write error: {e}')

    def _read_reg(self, reg: int) -> int | None:
        if self._port is None:
            return None
        frame = _read_regs_frame(self._mb_addr, reg, 1)
        try:
            self._port.write(frame)
            resp = self._port.read(7)   # addr + func + byte_count + 2 data + 2 CRC
            if len(resp) == 7 and resp[1] == 0x03:
                return struct.unpack('>H', resp[3:5])[0]
        except serial.SerialException as e:
            self.get_logger().warn(f'RS485 read error: {e}')
        return None

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)

    def _on_state(self, msg: String):
        was = self._manual
        self._manual = msg.data == 'MANUAL'
        if was and not self._manual:
            self._target = 0.0

    def _on_drive(self, msg: DriveCommand):
        if not self._manual:
            status = String()
            status.data = f'GATED:STANDBY  L={msg.left_velocity:+.2f}'
            self._tx_pub.publish(status)
            return
        self._target = max(-1.0, min(1.0, msg.left_velocity))

    # ── control loop (ramp + watchdog) ──────────────────────────────────────────

    def _control_loop(self):
        diff = self._target - self._current
        t    = self._ramp_time_up if diff > 0 else self._ramp_time_down
        step = 1.0 / (t * CONTROL_HZ)

        if abs(diff) <= step:
            self._current = self._target
        else:
            self._current += step * (1 if diff > 0 else -1)

        if not self._manual:
            if self._current == 0.0:
                return
            self._target = 0.0

        if self._port is None:
            return

        self._watchdog_ticks += 1
        force = self._watchdog_ticks >= self._watchdog_every
        if force:
            self._watchdog_ticks = 0

        if not force and self._sent is not None and abs(self._current - self._sent) <= self._dead_band:
            return

        self._send_velocity(self._current)
        self._sent = self._current

        status = String()
        status.data = f'TX  L={self._current:+.3f}'
        self._tx_pub.publish(status)

    def _send_velocity(self, v: float):
        if self._reverse:
            v = -v
        if abs(v) < self._dead_band:
            self._write_reg(REG_CONTROL, (CTRL_STOP << 8) | self._pole_pairs)
        else:
            rpm  = int(abs(v) * self._max_rpm)
            ctrl = CTRL_REVERSE if v < 0 else CTRL_FORWARD
            self._write_reg(REG_CONTROL, (ctrl << 8) | self._pole_pairs)
            self._write_reg(REG_SPEED,   rpm)

    def _stop(self):
        self._current = 0.0
        self._sent    = None
        if self._port:
            try:
                self._write_reg(REG_CONTROL, (CTRL_STOP << 8) | self._pole_pairs)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = RS485DriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
