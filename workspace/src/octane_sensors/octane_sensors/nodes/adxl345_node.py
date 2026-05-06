#!/usr/bin/env python3
"""ADXL345 accelerometer node.

Auto-detects the sensor on any I2C bus by probing /dev/i2c-* for the ADXL345
device-ID register (0x00 == 0xE5).  No extra Python dependencies — uses the
Linux i2c-dev kernel interface directly via fcntl/ioctl.

Publishes sensor_msgs/Imu with linear_acceleration populated.
Orientation and angular_velocity are marked unavailable (covariance[0] = -1).

Topic: sensors/imu/accel
Frame: imu_frame  (set imu_frame param to override)
"""

import fcntl
import struct
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

# Linux i2c-dev ioctl
_I2C_SLAVE = 0x0703

# ADXL345 registers
_DEVID        = 0x00
_DEVID_VAL    = 0xE5
_BW_RATE      = 0x2C
_POWER_CTL    = 0x2D
_DATA_FORMAT  = 0x31
_DATAX0       = 0x32  # first of 6 consecutive bytes: X0 X1 Y0 Y1 Z0 Z1

# Both possible I2C addresses (SDO low → 0x53, SDO high → 0x1D)
_ADDRS = [0x53, 0x1D]

# Full-resolution scale: 3.9 mg/LSB → m/s²
_SCALE = 0.0039 * 9.80665


class _I2C:
    def __init__(self, bus: int, addr: int):
        self._f = open(f'/dev/i2c-{bus}', 'r+b', buffering=0)
        fcntl.ioctl(self._f, _I2C_SLAVE, addr)

    def write(self, reg: int, val: int):
        self._f.write(bytes([reg, val]))

    def read(self, reg: int, n: int) -> bytes:
        self._f.write(bytes([reg]))
        return self._f.read(n)

    def read1(self, reg: int) -> int:
        return self.read(reg, 1)[0]

    def close(self):
        self._f.close()


def _find_adxl345() -> tuple[int, int] | None:
    buses = sorted(
        int(p.name[4:]) for p in Path('/dev').glob('i2c-*') if p.name[4:].isdigit()
    )
    for bus in buses:
        for addr in _ADDRS:
            try:
                dev = _I2C(bus, addr)
                devid = dev.read1(_DEVID)
                dev.close()
                if devid == _DEVID_VAL:
                    return bus, addr
            except Exception:
                pass
    return None


class Adxl345Node(Node):

    def __init__(self):
        super().__init__('adxl345_node')

        self.declare_parameter('i2c_bus',      -1)    # -1 = auto-detect
        self.declare_parameter('i2c_address',  -1)    # -1 = auto-detect
        self.declare_parameter('publish_rate',  50.0)
        self.declare_parameter('frame_id',     'imu_frame')
        self.declare_parameter('range_g',       2)    # 2 | 4 | 8 | 16

        bus      = int(self.get_parameter('i2c_bus').value)
        addr     = int(self.get_parameter('i2c_address').value)
        rate     = float(self.get_parameter('publish_rate').value)
        frame_id = self.get_parameter('frame_id').value
        range_g  = int(self.get_parameter('range_g').value)

        self._frame_id = frame_id

        if bus < 0 or addr < 0:
            self.get_logger().info('[ADXL345] Probing I2C buses for ADXL345 (DEVID=0xE5)...')
            result = _find_adxl345()
            if result is None:
                raise RuntimeError('[ADXL345] No ADXL345 found on any I2C bus')
            bus, addr = result
            self.get_logger().info(f'[ADXL345] Found on /dev/i2c-{bus} @ 0x{addr:02X}')
        else:
            self.get_logger().info(f'[ADXL345] Using /dev/i2c-{bus} @ 0x{addr:02X}')

        self._dev = _I2C(bus, addr)
        self._configure(range_g)

        self._pub = self.create_publisher(Imu, 'sensors/imu/accel', 10)
        self.create_timer(1.0 / rate, self._cb)
        self.get_logger().info(f'[ADXL345] Publishing at {rate} Hz  frame={frame_id}')

    def _configure(self, range_g: int):
        range_bits = {2: 0x00, 4: 0x01, 8: 0x02, 16: 0x03}.get(range_g, 0x00)
        self._dev.write(_BW_RATE,     0x0A)           # 100 Hz output
        self._dev.write(_DATA_FORMAT, 0x08 | range_bits)  # full-resolution
        self._dev.write(_POWER_CTL,   0x08)           # measure mode
        self.get_logger().info(f'[ADXL345] ±{range_g}g, full-resolution, 100 Hz ODR')

    def _cb(self):
        raw = self._dev.read(_DATAX0, 6)
        x, y, z = struct.unpack_from('<hhh', raw)

        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id

        msg.linear_acceleration.x = x * _SCALE
        msg.linear_acceleration.y = y * _SCALE
        msg.linear_acceleration.z = z * _SCALE
        # Leave covariance as zeros (known but uncalibrated)

        # Gyro and orientation not available on ADXL345
        msg.angular_velocity_covariance[0] = -1.0
        msg.orientation_covariance[0]      = -1.0
        msg.orientation.w                  =  1.0  # identity quaternion

        self._pub.publish(msg)

    def destroy_node(self):
        try:
            self._dev.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Adxl345Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
