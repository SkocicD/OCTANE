#!/usr/bin/env python3
"""ADXL345 accelerometer node for BTT ADXL345 V2.0 (RP2040 USB device).

Communicates via the Klipper MCU binary protocol over /dev/ttyACM*.
Auto-detects by USB VID:PID 1d50:614e or by scanning /dev/ttyACM*.

Publishes sensor_msgs/Imu to sensors/imu/accel.
"""

import re
import struct
import threading
import queue
import time
import zlib
import json
from pathlib import Path

import serial
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

# ── Klipper binary protocol ──────────────────────────────────────────────────

_SYNC     = 0x7E
_MSG_DEST = 0x10
_MSG_SEQ  = 0x0F
_MSG_MIN  = 5
_MSG_MAX  = 64

# BTT ADXL345 V2.0 hardware constants (RP2040 SPI1)
_SPI_BUS  = 1   # hardware SPI1
_SPI_PIN  = 9   # GPIO9 = CS
_SPI_MODE = 3
_SPI_RATE = 5_000_000
_DATA_RATE = 3200   # Hz ODR (register 0x0f)

# ADXL345 scale: 3.9 mg/LSB in full-res mode → m/s²
_SCALE = 0.0039 * 9.80665


def _crc16(buf: bytes) -> int:
    """Klipper CRC-16 variant."""
    crc = 0xFFFF
    for d in buf:
        d ^= crc & 0xFF
        d ^= (d & 0x0F) << 4
        crc = ((d << 8) | (crc >> 8)) ^ (d >> 4) ^ (d << 3)
    return crc & 0xFFFF


def _enc_vlq(v: int) -> bytes:
    out = []
    if v >=  0xC000000 or v < -0x4000000: out.append((v >> 28) & 0x7F | 0x80)
    if v >=  0x180000  or v <  -0x80000:  out.append((v >> 21) & 0x7F | 0x80)
    if v >=  0x3000    or v <   -0x1000:  out.append((v >> 14) & 0x7F | 0x80)
    if v >=  0x60      or v <     -0x20:  out.append((v >>  7) & 0x7F | 0x80)
    out.append(v & 0x7F)
    return bytes(out)


def _dec_vlq(data: bytes, pos: int):
    c = data[pos]; pos += 1
    v = c & 0x7F
    if (c & 0x60) == 0x60:
        v |= ~0x1F
    while c & 0x80:
        c = data[pos]; pos += 1
        v = (v << 7) | (c & 0x7F)
    return v, pos


def _frame(payload: bytes, seq: int) -> bytes:
    seq_byte = _MSG_DEST | (seq & _MSG_SEQ)
    total = 2 + len(payload) + 3
    body = bytes([total, seq_byte]) + payload
    crc = _crc16(body)
    return body + struct.pack('>H', crc) + bytes([_SYNC])


def _enc_params(fmt: str, args) -> bytes:
    """Encode command parameters from Klipper format string."""
    types = re.findall(r'%(\*s|\.\*s|hu|[cuds])', fmt)
    out = bytearray()
    for t, v in zip(types, args):
        if t == 'c':
            out.append(int(v) & 0xFF)
        elif t == 'hu':
            out += struct.pack('>H', int(v))
        elif t in ('*s', '.*s'):
            b = v if isinstance(v, (bytes, bytearray)) else bytes(v)
            out.append(len(b))
            out += b
        else:  # u, d, s
            out += _enc_vlq(int(v))
    return bytes(out)


# ── Klipper MCU client ───────────────────────────────────────────────────────

class KlipperMCU:
    def __init__(self, port: str):
        self._port = port
        self._ser: serial.Serial | None = None
        self._seq = 0
        self._buf = b''
        self._running = False
        self._lock = threading.Lock()
        self._handlers: dict[int, queue.Queue] = {}
        self._raw_q: queue.Queue | None = None
        self.cmds: dict[str, int] = {}
        self.resps: dict[str, int] = {}

    def connect(self, timeout: float = 15.0):
        self._ser = serial.Serial(self._port, 250000, timeout=0.1)
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._running = True
        threading.Thread(target=self._rx_loop, daemon=True).start()
        self._identify(timeout)

    def _rx_loop(self):
        while self._running:
            try:
                data = self._ser.read(256)
                if data:
                    self._buf += data
                    self._drain()
            except Exception:
                pass

    def _drain(self):
        while True:
            if len(self._buf) < _MSG_MIN:
                return
            n = self._buf[0]
            if n < _MSG_MIN or n > _MSG_MAX:
                self._buf = self._buf[1:]
                continue
            if len(self._buf) < n:
                return
            frame = self._buf[:n]
            self._buf = self._buf[n:]
            if frame[-1] != _SYNC:
                continue
            if _crc16(frame[:-3]) != struct.unpack('>H', frame[-3:-1])[0]:
                continue
            payload = frame[2:-3]
            if payload:
                self._dispatch(payload)

    def _dispatch(self, payload: bytes):
        raw_q = self._raw_q
        if raw_q is not None:
            raw_q.put(payload)
            return
        try:
            resp_id, pos = _dec_vlq(payload, 0)
        except Exception:
            return
        with self._lock:
            q = self._handlers.get(resp_id)
        if q:
            q.put(payload[pos:])

    def _send(self, payload: bytes):
        self._ser.write(_frame(payload, self._seq))
        self._seq = (self._seq + 1) & _MSG_SEQ

    def send_cmd(self, name: str, *args):
        cmd_id = self.cmds[name]
        fmt = name  # format string IS the key in Klipper's command dict
        payload = _enc_vlq(cmd_id) + _enc_params(fmt, args)
        self._send(payload)

    def register_resp(self, name: str) -> queue.Queue:
        resp_id = self.resps[name]
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._handlers[resp_id] = q
        return q

    def unregister_resp(self, name: str):
        resp_id = self.resps.get(name)
        if resp_id is not None:
            with self._lock:
                self._handlers.pop(resp_id, None)

    def _identify(self, timeout: float):
        CHUNK = 40
        total_size: int | None = None
        chunks: dict[int, bytes] = {}
        raw_q: queue.Queue = queue.Queue()
        self._raw_q = raw_q
        deadline = time.time() + timeout
        seq = 0

        try:
            offset = 0
            while time.time() < deadline:
                identify_pl = _enc_vlq(0) + _enc_vlq(offset) + _enc_vlq(CHUNK)
                self._ser.write(_frame(identify_pl, seq))
                seq = (seq + 1) & _MSG_SEQ

                try:
                    pl = raw_q.get(timeout=2.0)
                except queue.Empty:
                    continue

                try:
                    pos = 0
                    _resp_id, pos = _dec_vlq(pl, pos)
                    resp_off, pos = _dec_vlq(pl, pos)
                    resp_tot, pos = _dec_vlq(pl, pos)
                    data_len = pl[pos]; pos += 1
                    data = pl[pos:pos + data_len]
                except Exception:
                    continue

                if total_size is None:
                    total_size = resp_tot
                if data:
                    chunks[resp_off] = data
                next_off = resp_off + len(data)
                if total_size and next_off >= total_size:
                    break
                offset = next_off
        finally:
            self._raw_q = None

        if not chunks or total_size is None:
            raise RuntimeError('Klipper identify timed out')

        compressed = b''.join(chunks[k] for k in sorted(chunks))
        try:
            d = json.loads(zlib.decompress(compressed))
        except Exception as e:
            raise RuntimeError(f'Klipper identify dict decompress failed: {e}')

        self.cmds  = d.get('commands',  {})
        self.resps = {v: k for k, v in d.get('responses', {}).items()}

    def disconnect(self):
        self._running = False
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass


# ── ADXL345 data acquisition ─────────────────────────────────────────────────

def _decode_samples(raw: bytes) -> list[tuple[float, float, float]]:
    """Decode Klipper ADXL345 packed samples → (x, y, z) m/s².
    Each sample is 5 bytes: xlow, ylow, zlow, xzhigh, yzhigh.
    xzhigh carries high bits of X (bits 4:0) and Z (bits 7:5).
    yzhigh carries high bits of Y (bits 4:0), Z cont. (bits 6:5), error (bit 7).
    """
    samples = []
    for i in range(0, len(raw) - 4, 5):
        xlow, ylow, zlow, xzhigh, yzhigh = raw[i:i+5]
        if yzhigh & 0x80:
            continue
        rx = (xlow | ((xzhigh & 0x1F) << 8)) - ((xzhigh & 0x10) << 9)
        ry = (ylow | ((yzhigh & 0x1F) << 8)) - ((yzhigh & 0x10) << 9)
        rz = (zlow | ((xzhigh & 0xE0) << 3) | ((yzhigh & 0xE0) << 6)) - ((yzhigh & 0x40) << 7)
        samples.append((rx * _SCALE, ry * _SCALE, rz * _SCALE))
    return samples


# ── Auto-detect ───────────────────────────────────────────────────────────────

def _find_device() -> str | None:
    for tty in sorted(Path('/dev').glob('ttyACM*')):
        sys_path = Path(f'/sys/class/tty/{tty.name}/device')
        if sys_path.exists():
            uevent = (sys_path / '../uevent').read_text(errors='ignore')
            if '1d50/614e' in uevent or '1D50/614E' in uevent.upper():
                return str(tty)
    candidates = sorted(Path('/dev').glob('ttyACM*'))
    return str(candidates[0]) if candidates else None


# ── ROS 2 node ────────────────────────────────────────────────────────────────

class Adxl345Node(Node):

    def __init__(self):
        super().__init__('adxl345_node')
        self.declare_parameter('port',         '')
        self.declare_parameter('publish_rate',  50.0)
        self.declare_parameter('frame_id',     'imu_frame')

        port     = self.get_parameter('port').value or ''
        rate     = float(self.get_parameter('publish_rate').value)
        frame_id = self.get_parameter('frame_id').value

        self._frame_id = frame_id
        self._latest: tuple[float, float, float] | None = None
        self._latest_lock = threading.Lock()

        if not port:
            port = _find_device()
        if not port:
            raise RuntimeError('[ADXL345] No device found on /dev/ttyACM*')
        self.get_logger().info(f'[ADXL345] Connecting to {port}')

        self._mcu = KlipperMCU(port)
        self._mcu.connect(timeout=15.0)
        self.get_logger().info('[ADXL345] Klipper identify OK — configuring sensor')

        self._configure()

        self._pub = self.create_publisher(Imu, 'sensors/imu/accel', 10)
        self.create_timer(1.0 / rate, self._timer_cb)
        self.get_logger().info(f'[ADXL345] Publishing at {rate} Hz on sensors/imu/accel')

    def _configure(self):
        mcu = self._mcu

        # 1. Allocate 2 OIDs: 0=SPI, 1=ADXL345
        mcu.send_cmd('allocate_oids count=%c', 2)
        time.sleep(0.05)

        # 2. Configure SPI peripheral
        mcu.send_cmd(
            'config_spi oid=%c bus=%u pin=%u mode=%u rate=%u shutdown_msg=%*s',
            0, _SPI_BUS, _SPI_PIN, _SPI_MODE, _SPI_RATE, b'',
        )
        time.sleep(0.05)

        # 3. Configure ADXL345 OID
        cmd_name = next(
            (k for k in mcu.cmds if k.startswith('config_adxl345')), None
        )
        if cmd_name is None:
            raise RuntimeError('[ADXL345] config_adxl345 not found in firmware')
        # Handle both 'config_adxl345 oid=%c spi_oid=%c' and
        # 'config_adxl345 oid=%c spi_oid=%c axes_data=%u'
        axes_data = 0x00
        if 'axes_data' in cmd_name:
            mcu.send_cmd(cmd_name, 1, 0, axes_data)
        else:
            mcu.send_cmd(cmd_name, 1, 0)
        time.sleep(0.05)

        # 4. Finalize config (required by some Klipper versions)
        if 'finalize_config crc=%u' in mcu.cmds:
            mcu.send_cmd('finalize_config crc=%u', 0)
            time.sleep(0.05)

        # 5. Start bulk query
        query_name = next(
            (k for k in mcu.cmds if k.startswith('query_adxl345')), None
        )
        if query_name is None:
            raise RuntimeError('[ADXL345] query_adxl345 not found in firmware')

        # Register data response handler
        data_resp = next(
            (k for k in mcu.resps.values() if 'adxl345_data' in k), None
        )
        if data_resp:
            self._data_q = mcu.register_resp(data_resp)
            threading.Thread(target=self._data_loop, daemon=True).start()

        # rest_ticks=0 means run continuously
        if 'time=%u' in query_name:
            mcu.send_cmd(query_name, 1, 0, 0)
        else:
            mcu.send_cmd(query_name, 1, 0)

    def _data_loop(self):
        """Drain adxl345_data responses and cache latest sample."""
        while rclpy.ok():
            try:
                payload = self._data_q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                pos = 0
                _oid = payload[pos]; pos += 1
                _seq = struct.unpack_from('>H', payload, pos)[0]; pos += 2
                data_len = payload[pos]; pos += 1
                raw = payload[pos:pos + data_len]
                samples = _decode_samples(raw)
                if samples:
                    with self._latest_lock:
                        self._latest = samples[-1]
            except Exception:
                pass

    def _timer_cb(self):
        with self._latest_lock:
            sample = self._latest
        if sample is None:
            return

        msg = Imu()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = sample
        msg.angular_velocity_covariance[0] = -1.0
        msg.orientation_covariance[0]      = -1.0
        msg.orientation.w                  =  1.0
        self._pub.publish(msg)

    def destroy_node(self):
        try:
            self._mcu.disconnect()
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
