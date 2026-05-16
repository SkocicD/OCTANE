#!/usr/bin/env python3
"""Video stream node — compresses camera frames and sends them to the GUI over UDP.

Stream requests arrive on /network/stream_request (std_msgs/String):
    "<source_id>,<variant>,<quality>,<fps>"
    e.g.  "2,R,70,10"   → camera 2, RGB, quality 70, 10 fps
    e.g.  "6,D,40,5"    → mosaic, depth heatmap, quality 40, 5 fps
    e.g.  "255,R,0,10"  → stop all streams

source_id values:
    0 = orbbec_depth
    1 = near_rgb_left_side
    2 = near_rgb_left_front
    3 = near_rgb_right_side
    4 = near_rgb_right_front
    5 = near_rgb_back_rear
    6 = mosaic  (all 6 tiled 3×2)
    7 = terrain map (raw binary, variant T)
    255 = stop all

variant:  R = RGB,  D = depth heatmap (COLORMAP_INFERNO)
quality:  1–100 JPEG encode quality.  0 = use node's default_quality parameter.
          Higher = better image, larger frames.  Lower = more compression, smaller frames.

Spatial resolution is fixed server-side via the stream_scale parameter (default 100%).
Set stream_scale in network_params.yaml to reduce resolution for all streams.

UDP frame format (rover → GUI, port udp_port):
    [0x4F][0x56][source_id][variant][seq_hi][seq_lo][chunk_idx][chunk_total][...JPEG...]
    8-byte header, up to 1392 bytes of JPEG payload per packet.
    GUI reassembles by (seq, chunk_idx) and decodes JPEG.
"""

import os
import socket
import struct
import threading
import zlib
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from octane_msgs.msg import CameraFrame

SOURCE_MOSAIC = 6
SOURCE_MAP    = 7
SOURCE_STOP   = 255

VARIANT_RGB     = ord('R')
VARIANT_DEPTH   = ord('D')
VARIANT_TERRAIN = ord('T')

# Terrain payload layout (assembled before zlib compression):
#   [0]   version   uint8   = 1
#   [1]   flags     uint8   bit0=has_terrain  bit1=has_pose  bit2=has_nav  bit7=zlib_compressed
#   [2-3] reserved  uint16  = 0
#   ── terrain section (flags & 0x01) ──────────────────────────────
#   [4]   width     uint32  = 200
#   [8]   height    uint32  = 200
#   [12]  cell_m    float32 = 0.05  (metres per grid cell)
#   [16]  height    float32[200*200]  metres, row-major
#   [+]   rocks     uint8[200*200]   0-255
#   [+]   craters   uint8[200*200]   0-255
#   [+]   walls     uint8[200*200]   0-255
#   ── pose section (flags & 0x02, future) ─────────────────────────
#   pos_x/y/z float32×3 (metres), roll/pitch/yaw float32×3 (radians)
#   ── nav section (flags & 0x04, future) ──────────────────────────
#   left_motor float32, right_motor float32, bucket uint8, _pad uint8×3
_TERRAIN_TOPICS = {
    'height':  'mapping/terrain/height',
    'rocks':   'mapping/terrain/rocks',
    'craters': 'mapping/terrain/craters',
    'walls':   'mapping/terrain/walls',
}

_UDP_HEADER   = 8     # bytes
_CHUNK_SIZE   = 1392  # 1400 MTU - 8 header


def _build_source_map(cameras: dict) -> dict:
    """Index cameras by integer ID in yaml declaration order."""
    return {
        idx: {
            'name':        name,
            'rgb_topic':   cam['rgb_topic'],
            'depth_topic': cam['depth_topic'],
        }
        for idx, (name, cam) in enumerate(cameras.items())
    }


class VideoStreamNode(Node):

    def __init__(self):
        super().__init__('video_stream_node')

        self.declare_parameter('udp_port',            5002)
        self.declare_parameter('default_quality',     70)
        self.declare_parameter('stream_scale',        100)
        self.declare_parameter('depth_max_m',         8.0)
        self.declare_parameter('orbbec_depth_max_m',  5.0)
        self.declare_parameter('config_file',         '')

        self._udp_port       = self.get_parameter('udp_port').value
        self._default_quality = self.get_parameter('default_quality').value
        self._stream_scale   = max(1, min(100, self.get_parameter('stream_scale').value))
        self._depth_max_m    = self.get_parameter('depth_max_m').value
        self._orbbec_max_m   = self.get_parameter('orbbec_depth_max_m').value
        config_file          = self.get_parameter('config_file').value

        if not config_file:
            config_file = os.path.join(
                get_package_share_directory('octane'), 'config', 'cameras.yaml'
            )
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        self._source_map = _build_source_map(cfg['cameras'])

        self.bridge = CvBridge()
        self._lock  = threading.Lock()

        # Stream state
        self._gui_ip:          Optional[str] = None
        self._active_source:   Optional[int] = None
        self._active_variant:  int = VARIANT_RGB
        self._active_quality:  int = self._default_quality
        self._stream_timer         = None
        self._seq:             int = 0

        # Single-camera
        self._active_sub          = None
        self._latest_frame:  Optional[CameraFrame] = None

        # Mosaic: one sub + latest frame per source
        self._mosaic_subs:   list = []
        self._mosaic_frames: Dict[int, Optional[CameraFrame]] = {}

        # Terrain map: one sub per nexus output topic
        self._terrain_subs:   list = []
        self._terrain_frames: Dict[str, Optional[Image]] = {}

        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, '/network/client_ip',     self._on_client_ip, latched_qos)
        self.create_subscription(String, '/network/stream_request', self._on_request,   10)

        self.get_logger().info(
            f'Video stream node ready  UDP :{self._udp_port}  '
            f'default_quality={self._default_quality}  stream_scale={self._stream_scale}%'
        )

    # ── Connection ─────────────────────────────────────────────────────────────

    def _on_client_ip(self, msg: String):
        ip = msg.data.strip()
        if ip:
            self._gui_ip = ip
            self.get_logger().info(f'GUI IP set to {ip} — ready to stream')
        else:
            self._gui_ip = None
            self._stop_stream()

    # ── Request handling ───────────────────────────────────────────────────────

    def _on_request(self, msg: String):
        try:
            parts     = msg.data.strip().split(',')
            source_id = int(parts[0])
            variant   = ord(parts[1].upper()) if len(parts) > 1 else VARIANT_RGB
            quality   = int(parts[2])         if len(parts) > 2 else 0
            fps       = int(parts[3])         if len(parts) > 3 else 10
        except (ValueError, IndexError) as e:
            self.get_logger().error(f'Bad stream request "{msg.data}": {e}')
            return

        if source_id == SOURCE_STOP:
            self._stop_stream()
            return

        quality = quality if 1 <= quality <= 100 else self._default_quality
        fps     = max(1, min(fps, 30))
        self._start_stream(source_id, variant, quality, fps)

    def _start_stream(self, source_id: int, variant: int, quality: int, fps: int):
        self._stop_stream()

        self._active_source  = source_id
        self._active_variant = variant
        self._active_quality = quality

        if source_id == SOURCE_MAP:
            self._subscribe_terrain()
        elif source_id == SOURCE_MOSAIC:
            self._subscribe_mosaic(variant)
        else:
            self._subscribe_single(source_id, variant)

        self._stream_timer = self.create_timer(1.0 / fps, self._on_timer)

        name  = self._source_map.get(source_id, {}).get('name', f'src_{source_id}')
        vname = {VARIANT_RGB: 'RGB', VARIANT_DEPTH: 'depth', VARIANT_TERRAIN: 'terrain'}.get(variant, '?')
        self.get_logger().info(
            f'Streaming {name} {vname}  quality={quality}  '
            f'scale={self._stream_scale}%  {fps}fps'
        )

    def _stop_stream(self):
        if self._stream_timer:
            self._stream_timer.cancel()
            self._stream_timer = None
        if self._active_sub:
            self.destroy_subscription(self._active_sub)
            self._active_sub = None
        for sub in self._mosaic_subs:
            self.destroy_subscription(sub)
        self._mosaic_subs.clear()
        self._mosaic_frames.clear()
        for sub in self._terrain_subs:
            self.destroy_subscription(sub)
        self._terrain_subs.clear()
        self._terrain_frames.clear()
        self._latest_frame  = None
        self._active_source = None

    # ── Subscriptions ──────────────────────────────────────────────────────────

    def _subscribe_single(self, source_id: int, variant: int):
        src = self._source_map.get(source_id)
        if src is None:
            self.get_logger().error(f'Unknown source_id {source_id}')
            return
        topic = src['rgb_topic'] if variant == VARIANT_RGB else src['depth_topic']
        self._active_sub = self.create_subscription(
            CameraFrame, topic, self._single_cb, 10
        )

    def _subscribe_mosaic(self, variant: int):
        for idx, src in self._source_map.items():
            topic = src['rgb_topic'] if variant == VARIANT_RGB else src['depth_topic']
            sub = self.create_subscription(
                CameraFrame, topic,
                lambda msg, i=idx: self._mosaic_cb(i, msg), 10
            )
            self._mosaic_subs.append(sub)
            self._mosaic_frames[idx] = None

    def _subscribe_terrain(self):
        for key, topic in _TERRAIN_TOPICS.items():
            sub = self.create_subscription(
                Image, topic,
                lambda msg, k=key: self._terrain_cb(k, msg), 10
            )
            self._terrain_subs.append(sub)

    # ── Frame callbacks ────────────────────────────────────────────────────────

    def _single_cb(self, msg: CameraFrame):
        with self._lock:
            self._latest_frame = msg

    def _mosaic_cb(self, idx: int, msg: CameraFrame):
        with self._lock:
            self._mosaic_frames[idx] = msg

    def _terrain_cb(self, key: str, msg: Image):
        with self._lock:
            self._terrain_frames[key] = msg

    # ── Timer: encode + send ───────────────────────────────────────────────────

    def _on_timer(self):
        if self._gui_ip is None:
            return

        source_id = self._active_source
        variant   = self._active_variant
        quality   = self._active_quality

        if source_id == SOURCE_MAP:
            self._build_and_send_terrain()
            return

        if source_id == SOURCE_MOSAIC:
            bgr = self._build_mosaic(variant)
        else:
            with self._lock:
                frame = self._latest_frame
            if frame is None:
                return
            bgr = self._decode_frame(frame, variant, source_id)

        if bgr is None:
            return

        # Apply server-side spatial scale (set once in yaml, not GUI-controlled)
        if self._stream_scale != 100:
            w = max(1, int(bgr.shape[1] * self._stream_scale / 100))
            h = max(1, int(bgr.shape[0] * self._stream_scale / 100))
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        self._send_udp(bgr, source_id, variant, quality)

    # ── Image processing ───────────────────────────────────────────────────────

    def _decode_frame(self, frame: CameraFrame, variant: int, source_id: int):
        try:
            if variant == VARIANT_RGB:
                return self.bridge.imgmsg_to_cv2(frame.image, desired_encoding='bgr8')
            return self._depth_to_heatmap(frame.image)
        except Exception as e:
            self.get_logger().error(f'Frame decode error: {e}')
            return None

    def _depth_to_heatmap(self, img_msg) -> np.ndarray:
        enc = img_msg.encoding
        if enc == '32FC1':
            depth = self.bridge.imgmsg_to_cv2(img_msg, '32FC1')
            depth = np.clip(depth, 0.0, self._depth_max_m) / self._depth_max_m
        elif enc == '16UC1':
            depth = self.bridge.imgmsg_to_cv2(img_msg, '16UC1').astype(np.float32)
            depth = np.clip(depth, 0.0, self._orbbec_max_m * 1000.0) / (self._orbbec_max_m * 1000.0)
        else:
            depth = self.bridge.imgmsg_to_cv2(img_msg, 'passthrough').astype(np.float32)
            mn, mx = depth.min(), depth.max()
            depth = (depth - mn) / (mx - mn + 1e-6)
        return cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    def _build_mosaic(self, variant: int):
        with self._lock:
            frames = dict(self._mosaic_frames)

        cells = []
        for idx in sorted(self._source_map.keys()):
            frame = frames.get(idx)
            if frame is not None:
                cell = self._decode_frame(frame, variant, idx)
            else:
                cell = None
            if cell is None:
                cell = np.zeros((120, 160, 3), dtype=np.uint8)
            else:
                cell = cv2.resize(cell, (160, 120), interpolation=cv2.INTER_AREA)
            cells.append(cell)

        while len(cells) < 6:
            cells.append(np.zeros((120, 160, 3), dtype=np.uint8))

        return np.vstack([np.hstack(cells[0:3]), np.hstack(cells[3:6])])

    def _build_and_send_terrain(self):
        with self._lock:
            frames = dict(self._terrain_frames)

        if 'height' not in frames:
            return

        try:
            buf = bytearray()

            # Header: version=1, flags=bit0(terrain)|bit7(zlib_compressed)
            buf += struct.pack('<BBH', 1, 0x81, 0)  # version, flags, reserved

            # Terrain section
            buf += struct.pack('<IIf', 200, 200, 0.05)

            h_arr = self.bridge.imgmsg_to_cv2(frames['height'],  '32FC1')
            r_arr = self.bridge.imgmsg_to_cv2(frames.get('rocks',   frames['height']), '32FC1')
            c_arr = self.bridge.imgmsg_to_cv2(frames.get('craters', frames['height']), '32FC1')
            w_arr = self.bridge.imgmsg_to_cv2(frames.get('walls',   frames['height']), '32FC1')

            buf += h_arr.astype(np.float32).tobytes()
            buf += (np.clip(r_arr, 0.0, 1.0) * 255).astype(np.uint8).tobytes()
            buf += (np.clip(c_arr, 0.0, 1.0) * 255).astype(np.uint8).tobytes()
            buf += (np.clip(w_arr, 0.0, 1.0) * 255).astype(np.uint8).tobytes()

            # pose and nav sections not yet available — flags bits 1 and 2 stay 0

            compressed = zlib.compress(bytes(buf), level=6)
            self._send_raw_udp(compressed, SOURCE_MAP, VARIANT_TERRAIN)

        except Exception as e:
            self.get_logger().error(f'Terrain send error: {e}')

    # ── UDP ────────────────────────────────────────────────────────────────────

    def _send_raw_udp(self, data: bytes, source_id: int, variant: int):
        """Send pre-built binary data in chunks — no JPEG encoding."""
        chunks = [data[i:i + _CHUNK_SIZE] for i in range(0, len(data), _CHUNK_SIZE)]
        total  = len(chunks)
        seq    = self._seq
        self._seq = (self._seq + 1) % 65536

        for idx, chunk in enumerate(chunks):
            pkt = bytes([
                0x4F, ord('V'),
                source_id & 0xFF,
                variant   & 0xFF,
                (seq >> 8) & 0xFF, seq & 0xFF,
                idx, total,
            ]) + chunk
            try:
                self._udp_sock.sendto(pkt, (self._gui_ip, self._udp_port))
            except Exception as e:
                self.get_logger().error(f'UDP send error: {e}')
                return

    def _send_udp(self, bgr: np.ndarray, source_id: int, variant: int, quality: int):
        _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        data   = jpeg.tobytes()
        chunks = [data[i:i + _CHUNK_SIZE] for i in range(0, len(data), _CHUNK_SIZE)]
        total  = len(chunks)
        seq    = self._seq
        self._seq = (self._seq + 1) % 65536

        for idx, chunk in enumerate(chunks):
            pkt = bytes([
                0x4F, ord('V'),
                source_id & 0xFF,
                variant & 0xFF,
                (seq >> 8) & 0xFF, seq & 0xFF,
                idx, total,
            ]) + chunk
            try:
                self._udp_sock.sendto(pkt, (self._gui_ip, self._udp_port))
            except Exception as e:
                self.get_logger().error(f'UDP send error: {e}')
                return

    def destroy_node(self):
        self._stop_stream()
        self._udp_sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
