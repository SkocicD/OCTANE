#!/usr/bin/env python3
"""Video stream node — compresses camera frames and sends them to the GUI over UDP.

Stream requests arrive on /network/stream_request (std_msgs/String):
    "<source_id>,<variant>,<scale>,<fps>"
    e.g.  "2,R,50,10"   → camera 2, RGB, 50% scale, 10 fps
    e.g.  "6,D,25,5"    → mosaic, depth heatmap, 25% scale, 5 fps
    e.g.  "255,R,0,10"  → stop all streams

source_id values:
    0 = orbbec_depth
    1 = near_rgb_left_side
    2 = near_rgb_left_front
    3 = near_rgb_right_side
    4 = near_rgb_right_front
    5 = near_rgb_back_rear
    6 = mosaic  (all 6 tiled 3×2)
    7 = nvblox ESDF map slice
    255 = stop all

variant: R = RGB,  D = depth heatmap (COLORMAP_INFERNO)

scale: 1–100 (% of native resolution).  0 = use node's default_scale parameter.

UDP frame format (rover → GUI, port udp_port):
    [0x4F][0x56][source_id][variant][seq_hi][seq_lo][chunk_idx][chunk_total][...JPEG...]
    8-byte header, up to 1392 bytes of JPEG payload per packet.
    GUI reassembles by (seq, chunk_idx) and decodes JPEG.
"""

import os
import socket
import threading
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String

from octane_msgs.msg import CameraFrame

SOURCE_MOSAIC = 6
SOURCE_MAP    = 7
SOURCE_STOP   = 255

VARIANT_RGB   = ord('R')
VARIANT_DEPTH = ord('D')

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
        self.declare_parameter('default_scale',       50)
        self.declare_parameter('jpeg_quality',        70)
        self.declare_parameter('depth_max_m',         8.0)
        self.declare_parameter('orbbec_depth_max_m',  5.0)
        self.declare_parameter('config_file',         '')

        self._udp_port      = self.get_parameter('udp_port').value
        self._default_scale = self.get_parameter('default_scale').value
        self._jpeg_quality  = self.get_parameter('jpeg_quality').value
        self._depth_max_m   = self.get_parameter('depth_max_m').value
        self._orbbec_max_m  = self.get_parameter('orbbec_depth_max_m').value
        config_file         = self.get_parameter('config_file').value

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
        self._gui_ip:        Optional[str] = None
        self._active_source: Optional[int] = None
        self._active_variant: int = VARIANT_RGB
        self._active_scale:   int = self._default_scale
        self._stream_timer        = None
        self._seq:            int = 0

        # Single-camera
        self._active_sub          = None
        self._latest_frame:  Optional[CameraFrame] = None

        # Mosaic: one sub + latest frame per source
        self._mosaic_subs:   list = []
        self._mosaic_frames: Dict[int, Optional[CameraFrame]] = {}

        self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.create_subscription(String, '/network/client_ip',      self._on_client_ip, 10)
        self.create_subscription(String, '/network/stream_request',  self._on_request,   10)

        self.get_logger().info(
            f'Video stream node ready  UDP :{self._udp_port}  '
            f'default_scale={self._default_scale}%  quality={self._jpeg_quality}'
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
            scale     = int(parts[2])         if len(parts) > 2 else 0
            fps       = int(parts[3])         if len(parts) > 3 else 10
        except (ValueError, IndexError) as e:
            self.get_logger().error(f'Bad stream request "{msg.data}": {e}')
            return

        if source_id == SOURCE_STOP:
            self._stop_stream()
            return

        scale = scale if 1 <= scale <= 100 else self._default_scale
        fps   = max(1, min(fps, 30))
        self._start_stream(source_id, variant, scale, fps)

    def _start_stream(self, source_id: int, variant: int, scale: int, fps: int):
        self._stop_stream()

        self._active_source  = source_id
        self._active_variant = variant
        self._active_scale   = scale

        if source_id == SOURCE_MAP:
            self._subscribe_map()
        elif source_id == SOURCE_MOSAIC:
            self._subscribe_mosaic(variant)
        else:
            self._subscribe_single(source_id, variant)

        self._stream_timer = self.create_timer(1.0 / fps, self._on_timer)

        name  = self._source_map.get(source_id, {}).get('name', f'src_{source_id}')
        vname = 'RGB' if variant == VARIANT_RGB else 'depth'
        self.get_logger().info(f'Streaming {name} {vname} @ {scale}% {fps}fps')

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

    def _subscribe_map(self):
        try:
            from nvblox_msgs.msg import DistanceMapSlice
            self._active_sub = self.create_subscription(
                DistanceMapSlice, '/nvblox_node/static_map_slice', self._map_cb, 10
            )
        except ImportError:
            self.get_logger().warn('nvblox_msgs unavailable — map stream not supported')

    # ── Frame callbacks ────────────────────────────────────────────────────────

    def _single_cb(self, msg: CameraFrame):
        with self._lock:
            self._latest_frame = msg

    def _mosaic_cb(self, idx: int, msg: CameraFrame):
        with self._lock:
            self._mosaic_frames[idx] = msg

    def _map_cb(self, msg):
        with self._lock:
            self._latest_frame = msg

    # ── Timer: encode + send ───────────────────────────────────────────────────

    def _on_timer(self):
        if self._gui_ip is None:
            return

        source_id = self._active_source
        variant   = self._active_variant
        scale     = self._active_scale

        if source_id == SOURCE_MOSAIC:
            bgr = self._build_mosaic(variant)
        elif source_id == SOURCE_MAP:
            bgr = self._build_map()
        else:
            with self._lock:
                frame = self._latest_frame
            if frame is None:
                return
            bgr = self._decode_frame(frame, variant, source_id)

        if bgr is None:
            return

        if scale != 100:
            w = max(1, int(bgr.shape[1] * scale / 100))
            h = max(1, int(bgr.shape[0] * scale / 100))
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

        self._send_udp(bgr, source_id, variant)

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

    def _build_map(self):
        with self._lock:
            msg = self._latest_frame
        if msg is None:
            return None
        try:
            data = np.array(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            data = np.clip(data, 0.0, 2.0) / 2.0
            return cv2.applyColorMap((data * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        except Exception as e:
            self.get_logger().error(f'Map render error: {e}')
            return None

    # ── UDP ────────────────────────────────────────────────────────────────────

    def _send_udp(self, bgr: np.ndarray, source_id: int, variant: int):
        _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
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
