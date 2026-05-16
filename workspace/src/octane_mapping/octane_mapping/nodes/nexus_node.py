#!/usr/bin/env python3
"""Nexus terrain model inference node.

Subscribes to all 12 perception camera frames (5 near RGB, 5 near depth,
1 Orbbec RGB, 1 Orbbec depth) plus IMU for robot orientation, runs the
Nexus terrain model, and publishes height/rocks/craters/walls maps.

Published topics:
  mapping/terrain/height   — sensor_msgs/Image (32FC1, 200×200, metres)
  mapping/terrain/rocks    — sensor_msgs/Image (32FC1, 200×200, probability 0–1)
  mapping/terrain/craters  — sensor_msgs/Image (32FC1, 200×200, probability 0–1)
  mapping/terrain/walls    — sensor_msgs/Image (32FC1, 200×200, probability 0–1)

Parameters:
  model_path        (str)   — path to nexus.pt checkpoint (required)
  depth_stats_path  (str)   — path to depth_stats.json (optional, has defaults)
  inference_rate    (float) — inference Hz (default 5.0)
  device            (str)   — 'auto' | 'cuda' | 'cpu' (default 'auto')
  imu_topic         (str)   — IMU topic for roll/pitch (default 'sensors/imu/accel')
  use_imu           (bool)  — if False, roll/pitch are fixed at 0 (default False)
"""

import json
import math
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu

from octane_msgs.msg import CameraFrame

# torch and TerrainModel are imported lazily in __init__ when debug_terrain=False

# Must match training/dataset.py exactly
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_HEIGHT_SCALE  = 0.5    # GT was divided by this during training; multiply back for metres
_DEPTH_MAX_M   = 20.0   # depth images clipped 0–20 m → 0–1

# Topic order must match _SLOT_PATHS in training/dataset.py
_RGB_TOPICS = [
    'perception/camera/near/left_front/rgb/frame',
    'perception/camera/near/left_side/rgb/frame',
    'perception/camera/near/right_front/rgb/frame',
    'perception/camera/near/right_side/rgb/frame',
    'perception/camera/near/back_rear/rgb/frame',
    'perception/camera/depth_camera/rgb/frame',
]
_DEPTH_TOPICS = [
    'perception/camera/near/left_front/depth/frame',
    'perception/camera/near/left_side/depth/frame',
    'perception/camera/near/right_front/depth/frame',
    'perception/camera/near/right_side/depth/frame',
    'perception/camera/near/back_rear/depth/frame',
    'perception/camera/depth_camera/depth/frame',
]
_ALL_TOPICS  = _RGB_TOPICS + _DEPTH_TOPICS
_RGB_SLOTS   = set(range(6))
_DEPTH_SLOTS = set(range(6, 12))


def _depth_to_meters(img_msg: Image, bridge: CvBridge) -> np.ndarray:
    enc = img_msg.encoding
    if enc == '32FC1':
        return bridge.imgmsg_to_cv2(img_msg, desired_encoding='32FC1')
    if enc in ('16UC1', 'mono16'):
        return bridge.imgmsg_to_cv2(img_msg, desired_encoding='16UC1').astype(np.float32) / 1000.0
    if enc == '32SC1':
        return bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough').astype(np.float32) / 1000.0
    # Fallback: uint8 grayscale treated as 0–20 m
    gray = bridge.imgmsg_to_cv2(img_msg, desired_encoding='mono8').astype(np.float32)
    return gray / 255.0 * _DEPTH_MAX_M


def _normalize_chw(rgb_hwc: np.ndarray, mean: list, std: list) -> np.ndarray:
    x = rgb_hwc.astype(np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return x.transpose(2, 0, 1)


class NexusNode(Node):

    def __init__(self):
        super().__init__('nexus_node')

        self.declare_parameter('model_path',       '')
        self.declare_parameter('depth_stats_path', '')
        self.declare_parameter('inference_rate',   5.0)
        self.declare_parameter('device',           'auto')
        self.declare_parameter('imu_topic',        'sensors/imu/accel')
        self.declare_parameter('use_imu',          False)
        self.declare_parameter('debug_terrain',    True)

        model_path       = self.get_parameter('model_path').value
        depth_stats_path = self.get_parameter('depth_stats_path').value
        inference_rate   = self.get_parameter('inference_rate').value
        device_param     = self.get_parameter('device').value
        imu_topic        = self.get_parameter('imu_topic').value
        self._use_imu    = self.get_parameter('use_imu').value
        debug_terrain    = self.get_parameter('debug_terrain').value

        self.bridge = CvBridge()

        # Publishers (created regardless of mode)
        self._pub_height  = self.create_publisher(Image, 'mapping/terrain/height',  10)
        self._pub_rocks   = self.create_publisher(Image, 'mapping/terrain/rocks',   10)
        self._pub_craters = self.create_publisher(Image, 'mapping/terrain/craters', 10)
        self._pub_walls   = self.create_publisher(Image, 'mapping/terrain/walls',   10)

        if debug_terrain:
            self.get_logger().info(
                f'debug_terrain=True — publishing synthetic terrain at {inference_rate} Hz'
            )
            self.create_timer(1.0 / inference_rate, self._run_debug)
            return

        if not model_path:
            raise RuntimeError('nexus_node: model_path parameter is required')

        import torch
        from octane_mapping.terrain_model import TerrainModel
        self._torch = torch

        # Device
        if device_param == 'auto':
            self.device = self._torch.device('cuda' if self._torch.cuda.is_available() else 'cpu')
        else:
            self.device = self._torch.device(device_param)

        # Load model
        self.get_logger().info(f'Loading terrain model: {model_path}')
        ckpt = self._torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = TerrainModel().to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self._torch.backends.cudnn.benchmark = True
        self.get_logger().info(
            f'Terrain model loaded  epoch={ckpt.get("epoch", "?")}  device={self.device}'
        )

        # Depth stats (normalisation must match training)
        if depth_stats_path:
            try:
                with open(depth_stats_path) as f:
                    ds = json.load(f)
                self._depth_mean = ds['mean']
                self._depth_std  = ds['std']
                self.get_logger().info(f'Depth stats loaded from {depth_stats_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load depth_stats_path: {e} — using defaults')
                self._depth_mean = [0.5, 0.5, 0.5]
                self._depth_std  = [0.25, 0.25, 0.25]
        else:
            self.get_logger().warn('depth_stats_path not set — using default depth normalisation')
            self._depth_mean = [0.5, 0.5, 0.5]
            self._depth_std  = [0.25, 0.25, 0.25]

        # Frame buffer: latest CameraFrame per topic
        self._frames: dict[str, CameraFrame] = {}
        self._lock = threading.Lock()
        self._inferring = False

        # Robot orientation from IMU gravity vector (yaw requires magnetometer/localization)
        self._roll  = 0.0
        self._pitch = 0.0
        self._yaw   = 0.0
        self._imu_lock = threading.Lock()

        # Subscriptions
        for topic in _ALL_TOPICS:
            self.create_subscription(
                CameraFrame, topic,
                lambda msg, t=topic: self._frame_cb(t, msg),
                10,
            )
        if self._use_imu:
            self.create_subscription(Imu, imu_topic, self._imu_cb, 10)
        else:
            self.get_logger().info('use_imu=False — roll/pitch fixed at 0')

        self.create_timer(1.0 / inference_rate, self._run_inference)
        self.get_logger().info(
            f'Nexus node ready  ({inference_rate} Hz, {len(_ALL_TOPICS)} camera topics, imu={self._use_imu})'
        )

    # ── Debug mode ────────────────────────────────────────────────────────────

    def _run_debug(self):
        stamp = self.get_clock().now().to_msg()
        N = 200

        # Smooth height field: sum of random Gaussian bumps
        height = np.zeros((N, N), dtype=np.float32)
        for _ in range(8):
            cx, cy   = np.random.randint(20, N - 20, 2)
            amp      = np.random.uniform(-0.4, 0.6)
            sigma    = np.random.uniform(10, 45)
            y, x     = np.ogrid[:N, :N]
            height  += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

        # Rocks: scattered tight Gaussian peaks
        rocks = np.zeros((N, N), dtype=np.float32)
        for _ in range(np.random.randint(5, 20)):
            cx, cy  = np.random.randint(0, N, 2)
            sigma   = np.random.uniform(2, 7)
            y, x    = np.ogrid[:N, :N]
            rocks  += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        rocks = np.clip(rocks, 0.0, 1.0).astype(np.float32)

        # Craters: wider, lower peaks
        craters = np.zeros((N, N), dtype=np.float32)
        for _ in range(np.random.randint(2, 8)):
            cx, cy    = np.random.randint(10, N - 10, 2)
            sigma     = np.random.uniform(6, 18)
            y, x      = np.ogrid[:N, :N]
            craters  += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        craters = np.clip(craters, 0.0, 1.0).astype(np.float32)

        # Walls: 1-3 random line segments rendered as Gaussian ridges
        walls = np.zeros((N, N), dtype=np.float32)
        y, x  = np.ogrid[:N, :N]
        for _ in range(np.random.randint(1, 4)):
            theta  = np.random.uniform(0, np.pi)
            rho    = np.random.uniform(30, N - 30)
            dist   = np.abs(x * np.cos(theta) + y * np.sin(theta) - rho)
            walls += np.exp(-dist ** 2 / (2 * 3 ** 2))
        walls = np.clip(walls, 0.0, 1.0).astype(np.float32)

        self._pub_height.publish(self._to_img(height,  stamp))
        self._pub_rocks.publish(self._to_img(rocks,    stamp))
        self._pub_craters.publish(self._to_img(craters, stamp))
        self._pub_walls.publish(self._to_img(walls,    stamp))

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _frame_cb(self, topic: str, msg: CameraFrame):
        with self._lock:
            self._frames[topic] = msg

    def _imu_cb(self, msg: Imu):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        with self._imu_lock:
            self._roll  = math.atan2(ay, az)
            self._pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))

    # ── Inference timer ───────────────────────────────────────────────────────

    def _run_inference(self):
        if self._inferring:
            return
        self._inferring = True
        try:
            self._do_inference()
        finally:
            self._inferring = False

    def _do_inference(self):
        with self._lock:
            frames = dict(self._frames)
        with self._imu_lock:
            roll, pitch, yaw = self._roll, self._pitch, self._yaw

        missing = [t for t in _ALL_TOPICS if t not in frames]
        if missing:
            self.get_logger().warn(
                f'Waiting for {len(missing)} frame(s), e.g. {missing[0]}',
                throttle_duration_sec=5.0,
            )
            return

        try:
            imgs = self._preprocess(frames)
        except Exception as e:
            self.get_logger().error(f'Preprocessing error: {e}')
            return

        rot = self._torch.tensor([
            math.sin(roll),  math.cos(roll),
            math.sin(pitch), math.cos(pitch),
            math.sin(yaw),   math.cos(yaw),
        ], dtype=self._torch.float32, device=self.device).unsqueeze(0)  # (1, 6)

        with self._torch.inference_mode():
            preds = self.model(imgs, rot)

        stamp = frames[_ALL_TOPICS[0]].image.header.stamp

        height  = (preds['height'].squeeze(0).cpu().numpy() * _HEIGHT_SCALE).astype(np.float32)
        rocks   = preds['rocks'].squeeze(0).cpu().numpy().astype(np.float32)
        craters = preds['craters'].squeeze(0).cpu().numpy().astype(np.float32)
        walls   = preds['walls'].squeeze(0).cpu().numpy().astype(np.float32)

        self._pub_height.publish(self._to_img(height,  stamp))
        self._pub_rocks.publish(self._to_img(rocks,   stamp))
        self._pub_craters.publish(self._to_img(craters, stamp))
        self._pub_walls.publish(self._to_img(walls,   stamp))

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, frames: dict):
        tensors = []
        for i, topic in enumerate(_ALL_TOPICS):
            frame = frames[topic]
            if i in _RGB_SLOTS:
                bgr = self.bridge.imgmsg_to_cv2(frame.image, desired_encoding='bgr8')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                chw = _normalize_chw(rgb, _IMAGENET_MEAN, _IMAGENET_STD)
            else:
                meters = _depth_to_meters(frame.image, self.bridge)
                norm   = np.clip(meters / _DEPTH_MAX_M, 0.0, 1.0).astype(np.float32)
                rgb    = np.stack([norm, norm, norm], axis=-1)
                rgb    = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
                mean   = np.array(self._depth_mean, dtype=np.float32)
                std    = np.array(self._depth_std,  dtype=np.float32)
                chw    = ((rgb - mean) / std).transpose(2, 0, 1)
            tensors.append(self._torch.from_numpy(chw))

        return self._torch.stack(tensors).unsqueeze(0).to(self.device)  # (1, 12, 3, 224, 224)

    def _to_img(self, arr: np.ndarray, stamp) -> Image:
        msg = self.bridge.cv2_to_imgmsg(arr, encoding='32FC1')
        msg.header.stamp    = stamp
        msg.header.frame_id = 'base_link'
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = NexusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
