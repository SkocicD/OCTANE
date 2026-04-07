#!/usr/bin/env python3
"""Depth estimation node using Depth Anything V3.

Subscribes to N CameraFrame topics, runs batched DA3 metric depth inference
on the latest frames, and publishes depth images (32FC1, meters) on
corresponding output topics.

Example:
    ros2 run octane_perception depth_estimation_node \
        --ros-args \
        -p input_topics:="['/cam0/frame', '/cam1/frame', '/cam2/frame']" \
        -p model_name:="depth-anything/DA3METRIC-LARGE" \
        -p inference_rate:=10.0
"""

import threading

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from octane_msgs.msg import CameraFrame


class DepthEstimationNode(Node):

    def __init__(self):
        super().__init__('depth_estimation_node')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_name', 'depth-anything/DA3METRIC-LARGE')
        self.declare_parameter('model_cache_dir', '')  # empty = auto (workspace/models/da3)
        self.declare_parameter('input_topics', ['/cam0/frame'])
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('process_res', 504)

        model_name = self.get_parameter('model_name').value
        model_cache_dir = self.get_parameter('model_cache_dir').value
        input_topics = self.get_parameter('input_topics').value
        inference_rate = self.get_parameter('inference_rate').value
        self.process_res = self.get_parameter('process_res').value

        self.bridge = CvBridge()

        # ── resolve model cache directory ─────────────────────────────────────
        # Default: workspace/models/da3 (next to src/, easy to find)
        if not model_cache_dir:
            import os
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            model_cache_dir = os.path.abspath(
                os.path.join(pkg_dir, '..', '..', '..', '..', '..', 'models', 'da3')
            )
        import os
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = model_cache_dir

        # ── load DA3 model ────────────────────────────────────────────────────
        self.get_logger().info(f'Loading model: {model_name}  (cache: {model_cache_dir})')
        from depth_anything_3.api import DepthAnything3

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DepthAnything3.from_pretrained(model_name).to(device)
        self.device = device
        self.is_metric = 'METRIC' in model_name.upper() or 'NESTED' in model_name.upper()
        self.get_logger().info(
            f'Model loaded on {device}  (metric={self.is_metric})'
        )

        # ── subscriptions + publishers ────────────────────────────────────────
        # For each input topic "/camN/frame", publish depth on "/camN/depth"
        self._latest_frames: dict[str, CameraFrame] = {}
        self._lock = threading.Lock()
        self._depth_pubs: dict[str, object] = {}

        for topic in input_topics:
            # Derive depth topic: /camN/frame → /camN/depth
            if topic.endswith('/frame'):
                depth_topic = topic.rsplit('/frame', 1)[0] + '/depth'
            else:
                depth_topic = topic + '/depth'

            self._depth_pubs[topic] = self.create_publisher(Image, depth_topic, 10)
            self.create_subscription(
                CameraFrame, topic,
                lambda msg, t=topic: self._frame_callback(t, msg),
                10,
            )
            self.get_logger().info(f'  {topic} → {depth_topic}')

        # ── inference timer ───────────────────────────────────────────────────
        self.create_timer(1.0 / inference_rate, self._run_inference)
        self.get_logger().info(
            f'Depth estimation node started  ({len(input_topics)} cameras, '
            f'{inference_rate} Hz inference)'
        )

    def _frame_callback(self, topic: str, msg: CameraFrame):
        with self._lock:
            self._latest_frames[topic] = msg

    def _run_inference(self):
        # Grab the latest frame from each subscribed topic
        with self._lock:
            snapshot = dict(self._latest_frames)

        if not snapshot:
            return

        topics = list(snapshot.keys())
        frames = [snapshot[t] for t in topics]

        # Convert ROS images → RGB numpy arrays
        rgb_images = []
        for frame in frames:
            try:
                bgr = self.bridge.imgmsg_to_cv2(frame.image, desired_encoding='bgr8')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_images.append(rgb)
            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')
                return

        # Batched inference
        with torch.inference_mode():
            prediction = self.model.inference(rgb_images)

        # Convert depth output → metric meters and publish
        for i, topic in enumerate(topics):
            depth = prediction.depth[i]  # (H_proc, W_proc) float32

            if self.is_metric:
                # DA3METRIC: canonical metric → meters
                fx = prediction.intrinsics[i, 0, 0]
                fy = prediction.intrinsics[i, 1, 1]
                focal_px = (fx + fy) / 2.0
                depth_meters = (focal_px * depth / 300.0).astype(np.float32)
            else:
                # Relative depth (unitless) — publish as-is
                depth_meters = depth.astype(np.float32)

            # Resize to original image dimensions
            orig_h = frames[i].image.height
            orig_w = frames[i].image.width
            if depth_meters.shape[0] != orig_h or depth_meters.shape[1] != orig_w:
                depth_meters = cv2.resize(
                    depth_meters, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

            # Publish as 32FC1 Image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_meters, encoding='32FC1')
            depth_msg.header = frames[i].image.header
            self._depth_pubs[topic].publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
