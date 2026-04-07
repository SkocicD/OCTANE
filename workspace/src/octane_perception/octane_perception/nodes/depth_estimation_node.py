#!/usr/bin/env python3
"""DA3 depth estimation node.

Subscribes to N RGB CameraFrame topics, runs one shared DA3 model on the
batched frames, and publishes a depth CameraFrame for each input. Output
topic is the input topic with /rgb/ swapped to /depth/.
"""

import os
import threading

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node

from octane_msgs.msg import CameraFrame


def _derive_depth_topic(rgb_topic: str) -> str:
    if '/rgb/' in rgb_topic:
        return rgb_topic.replace('/rgb/', '/depth/', 1)
    if rgb_topic.endswith('/frame'):
        return rgb_topic.rsplit('/frame', 1)[0] + '/depth/frame'
    return rgb_topic + '_depth'


class DepthEstimationNode(Node):

    def __init__(self):
        super().__init__('depth_estimation_node')

        # Parameters
        self.declare_parameter('model_name', 'depth-anything/DA3METRIC-LARGE')
        self.declare_parameter('model_cache_dir', '')
        self.declare_parameter('input_topics', ['/cam0/frame'])
        self.declare_parameter('inference_rate', 10.0)
        self.declare_parameter('process_res', 504)

        model_name = self.get_parameter('model_name').value
        model_cache_dir = self.get_parameter('model_cache_dir').value
        input_topics = self.get_parameter('input_topics').value
        inference_rate = self.get_parameter('inference_rate').value
        self.process_res = self.get_parameter('process_res').value

        self.bridge = CvBridge()

        # Resolve model cache dir (default: workspace/models/da3)
        if not model_cache_dir:
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            model_cache_dir = os.path.abspath(
                os.path.join(pkg_dir, '..', '..', '..', '..', '..', 'models', 'da3')
            )
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = model_cache_dir

        # Load DA3 (single shared model for all cameras)
        self.get_logger().info(f'Loading model: {model_name}  (cache: {model_cache_dir})')
        from depth_anything_3.api import DepthAnything3

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DepthAnything3.from_pretrained(model_name).to(device)
        self.device = device
        self.is_metric = 'METRIC' in model_name.upper() or 'NESTED' in model_name.upper()
        self.get_logger().info(f'Model loaded on {device}  (metric={self.is_metric})')

        # Per-camera subs and pubs
        self._latest_frames: dict[str, CameraFrame] = {}
        self._lock = threading.Lock()
        self._depth_pubs: dict[str, object] = {}

        for topic in input_topics:
            depth_topic = _derive_depth_topic(topic)
            self._depth_pubs[topic] = self.create_publisher(CameraFrame, depth_topic, 10)
            self.create_subscription(
                CameraFrame, topic,
                lambda msg, t=topic: self._frame_callback(t, msg),
                10,
            )
            self.get_logger().info(f'  {topic} → {depth_topic}')

        self.create_timer(1.0 / inference_rate, self._run_inference)
        self.get_logger().info(
            f'Depth estimation node started  ({len(input_topics)} cameras, '
            f'{inference_rate} Hz inference)'
        )

    def _frame_callback(self, topic: str, msg: CameraFrame):
        with self._lock:
            self._latest_frames[topic] = msg

    def _run_inference(self):
        with self._lock:
            snapshot = dict(self._latest_frames)

        if not snapshot:
            return

        topics = list(snapshot.keys())
        frames = [snapshot[t] for t in topics]

        # ROS Image → RGB numpy
        rgb_images = []
        for frame in frames:
            try:
                bgr = self.bridge.imgmsg_to_cv2(frame.image, desired_encoding='bgr8')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_images.append(rgb)
            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')
                return

        # Single batched forward pass for all cameras
        with torch.inference_mode():
            prediction = self.model.inference(rgb_images)

        for i, topic in enumerate(topics):
            depth = prediction.depth[i]

            if self.is_metric:
                # Canonical metric → meters
                fx = prediction.intrinsics[i, 0, 0]
                fy = prediction.intrinsics[i, 1, 1]
                focal_px = (fx + fy) / 2.0
                depth_meters = (focal_px * depth / 300.0).astype(np.float32)
            else:
                depth_meters = depth.astype(np.float32)

            # Resize back to source resolution
            orig_h = frames[i].image.height
            orig_w = frames[i].image.width
            if depth_meters.shape[0] != orig_h or depth_meters.shape[1] != orig_w:
                depth_meters = cv2.resize(
                    depth_meters, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

            depth_img_msg = self.bridge.cv2_to_imgmsg(depth_meters, encoding='32FC1')
            depth_img_msg.header = frames[i].image.header

            # Wrap in CameraFrame, copy metadata from source
            out = CameraFrame()
            out.serial = frames[i].serial
            out.image = depth_img_msg
            out.info = frames[i].info
            out.param = frames[i].param
            out.offset = frames[i].offset
            self._depth_pubs[topic].publish(out)


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
