import os
import threading
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from octane_msgs.msg import CameraFrame


def _topic_to_filename(topic: str) -> str:
    # "perception/camera/near/left_side/rgb/frame" -> "near_left_side_rgb"
    name = topic
    if name.startswith('perception/camera/'):
        name = name[len('perception/camera/'):]
    if name.endswith('/frame'):
        name = name[:-len('/frame')]
    return name.replace('/', '_')


class CameraRecorderNode(Node):

    def __init__(self):
        super().__init__('camera_recorder_node')

        self.declare_parameter('output_base', '/media/csulunabotics/SSD2/OCTANE/cam_storage')
        self.declare_parameter('rgb_topics', [''])
        self.declare_parameter('depth_topics', [''])
        self.declare_parameter('video_fps', 30.0)

        output_base = self.get_parameter('output_base').value
        rgb_topics = [t for t in self.get_parameter('rgb_topics').value if t]
        depth_topics = [t for t in self.get_parameter('depth_topics').value if t]
        self._fps = float(self.get_parameter('video_fps').value)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._run_dir = os.path.join(output_base, timestamp)
        os.makedirs(self._run_dir, exist_ok=True)
        self.get_logger().info(f'Recording run: {self._run_dir}')

        self._bridge = CvBridge()
        self._writers: dict[str, cv2.VideoWriter | None] = {}
        self._lock = threading.Lock()

        for topic in rgb_topics:
            self._subscribe(topic, is_depth=False)
        for topic in depth_topics:
            self._subscribe(topic, is_depth=True)

        self.get_logger().info(
            f'Camera recorder started  ({len(rgb_topics)} RGB + {len(depth_topics)} depth topics)'
        )

    def _subscribe(self, topic: str, is_depth: bool) -> None:
        self.create_subscription(
            CameraFrame,
            topic,
            lambda msg, t=topic, d=is_depth: self._frame_cb(t, msg, d),
            10,
        )

    def _frame_cb(self, topic: str, msg: CameraFrame, is_depth: bool) -> None:
        try:
            if is_depth:
                depth_f32 = self._bridge.imgmsg_to_cv2(msg.image, desired_encoding='32FC1')
                normed = cv2.normalize(depth_f32, None, 0, 255, cv2.NORM_MINMAX)
                frame_bgr = cv2.applyColorMap(normed.astype(np.uint8), cv2.COLORMAP_INFERNO)
            else:
                frame_bgr = self._bridge.imgmsg_to_cv2(msg.image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image convert error [{topic}]: {e}', throttle_duration_sec=5.0)
            return

        h, w = frame_bgr.shape[:2]
        with self._lock:
            writer = self._get_or_create_writer(topic, w, h)
        if writer is not None:
            writer.write(frame_bgr)

    def _get_or_create_writer(self, topic: str, w: int, h: int) -> 'cv2.VideoWriter | None':
        if topic not in self._writers:
            filename = _topic_to_filename(topic) + '.mp4'
            path = os.path.join(self._run_dir, filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
            if not writer.isOpened():
                self.get_logger().error(f'Failed to open VideoWriter: {path}')
                self._writers[topic] = None
            else:
                self.get_logger().info(f'  {topic}  →  {filename}')
                self._writers[topic] = writer
        return self._writers[topic]

    def destroy_node(self) -> None:
        with self._lock:
            for writer in self._writers.values():
                if writer is not None:
                    writer.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
