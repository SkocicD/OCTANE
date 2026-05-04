"""Publish Isaac Sim camera renders as sensor_msgs/Image + CameraInfo via rclpy.

Publishes raw sensor_msgs topics.  A separate bridge node in WSL
(isaac_sim_camera_bridge_node) wraps them into octane_msgs/CameraFrame
on the topics the perception stack expects.

Isaac Sim raw topics (sensor_msgs/Image):
    /isaac_sim/near/left_front/image
    /isaac_sim/near/left_side/image
    /isaac_sim/near/right_front/image
    /isaac_sim/near/right_side/image
    /isaac_sim/near/back_rear/image
    /isaac_sim/depth_camera/rgb/image
    /isaac_sim/depth_camera/depth/image   (32FC1, metres)

Paired camera_info on  /isaac_sim/<name>/camera_info
"""

from __future__ import annotations
import numpy as np

_ROBOT_ROOT = "/World/envs/env_0/Robot"
_CAM_BASE   = f"{_ROBOT_ROOT}/CSU_Lunabotics_Isaac_Lab_Model/tn__base_link1_wJ/tn__Cameras1_XG"

# (prim_path, isaac_topic_name, serial, is_depth, width, height, fx, fy, cx, cy)
_CAMERA_CONFIGS: list[tuple] = [
    (_CAM_BASE + "/Innomaker_RGB_130_LEFT_FRONT",  "near/left_front",   "left_front",    False, 480, 360, 461.0, 461.0, 240.0, 180.0),
    (_CAM_BASE + "/Innomaker_RGB_130_LEFT_SIDE",   "near/left_side",    "left_side",     False, 480, 360, 461.0, 461.0, 240.0, 180.0),
    (_CAM_BASE + "/Innomaker_RGB_130_RIGHT_FRONT", "near/right_front",  "right_front",   False, 480, 360, 461.0, 461.0, 240.0, 180.0),
    (_CAM_BASE + "/Innomaker_RGB_130_RIGHT_SIDE",  "near/right_side",   "right_side",    False, 480, 360, 461.0, 461.0, 240.0, 180.0),
    (_CAM_BASE + "/Innomaker_RGB_130_BACK_REAR",   "near/back_rear",    "back_rear",     False, 480, 360, 461.0, 461.0, 240.0, 180.0),
    (_CAM_BASE + "/Orbbec_Astra_Pro_RGB",          "depth_camera/rgb",  "depth_camera",  False, 640, 480, 525.0, 525.0, 320.0, 240.0),
    (_CAM_BASE + "/Orbbec_Astra_Pro_D",            "depth_camera/depth","depth_camera_d",True,  640, 480, 525.0, 525.0, 320.0, 240.0),
]


class CameraRosPublisher:
    """Manages annotators and rclpy publishers for all simulation cameras."""

    def __init__(self):
        self._ready = False
        self._cameras: list[dict] = []
        self._node = None

    # ──────────────────────────────────────────────────────────────────────
    def setup(self) -> None:
        """Create render products, annotators, and publishers.  Call once after scene loads."""
        try:
            import rclpy
            import omni.replicator.core as rep
            from sensor_msgs.msg import Image, CameraInfo
            from cv_bridge import CvBridge
        except ImportError as e:
            print(f"[CameraRosPublisher] SKIP — missing dependency: {e}")
            return

        if not rclpy.ok():
            rclpy.init()
        self._node   = rclpy.create_node("isaac_sim_cameras")
        self._bridge = CvBridge()

        for (prim, name, serial, is_depth, w, h, fx, fy, cx, cy) in _CAMERA_CONFIGS:
            try:
                rp     = rep.create.render_product(prim, resolution=(w, h))
                annot  = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_image_plane" if is_depth else "rgb"
                )
                annot.attach([rp])
            except Exception as e:
                print(f"[CameraRosPublisher] WARNING: could not attach annotator for {prim}: {e}")
                continue

            img_topic  = f"/isaac_sim/{name}/image"
            info_topic = f"/isaac_sim/{name}/camera_info"
            img_pub    = self._node.create_publisher(Image,       img_topic,  1)
            info_pub   = self._node.create_publisher(CameraInfo,  info_topic, 1)

            self._cameras.append({
                "prim": prim, "name": name, "serial": serial,
                "is_depth": is_depth,
                "w": w, "h": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                "annotator": annot,
                "img_pub": img_pub, "info_pub": info_pub,
            })
            print(f"[CameraRosPublisher]  {img_topic}")

        self._ready = bool(self._cameras)
        print(f"[CameraRosPublisher] {len(self._cameras)}/{len(_CAMERA_CONFIGS)} cameras ready")

    # ──────────────────────────────────────────────────────────────────────
    def publish(self) -> None:
        """Read latest rendered frames and publish.  Call each step after rendering."""
        if not self._ready:
            return

        import rclpy
        from sensor_msgs.msg import Image, CameraInfo

        now = self._node.get_clock().now().to_msg()

        for cam in self._cameras:
            data = cam["annotator"].get_data()
            if data is None or data.size == 0:
                continue

            w, h = cam["w"], cam["h"]

            # ── image ────────────────────────────────────────────────────
            if cam["is_depth"]:
                img_arr = data.reshape(h, w).astype(np.float32)
                img_msg = self._bridge.cv2_to_imgmsg(img_arr, encoding="32FC1")
            else:
                rgba    = data.reshape(h, w, 4)
                bgr     = rgba[:, :, :3][:, :, ::-1].copy()
                img_msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")

            img_msg.header.stamp    = now
            img_msg.header.frame_id = cam["serial"]
            cam["img_pub"].publish(img_msg)

            # ── camera_info ──────────────────────────────────────────────
            info             = CameraInfo()
            info.header.stamp    = now
            info.header.frame_id = cam["serial"]
            info.width   = w
            info.height  = h
            info.distortion_model = "plumb_bob"
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
            fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            cam["info_pub"].publish(info)

        rclpy.spin_once(self._node, timeout_sec=0.0)

    # ──────────────────────────────────────────────────────────────────────
    def destroy(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        self._ready = False
