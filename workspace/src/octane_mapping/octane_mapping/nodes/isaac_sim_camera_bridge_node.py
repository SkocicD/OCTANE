#!/usr/bin/env python3
"""Isaac Sim camera bridge — republishes raw sim images as CameraFrame messages.

Subscribes to sensor_msgs/Image + sensor_msgs/CameraInfo pairs published by
Isaac Sim and republishes them as octane_msgs/CameraFrame on the topics the
perception stack expects.

Purely passive — if Isaac Sim is not running the node sits idle and has no
effect on the real camera pipeline.
"""

import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from octane_msgs.msg import CameraFrame, CameraOffset, CameraParam

CAMERAS = [
    {
        'serial':    'left_front',
        'image_in':  '/isaac_sim/near/left_front/image',
        'info_in':   '/isaac_sim/near/left_front/camera_info',
        'frame_out': 'perception/camera/near/left_front/rgb/frame',
    },
    {
        'serial':    'left_side',
        'image_in':  '/isaac_sim/near/left_side/image',
        'info_in':   '/isaac_sim/near/left_side/camera_info',
        'frame_out': 'perception/camera/near/left_side/rgb/frame',
    },
    {
        'serial':    'right_front',
        'image_in':  '/isaac_sim/near/right_front/image',
        'info_in':   '/isaac_sim/near/right_front/camera_info',
        'frame_out': 'perception/camera/near/right_front/rgb/frame',
    },
    {
        'serial':    'right_side',
        'image_in':  '/isaac_sim/near/right_side/image',
        'info_in':   '/isaac_sim/near/right_side/camera_info',
        'frame_out': 'perception/camera/near/right_side/rgb/frame',
    },
    {
        'serial':    'back_rear',
        'image_in':  '/isaac_sim/near/back_rear/image',
        'info_in':   '/isaac_sim/near/back_rear/camera_info',
        'frame_out': 'perception/camera/near/back_rear/rgb/frame',
    },
    {
        'serial':    'depth_camera_rgb',
        'image_in':  '/isaac_sim/depth_camera/rgb/image',
        'info_in':   '/isaac_sim/depth_camera/rgb/camera_info',
        'frame_out': 'perception/camera/depth_camera/rgb/frame',
    },
    {
        'serial':    'depth_camera_depth',
        'image_in':  '/isaac_sim/depth_camera/depth/image',
        'info_in':   '/isaac_sim/depth_camera/depth/camera_info',
        'frame_out': 'perception/camera/depth_camera/depth/frame',
    },
]


class IsaacSimCameraBridgeNode(Node):

    def __init__(self):
        super().__init__('isaac_sim_camera_bridge_node')

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        for cam in CAMERAS:
            pub = self.create_publisher(CameraFrame, cam['frame_out'], 10)

            img_sub  = message_filters.Subscriber(self, Image,      cam['image_in'], qos_profile=qos)
            info_sub = message_filters.Subscriber(self, CameraInfo, cam['info_in'],  qos_profile=qos)

            sync = message_filters.ApproximateTimeSynchronizer(
                [img_sub, info_sub], queue_size=10, slop=0.05
            )
            sync.registerCallback(self._make_cb(pub, cam['serial']))

            # Keep references alive
            cam['_pub']      = pub
            cam['_img_sub']  = img_sub
            cam['_info_sub'] = info_sub
            cam['_sync']     = sync

        self.get_logger().info(
            f'Isaac Sim camera bridge ready — {len(CAMERAS)} cameras bridged'
        )

    def _make_cb(self, pub, serial: str):
        def cb(image: Image, info: CameraInfo):
            msg         = CameraFrame()
            msg.serial  = serial
            msg.image   = image
            msg.info    = info
            msg.param   = CameraParam()
            msg.offset  = CameraOffset()
            pub.publish(msg)
        return cb


def main(args=None):
    rclpy.init(args=args)
    node = IsaacSimCameraBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
