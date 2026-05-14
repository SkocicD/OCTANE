import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node

# Subscribes to all 6 RGB topics and all 6 depth topics (1 true Orbbec depth +
# 5 DA3 synthetic), writing each stream to an MP4 in a timestamped run folder
# under /media/csulunabotics/SSD2/OCTANE/cam_storage/.
#
# RGB videos are stored as bgr8.  Depth videos are normalised and colourised
# with COLORMAP_INFERNO for visual review.


def generate_launch_description():
    octane_share = get_package_share_directory('octane')
    config_file = os.path.join(octane_share, 'config', 'cameras.yaml')
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    rgb_topics = []
    depth_topics = []
    for cam_cfg in cfg['cameras'].values():
        rgb_topics.append(cam_cfg['rgb_topic'])
        depth_topics.append(cam_cfg['depth_topic'])

    return LaunchDescription([
        LogInfo(msg='Starting logging subsystem'),
        Node(
            package='octane_logging',
            executable='camera_recorder_node',
            name='camera_recorder_node',
            output='log',
            parameters=[{
                'output_base':  '/media/csulunabotics/SSD2/OCTANE/cam_storage',
                'rgb_topics':   rgb_topics,
                'depth_topics': depth_topics,
                'video_fps':    30.0,
            }],
        ),
        LogInfo(msg='Logging subsystem online'),
    ])
