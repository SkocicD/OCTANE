import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


# Mapping subsystem launch:
#   - One CameraFrameSplitter per camera (CameraFrame → Image+CameraInfo + TF)
#   - nvblox node consuming the shimmed topics for multi-camera TSDF mapping
#
# All cameras are read from the central cameras.yaml config.

def generate_launch_description():
    nodes = [LogInfo(msg='Starting mapping subsystem')]

    # Load camera config
    config_file = os.path.join(
        get_package_share_directory('octane'), 'config', 'cameras.yaml'
    )
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    # ── Camera frame splitters (one per camera) ──────────────────────────────
    # Each splitter takes a CameraFrame topic and re-publishes Image +
    # CameraInfo on standard topic names that nvblox subscribes to.
    for cam_name, cam_cfg in cfg['cameras'].items():
        # RGB stream: rgb_topic → mapping/<cam>/rgb/{image,camera_info}
        nodes.append(
            Node(
                package='octane_mapping',
                executable='camera_frame_splitter',
                name=f'{cam_name}_rgb_splitter',
                output='screen',
                parameters=[{
                    'camera_name': cam_name,
                    'input_topic': cam_cfg['rgb_topic'],
                    'output_namespace': f'mapping/{cam_name}/rgb',
                    'frame_id': f'{cam_name}_frame',
                    'parent_frame': 'base_link',
                    'config_file': config_file,
                }],
            )
        )
        # Depth stream: depth_topic → mapping/<cam>/depth/{image,camera_info}
        nodes.append(
            Node(
                package='octane_mapping',
                executable='camera_frame_splitter',
                name=f'{cam_name}_depth_splitter',
                output='screen',
                parameters=[{
                    'camera_name': cam_name,
                    'input_topic': cam_cfg['depth_topic'],
                    'output_namespace': f'mapping/{cam_name}/depth',
                    'frame_id': f'{cam_name}_frame',
                    'parent_frame': 'base_link',
                    'config_file': config_file,
                }],
            )
        )

    # ── nvblox (TODO) ────────────────────────────────────────────────────────
    # Will subscribe to mapping/<cam>/depth/image + mapping/<cam>/depth/camera_info
    # for each camera and produce TSDF / mesh / occupancy / ESDF outputs.
    # Uses the TF tree the splitters publish for camera poses.

    nodes.append(LogInfo(msg='Mapping subsystem online'))
    return LaunchDescription(nodes)
