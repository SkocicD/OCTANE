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
                output='log',
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
                output='log',
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

    # ── nvblox (multi-camera TSDF mapping) ───────────────────────────────────
    # Subscribes to the splitter outputs for each camera and produces TSDF,
    # mesh, occupancy grid, and ESDF.  Camera poses come from the TF tree
    # published by the splitters (base_link → <cam>_frame).
    #
    # Uses input remappings to route each camera's shimmed topics into
    # nvblox's expected topic names.  Single nvblox instance handles all
    # cameras via the multi-camera input mode.
    nvblox_remappings = []
    for i, cam_name in enumerate(cfg['cameras'].keys()):
        nvblox_remappings.extend([
            (f'camera_{i}/depth/image',       f'mapping/{cam_name}/depth/image'),
            (f'camera_{i}/depth/camera_info', f'mapping/{cam_name}/depth/camera_info'),
            (f'camera_{i}/color/image',       f'mapping/{cam_name}/rgb/image'),
            (f'camera_{i}/color/camera_info', f'mapping/{cam_name}/rgb/camera_info'),
        ])

    try:
        from ament_index_python.packages import get_package_share_directory as _gpsd
        _gpsd('nvblox_ros')
        nodes.append(
            Node(
                package='nvblox_ros',
                executable='nvblox_node',
                name='nvblox_node',
                output='log',
                parameters=[{
                    'global_frame': 'odom',
                    'pose_frame':   'base_link',
                    'mapping_type': 'static_tsdf',
                    'voxel_size':   0.05,
                    'num_cameras':  len(cfg['cameras']),
                    'use_color':    True,
                    'use_depth':    True,
                    'use_lidar':    False,
                    'max_integration_distance_m': 5.0,
                    'integrate_color_radius_m':   5.0,
                    'esdf_mode':    'esdf_3d',
                }],
                remappings=nvblox_remappings,
            )
        )
    except Exception:
        print("[WARN] nvblox_ros not found — nvblox node skipped (build with --external to enable)")

    nodes.append(LogInfo(msg='Mapping subsystem online'))
    return LaunchDescription(nodes)
