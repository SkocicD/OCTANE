import os

import yaml
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


# Zone definitions live in octane/config/zones.yaml — edit there.


def generate_launch_description():
    nodes = [
        LogInfo(msg='Starting mapping subsystem'),
        # Stub identity odom→base_link until octane_localization is ready
        # (April tags + IMU fusion).  Replace this node with the real source then.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
            output='log',
        ),
    ]

    # Load camera config
    try:
        octane_share = get_package_share_directory('octane')
        config_file  = os.path.join(octane_share, 'config', 'cameras.yaml')
        zones_file   = os.path.join(octane_share, 'config', 'zones.yaml')
        nvblox_params = os.path.join(octane_share, 'config', 'nvblox.yaml')
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        with open(zones_file) as f:
            zones_cfg = yaml.safe_load(f)
    except Exception as e:
        print(f'[WARN] Could not load config: {e} — skipping mapping nodes')
        nodes.append(LogInfo(msg='Mapping subsystem skipped (config unavailable)'))
        return LaunchDescription(nodes)

    # ── Camera frame splitters (one per camera, per stream) ───────────────────
    # Unpacks each CameraFrame into Image + CameraInfo on nvblox-compatible
    # topic names, and broadcasts the static TF base_link → <cam>_frame.
    for cam_name, cam_cfg in cfg['cameras'].items():
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

    # ── nvblox — live 3D view around the robot ────────────────────────────────
    # Integrates depth from all 6 cameras into a rolling TSDF/ESDF.
    # TSDF decay + 4 m radius clearing keeps it a live view, not a persistent
    # global map.  Camera poses come from the TF tree above.
    nvblox_remappings = []
    for i, cam_name in enumerate(cfg['cameras'].keys()):
        nvblox_remappings.extend([
            (f'camera_{i}/depth/image',       f'mapping/{cam_name}/depth/image'),
            (f'camera_{i}/depth/camera_info', f'mapping/{cam_name}/depth/camera_info'),
            (f'camera_{i}/color/image',       f'mapping/{cam_name}/rgb/image'),
            (f'camera_{i}/color/camera_info', f'mapping/{cam_name}/rgb/camera_info'),
        ])

    try:
        get_package_share_directory('nvblox_ros')
        nodes.append(
            Node(
                package='nvblox_ros',
                executable='nvblox_node',
                name='nvblox_node',
                output='log',
                parameters=[
                    nvblox_params,
                    {
                        'global_frame': 'odom',
                        'pose_frame':   'base_link',
                    },
                ],
                remappings=nvblox_remappings,
            )
        )
    except PackageNotFoundError:
        print('[WARN] nvblox_ros not found — build isaac_ros_nvblox first')
        nodes.append(LogInfo(msg='nvblox skipped (nvblox_ros not built)'))

    nodes.append(LogInfo(msg='Mapping subsystem online'))
    return LaunchDescription(nodes)
