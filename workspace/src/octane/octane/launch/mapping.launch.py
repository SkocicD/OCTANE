import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    nodes = [
        LogInfo(msg='Starting mapping subsystem'),
        # Stub identity odom→base_link until octane_localization is ready.
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
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f'[WARN] Could not load config: {e} — skipping mapping nodes')
        nodes.append(LogInfo(msg='Mapping subsystem skipped (config unavailable)'))
        return LaunchDescription(nodes)

    # ── Camera frame splitters (one per camera, per stream) ───────────────────
    # Unpacks each CameraFrame into Image + CameraInfo and broadcasts the
    # static TF base_link → <cam>_frame.
    for cam_name, cam_cfg in cfg['cameras'].items():
        nodes.append(
            Node(
                package='octane_mapping',
                executable='camera_frame_splitter',
                name=f'{cam_name}_rgb_splitter',
                output='log',
                parameters=[{
                    'camera_name':      cam_name,
                    'input_topic':      cam_cfg['rgb_topic'],
                    'output_namespace': f'mapping/{cam_name}/rgb',
                    'frame_id':         f'{cam_name}_frame',
                    'parent_frame':     'base_link',
                    'config_file':      config_file,
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
                    'camera_name':      cam_name,
                    'input_topic':      cam_cfg['depth_topic'],
                    'output_namespace': f'mapping/{cam_name}/depth',
                    'frame_id':         f'{cam_name}_frame',
                    'parent_frame':     'base_link',
                    'config_file':      config_file,
                }],
            )
        )

    # ── Colored point clouds — one per camera via depth_image_proc ────────────
    for cam_name in cfg['cameras']:
        nodes.append(
            ComposableNodeContainer(
                name=f'{cam_name}_pc_container',
                namespace=f'mapping/{cam_name}',
                package='rclcpp_components',
                executable='component_container',
                composable_node_descriptions=[
                    ComposableNode(
                        package='depth_image_proc',
                        plugin='depth_image_proc::PointCloudXyzrgbNode',
                        name='point_cloud',
                        namespace=f'mapping/{cam_name}',
                        parameters=[{'exact_sync': False, 'queue_size': 10}],
                        remappings=[
                            ('rgb/image_rect_color',        'rgb/image'),
                            ('depth_registered/image_rect', 'depth/image'),
                        ],
                    )
                ],
                output='log',
            )
        )

    # ── Combined point cloud (all 6 merged into base_link frame) ─────────────
    nodes.append(
        Node(
            package='octane_mapping',
            executable='point_cloud_mux_node',
            name='point_cloud_mux_node',
            output='log',
            parameters=[{'config_file': config_file, 'publish_rate': 5.0}],
        )
    )

    nodes.append(LogInfo(msg='Mapping subsystem online'))
    return LaunchDescription(nodes)
