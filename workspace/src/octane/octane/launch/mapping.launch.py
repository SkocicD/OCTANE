import os

import yaml
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


# Zone definitions live in octane/config/zones.yaml — edit there.


def generate_launch_description():
    nodes = [
        DeclareLaunchArgument('model_path',       default_value='',     description='Path to terrain model .pt checkpoint'),
        DeclareLaunchArgument('depth_stats_path', default_value='',     description='Path to depth_stats.json'),
        DeclareLaunchArgument('inference_rate',   default_value='5.0',  description='Terrain inference Hz'),
        DeclareLaunchArgument('device',           default_value='auto', description='auto | cuda | cpu'),
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

    # ── nvblox — disabled (point_cloud_mux_node is the 3D data source) ──────────
    # Uncomment to re-enable GPU TSDF/ESDF fusion if needed in future.
    #
    # nvblox_remappings = []
    # for i, cam_name in enumerate(cfg['cameras'].keys()):
    #     nvblox_remappings.extend([
    #         (f'camera_{i}/depth/image',       f'mapping/{cam_name}/depth/image'),
    #         (f'camera_{i}/depth/camera_info', f'mapping/{cam_name}/depth/camera_info'),
    #         (f'camera_{i}/color/image',       f'mapping/{cam_name}/rgb/image'),
    #         (f'camera_{i}/color/camera_info', f'mapping/{cam_name}/rgb/camera_info'),
    #     ])
    # try:
    #     get_package_share_directory('nvblox_ros')
    #     nodes.append(
    #         Node(
    #             package='nvblox_ros',
    #             executable='nvblox_node',
    #             name='nvblox_node',
    #             output='log',
    #             parameters=[nvblox_params, {'global_frame': 'odom', 'pose_frame': 'base_link'}],
    #             remappings=nvblox_remappings,
    #         )
    #     )
    # except PackageNotFoundError:
    #     nodes.append(LogInfo(msg='nvblox skipped (nvblox_ros not built)'))

    # ── Colored point clouds — one per near camera via depth_image_proc ──────────
    # Each camera gets its own container + namespace so topic routing never
    # collides.  Within namespace mapping/<cam>, the node's default topics are:
    #   rgb/image_rect_color  →  remapped to rgb/image      (our splitter output)
    #   rgb/camera_info       →  matches directly            (no remap needed)
    #   depth_registered/image_rect → remapped to depth/image
    #   points                →  resolves to mapping/<cam>/points automatically
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

    # ── Combined point cloud (all 6 cameras merged into base_link frame) ─────
    nodes.append(
        Node(
            package='octane_mapping',
            executable='point_cloud_mux_node',
            name='point_cloud_mux_node',
            output='log',
            parameters=[{'config_file': config_file, 'publish_rate': 5.0}],
        )
    )

    # ── Terrain model inference ───────────────────────────────────────────────
    nodes.append(
        Node(
            package='octane_mapping',
            executable='terrain_inference_node',
            name='terrain_inference_node',
            output='screen',
            parameters=[{
                'model_path':       LaunchConfiguration('model_path'),
                'depth_stats_path': LaunchConfiguration('depth_stats_path'),
                'inference_rate':   LaunchConfiguration('inference_rate'),
                'device':           LaunchConfiguration('device'),
            }],
        )
    )

    nodes.append(LogInfo(msg='Mapping subsystem online'))
    return LaunchDescription(nodes)
