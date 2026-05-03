import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Camera scheme (driven by octane/config/cameras.yaml):
#   1× Orbbec (depth + RGB)   → perception/camera/depth_camera/{depth,rgb}/frame
#   5× standalone RGB cameras → perception/camera/near/<position>/rgb/frame
#   DA3 depth estimation      → perception/camera/near/<position>/depth/frame
#
# With debug_images:=true each camera also publishes sensor_msgs/Image on
#   perception/camera/near/<position>/rgb/image  (for RViz / rqt_image_view)
#
# Device paths are udev symlinks — see setup_cameras.sh in repo root.


def generate_launch_description():
    octane_share = get_package_share_directory('octane')
    config_file = os.path.join(octane_share, 'config', 'cameras.yaml')
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    debug_images_arg = DeclareLaunchArgument(
        'debug_images', default_value='true',
        description='Publish raw sensor_msgs/Image alongside CameraFrame (for RViz/rqt)',
    )
    debug_images = LaunchConfiguration('debug_images')

    nodes = [
        debug_images_arg,
        LogInfo(msg='Starting perception subsystem'),
    ]

    # ── Orbbec depth camera ──────────────────────────────────────────────────
    orbbec_cfg = cfg['cameras']['orbbec_depth']
    try:
        astra_camera_dir = get_package_share_directory('astra_camera')
        nodes.append(
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    os.path.join(astra_camera_dir, 'launch', 'astra_pro.launch.xml')
                ),
                launch_arguments={
                    'enable_ir':          'false',
                    'enable_point_cloud': 'false',
                }.items(),
            )
        )
    except Exception:
        print('[WARN] astra_camera not found — skipping Orbbec launch include')
    nodes.append(
        Node(
            package='octane_perception',
            executable='astra_depth_node',
            name='astra_depth_node',
            output='log',
            remappings=[
                ('depth_camera/depth', orbbec_cfg['depth_topic']),
                ('depth_camera/color', orbbec_cfg['rgb_topic']),
            ],
        )
    )

    # ── 5× near RGB cameras (any camera entry with a device_path) ────────────
    near_rgb_topics = []
    for cam_name, cam_cfg in cfg['cameras'].items():
        if 'device_path' not in cam_cfg:
            continue
        rgb_topic = cam_cfg['rgb_topic']
        near_rgb_topics.append(rgb_topic)
        nodes.append(
            Node(
                package='octane_perception',
                executable='rgb_camera_node',
                name=cam_name,
                output='log',
                parameters=[{
                    'device_path': cam_cfg['device_path'],
                    'serial':      cam_cfg['serial'],
                    'frame_rate':  cam_cfg.get('frame_rate', 30),
                    'width':       cam_cfg['width'],
                    'height':      cam_cfg['height'],
                    'debug_images': ParameterValue(debug_images, value_type=bool),
                }],
                remappings=[
                    ('camera/frame', rgb_topic),
                    ('camera/image', rgb_topic.replace('/frame', '/image')),
                ],
            )
        )

    # ── DA3 depth estimation — runs on all near RGB cameras ──────────────────
    nodes.append(
        Node(
            package='octane_perception',
            executable='depth_estimation_node',
            name='depth_estimation_node',
            output='log',
            parameters=[{
                'model_name':    'depth-anything/DA3METRIC-LARGE',
                'input_topics':  near_rgb_topics,
                'inference_rate': 10.0,
                'process_res':   392,
                'debug_images':  ParameterValue(debug_images, value_type=bool),
            }],
        )
    )

    nodes.append(LogInfo(msg='Perception subsystem online'))
    return LaunchDescription(nodes)
