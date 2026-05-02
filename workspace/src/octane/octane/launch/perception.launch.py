from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


# Camera scheme:
#   1× Orbbec (depth + RGB)   → perception/camera/depth_camera/{depth,rgb}/frame
#   5× standalone RGB cameras → perception/camera/near/{position}/rgb/frame
#
# With debug_images:=true each camera also publishes sensor_msgs/Image on
#   perception/camera/near/{position}/rgb/image  (for RViz / rqt_image_view)
#
# Device paths are udev symlinks — see setup_cameras.sh in repo root.
NEAR_RGB_CAMERAS = [
    # (node_name,            position,        device_path)
    ('near_rgb_left_front',  'left_front',    '/dev/cam_left_front'),
    ('near_rgb_left_side',   'left_side',     '/dev/cam_left_side'),
    ('near_rgb_right_front', 'right_front',   '/dev/cam_right_front'),
    ('near_rgb_right_side',  'right_side',    '/dev/cam_right_side'),
    ('near_rgb_back_rear',   'back_rear',     '/dev/cam_back_rear'),
]


def generate_launch_description():
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
    try:
        astra_camera_dir = get_package_share_directory('astra_camera')
        nodes.append(
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    os.path.join(astra_camera_dir, 'launch', 'astra_pro.launch.xml')
                ),
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
                ('depth_camera/depth', 'perception/camera/depth_camera/depth/frame'),
                ('depth_camera/color', 'perception/camera/depth_camera/rgb/frame'),
            ],
        )
    )

    # ── 5× near RGB cameras ──────────────────────────────────────────────────
    for node_name, position, device_path in NEAR_RGB_CAMERAS:
        nodes.append(
            Node(
                package='octane_perception',
                executable='rgb_camera_node',
                name=node_name,
                output='log',
                parameters=[{
                    'device_path': device_path,
                    'serial': position,
                    'frame_rate': 30,
                    'width': 480,
                    'height': 360,
                    'debug_images': debug_images,
                }],
                remappings=[
                    ('camera/frame', f'perception/camera/near/{position}/rgb/frame'),
                    ('camera/image', f'perception/camera/near/{position}/rgb/image'),
                ],
            )
        )

    # ── DA3 depth estimation — runs on all 5 near RGB cameras ────────────────
    near_rgb_topics = [
        f'perception/camera/near/{position}/rgb/frame'
        for _, position, _ in NEAR_RGB_CAMERAS
    ]
    nodes.append(
        Node(
            package='octane_perception',
            executable='depth_estimation_node',
            name='depth_estimation_node',
            output='log',
            parameters=[{
                'model_name': 'depth-anything/DA3METRIC-LARGE',
                'input_topics': near_rgb_topics,
                'inference_rate': 10.0,
                'process_res': 504,
                'debug_images': debug_images,
            }],
        )
    )

    nodes.append(LogInfo(msg='Perception subsystem online'))
    return LaunchDescription(nodes)
