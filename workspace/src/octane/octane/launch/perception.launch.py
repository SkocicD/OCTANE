from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


# Camera scheme:
#   1× Orbbec (depth + RGB)            → perception/camera/depth_camera/{depth,rgb}/frame
#   5× standalone RGB cameras          → perception/camera/near/{position}/rgb/frame
#
# Device paths use /dev/v4l/by-path/ (port-based, stable while cables stay put).
# Update these once cameras are physically assigned to rover positions.
NEAR_RGB_CAMERAS = [
    # (node_name,            position,        device_path)
    ('near_rgb_left_side',   'left_side',     '/dev/v4l/by-path/****-left-side'),
    ('near_rgb_left_front',  'left_front',    '/dev/v4l/by-path/****-left-front'),
    ('near_rgb_right_side',  'right_side',    '/dev/v4l/by-path/****-right-side'),
    ('near_rgb_right_front', 'right_front',   '/dev/v4l/by-path/****-right-front'),
    ('near_rgb_back_rear',   'back_rear',     '/dev/v4l/by-path/****-back-rear'),
]


def generate_launch_description():
    nodes = [
        LogInfo(msg='Starting perception subsystem'),
    ]

    # ── Orbbec depth camera ──────────────────────────────────────────────────
    try:
        orbbec_camera_dir = get_package_share_directory('orbbec_camera')
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(orbbec_camera_dir, 'launch', 'astra.launch.py')
                ),
            )
        )
    except Exception:
        print('[WARN] orbbec_camera not found — skipping Orbbec launch include')
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
                    'width': 640,
                    'height': 480,
                }],
                remappings=[('camera/frame', f'perception/camera/near/{position}/rgb/frame')],
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
            }],
        )
    )

    nodes.append(LogInfo(msg='Perception subsystem online'))
    return LaunchDescription(nodes)
