from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


# Camera scheme:
#   1× Orbbec Astra Pro (depth + RGB)  → usb path 0:4.1.2.1  (Hub 1, port 2)
#   5× Innomaker U20CAM-1080p-S1       → by-path entries below
#
# All cameras are identified by USB port path — stable across reboots as long
# as each camera stays in the same physical port. Reassign positions here once
# physically mounted; device paths do not need to change.
#
# Orbbec device paths (passed to astra_camera driver):
#   RGB:   /dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.2.1:1.0-video-index0
#   Depth: /dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.2.1:1.0-video-index1

BY_PATH = 'platform-3610000.usb-usb-0'

NEAR_RGB_CAMERAS = [
    # (node_name,            position,       camera_id, usb_bridge_port, device_path)
    # Hub 1 — ports 3, 4
    ('near_rgb_left_side',   'left_side',    0, 5557, f'/dev/v4l/by-path/{BY_PATH}:4.1.3:1.0-video-index0'),
    ('near_rgb_left_front',  'left_front',   1, 5558, f'/dev/v4l/by-path/{BY_PATH}:4.1.4:1.0-video-index0'),
    # Hub 2 — ports 2, 3, 4
    ('near_rgb_right_side',  'right_side',   2, 5559, f'/dev/v4l/by-path/{BY_PATH}:4.2.2:1.0-video-index0'),
    ('near_rgb_right_front', 'right_front',  3, 5560, f'/dev/v4l/by-path/{BY_PATH}:4.2.3:1.0-video-index0'),
    ('near_rgb_back_rear',   'back_rear',    4, 5561, f'/dev/v4l/by-path/{BY_PATH}:4.2.4:1.0-video-index0'),
]

ORBBEC_RGB_PATH   = f'/dev/v4l/by-path/{BY_PATH}:4.1.2.1:1.0-video-index0'
ORBBEC_DEPTH_PATH = f'/dev/v4l/by-path/{BY_PATH}:4.1.2.1:1.0-video-index1'


def generate_launch_description():
    use_usb_bridge_arg = DeclareLaunchArgument(
        'use_usb_bridge',
        default_value='false',
        description='Use USB bridge for Mac development (true/false)',
    )
    enable_depth_estimation_arg = DeclareLaunchArgument(
        'enable_depth_estimation',
        default_value='false',
        description='Enable depth estimation node (true/false)',
    )
    use_usb_bridge = LaunchConfiguration('use_usb_bridge')
    enable_depth_estimation = LaunchConfiguration('enable_depth_estimation')

    nodes = [
        use_usb_bridge_arg,
        enable_depth_estimation_arg,
        LogInfo(msg='Starting perception subsystem'),
    ]

    # ════════════════════════════════════════════════════════════════════════
    # Native cameras (Jetson)
    # ════════════════════════════════════════════════════════════════════════

    # ── Orbbec depth camera ──────────────────────────────────────────────────
    try:
        orbbec_camera_dir = get_package_share_directory('astra_camera')
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(orbbec_camera_dir, 'launch', 'astra.launch.py')
                ),
                launch_arguments={
                    'color_device_path': ORBBEC_RGB_PATH,
                    'depth_device_path': ORBBEC_DEPTH_PATH,
                }.items(),
                condition=UnlessCondition(use_usb_bridge),
            )
        )
    except Exception:
        print("[WARN] astra_camera not found — Orbbec launch skipped (build with --orbbec to enable)")

    nodes.append(
        Node(
            package='octane_perception',
            executable='astra_depth_node',
            name='astra_depth_node',
            output='log',
            condition=UnlessCondition(use_usb_bridge),
            parameters=[{
                'rgb_device_path':   ORBBEC_RGB_PATH,
                'depth_device_path': ORBBEC_DEPTH_PATH,
            }],
            remappings=[
                ('depth_camera/depth', 'perception/camera/depth_camera/depth/frame'),
                ('depth_camera/color', 'perception/camera/depth_camera/rgb/frame'),
            ],
        )
    )

    # ── 5× near RGB cameras ──────────────────────────────────────────────────
    for node_name, position, cam_id, _bridge_port, device_path in NEAR_RGB_CAMERAS:
        nodes.append(
            Node(
                package='octane_perception',
                executable='rgb_camera_node',
                name=node_name,
                output='log',
                parameters=[{
                    'device_path': device_path,
                    'camera_id': cam_id,
                    'serial': position,
                    'frame_rate': 30,
                    'width': 640,
                    'height': 480,
                }],
                remappings=[('camera/frame', f'perception/camera/near/{position}/rgb/frame')],
                condition=UnlessCondition(use_usb_bridge),
            )
        )

    # ════════════════════════════════════════════════════════════════════════
    # USB bridge (Mac dev — TCP stream from host)
    # ════════════════════════════════════════════════════════════════════════

    # ── Orbbec depth + RGB ───────────────────────────────────────────────────
    nodes.append(
        Node(
            package='octane_perception',
            executable='usb_bridge_camera_node',
            name='orbbec_depth_bridge',
            output='log',
            parameters=[{
                'bridge_host': 'host.docker.internal',
                'bridge_port': 5555,
                'serial': 'orbbec_depth',
                'topic': 'camera/frame',
                'frame_id': 'orbbec_depth_frame',
                'width': 640,
                'height': 480,
                'encoding': 'mono16',
            }],
            remappings=[('camera/frame', 'perception/camera/depth_camera/depth/frame')],
            condition=IfCondition(use_usb_bridge),
        )
    )
    nodes.append(
        Node(
            package='octane_perception',
            executable='usb_bridge_camera_node',
            name='orbbec_rgb_bridge',
            output='log',
            parameters=[{
                'bridge_host': 'host.docker.internal',
                'bridge_port': 5556,
                'serial': 'orbbec_rgb',
                'topic': 'camera/frame',
                'frame_id': 'orbbec_rgb_frame',
                'width': 640,
                'height': 480,
                'encoding': 'bgr8',
            }],
            remappings=[('camera/frame', 'perception/camera/depth_camera/rgb/frame')],
            condition=IfCondition(use_usb_bridge),
        )
    )

    # ── 5× near RGB camera bridges ───────────────────────────────────────────
    for node_name, position, _cam_id, bridge_port, _device_path in NEAR_RGB_CAMERAS:
        nodes.append(
            Node(
                package='octane_perception',
                executable='usb_bridge_camera_node',
                name=f'{node_name}_bridge',
                output='log',
                parameters=[{
                    'bridge_host': 'host.docker.internal',
                    'bridge_port': bridge_port,
                    'serial': position,
                    'topic': 'camera/frame',
                    'frame_id': f'{position}_frame',
                    'width': 640,
                    'height': 480,
                    'encoding': 'bgr8',
                }],
                remappings=[('camera/frame', f'perception/camera/near/{position}/rgb/frame')],
                condition=IfCondition(use_usb_bridge),
            )
        )

    # ════════════════════════════════════════════════════════════════════════
    # Depth estimation (DA3) — runs on all 5 near RGB cameras
    # ════════════════════════════════════════════════════════════════════════
    near_rgb_topics = [
        f'perception/camera/near/{position}/rgb/frame'
        for _, position, _, _, _ in NEAR_RGB_CAMERAS
    ]
    nodes.append(
        Node(
            package='octane_perception',
            executable='depth_estimation_node',
            name='depth_estimation_node',
            condition=IfCondition(enable_depth_estimation),
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
