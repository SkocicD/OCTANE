from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

# Generate launch description for perception subsystem


def generate_launch_description():

    # Declare launch argument for USB bridge mode
    use_usb_bridge_arg = DeclareLaunchArgument(
        'use_usb_bridge',
        default_value='true',
        description='Use USB bridge for Mac development (true/false)'
    )

    use_usb_bridge = LaunchConfiguration('use_usb_bridge')

    # ========================================
    # Camera Driver (Native or USB Bridge)
    # ========================================


# Native RGB camera nodes (for Jetson)
    # TODO: Update device_path with actual camera serial numbers from /dev/v4l/by-id/
    # Run: ls -l /dev/v4l/by-id/ to find camera identifiers

    # Left Side RGB camera
    rgb_camera_left = Node(
        package='octane_perception',
        executable='rgb_camera_node',
        name='rgb_camera_left',
        output='screen',
        parameters=[{
            'device_path': '/dev/v4l/by-id/usb-****-video-index0',
            'camera_id': 0,
            'frame_rate': 30,
            'width': 640,
            'height': 480
        }],
        remappings=[
            ('camera/image_raw', 'camera/left_side/image_raw'),
            ('camera/camera_info', 'camera/left_side/camera_info')
        ],
        condition=UnlessCondition(use_usb_bridge)
    )

    # Right Side RGB camera
    rgb_camera_right = Node(
        package='octane_perception',
        executable='rgb_camera_node',
        name='rgb_camera_right',
        output='screen',
        parameters=[{
            'device_path': '/dev/v4l/by-id/usb-****-video-index0',
            'camera_id': 1,
            'frame_rate': 30,
            'width': 640,
            'height': 480
        }],
        remappings=[
            ('camera/image_raw', 'camera/right_side/image_raw'),
            ('camera/camera_info', 'camera/right_side/camera_info')
        ],
        condition=UnlessCondition(use_usb_bridge)
    )

    # Left Rear RGB camera
    rgb_camera_left_rear = Node(
        package='octane_perception',
        executable='rgb_camera_node',
        name='rgb_camera_left_rear',
        output='screen',
        parameters=[{
            'device_path': '/dev/v4l/by-id/usb-****-video-index0',
            'camera_id': 2,
            'frame_rate': 30,
            'width': 640,
            'height': 480
        }],
        remappings=[
            ('camera/image_raw', 'camera/left_rear/image_raw'),
            ('camera/camera_info', 'camera/left_rear/camera_info')
        ],
        condition=UnlessCondition(use_usb_bridge)
    )

    # Right Rear RGB camera
    rgb_camera_right_rear = Node(
        package='octane_perception',
        executable='rgb_camera_node',
        name='rgb_camera_right_rear',
        output='screen',
        parameters=[{
            'device_path': '/dev/v4l/by-id/usb-****-video-index0',
            'camera_id': 3,
            'frame_rate': 30,
            'width': 640,
            'height': 480
        }],
        remappings=[
            ('camera/image_raw', 'camera/right_rear/image_raw'),
            ('camera/camera_info', 'camera/right_rear/camera_info')
        ],
        condition=UnlessCondition(use_usb_bridge)
    )

# USB bridge nodes (for Mac development)
    # Orbbec camera bridges (depth + RGB from same camera)
    orbbec_depth_bridge = Node(
        package='octane_perception',
        executable='usb_bridge_camera_node',
        name='orbbec_depth_bridge',
        output='screen',
        parameters=[{
            'bridge_host': 'host.docker.internal',
            'bridge_port': 5555,
            'image_topic': 'camera/orbbec/depth/image_raw',
            'camera_info_topic': 'camera/orbbec/depth/camera_info',
            'frame_id': 'orbbec_depth_frame',
            'width': 640,
            'height': 480,
            'encoding': 'mono16'
        }],
        condition=IfCondition(use_usb_bridge)
    )

    orbbec_rgb_bridge = Node(
        package='octane_perception',
        executable='usb_bridge_camera_node',
        name='orbbec_rgb_bridge',
        output='screen',
        parameters=[{
            'bridge_host': 'host.docker.internal',
            'bridge_port': 5556,
            'image_topic': 'perception/camera/far/image_raw',
            'camera_info_topic': 'camera/orbbec/rgb/camera_info',
            'frame_id': 'orbbec_rgb_frame',
            'width': 640,
            'height': 480,
            'encoding': 'bgr8'
        }],
        condition=IfCondition(use_usb_bridge)
    )

    # Standalone RGB camera bridges (4 cameras)
    rgb_left_bridge = Node(
        package='octane_perception',
        executable='usb_bridge_camera_node',
        name='rgb_left_bridge',
        output='screen',
        parameters=[{
            'bridge_host': 'host.docker.internal',
            'bridge_port': 5557,
            'image_topic': 'camera/left/image_raw',
            'camera_info_topic': 'camera/left/camera_info',
            'frame_id': 'camera_left_frame',
            'width': 640,
            'height': 480,
            'encoding': 'bgr8'
        }],
        condition=IfCondition(use_usb_bridge)
    )

    rgb_right_bridge = Node(
        package='octane_perception',
        executable='usb_bridge_camera_node',
        name='rgb_right_bridge',
        output='screen',
        parameters=[{
            'bridge_host': 'host.docker.internal',
            'bridge_port': 5558,
            'image_topic': 'camera/right/image_raw',
            'camera_info_topic': 'camera/right/camera_info',
            'frame_id': 'camera_right_frame',
            'width': 640,
            'height': 480,
            'encoding': 'bgr8'
        }],
        condition=IfCondition(use_usb_bridge)
    )

    rgb_left_rear_bridge = Node(
        package='octane_perception',
        executable='usb_bridge_camera_node',
        name='rgb_left_rear_bridge',
        output='screen',
        parameters=[{
            'bridge_host': 'host.docker.internal',
            'bridge_port': 5559,
            'image_topic': 'camera/left_rear/image_raw',
            'camera_info_topic': 'camera/left_rear/camera_info',
            'frame_id': 'camera_left_rear_frame',
            'width': 640,
            'height': 480,
            'encoding': 'bgr8'
        }],
        condition=IfCondition(use_usb_bridge)
    )

    rgb_right_rear_bridge = Node(
        package='octane_perception',
        executable='usb_bridge_camera_node',
        name='rgb_right_rear_bridge',
        output='screen',
        parameters=[{
            'bridge_host': 'host.docker.internal',
            'bridge_port': 5560,
            'image_topic': 'camera/right_rear/image_raw',
            'camera_info_topic': 'camera/right_rear/camera_info',
            'frame_id': 'camera_right_rear_frame',
            'width': 640,
            'height': 480,
            'encoding': 'bgr8'
        }],
        condition=IfCondition(use_usb_bridge)
    )

    # ========================================
    # Perception Nodes
    # ========================================

    # NOTE: Camera viewer removed - use ROS visualization tools instead
    # Use rqt_image_view or rviz2 to view camera feeds via VNC

    return LaunchDescription([
        use_usb_bridge_arg,
        LogInfo(msg='Starting perception subsystem'),

        # Camera driver (native or bridge)
        rgb_camera_left,
        rgb_camera_right,
        rgb_camera_left_rear,
        rgb_camera_right_rear,
        orbbec_depth_bridge,
        orbbec_rgb_bridge,
        rgb_left_bridge,
        rgb_right_bridge,
        rgb_left_rear_bridge,
        rgb_right_rear_bridge,

        LogInfo(msg='Perception subsystem online'),
    ])
