from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Generate launch description for perception subsystem


def generate_launch_description():

    # Native RGB camera nodes (for Jetson)
    # TODO: Update device_path with actual camera serial numbers from /dev/v4l/by-id/
    # Run: ls -l /dev/v4l/by-id/ to find camera identifiers

    # Left Side RGB camera
    april_tag_detector = Node(
        package='octane_localization',
        executable='april_tag_detector',
        name='april_tag_detector'
    )

    return LaunchDescription([
        LogInfo(msg='Starting localization subsystem'),
        april_tag_detector,
        LogInfo(msg='Perception subsystem online'),
    ])
