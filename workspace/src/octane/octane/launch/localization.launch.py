from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port',         default_value='5051',  description='TCP port the Pi connects to'),
        DeclareLaunchArgument('publish_rate', default_value='10.0',  description='Triangulator output Hz'),
        DeclareLaunchArgument('obs_timeout',  default_value='5.0',   description='Seconds before a tag detection is considered stale'),
        DeclareLaunchArgument('min_tags',     default_value='2',     description='Minimum distinct tags required to publish a pose'),

        LogInfo(msg='Starting localization subsystem'),

        Node(
            package='octane_localization',
            executable='far_camera_receiver_node',
            name='far_camera_receiver_node',
            output='screen',
            parameters=[{
                'port': LaunchConfiguration('port'),
            }],
        ),

        Node(
            package='octane_localization',
            executable='triangulator_node',
            name='triangulator_node',
            output='screen',
            parameters=[{
                'publish_rate': LaunchConfiguration('publish_rate'),
                'obs_timeout':  LaunchConfiguration('obs_timeout'),
                'min_tags':     LaunchConfiguration('min_tags'),
            }],
        ),

        LogInfo(msg='Localization subsystem online'),
    ])
