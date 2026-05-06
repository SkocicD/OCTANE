from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg='Starting sensors subsystem'),
        Node(
            package='octane_sensors',
            executable='adxl345_node',
            name='adxl345_node',
            output='screen',
            parameters=[{
                'publish_rate': 50.0,
                'frame_id':     'imu_frame',
            }],
        ),
        LogInfo(msg='Sensors subsystem online'),
    ])
