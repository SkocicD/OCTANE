from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
from launch_ros.actions import Node


def generate_launch_description():
    imu_monitor_terminal = ExecuteProcess(
        cmd=[
            'xterm', '-title', 'OCTANE | IMU Monitor',
            '-fa', 'Monospace', '-fs', '10',
            '-bg', '#0d1117', '-fg', '#aaff00', '-hold',
            '-e', 'ros2', 'run', 'octane_sensors', 'imu_monitor_node',
        ],
        output='log',
    )

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
        imu_monitor_terminal,
        LogInfo(msg='Sensors subsystem online'),
    ])
