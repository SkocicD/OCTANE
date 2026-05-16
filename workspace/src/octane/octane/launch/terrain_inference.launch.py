"""Launch the terrain model inference node.

Usage:
  ros2 launch octane terrain_inference.launch.py \\
    model_path:=/path/to/best.pt \\
    depth_stats_path:=/path/to/depth_stats.json \\
    inference_rate:=5.0

The node subscribes to all 12 perception camera topics (must have perception
stack running) and publishes to:
  /mapping/terrain/height   (32FC1 float metres,  200×200)
  /mapping/terrain/rocks    (32FC1 float 0–1,     200×200)
  /mapping/terrain/craters  (32FC1 float 0–1,     200×200)
  /mapping/terrain/walls    (32FC1 float 0–1,     200×200)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model_path',       default_value='',    description='Path to terrain model .pt checkpoint'),
        DeclareLaunchArgument('depth_stats_path', default_value='',    description='Path to depth_stats.json'),
        DeclareLaunchArgument('inference_rate',   default_value='5.0', description='Inference Hz'),
        DeclareLaunchArgument('device',           default_value='auto', description='auto | cuda | cpu'),
        DeclareLaunchArgument('imu_topic',        default_value='sensors/imu/accel', description='IMU topic for roll/pitch'),

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
                'imu_topic':        LaunchConfiguration('imu_topic'),
            }],
        ),
    ])
