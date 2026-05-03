#!/usr/bin/env python3
"""Launch file for OCTANE network communication.

Launches:
  - network_comm_node:  TCP server (mode commands, telemetry, heartbeat)
  - video_stream_node:  UDP video/depth/map streamer (on-demand, GUI-controlled)
  - network_monitor_node: blue xterm live view of commands + heartbeat status

Defaults live in octane_network/config/network_params.yaml.
Override any value at the CLI, e.g.:
    ros2 launch octane network.launch.py tcp_port:=5001 default_scale:=25
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('octane_network'), 'config', 'network_params.yaml'
    )

    # CLI-overridable args — defaults pulled from network_params.yaml comments,
    # but the yaml file is passed directly to each node so values are not duplicated.
    args = [
        DeclareLaunchArgument('tcp_host',      default_value='0.0.0.0',
                              description='TCP bind address'),
        DeclareLaunchArgument('tcp_port',      default_value='5000',
                              description='TCP port for control traffic'),
        DeclareLaunchArgument('udp_port',      default_value='5002',
                              description='UDP port for video frames (rover → GUI)'),
        DeclareLaunchArgument('default_scale', default_value='50',
                              description='Fallback video scale %% (1-100) when GUI sends 0'),
        DeclareLaunchArgument('jpeg_quality',  default_value='70',
                              description='JPEG encode quality for video frames (1-100)'),
    ]

    comm_node = Node(
        package='octane_network',
        executable='network_comm_node',
        name='network_comm_node',
        output='log',
        parameters=[
            params_file,
            {
                'host': LaunchConfiguration('tcp_host'),
                'port': LaunchConfiguration('tcp_port'),
            },
        ],
    )

    video_node = Node(
        package='octane_network',
        executable='video_stream_node',
        name='video_stream_node',
        output='log',
        parameters=[
            params_file,
            {
                'udp_port':      LaunchConfiguration('udp_port'),
                'default_scale': LaunchConfiguration('default_scale'),
                'jpeg_quality':  LaunchConfiguration('jpeg_quality'),
            },
        ],
    )

    network_monitor_terminal = ExecuteProcess(
        cmd=[
            'xterm', '-title', 'OCTANE | Network Monitor',
            '-fa', 'Monospace', '-fs', '10',
            '-bg', '#0d1117', '-fg', '#58a6ff', '-hold',
            '-e', 'ros2', 'run', 'octane_network', 'network_monitor_node',
        ],
        output='log',
    )

    return LaunchDescription([*args, comm_node, video_node, network_monitor_terminal])
