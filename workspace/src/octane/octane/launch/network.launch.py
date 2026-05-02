#!/usr/bin/env python3
"""Launch file for OCTANE network communication.

Launches:
  - network_comm_node: TCP server (mode commands, telemetry, heartbeat)
  - network_monitor_node: blue xterm live view of commands + heartbeat status
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    tcp_host_arg = DeclareLaunchArgument(
        'tcp_host', default_value='0.0.0.0',
        description='TCP bind address'
    )
    tcp_port_arg = DeclareLaunchArgument(
        'tcp_port', default_value='5000',
        description='TCP port for mode commands, telemetry, and heartbeat'
    )
    telemetry_rate_arg = DeclareLaunchArgument(
        'telemetry_rate', default_value='10.0',
        description='Telemetry rate in Hz'
    )
    heartbeat_rate_arg = DeclareLaunchArgument(
        'heartbeat_rate', default_value='0.33',
        description='Heartbeat rate in Hz (0.33 = once every ~3 s)'
    )

    comm_node = Node(
        package='octane_network',
        executable='network_comm_node',
        name='network_comm_node',
        output='log',
        parameters=[{
            'host':           LaunchConfiguration('tcp_host'),
            'port':           LaunchConfiguration('tcp_port'),
            'telemetry_rate': LaunchConfiguration('telemetry_rate'),
            'heartbeat_rate': LaunchConfiguration('heartbeat_rate'),
        }],
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

    return LaunchDescription([
        tcp_host_arg,
        tcp_port_arg,
        telemetry_rate_arg,
        heartbeat_rate_arg,
        network_monitor_terminal,
        comm_node,
    ])
