#!/usr/bin/env python3
"""Launch file for OCTANE network communication with heartbeat.

Located in octane package for centralized launch management.

Launches:
  - network_comm_node: TCP server for mode commands and telemetry
  - heartbeat_sender: UDP heartbeat for connection monitoring
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description."""

    # TCP parameters
    tcp_host_arg = DeclareLaunchArgument(
        'tcp_host',
        default_value='0.0.0.0',
        description='TCP host address to bind'
    )

    tcp_port_arg = DeclareLaunchArgument(
        'tcp_port',
        default_value='5000',
        description='TCP port for mode commands'
    )

    telemetry_rate_arg = DeclareLaunchArgument(
        'telemetry_rate',
        default_value='10.0',
        description='Telemetry broadcast rate in Hz'
    )

    # UDP heartbeat parameters
    udp_host_arg = DeclareLaunchArgument(
        'udp_host',
        default_value='255.255.255.255',
        description='UDP broadcast address for heartbeat'
    )

    udp_port_arg = DeclareLaunchArgument(
        'udp_port',
        default_value='5001',
        description='UDP port for heartbeat'
    )

    heartbeat_rate_arg = DeclareLaunchArgument(
        'heartbeat_rate',
        default_value='0.33',
        description='Heartbeat rate in Hz (0.33 = once every ~3 s)'
    )

    udp_bind_ip_arg = DeclareLaunchArgument(
        'udp_bind_ip',
        default_value='',
        description='Local IP to bind heartbeat socket to (pins to a specific interface). '
                    'Set to rover WiFi/Ethernet IP if broadcast goes to wrong interface.'
    )

    # TCP node (mode commands + telemetry)
    tcp_node = Node(
        package='octane_network',
        executable='network_comm_node',
        name='network_comm_node',
        output='log',
        parameters=[
            {
                'host': LaunchConfiguration('tcp_host'),
                'port': LaunchConfiguration('tcp_port'),
                'telemetry_rate': LaunchConfiguration('telemetry_rate'),
            }
        ],
    )

    # UDP heartbeat node
    udp_node = Node(
        package='octane_network',
        executable='heartbeat_sender',
        name='heartbeat_sender',
        output='screen',   # show HB log lines in the launch terminal
        parameters=[
            {
                'host':    LaunchConfiguration('udp_host'),
                'port':    LaunchConfiguration('udp_port'),
                'rate_hz': LaunchConfiguration('heartbeat_rate'),
                'bind_ip': LaunchConfiguration('udp_bind_ip'),
            }
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

    return LaunchDescription([
        tcp_host_arg,
        tcp_port_arg,
        telemetry_rate_arg,
        udp_host_arg,
        udp_port_arg,
        heartbeat_rate_arg,
        udp_bind_ip_arg,
        tcp_node,
        udp_node,
        network_monitor_terminal,
    ])
