#!/usr/bin/env python3
"""Launch file for OCTANE network communication node.

Located in octane package for centralized launch management.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    """Generate launch description."""

    # Declare launch arguments
    host_arg = DeclareLaunchArgument(
        'host',
        default_value='0.0.0.0',
        description='Host address to bind TCP server'
    )

    port_arg = DeclareLaunchArgument(
        'port',
        default_value='5000',
        description='TCP port for ground station connection'
    )

    telemetry_rate_arg = DeclareLaunchArgument(
        'telemetry_rate',
        default_value='10.0',
        description='Telemetry broadcast rate in Hz'
    )

    # Config file path
    config_path = PathJoinSubstitution([
        FindPackageShare('octane_network'),
        'config',
        'network_params.yaml',
    ])

    node = Node(
        package='octane_network',
        executable='network_comm_node',
        name='network_comm_node',
        output='screen',
        parameters=[
            config_path,
            {
                'host': LaunchConfiguration('host'),
                'port': LaunchConfiguration('port'),
                'telemetry_rate': LaunchConfiguration('telemetry_rate'),
            }
        ],
    )

    return LaunchDataset([
        host_arg,
        port_arg,
        telemetry_rate_arg,
        node,
    ])
