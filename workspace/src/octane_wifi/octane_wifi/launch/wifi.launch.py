#!/usr/bin/env python3
"""Launch file for OCTANE WiFi communication node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Configure and start wifi_comm_node."""
    config_path = PathJoinSubstitution([
        FindPackageShare("octane_wifi"),
        "config",
        "wifi_params.yaml",
    ])

    nodes = [
        Node(
            package="octane_wifi",
            executable="wifi_comm_node",
            name="wifi_comm_node",
            output="screen",
            parameters=[config_path],
        ),
    ]

    return nodes


def generate_launch_description():
    """Generate launch description."""
    return LaunchDescription([
        DeclareLaunchArgument(
            'host',
            default_value='0.0.0.0',
            description='Host address to bind TCP server'
        ),
        DeclareLaunchArgument(
            'port',
            default_value='5000',
            description='TCP port for ground station connection'
        ),
        OpaqueFunction(function=launch_setup),
    ])
