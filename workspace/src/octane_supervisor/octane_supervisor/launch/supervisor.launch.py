#!/usr/bin/env python3
"""Launch file for OCTANE supervisor package.

Launches:
    - mode_manager_node: State machine for mode transitions
    - fault_manager_node: Fault aggregation and status
    - fault_checker_node: Evaluates fault conditions from sensors

Usage:
    ros2 launch octane_supervisor supervisor.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Configure and return nodes."""
    config_path = PathJoinSubstitution([
        FindPackageShare("octane_supervisor"),
        "config",
        "faults.yaml",
    ])

    nodes = [
        Node(
            package="octane_supervisor",
            executable="mode_manager_node",
            name="mode_manager_node",
            output="screen",
            parameters=[{"check_rate": 10.0}],
        ),
        Node(
            package="octane_supervisor",
            executable="fault_manager_node",
            name="fault_manager_node",
            output="screen",
            parameters=[{"check_rate": 10.0}],
        ),
        Node(
            package="octane_supervisor",
            executable="fault_checker_node",
            name="fault_checker_node",
            output="screen",
            parameters=[
                {"faults_config": config_path},
                {"fault_check_rate": 10.0},
            ],
        ),
    ]

    return nodes


def generate_launch_description():
    """Generate launch description."""
    return LaunchDescription([
        OpaqueFunction(function=launch_setup),
    ])
