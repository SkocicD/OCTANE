#!/usr/bin/env python3
"""Launch file for OCTANE supervisor nodes.

Located in octane package for centralized launch management.

Launches:
- mode_manager_node: State machine for mode transitions
- fault_manager_node: Fault aggregation and status
- fault_checker_node: Evaluates fault conditions from sensors
- state_monitor_node: Displays state changes (opens in new terminal)
"""

import os
import sys
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    """Generate launch description."""

    # Declare launch arguments
    check_rate_arg = DeclareLaunchArgument(
        'check_rate',
        default_value='10.0',
        description='Node check rate in Hz'
    )

    fault_check_rate_arg = DeclareLaunchArgument(
        'fault_check_rate',
        default_value='10.0',
        description='Fault detection check rate in Hz'
    )

    disable_faults_arg = DeclareLaunchArgument(
        'disable_faults',
        default_value='false',
        description='DEV: suppress all fault signals so FAULT state is never entered'
    )

    # Config file path
    config_path = PathJoinSubstitution([
        FindPackageShare('octane_supervisor'),
        'config',
        'faults.yaml',
    ])

    nodes = [
        Node(
            package='octane_supervisor',
            executable='mode_manager_node',
            name='mode_manager_node',
            output='screen',
            parameters=[
                {'check_rate': LaunchConfiguration('check_rate')},
                {'disable_faults': LaunchConfiguration('disable_faults')},
            ],
        ),
        Node(
            package='octane_supervisor',
            executable='fault_manager_node',
            name='fault_manager_node',
            output='screen',
            parameters=[
                {'check_rate': LaunchConfiguration('check_rate')},
            ],
        ),
        Node(
            package='octane_supervisor',
            executable='fault_checker_node',
            name='fault_checker_node',
            output='screen',
            parameters=[
                {'faults_config': config_path},
                {'fault_check_rate': LaunchConfiguration('fault_check_rate')},
            ],
        ),
    ]

    # State monitor - opens in new terminal window on Windows
    # This allows you to watch state transitions as they happen on the robot
    state_monitor = ExecuteProcess(
        cmd=[
            'start', 'cmd', '/k',
            'ros2', 'run', 'octane_supervisor', 'state_monitor_node'
        ],
        shell=True,
        output='screen',
        description='Opens state monitor in new terminal window'
    )

    return LaunchDescription([
        check_rate_arg,
        fault_check_rate_arg,
        disable_faults_arg,
        *nodes,
        state_monitor,
    ])
