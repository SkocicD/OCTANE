#!/usr/bin/env python3
"""Launch file for OCTANE supervisor nodes."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    check_rate_arg = DeclareLaunchArgument(
        'check_rate',
        default_value='10.0',
        description='Node check rate in Hz',
    )
    fault_check_rate_arg = DeclareLaunchArgument(
        'fault_check_rate',
        default_value='10.0',
        description='Fault detection check rate in Hz',
    )

    config_path = PathJoinSubstitution([
        FindPackageShare('octane_supervisor'),
        'config',
        'faults.yaml',
    ])

    mode_manager = Node(
        package='octane_supervisor',
        executable='mode_manager_node',
        name='mode_manager_node',
        output='log',
        parameters=[{'check_rate': LaunchConfiguration('check_rate')}],
    )

    fault_checker = Node(
        package='octane_supervisor',
        executable='fault_checker_node',
        name='fault_checker_node',
        output='log',
        parameters=[
            {'faults_config': config_path},
            {'fault_check_rate': LaunchConfiguration('fault_check_rate')},
        ],
    )

    # Open state monitor and fault manager in their own xterm windows so their
    # live displays don't clutter the main ROS terminal.
    state_monitor_terminal = ExecuteProcess(
        cmd=[
            'xterm',
            '-title', 'OCTANE | State Monitor',
            '-fa', 'Monospace', '-fs', '10',
            '-bg', '#0d1117', '-fg', '#39d353',
            '-hold',
            '-e', 'ros2', 'run', 'octane_supervisor', 'state_monitor_node',
        ],
        output='log',
    )

    fault_manager_terminal = ExecuteProcess(
        cmd=[
            'xterm',
            '-title', 'OCTANE | Fault Manager',
            '-fa', 'Monospace', '-fs', '10',
            '-bg', '#0d1117', '-fg', '#ffa500',
            '-hold',
            '-e', 'ros2', 'run', 'octane_supervisor', 'fault_manager_node',
            '--ros-args', '-p', 'check_rate:=10.0',
        ],
        output='log',
    )

    return LaunchDescription([
        check_rate_arg,
        fault_check_rate_arg,
        mode_manager,
        fault_checker,
        state_monitor_terminal,
        fault_manager_terminal,
    ])
