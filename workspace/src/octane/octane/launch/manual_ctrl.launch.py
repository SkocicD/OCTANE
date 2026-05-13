#!/usr/bin/env python3
"""Launch file for OCTANE manual control nodes.

Launches:
  - manual_drive_node:    WASD key state -> DriveCommand (left/right velocity)
  - can_worker:           combined process — can_drain_node (RX drain) + can_drive_node (PDO TX)
  - can_debug_node:       purple xterm live view (transceiver status, keys, velocity bars)
  - rs485_drive_node:     left_velocity -> Modbus RTU BLD-510B (replaces broken CAN motor #2)
  - rs485_debug_node:     violet xterm live view for RS485 motor
  - manual_actuator_node: arrow key state -> ActuatorCommand (arm + bucket)
  - serial_actuator_node: ActuatorCommand -> 4-bit relay state byte over UART (Arduino drives relays)
  - actuator_debug_node:  sky-blue xterm live view (arrow keys, arm/bucket state)
"""

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    robot_params = os.path.join(
        get_package_share_directory('octane'), 'config', 'robot_params.yaml'
    )

    throttle_scale_arg = DeclareLaunchArgument(
        'throttle_scale',
        default_value='1.0',
        description='Max throttle output (0.0 to 1.0)'
    )

    drive_node = Node(
        package='octane_manual_ctrl',
        executable='manual_drive_node',
        name='manual_drive_node',
        output='log',
        parameters=[
            robot_params,
            {'throttle_scale': LaunchConfiguration('throttle_scale')},
        ],
    )

    can_worker_node = Node(
        package='octane_manual_ctrl',
        executable='can_worker',
        output='log',
        parameters=[robot_params],
    )

    can_debug_terminal = ExecuteProcess(
        cmd=[
            'xterm', '-title', 'OCTANE | CAN Debug',
            '-fa', 'Monospace', '-fs', '10',
            '-bg', '#0d1117', '-fg', '#bf80ff', '-hold',
            '-e', 'ros2', 'run', 'octane_manual_ctrl', 'can_debug_node',
        ],
        output='log',
    )

    manual_actuator_node = Node(
        package='octane_manual_ctrl',
        executable='manual_actuator_node',
        name='manual_actuator_node',
        output='log',
    )

    serial_actuator_node = Node(
        package='octane_serial',
        executable='serial_actuator_node',
        name='serial_actuator_node',
        output='log',
    )

    actuator_debug_terminal = ExecuteProcess(
        cmd=[
            'xterm', '-title', 'OCTANE | Actuator Debug',
            '-fa', 'Monospace', '-fs', '10',
            '-bg', '#0d1117', '-fg', '#00cfff', '-hold',
            '-e', 'ros2', 'run', 'octane_manual_ctrl', 'actuator_debug_node',
        ],
        output='log',
    )

    rs485_drive_node = Node(
        package='octane_rs485',
        executable='rs485_drive_node',
        name='rs485_drive_node',
        output='log',
        parameters=[
            robot_params,
            {
                'port':           '/dev/rs485_drive',  # udev symlink for CH340 (VID 1a86:7523)
                'baud_rate':      9600,
                'modbus_address': 1,
                'max_rpm':        3000,
                'pole_pairs':     4,
                'reverse':        True,
            },
        ],
    )

    rs485_debug_terminal = ExecuteProcess(
        cmd=[
            'xterm', '-title', 'OCTANE | RS485 Debug',
            '-fa', 'Monospace', '-fs', '10',
            '-bg', '#0d1117', '-fg', '#5603fc', '-hold',
            '-e', 'ros2', 'run', 'octane_rs485', 'rs485_debug_node',
        ],
        output='log',
    )

    return LaunchDescription([
        throttle_scale_arg,
        # xterms first — ensures terminals open even if a background node crashes
        can_debug_terminal,
        actuator_debug_terminal,
        rs485_debug_terminal,
        # background nodes
        drive_node,
        can_worker_node,
        rs485_drive_node,
        manual_actuator_node,
        serial_actuator_node,
    ])
