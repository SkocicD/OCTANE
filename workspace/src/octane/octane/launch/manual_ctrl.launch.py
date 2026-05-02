#!/usr/bin/env python3
"""Launch file for OCTANE manual control nodes.

Launches:
  - manual_drive_node:    WASD key state -> DriveCommand (left/right velocity)
  - can_drive_node:       DriveCommand -> CANOpen PDO motor commands
  - can_debug_node:       purple xterm live view (transceiver status, keys, velocity bars)
  - manual_actuator_node: arrow key state -> ActuatorCommand (arm + bucket)
  - serial_actuator_node: ActuatorCommand -> 4-bit relay state byte over UART (Arduino drives relays)
  - actuator_debug_node:  sky-blue xterm live view (arrow keys, arm/bucket state)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    throttle_scale_arg = DeclareLaunchArgument(
        'throttle_scale',
        default_value='1.0',
        description='Max throttle output (0.0 to 1.0)'
    )

    turn_scale_arg = DeclareLaunchArgument(
        'turn_scale',
        default_value='0.6',
        description='Turn differential scale (0.0 to 1.0)'
    )

    drive_node = Node(
        package='octane_manual_ctrl',
        executable='manual_drive_node',
        name='manual_drive_node',
        output='log',
        parameters=[{
            'throttle_scale': LaunchConfiguration('throttle_scale'),
            'turn_scale': LaunchConfiguration('turn_scale'),
        }],
    )

    can_drive_node = Node(
        package='octane_manual_ctrl',
        executable='can_drive_node',
        name='can_drive_node',
        output='log',
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
        package='octane_gpio',
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

    return LaunchDescription([
        throttle_scale_arg,
        turn_scale_arg,
        # xterms first — ensures terminals open even if a background node crashes
        can_debug_terminal,
        actuator_debug_terminal,
        # background nodes
        drive_node,
        can_drive_node,
        manual_actuator_node,
        serial_actuator_node,
    ])
