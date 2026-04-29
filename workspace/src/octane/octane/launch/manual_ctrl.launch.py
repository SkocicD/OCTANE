#!/usr/bin/env python3
"""Launch file for OCTANE manual control nodes.

Launches:
  - manual_drive_node:    interprets WASD keys -> left/right drive velocity
  - manual_actuator_node: interprets arrow keys -> arm/bucket relay commands
  - can_drive_node:       drive velocity -> CAN bus (TODO: hw not implemented)
  - gpio_actuator_node:   actuator command -> GPIO relays (TODO: hw not implemented)
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
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
        output='screen',
        parameters=[{
            'throttle_scale': LaunchConfiguration('throttle_scale'),
            'turn_scale': LaunchConfiguration('turn_scale'),
        }],
    )

    actuator_node = Node(
        package='octane_manual_ctrl',
        executable='manual_actuator_node',
        name='manual_actuator_node',
        output='screen',
    )

    can_drive_node = Node(
        package='octane_can',
        executable='can_drive_node',
        name='can_drive_node',
        output='screen',
    )

    gpio_actuator_node = Node(
        package='octane_gpio',
        executable='gpio_actuator_node',
        name='gpio_actuator_node',
        output='screen',
    )

    return LaunchDescription([
        throttle_scale_arg,
        turn_scale_arg,
        drive_node,
        actuator_node,
        can_drive_node,
        gpio_actuator_node,
    ])
