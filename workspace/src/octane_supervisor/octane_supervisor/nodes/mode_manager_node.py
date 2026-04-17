#!/usr/bin/env python3
"""Mode manager node for OCTANE supervisor.

Manages state transitions based on commands from ground station (via WiFi)
and fault signals from the fault manager.

Topics:
    /supervisor/mode_command (std_msgs/String)
        Input: Mode command from ground station (manual, autonomous, standby, fault_reset)
    /supervisor/fault_signal (std_msgs/String)
        Input: Fault type from fault detectors
    /supervisor/state (std_msgs/String)
        Output: Current state (STANDBY, MANUAL, AUTONOMOUS, FAULT)
    /supervisor/control_status (octane_supervisor/ControlStatus)
        Output: navigation_enabled, manual_enabled, emergency_stop
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from octane_supervisor.state_machine import StateMachine, Mode, State


class ControlStatusMsg:
    """Custom message type for control status.

    TODO: Replace with proper ROS2 message definition
    """
    def __init__(self):
        self.navigation_enabled = False
        self.manual_enabled = False
        self.emergency_stop = False
        self.fault_type = ""


class ModeManagerNode(Node):
    """Node that manages system state transitions."""

    def __init__(self):
        super().__init__("mode_manager_node")

        self.state_machine = StateMachine()
        self.declare_parameters(
            namespace="",
            parameters=[
                ("check_rate", 10.0),
            ],
        )
        check_rate = self.get_parameter("check_rate").get_parameter_value().double_value

        # QoS for reliable command receipt
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Subscribers
        self.create_subscription(
            String,
            "/supervisor/mode_command",
            self.mode_command_callback,
            qos,
        )
        self.create_subscription(
            String,
            "/supervisor/fault_signal",
            self.fault_signal_callback,
            qos,
        )

        # Publishers
        self.state_pub = self.create_publisher(String, "/supervisor/state", qos)
        self.control_status_pub = self.create_publisher(ControlStatusMsg, "/supervisor/control_status", qos)

        # Timer for state publishing
        self.timer = self.create_timer(1.0 / check_rate, self.publish_state)

        self.get_logger().info("Mode manager node initialized")
        self.publish_state()

    def mode_command_callback(self, msg: String):
        """Handle mode command from ground station."""
        mode_str = msg.data.lower()
        mode_map = {
            "standby": Mode.STANDBY,
            "manual": Mode.MANUAL,
            "autonomous": Mode.AUTONOMOUS,
            "fault_reset": Mode.FAULT_RESET,
        }

        if mode_str not in mode_map:
            self.get_logger().warn(f"Invalid mode command: {mode_str}")
            return

        target_mode = mode_map[mode_str]

        # If fault reset requested, check if faults are cleared
        if target_mode == Mode.FAULT_RESET:
            # TODO: Check with fault_manager if all faults cleared
            pass

        transition = self.state_machine.transition(target_mode)

        if transition.success:
            self.get_logger().info(f"Mode changed to: {transition.to_state.name}")
        else:
            self.get_logger().warn(f"Mode transition failed: {transition.reason}")

        self.publish_state()

    def fault_signal_callback(self, msg: String):
        """Handle fault signal from fault detector."""
        fault_type = msg.data
        self.get_logger().error(f"Fault signal received: {fault_type}")

        transition = self.state_machine.transition(Mode.STANDBY, fault_type=fault_type)

        if transition.success:
            self.get_logger().error(f"Entered FAULT state due to: {fault_type}")
        self.publish_state()

    def publish_state(self):
        """Publish current state and control status."""
        # Publish state string
        state_msg = String()
        state_msg.data = self.state_machine.state.name
        self.state_pub.publish(state_msg)

        # Publish control status
        status_msg = ControlStatusMsg()
        status_msg.navigation_enabled = self.state_machine.navigation_enabled
        status_msg.manual_enabled = self.state_machine.manual_enabled
        status_msg.emergency_stop = self.state_machine.emergency_stop
        status_msg.fault_type = self.state_machine.fault_type or ""
        self.control_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ModeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
