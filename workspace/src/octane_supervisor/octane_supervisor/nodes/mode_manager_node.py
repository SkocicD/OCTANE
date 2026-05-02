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
    /supervisor/navigation_enabled (std_msgs/Bool)
        Output: Navigation control active
/supervisor/manual_enabled (std_msgs/Bool)
Output: Manual control active
/supervisor/e_suggestion (std_msgs/Bool)
Output: E-suggestion active (software fault)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Bool
from octane_supervisor.state_machine import StateMachine, Mode, State


class ModeManagerNode(Node):
    """Node that manages system state transitions."""

    def __init__(self):
        super().__init__("mode_manager_node")

        self.state_machine = StateMachine()
        self.declare_parameters(
            namespace="",
            parameters=[
                ("check_rate", 10.0),
                ("disable_faults", False),
            ],
        )
        check_rate = self.get_parameter("check_rate").get_parameter_value().double_value
        self.disable_faults = self.get_parameter("disable_faults").get_parameter_value().bool_value

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
        self.create_subscription(
            String,
            "/supervisor/fault_status",
            self.fault_status_callback,
            qos,
        )

        # Track fault status for fault reset validation
        self.faults_active = False

        # Publishers
        self.state_pub = self.create_publisher(String, "/supervisor/state", qos)
        self.navigation_enabled_pub = self.create_publisher(Bool, "/supervisor/navigation_enabled", qos)
        self.manual_enabled_pub = self.create_publisher(Bool, "/supervisor/manual_enabled", qos)
        self.e_suggestion_pub = self.create_publisher(Bool, "/supervisor/e_suggestion", qos)

        # Timer for state publishing
        self.timer = self.create_timer(1.0 / check_rate, self.publish_state)

        if self.disable_faults:
            self.get_logger().warn("DEV MODE: fault tripping is DISABLED — FAULT state will never be entered")

        self.get_logger().info("Mode manager node initialized")
        self.publish_state()

    def fault_status_callback(self, msg: String):
        """Update fault status from fault manager."""
        # Simple check: if message contains "NO_faults", then no active faults
        self.faults_active = "NO_faults" not in msg.data
        if self.faults_active:
            self.get_logger().debug(f"Faults active: {msg.data}")
        else:
            self.get_logger().debug("No faults active")

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
        if target_mode == Mode.FAULT_RESET and self.faults_active:
            self.get_logger().warn("Fault reset ignored: faults are still active")
            return

        transition = self.state_machine.transition(target_mode)

        if transition.success:
            self.get_logger().info(f"Mode changed to: {transition.to_state.name}")
        else:
            self.get_logger().warn(f"Mode transition failed: {transition.reason}")

        self.publish_state()

    def fault_signal_callback(self, msg: String):
        """Handle fault signal from fault detector."""
        fault_type = msg.data

        if self.disable_faults:
            self.get_logger().warn(f"DEV MODE: fault suppressed ({fault_type}) — FAULT transition skipped")
            return

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

        # Publish individual control status flags
        nav_msg = Bool()
        nav_msg.data = self.state_machine.navigation_enabled
        self.navigation_enabled_pub.publish(nav_msg)

        manual_msg = Bool()
        manual_msg.data = self.state_machine.manual_enabled
        self.manual_enabled_pub.publish(manual_msg)

        esug_msg = Bool()
        esug_msg.data = self.state_machine.e_suggestion
        self.e_suggestion_pub.publish(esug_msg)


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
