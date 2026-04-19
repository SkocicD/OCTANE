#!/usr/bin/env python3
"""Fault manager node for OCTANE supervisor.

Aggregates fault signals from multiple detectors, maintains fault status,
and provides fault reset functionality.

Topics:
    /supervisor/fault_signal (std_msgs/String)
        Input: Fault trigger signals
    /supervisor/fault_status (std_msgs/String)
        Output: Current fault status for all monitored faults
    /supervisor/fault_reset (std_msgs/Empty)
        Input: Reset command from ground station
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Empty
from dataclasses import dataclass
from typing import Dict
import time


@dataclass
class FaultEntry:
    """Single fault entry."""
    name: str
    severity: str
    description: str
    active: bool
    timestamp: float
    auto_recover: bool


class FaultManagerNode(Node):
    """Node that manages fault aggregation and status."""

    SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    def __init__(self):
        super().__init__("fault_manager_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("check_rate", 10.0),
            ],
        )
        check_rate = self.get_parameter("check_rate").get_parameter_value().double_value

        # Fault registry
        self.faults: Dict[str, FaultEntry] = {}

        # Initialize with known faults from config
        fault_defs = [
            ("battery_undervoltage", "critical", "Battery undervoltage"),
            ("battery_overcurrent", "high", "Battery overcurrent"),
            ("motor_overcurrent", "medium", "Motor overcurrent"),
            ("motor_driver_failure", "high", "Motor driver failure"),
            ("wifi_connection_loss", "high", "WiFi connection lost"),
            ("ml_inference_crash", "critical", "ML inference crash"),
            ("can_bus_down", "critical", "CAN bus down"),
            ("april_tag_lost", "high", "AprilTag localization lost"),
        ]

        for name, severity, desc in fault_defs:
            self.faults[name] = FaultEntry(
                name=name,
                severity=severity,
                description=desc,
                active=False,
                timestamp=0,
                auto_recover=severity in ["low", "medium"],
            )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Subscriptions
        self.create_subscription(
            String,
            "/supervisor/fault_signal",
            self.fault_signal_callback,
            qos,
        )
        self.create_subscription(
            Empty,
            "/supervisor/fault_reset",
            self.fault_reset_callback,
            qos,
        )

        # Publishers
        self.fault_status_pub = self.create_publisher(
            String, "/supervisor/fault_status", qos
        )

        # Timer
        self.timer = self.create_timer(1.0 / check_rate, self.publish_status)

        self.get_logger().info("Fault manager node initialized")

    def fault_signal_callback(self, msg: String):
        """Handle fault trigger signal."""
        fault_name = msg.data

        if fault_name not in self.faults:
            self.faults[fault_name] = FaultEntry(
                name=fault_name,
                severity="medium",
                description="Unknown fault",
                active=False,
                timestamp=0,
                auto_recover=True,
            )

        fault = self.faults[fault_name]
        fault.active = True
        fault.timestamp = time.time()

        self.get_logger().error(f"Fault registered: {fault_name} ({fault.severity})")

    def fault_reset_callback(self, msg: Empty):
        """Handle fault reset command."""
        # Only allow reset if all non-auto-recover faults are cleared
        critical_active = [
            name for name, f in self.faults.items()
            if f.active and not f.auto_recover
        ]

        if critical_active:
            self.get_logger().warn(
                f"Cannot reset: critical faults still active: {critical_active}"
            )
            return

        # Clear all auto-recoverable faults
        for name, fault in self.faults.items():
            if fault.auto_recover and fault.active:
                fault.active = False
                self.get_logger().info(f"Fault cleared: {name}")

        self.get_logger().info("Fault reset completed")

    def publish_status(self):
        """Publish current fault status."""
        active_faults = [f for f in self.faults.values() if f.active]

        if active_faults:
            status_parts = []
            max_severity = "low"
            for fault in active_faults:
                status_parts.append(f"{fault.name}({fault.severity})")
                if self.SEVERITY_ORDER.get(fault.severity, 0) > self.SEVERITY_ORDER.get(max_severity, 0):
                    max_severity = fault.severity

            status = f"FAULTS_ACTIVE [{max_severity}]: {', '.join(status_parts)}"
            self.get_logger().debug(status)
        else:
            status = "NO_faults"

        msg = String()
        msg.data = status
        self.fault_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaultManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
