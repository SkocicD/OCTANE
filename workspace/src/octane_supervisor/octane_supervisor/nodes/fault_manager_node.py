#!/usr/bin/env python3
"""Fault manager node for OCTANE supervisor.

Aggregates fault signals, maintains fault registry, and provides
a live terminal display (runs in its own xterm window).

Topics:
    /supervisor/fault_signal (std_msgs/String)   — input fault trigger
    /supervisor/fault_status (std_msgs/String)   — output current status
    /supervisor/fault_reset  (std_msgs/Empty)    — input reset command
"""

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, Empty
from dataclasses import dataclass
from typing import Dict


@dataclass
class FaultEntry:
    name: str
    severity: str
    description: str
    active: bool
    timestamp: float
    auto_recover: bool


class FaultManagerNode(Node):

    SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}

    FAULT_DEFS = [
        ("battery_undervoltage", "critical", "Battery undervoltage"),
        ("battery_overcurrent",  "high",     "Battery overcurrent"),
        ("motor_overcurrent",    "medium",   "Motor overcurrent"),
        ("motor_driver_failure", "high",     "Motor driver failure"),
        ("wifi_connection_loss", "high",     "WiFi connection lost"),
        ("ml_inference_crash",   "critical", "ML inference crash"),
        ("can_bus_down",         "critical", "CAN bus down"),
        ("april_tag_lost",       "high",     "AprilTag localization lost"),
    ]

    SEV_COLOR = {
        "critical": "\033[91m",  # bright red
        "high":     "\033[93m",  # bright yellow
        "medium":   "\033[33m",  # yellow
        "low":      "\033[36m",  # cyan
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def __init__(self):
        super().__init__("fault_manager_node")

        self.declare_parameters(
            namespace="",
            parameters=[("check_rate", 10.0)],
        )
        check_rate = self.get_parameter("check_rate").get_parameter_value().double_value

        self.faults: Dict[str, FaultEntry] = {}
        for name, severity, desc in self.FAULT_DEFS:
            self.faults[name] = FaultEntry(
                name=name,
                severity=severity,
                description=desc,
                active=False,
                timestamp=0.0,
                auto_recover=severity in ("low", "medium"),
            )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(String, "/supervisor/fault_signal", self.fault_signal_callback, qos)
        self.create_subscription(Empty,  "/supervisor/fault_reset",  self.fault_reset_callback,  qos)

        self.fault_status_pub = self.create_publisher(String, "/supervisor/fault_status", qos)

        self.create_timer(1.0 / check_rate, self.publish_status)
        self.create_timer(1.0, self._refresh_display)

        self._print_header()

    # ── display ──────────────────────────────────────────────────────────────

    def _print_header(self):
        print("\033[2J\033[H", end="")
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        print(f"{self.BOLD}    OCTANE FAULT MANAGER{self.RESET}")
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        print(f"    Monitoring {len(self.faults)} fault definitions")
        print(f"    Watching: /supervisor/fault_signal")
        print(f"    Press Ctrl+C to stop\n")
        sys.stdout.flush()

    def _refresh_display(self):
        active = [f for f in self.faults.values() if f.active]
        now = time.time()

        print("\033[2J\033[H", end="")
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        print(f"{self.BOLD}    OCTANE FAULT MANAGER{self.RESET}")
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")

        if not active:
            print(f"\n    \033[92m{self.BOLD}STATUS: ALL CLEAR{self.RESET}  — no active faults\n")
        else:
            worst = max(active, key=lambda f: self.SEVERITY_ORDER.get(f.severity, 0))
            color = self.SEV_COLOR.get(worst.severity, "")
            print(f"\n    {color}{self.BOLD}STATUS: {len(active)} ACTIVE FAULT(S) [{worst.severity.upper()}]{self.RESET}\n")

            for f in sorted(active, key=lambda f: -self.SEVERITY_ORDER.get(f.severity, 0)):
                c = self.SEV_COLOR.get(f.severity, "")
                age = now - f.timestamp
                print(f"    {c}{self.BOLD}[{f.severity.upper():8}]{self.RESET}  {f.name}")
                print(f"               {f.description}  ({age:.0f}s ago)")
                if f.auto_recover:
                    print(f"               \033[36m(auto-recoverable){self.RESET}")
                print()

        inactive_count = sum(1 for f in self.faults.values() if not f.active)
        print(f"{self.BOLD}{'-' * 60}{self.RESET}")
        print(f"    {inactive_count}/{len(self.faults)} faults clear   |   Press Ctrl+C to stop")
        print(f"{self.BOLD}{'=' * 60}{self.RESET}")
        sys.stdout.flush()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def fault_signal_callback(self, msg: String):
        fault_name = msg.data

        if fault_name not in self.faults:
            self.faults[fault_name] = FaultEntry(
                name=fault_name,
                severity="medium",
                description="Unknown fault",
                active=False,
                timestamp=0.0,
                auto_recover=True,
            )

        fault = self.faults[fault_name]
        fault.active = True
        fault.timestamp = time.time()
        self.get_logger().error(f"Fault registered: {fault_name} ({fault.severity})")

    def fault_reset_callback(self, msg: Empty):
        critical_active = [
            name for name, f in self.faults.items() if f.active and not f.auto_recover
        ]
        if critical_active:
            self.get_logger().warn(f"Cannot reset: critical faults still active: {critical_active}")
            return

        for name, fault in self.faults.items():
            if fault.auto_recover and fault.active:
                fault.active = False
                self.get_logger().info(f"Fault cleared: {name}")

        self.get_logger().info("Fault reset completed")

    def publish_status(self):
        active = [f for f in self.faults.values() if f.active]

        if active:
            max_sev = max(active, key=lambda f: self.SEVERITY_ORDER.get(f.severity, 0)).severity
            parts = [f"{f.name}({f.severity})" for f in active]
            status = f"FAULTS_ACTIVE [{max_sev}]: {', '.join(parts)}"
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
