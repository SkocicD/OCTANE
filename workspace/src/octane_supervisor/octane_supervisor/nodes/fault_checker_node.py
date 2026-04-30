#!/usr/bin/env python3
"""Fault checker node for OCTANE supervisor.

Reads fault conditions from YAML config, subscribes to sensor topics,
and publishes fault signals when conditions are met.

Config: workspace/src/octane_supervisor/config/faults.yaml

Topics:
    /supervisor/fault_signal (std_msgs/String)
        Output: Fault type when triggered
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
import yaml


class FaultCheckerNode(Node):
    """Node that evaluates fault conditions from sensor data."""

    def __init__(self):
        super().__init__("fault_checker_node")

        # Load fault configuration — use the faults_config parameter from the launch file
        self.declare_parameter('faults_config', '')
        config_path = self.get_parameter('faults_config').get_parameter_value().string_value
        if not config_path or not os.path.exists(config_path):
            config_path = self.find_param_file("faults.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.faults = self.config.get("faults", {})
        self.supervisor_config = self.config.get("supervisor", {})

        self.get_logger().info(f"Loaded {len(self.faults)} fault definitions")

        # Track active faults and sensor data
        self.active_faults = {}
        self.sensor_data = {}

        # Subscriptions to sensor topics (created dynamically)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        for fault_name, fault_cfg in self.faults.items():
            source = fault_cfg.get("source")
            if source:
                self.create_subscription(
                    String,
                    source,
                    lambda msg, fn=fault_name: self.sensor_callback(msg, fn),
                    qos,
                )

        # Fault check timer
        check_rate = self.supervisor_config.get("fault_check_rate", 10.0)
        self.timer = self.create_timer(1.0 / check_rate, self.check_faults)

        # Fault signal publisher
        self.fault_pub = self.create_publisher(String, "/supervisor/fault_signal", qos)

        self.get_logger().info("Fault checker node initialized")

    def find_param_file(self, filename: str) -> str:
        """Find config file in package share directory."""
        # Relative to install location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "config", filename)
        if os.path.exists(config_path):
            return config_path

        # Fallback to workspace path
        workspace_root = os.environ.get("WORKSPACE_ROOT", "/workspace")
        config_path = os.path.join(
            workspace_root, "src", "octane_supervisor", "octane_supervisor", "config", filename
        )
        if os.path.exists(config_path):
            return config_path

        raise FileNotFoundError(f"Config file not found: {filename}")

    def sensor_callback(self, msg: String, fault_name: str):
        """Handle sensor data update."""
        try:
            import json
            try:
                data = json.loads(msg.data)
                self.sensor_data[fault_name] = data
            except json.JSONDecodeError:
                self.sensor_data[fault_name] = {"value": msg.data}
        except Exception as e:
            self.get_logger().warn(f"Error parsing sensor data for {fault_name}: {e}")

    def check_faults(self):
        """Evaluate all fault conditions."""
        for fault_name, fault_cfg in self.faults.items():
            if fault_name in self.active_faults:
                # Fault already active, check if auto-recoverable
                if fault_cfg.get("auto_recover", False):
                    if not self._evaluate_condition(fault_name, fault_cfg):
                        self._clear_fault(fault_name, fault_cfg)
                continue

            # Evaluate condition
            if self._evaluate_condition(fault_name, fault_cfg):
                self._trigger_fault(fault_name, fault_cfg)

    def _evaluate_condition(self, fault_name: str, fault_cfg: dict) -> bool:
        """Evaluate fault condition expression."""
        condition = fault_cfg.get("condition", "")
        if not condition:
            return False

        try:
            context = {}

            # Add sensor data
            if fault_name in self.sensor_data:
                context.update(self.sensor_data[fault_name])

            # Add threshold for comparison
            if "threshold" in fault_cfg:
                context["threshold"] = fault_cfg["threshold"]

            result = eval(condition, {"__builtins__": {}}, context)
            return bool(result)
        except NameError:
            return False  # sensor data not yet received — not an error
        except Exception as e:
            self.get_logger().warn(f"Error evaluating condition for {fault_name}: {e}")
            return False

    def _trigger_fault(self, fault_name: str, fault_cfg: dict):
        """Trigger a fault."""
        severity = fault_cfg.get("severity", "medium")
        description = fault_cfg.get("description", fault_name)

        self.get_logger().error(f"FAULT TRIGGERED: {fault_name} ({severity}) - {description}")

        self.active_faults[fault_name] = fault_cfg

        # Publish fault signal
        msg = String()
        msg.data = fault_name
        self.fault_pub.publish(msg)

    def _clear_fault(self, fault_name: str, fault_cfg: dict):
        """Clear an auto-recoverable fault."""
        self.get_logger().info(f"Fault cleared: {fault_name}")
        del self.active_faults[fault_name]


def main(args=None):
    rclpy.init(args=args)
    node = FaultCheckerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
