#!/usr/bin/env python3
"""
State Monitor Node for OCTANE Supervisor
Displays the current rover mode state in a clean, readable format.
Use this to verify state transitions are happening on the robot side.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import signal
import sys


class StateMonitorNode(Node):
    def __init__(self):
        super().__init__('state_monitor_node')

        # Subscribe to supervisor state
        self.subscription = self.create_subscription(
            String,
            '/supervisor/state',
            self.state_callback,
            10
        )

        self.current_state = None

        # Print header
        print("\n" + "=" * 50)
        print("    OCTANE ROVER STATE MONITOR (Supervisor)")
        print("=" * 50)
        print("    Watching: /supervisor/state")
        print("    Press Ctrl+C to stop\n")

    def state_callback(self, msg):
        """Handle incoming state messages"""
        new_state = msg.data

        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state

            # Get timestamp
            timestamp = self.get_clock().now().to_msg()
            time_str = f"{timestamp.sec}.{timestamp.nanosec // 1000000:03d}"

            # Print formatted state change
            print(f"[{time_str}] STATE CHANGE:")
            print(f"    FROM: {old_state or 'UNKNOWN':^15}")
            print(f"    TO:   {new_state:^15}")
            print("-" * 50)

            # Add state-specific formatting
            if new_state == 'STANDBY':
                print("    ⏸️  Status: READY - Awaiting command")
            elif new_state == 'MANUAL':
                print("    🎮 Status: MANUAL CONTROL ACTIVE")
            elif new_state == 'AUTONOMOUS':
                print("    🤖 Status: AUTONOMOUS MODE ACTIVE")
            elif new_state == 'FAULT':
                print("    ⚠️  Status: FAULT - Emergency stop engaged")
            else:
                print(f"    State: {new_state}")
            print()


def main(args=None):
    print("Initializing state monitor...")
    rclpy.init(args=args)

    node = StateMonitorNode()

    def signal_handler(sig, frame):
        print("\n" + "=" * 50)
        print("    State monitor stopped")
        print("=" * 50)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
