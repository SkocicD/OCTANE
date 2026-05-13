#!/usr/bin/env python3
"""Combined CAN worker — one process, one device handle, two nodes.

Owns the gs_usb CAN adapter and runs CANDrainNode + CANDriveNode inside a
single Python process sharing the same CANTransceiver instance.  This avoids
USB device exclusivity conflicts while ensuring the drain loop is always alive
regardless of robot state.

Used by manual_ctrl.launch.py in place of launching can_drive_node alone.
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from octane_manual_ctrl.can.can_transceiver import CANTransceiver
from octane_manual_ctrl.nodes.can_drain_node import CANDrainNode
from octane_manual_ctrl.nodes.can_drive_node import CANDriveNode


def main(args=None):
    rclpy.init(args=args)

    try:
        tx = CANTransceiver()
    except Exception as e:
        import sys
        print(f'[can_worker] CAN device not found: {e}', file=sys.stderr)
        tx = None

    drain_node = CANDrainNode(transceiver=tx)
    drive_node = CANDriveNode(transceiver=tx)

    executor = MultiThreadedExecutor()
    executor.add_node(drain_node)
    executor.add_node(drive_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        drive_node._stop_all()
        drain_node.destroy_node()
        drive_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
