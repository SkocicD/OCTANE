#!/usr/bin/env python3
"""CAN drain node — keeps the USB RX queue clear so TX is never flow-controlled.

The gs_usb adapter queues every motor heartbeat and PDO reply frame in a USB
RX buffer.  Once that buffer fills up the adapter stops accepting new TX
submissions, so motor commands simply get dropped.  This node owns the
CANTransceiver (which already runs a tight 1 ms drain thread internally) and
stays alive as long as the launch is running.

Standalone use (no can_drive_node in the same launch):
  ros2 run octane_manual_ctrl can_drain_node

Combined use (see can_worker.py):
  The can_worker entry-point passes a pre-created CANTransceiver so both
  this node and CANDriveNode share one device handle inside one process.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String

from octane_manual_ctrl.can.can_transceiver import CANTransceiver


class CANDrainNode(Node):

    def __init__(self, transceiver: CANTransceiver | None = None):
        super().__init__('can_drain_node')

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status_pub = self.create_publisher(
            String, '/manual_ctrl/can_drain_status', status_qos)

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        self._rx_pub = self.create_publisher(String, '/can/rx', qos)
        self._tx_pub = self.create_publisher(String, '/can/tx_raw', qos)

        if transceiver is not None:
            self._transceiver = transceiver
            self._publish_status('OK — drain active (shared transceiver)')
            self.get_logger().info('CAN drain node ready (shared transceiver)')
        else:
            try:
                self._transceiver = CANTransceiver()
                self._publish_status('OK — drain active (standalone)')
                self.get_logger().info('CAN drain node ready (standalone)')
            except Exception as e:
                self._transceiver = None
                self._publish_status(f'ERROR: {e}')
                self.get_logger().error(f'CAN drain init failed: {e}')

        if self._transceiver is not None:
            self._transceiver.set_rx_callback(self._on_rx)
            self._transceiver.set_tx_callback(self._on_tx)

    def _on_rx(self, can_id: int, data: bytes) -> None:
        msg = String()
        msg.data = f'RX  0x{can_id:03X}  [{len(data)}]  {data.hex(" ").upper()}'
        self._rx_pub.publish(msg)

    def _on_tx(self, can_id: int, data: bytes) -> None:
        msg = String()
        msg.data = f'TX  0x{can_id:03X}  [{len(data)}]  {data.hex(" ").upper()}'
        self._tx_pub.publish(msg)

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CANDrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
