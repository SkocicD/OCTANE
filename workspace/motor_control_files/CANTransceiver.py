from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame


class CANTransceiver:
    def __init__(self):
        devs = GsUsb.scan()
        if len(devs) == 0:
            print("Can not find gs_usb device")
            return

        dev = devs[0]

        dev.gs_usb.detach_kernel_driver(0)
        if not dev.set_bitrate(1000000):
            print("Can not set bitrate for gs_usb")
            return

        dev.start()
        self.dev = dev

    def send_CAN_message(self, can_id, data):
        frame = GsUsbFrame(can_id=can_id, data=data)
        if self.dev.send(frame):
            print(f"TX  {frame}")

    def read_CAN_bus(self):
        frame = GsUsbFrame()
        while self.dev.read(frame, 1):
            print(f"RX  {frame}")


# def main():
#     cont = MotorController(node_id=0x01)
#     cont.start_CAN_node()
#     time.sleep(.1)
#     cont.set_motor_mode()
#     time.sleep(.1)
#     cont.set_motor_control(start=1)
#     time.sleep(.1)
#     cont.set_heartbeat(1000)
#     time.sleep(.1)
#     cont.send(GsUsbFrame(PDO_TX, data=b'\x00\x00\x00\x00\x00\x00\x00\x00'))
#     time.sleep(1)
#     # random choice of speed
#     b = 1
#     for speed in range(100, 2000, 20):  # 75, 500, 5):
#         speedbytes = speed.to_bytes(2, byteorder='little')
#         cont.send(GsUsbFrame(PDO_TX, data=b.to_bytes(1) +
#                   speedbytes + b'\x00\x00\x00\x00\x00'))
#         time.sleep(3)
#         cont.read_all_rx()
#         if b == 3:
#             b = 1
#         else:
#             b = 3
#     # cont.send(GsUsbFrame(PDO_TX, data=b'\x07\x11\x11\x00\x00\x00\x00\x00'))
#     # cont.set_speed(speed)
#     # time.sleep(.1)
#     # # cont.set_speed(128)
#     # cont.reset_CAN_node()
#     # cont.start_CAN_node()
#     # cont.set_object_value(b'\x61\x00', b'\x01', b'\x00\x00\x00\x00')
#     cont.read_object_value(b'\x61\x0B', b'\x02')
#     cont.read_object_value(b'\x63\x03', b'\x00')
#     while 1:
#         cont.read_all_rx()
#         time.sleep(.1)
#
#     # GsUsbFrame(can_id=0x201, data=b"\x03\x66\x66\x00\x00\x00\x00\x00"),
#
#
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         pass
