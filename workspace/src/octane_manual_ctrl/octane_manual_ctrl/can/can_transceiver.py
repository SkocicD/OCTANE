from gs_usb.gs_usb import GsUsb
from gs_usb.gs_usb_frame import GsUsbFrame


class CANTransceiver:
    def __init__(self, bitrate: int = 1_000_000):
        devs = GsUsb.scan()
        if not devs:
            raise RuntimeError('No gs_usb CAN adapter found — is the transceiver plugged in?')

        self._dev = devs[0]
        try:
            self._dev.gs_usb.detach_kernel_driver(0)
        except Exception:
            pass  # no kernel driver attached — fine

        if not self._dev.set_bitrate(bitrate):
            raise RuntimeError(f'Could not set CAN bitrate to {bitrate}')

        self._dev.start()

    def send(self, can_id: int, data: bytes):
        self._dev.send(GsUsbFrame(can_id=can_id, data=data))

    def read_all(self, timeout_ms: int = 1):
        frame = GsUsbFrame()
        frames = []
        while self._dev.read(frame, timeout_ms):
            frames.append((frame.can_id, bytes(frame.data)[:frame.can_dlc]))
        return frames
