import threading

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
            pass

        if not self._dev.set_bitrate(bitrate):
            raise RuntimeError(f'Could not set CAN bitrate to {bitrate}')

        self._dev.start()

        # Drain RX buffer continuously so the adapter never gets flow-controlled.
        # Without this, motor reply frames fill the USB RX queue and TX stops working.
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def _drain_loop(self):
        frame = GsUsbFrame()
        while True:
            self._dev.read(frame, 1)  # 1 ms timeout, loops forever

    def send(self, can_id: int, data: bytes):
        self._dev.send(GsUsbFrame(can_id=can_id, data=data))
