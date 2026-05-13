import threading
from typing import Callable

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

        self._rx_callback: Callable[[int, bytes], None] | None = None
        self._tx_callback: Callable[[int, bytes], None] | None = None

        # Drain RX buffer continuously so the adapter never gets flow-controlled.
        # Without this, motor reply frames fill the USB RX queue and TX stops working.
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()

    def set_rx_callback(self, cb: Callable[[int, bytes], None]) -> None:
        self._rx_callback = cb

    def set_tx_callback(self, cb: Callable[[int, bytes], None]) -> None:
        self._tx_callback = cb

    def _drain_loop(self):
        frame = GsUsbFrame()
        while True:
            if self._dev.read(frame, 1):
                cb = self._rx_callback
                if cb:
                    try:
                        cb(frame.arbitration_id, bytes(frame.data[:frame.can_dlc]))
                    except Exception:
                        pass

    def send(self, can_id: int, data: bytes):
        cb = self._tx_callback
        if cb:
            try:
                cb(can_id, data)
            except Exception:
                pass
        self._dev.send(GsUsbFrame(can_id=can_id, data=data))
