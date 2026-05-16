# CAN Recovery Scripts

## octane-can-reset

Soft-replug the gs_usb CAN transceiver (1d50:606f) without physical access.
Uses the USB driver unbind/bind sysfs interface — the Jetson-safe alternative
to USBDEVFS_RESET, which crashes CANable firmware.

Install via setup_can.sh (run once with sudo):
```bash
sudo ./setup_can.sh
```

Then call manually or from launch_system.sh:
```bash
sudo octane-can-reset
```

## gs_usb_module

Out-of-tree kernel module build for gs_usb (CANable/candleLight USB CAN driver).
Compiled against linux-headers-5.15.185-tegra-ubuntu22.04_aarch64.

NOT currently used — the Tegra kernel lacks the AF_CAN socket layer so SocketCAN
does not work. The Python gs_usb library (libusb) is used instead via can_transceiver.py.

Kept here in case a future kernel update adds AF_CAN support, at which point
switching to SocketCAN (python-can, can2 interface) would be straightforward.

To rebuild:
```bash
cd gs_usb_module && make
sudo cp gs_usb.ko /lib/modules/$(uname -r)/kernel/drivers/net/can/usb/
sudo depmod -a
sudo modprobe gs_usb
```
