#!/bin/bash
# Remove brltty, which Ubuntu installs by default and which hijacks CH340/CH341
# USB serial devices (Arduino Nano, RS485 transceiver) before the ch341 driver
# can bind — preventing /dev/ttyUSB* from appearing.
#
# Run once with sudo after a fresh flash if /dev/arduino_nano is missing:
#   sudo ./fix_brltty.sh

set -e

if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] Run with sudo: sudo ./fix_brltty.sh"
    exit 1
fi

if dpkg -s brltty &>/dev/null 2>&1; then
    echo "[SETUP] Removing brltty..."
    apt-get remove --purge -y brltty
    echo "[OK] brltty removed"
else
    echo "[OK] brltty not present"
fi

# Force USB re-enumeration for the Arduino (1-4.2.1) so ch341 binds immediately
# without needing a physical replug.
ARDUINO_USB="/sys/bus/usb/devices/1-4.2.1"
if [ -d "$ARDUINO_USB" ]; then
    echo "[SETUP] Re-binding Arduino Nano USB port..."
    echo 0 > "${ARDUINO_USB}/authorized"
    sleep 1
    echo 1 > "${ARDUINO_USB}/authorized"
    sleep 2
fi

udevadm control --reload-rules
udevadm trigger

sleep 2

echo ""
if [ -L /dev/arduino_nano ]; then
    echo "[OK] /dev/arduino_nano -> $(readlink /dev/arduino_nano)"
else
    echo "[WARN] /dev/arduino_nano still missing — try unplugging and replugging the Arduino"
fi
if [ -L /dev/rs485_drive ]; then
    echo "[OK] /dev/rs485_drive  -> $(readlink /dev/rs485_drive)"
fi
