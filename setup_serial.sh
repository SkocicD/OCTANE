#!/bin/bash
# One-time setup for USB serial devices (RS485 transceiver + Arduino Nano).
# Run once with sudo.
#
# Both devices use CH340 (1a86:7523) with no serial number so they are
# distinguished by physical USB port path (KERNELS in udev):
#   RS485 transceiver : 1-4.3   (Jetson carrier board USB port, direct)
#   Arduino Nano      : 1-4.2.1 (external 4-port USB 3.0 hub)

set -e

# ── 0. Remove brltty ──────────────────────────────────────────────────────────
# Ubuntu installs brltty (Braille TTY) by default. It claims CH340/CH341 USB
# serial devices as Braille displays before the ch341 driver can bind, which
# prevents /dev/ttyUSB* from appearing for the Arduino Nano.
if dpkg -s brltty &>/dev/null 2>&1; then
    echo "[SETUP] Removing brltty (hijacks CH340 devices)..."
    apt-get remove --purge -y brltty
    echo "[OK] brltty removed"
else
    echo "[OK] brltty not present"
fi

# Add user to dialout for future sessions; plugdev rule below gives immediate access
REAL_USER="${SUDO_USER:-$USER}"
if ! groups "$REAL_USER" | grep -q dialout; then
    echo "[SETUP] Adding ${REAL_USER} to dialout group..."
    usermod -aG dialout "$REAL_USER"
fi

echo "Writing RS485 udev rule..."
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", KERNELS=="1-4.3", GROUP="plugdev", MODE="0660", SYMLINK+="rs485_drive"' \
    | sudo tee /etc/udev/rules.d/99-rs485-drive.rules > /dev/null

echo "Writing Arduino Nano udev rule..."
echo 'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", KERNELS=="1-4.2.1", GROUP="plugdev", MODE="0660", SYMLINK+="arduino_nano"' \
    | sudo tee /etc/udev/rules.d/99-arduino-nano.rules > /dev/null

sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "[OK] Serial setup complete."
echo "     RS485 transceiver : /dev/rs485_drive  (plug into Jetson carrier USB port)"
echo "     Arduino Nano      : /dev/arduino_nano (plug into external USB hub)"
