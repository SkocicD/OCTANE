#!/bin/bash
# One-time setup for gs_usb CAN adapter permissions.
# Run once with sudo — lets any user open the adapter without sudo.

set -e

RULE_FILE="/etc/udev/rules.d/90-gs-usb.rules"

echo "Writing udev rule to ${RULE_FILE}..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", MODE="0666", GROUP="plugdev"' \
    | sudo tee "$RULE_FILE" > /dev/null

echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

REAL_USER="${SUDO_USER:-$USER}"
if ! groups "$REAL_USER" | grep -q plugdev; then
    echo "Adding ${REAL_USER} to plugdev group..."
    sudo usermod -aG plugdev "$REAL_USER"
    echo "Done — log out and back in for group change to take effect."
else
    echo "${REAL_USER} already in plugdev group."
fi

echo ""
echo "[OK] Unplug and replug the CAN adapter, then ros2 nodes can access it without sudo."
