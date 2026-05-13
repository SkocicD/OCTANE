#!/bin/bash
# One-time setup for gs_usb CAN adapter.
# Run once with sudo.

set -e

REAL_USER="${SUDO_USER:-$USER}"

# USB device permissions
echo "Writing USB permissions rule..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", MODE="0666", GROUP="plugdev"' \
    | sudo tee /etc/udev/rules.d/90-gs-usb.rules > /dev/null

# Auto-configure and bring up can0 whenever the gs_usb net interface appears.
# This fires after USB reset in launch_system.sh so no manual replug is needed.
echo "Writing CAN auto-start rule..."
cat <<'EOF' | sudo tee /etc/udev/rules.d/91-can-autostart.rules > /dev/null
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", \
    RUN+="/usr/sbin/ip link set %k type can bitrate 1000000", \
    RUN+="/usr/sbin/ip link set %k up"
EOF

# Sudoers entry so launch_system.sh can bring can0 DOWN without a password prompt.
echo "Writing sudoers entry for can0..."
echo "${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can0 down" \
    | sudo tee /etc/sudoers.d/octane-can0 > /dev/null
sudo chmod 440 /etc/sudoers.d/octane-can0

sudo udevadm control --reload-rules
sudo udevadm trigger

if ! groups "$REAL_USER" | grep -q plugdev; then
    echo "Adding ${REAL_USER} to plugdev group..."
    sudo usermod -aG plugdev "$REAL_USER"
    echo "Done — log out and back in for group change to take effect."
else
    echo "${REAL_USER} already in plugdev group."
fi

echo ""
echo "[OK] CAN setup complete. No replug needed after launch_system.sh."
