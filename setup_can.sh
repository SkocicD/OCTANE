#!/bin/bash
# One-time setup for gs_usb CAN adapter.
# Run once with sudo.

set -e

REAL_USER="${SUDO_USER:-$USER}"

# USB device permissions
echo "Writing USB permissions rule..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", MODE="0666", GROUP="plugdev"' \
    | sudo tee /etc/udev/rules.d/90-gs-usb.rules > /dev/null

# Auto-configure and bring up any gs_usb-backed CAN interface when it appears.
# Fires after the authorized toggle in octane-can-reset, which is the software
# equivalent of a physical replug and is safe for CANable firmware.
echo "Writing CAN auto-start rule..."
cat <<'EOF' | sudo tee /etc/udev/rules.d/91-can-autostart.rules > /dev/null
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", DRIVERS=="gs_usb", \
    RUN+="/usr/sbin/ip link set %k type can bitrate 1000000", \
    RUN+="/usr/sbin/ip link set %k up"
EOF

# Helper script: soft-replug the CAN transceiver by toggling the USB authorized
# attribute. This is what the kernel does on physical unplug/replug without
# sending USBDEVFS_RESET which can crash CANable firmware.
echo "Installing /usr/local/bin/octane-can-reset..."
cat <<'EOF' | sudo tee /usr/local/bin/octane-can-reset > /dev/null
#!/bin/bash
VID="1d50"
PID="606f"
SYSFS_PATH=""
for d in /sys/bus/usb/devices/*/; do
    [ "$(cat ${d}idVendor 2>/dev/null)" = "$VID" ] || continue
    [ "$(cat ${d}idProduct 2>/dev/null)" = "$PID" ] || continue
    SYSFS_PATH="$d"
    break
done
if [ -z "$SYSFS_PATH" ]; then
    echo "[ERROR] CAN transceiver (1d50:606f) not found on USB bus" >&2
    exit 1
fi
echo "[CAN] Disconnecting transceiver (${SYSFS_PATH})..."
echo 0 > "${SYSFS_PATH}authorized"
sleep 1
echo "[CAN] Reconnecting transceiver..."
echo 1 > "${SYSFS_PATH}authorized"
EOF
sudo chmod 755 /usr/local/bin/octane-can-reset

# Sudoers: allow launch_system.sh to call the reset helper and bring canN up/down
echo "Writing sudoers entry..."
cat <<EOF | sudo tee /etc/sudoers.d/octane-can > /dev/null
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/local/bin/octane-can-reset
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can0 down
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can0 up
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can0 type can bitrate 1000000
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can1 down
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can1 up
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can1 type can bitrate 1000000
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can2 down
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can2 up
${REAL_USER} ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can2 type can bitrate 1000000
EOF
sudo chmod 440 /etc/sudoers.d/octane-can

# Remove old sudoers file if it exists
sudo rm -f /etc/sudoers.d/octane-can0

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
echo "[OK] CAN setup complete."
echo "     Run 'sudo octane-can-reset' to soft-replug the transceiver at any time."
