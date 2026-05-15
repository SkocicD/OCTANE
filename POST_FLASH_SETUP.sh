#!/bin/bash
# Run this once after every Jetson OS reflash to restore all hardware setup.
# Must be run with sudo from the OCTANE directory on the SSD.
#
#   sudo ./POST_FLASH_SETUP.sh
#
# What it does (in order):
#   1. mDNS      — octane.local hostname via avahi
#   2. Serial    — removes brltty, udev symlinks, group permissions for Arduino + RS485
#   3. CAN       — builds gs_usb.ko, udev auto-start rules, sudoers, octane-can-reset
#   4. Cameras   — udev symlinks for all 5 Innomaker cams + Orbbec
#
# After this completes, run build_system.sh to (re)build the ROS workspace.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] Run with sudo: sudo ./POST_FLASH_SETUP.sh"
    exit 1
fi

run_step() {
    local name="$1"
    local script="$2"
    echo ""
    echo "════════════════════════════════════════"
    echo "  $name"
    echo "════════════════════════════════════════"
    bash "${SCRIPT_DIR}/${script}"
}

run_step "1/4  mDNS (octane.local)"        setup_mdns.sh
run_step "2/4  Serial (Arduino + RS485)"   setup_serial.sh
run_step "3/4  CAN (gs_usb + CANable)"    setup_can.sh
run_step "4/4  Cameras (udev symlinks)"    setup_cameras.sh

echo ""
echo "════════════════════════════════════════"
echo "  POST-FLASH SETUP COMPLETE"
echo "════════════════════════════════════════"
echo ""
echo "  Next step: run ./build_system.sh to build the ROS workspace."
echo ""
echo "  Verify devices:"
echo "    /dev/rs485_drive   -> $(readlink /dev/rs485_drive 2>/dev/null || echo MISSING)"
echo "    /dev/arduino_nano  -> $(readlink /dev/arduino_nano 2>/dev/null || echo MISSING)"
echo "    /dev/cam_left_front  -> $(readlink /dev/cam_left_front 2>/dev/null || echo MISSING)"
echo "    /dev/cam_right_front -> $(readlink /dev/cam_right_front 2>/dev/null || echo MISSING)"
echo "    /dev/cam_left_side   -> $(readlink /dev/cam_left_side 2>/dev/null || echo MISSING)"
echo "    /dev/cam_right_side  -> $(readlink /dev/cam_right_side 2>/dev/null || echo MISSING)"
echo "    /dev/cam_back_rear   -> $(readlink /dev/cam_back_rear 2>/dev/null || echo MISSING)"
echo "    can2 (gs_usb):     $(ip link show | grep -o 'can[0-9]*' | while read i; do
        driver=\$(readlink -f /sys/class/net/\$i/device/driver 2>/dev/null | xargs basename 2>/dev/null)
        [ "\$driver" = "gs_usb" ] && echo "\$i (UP)" && break
    done || echo MISSING)"
echo ""
