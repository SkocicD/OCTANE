#!/bin/bash
# assign_node_id.sh — Flash a new CANopen node ID onto a BLD_CAN motor controller.
# Based on David's change_node_id.py from the motor_control branch.
# Run once per motor with ONLY THAT MOTOR powered on the CAN bus.

set -euo pipefail

# ── 1. Check CAN adapter ──────────────────────────────────────────────────────
echo "Checking for gs_usb CAN adapter (1d50:606f)..."
if ! lsusb | grep -qi "1d50:606f"; then
    echo "ERROR: gs_usb adapter not found. Plug in the CAN transceiver and retry."
    exit 1
fi
echo "[OK] CAN adapter found."
echo ""

# ── 2. Prompt for node IDs ────────────────────────────────────────────────────
read -rp "Current node ID [default: 1]: " CURRENT_ID
CURRENT_ID="${CURRENT_ID:-1}"

read -rp "New node ID (1–127): " NEW_ID
if [[ -z "$NEW_ID" ]]; then
    echo "ERROR: New node ID is required."
    exit 1
fi

echo ""
printf "  Current ID : %s\n" "$CURRENT_ID"
printf "  New ID     : %s\n" "$NEW_ID"
echo ""
read -rp "Assign? [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
echo ""

# ── 3. Run assignment ─────────────────────────────────────────────────────────
python3 - "$CURRENT_ID" "$NEW_ID" <<'PYEOF'
import sys, time

def parse_id(s):
    return int(s, 0)  # handles both decimal and 0x hex

current_id = parse_id(sys.argv[1])
new_id     = parse_id(sys.argv[2])

if not (1 <= new_id <= 127):
    print(f"ERROR: New node ID must be 1–127, got {new_id}")
    sys.exit(1)

try:
    from gs_usb.gs_usb import GsUsb
    from gs_usb.gs_usb_frame import GsUsbFrame
except ImportError:
    print("ERROR: gs_usb not installed — run: pip3 install gs_usb")
    sys.exit(1)

devs = GsUsb.scan()
if not devs:
    print("ERROR: No gs_usb device found.")
    sys.exit(1)

dev = devs[0]
try:
    dev.gs_usb.detach_kernel_driver(0)
except Exception:
    pass

if not dev.set_bitrate(1_000_000):
    print("ERROR: Could not set 1 Mbit/s bitrate.")
    sys.exit(1)

dev.start()

SDO_TX = 0x600
SPEC = {1: b'\x2f', 2: b'\x2b', 3: b'\x27', 4: b'\x23'}

def sdo_write(node_id, index, subindex, value, flip=True):
    data = SPEC[len(value)] + index[::-1] + subindex + (value[::-1] if flip else value)
    data += b'\x00' * (8 - len(data))
    dev.send(GsUsbFrame(can_id=SDO_TX | node_id, data=data))
    time.sleep(0.05)

# NMT: start node so it accepts SDO writes
nmt = b'\x01' + current_id.to_bytes(1, 'little') + b'\x00' * 6
dev.send(GsUsbFrame(can_id=0x000, data=nmt))
time.sleep(0.1)

print(f"Setting node ID {current_id} → {new_id} ...")

sdo_write(current_id, b'\x10\x06', b'\x00', new_id.to_bytes(1, 'little'))   # node ID
sdo_write(current_id, b'\x10\x17', b'\x00', (1000).to_bytes(2, 'big'))      # heartbeat 1000 ms
sdo_write(current_id, b'\x10\x10', b'\x00', b'\x45\x56\x41\x53', flip=False)  # save to flash

time.sleep(0.2)
print(f"[OK] Saved. Power-cycle the motor controller to activate node ID {new_id}.")
PYEOF
