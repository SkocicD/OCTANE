#!/usr/bin/env bash
# ================================================================
#  USD Isaac Lab Compliance + Fix Pipeline
#
#  Runs three tools in sequence:
#    1. check_usd_compliance  — read-only audit (PASS/WARN/FAIL)
#    2. disable_nonwheel_collision — disables arm/non-drive collision
#    3. lock_arm_joints       — replaces arm drives with position-hold
#
#  Usage:
#    ./run_usd_tools.sh "/path/to/robot.usd"
#
#  Edit the CONFIG section below once per robot.
# ================================================================
set -euo pipefail

# ----------------------------------------------------------------
#  CONFIG — edit these for your robot
# ----------------------------------------------------------------

# Path to isaaclab.sh
ISAACLAB="/home/$USER/IsaacLab/isaaclab.sh"

# Rigid body prim NAMES to keep collision ON (chassis + all wheels)
KEEP_BODIES="tn__base_link1_wJ,tn__Wheel11_i7t6,tn__Wheel21_i7t6,tn__Wheel31_i7t6,tn__Wheel41_i7t6,tn__Wheel51_i7t6,tn__Wheel61_i7t6"

# Joint prim NAMES to skip when locking arm joints
# (Python-actuated / wheel joints — leave their drives alone)
SKIP_JOINTS="Left_Front_Wheel,Left_Center_Wheel,Left_Rear_Wheel,Right_Front_Wheel,Right_Center_Wheel,Right_Rear_Wheel"

# ----------------------------------------------------------------
#  ARGUMENT CHECK
# ----------------------------------------------------------------
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo ""
    echo "  Usage: ./run_usd_tools.sh \"/path/to/robot.usd\""
    echo ""
    exit 1
fi
USD="$1"

echo ""
echo "================================================================"
echo "  USD Isaac Lab Tool Pipeline"
echo "  $USD"
echo "================================================================"

# ----------------------------------------------------------------
#  STEP 1 — Compliance check (read-only)
# ----------------------------------------------------------------
echo ""
echo "---- STEP 1 / 3 :  Compliance Check  (read-only) ----"
echo ""
"$ISAACLAB" -p "$TOOLS_DIR/check_usd_compliance.py" \
    --usd "$USD" \
    --headless

# ----------------------------------------------------------------
#  STEP 2 — Disable non-wheel collision
# ----------------------------------------------------------------
echo ""
echo "---- STEP 2 / 3 :  Disable Non-Wheel Collision ----"
echo ""
"$ISAACLAB" -p "$TOOLS_DIR/disable_nonwheel_collision.py" \
    --usd "$USD" \
    --keep "$KEEP_BODIES" \
    --headless

# ----------------------------------------------------------------
#  STEP 3 — Lock arm joints to position-hold
# ----------------------------------------------------------------
echo ""
echo "---- STEP 3 / 4 :  Lock Arm Joints ----"
echo ""
"$ISAACLAB" -p "$TOOLS_DIR/lock_arm_joints.py" \
    --usd "$USD" \
    --skip "$SKIP_JOINTS" \
    --headless

# ----------------------------------------------------------------
#  STEP 4 — Deactivate OmniGraph nodes (ROS2 controllers etc.)
# ----------------------------------------------------------------
echo ""
echo "---- STEP 4 / 4 :  Deactivate OmniGraph Nodes ----"
echo ""
"$ISAACLAB" -p "$TOOLS_DIR/deactivate_omnigraphs.py" \
    --usd "$USD" \
    --headless

echo ""
echo "================================================================"
echo "  Done."
echo "================================================================"
echo ""
