#!/usr/bin/env bash
# ── Lunabotics Isaac Lab project wrapper (Linux/macOS) ───────────────────────
# Reads ISAACLAB_PATH from isaaclab_path.cfg (next to this script) and
# forwards all arguments to the real isaaclab.sh in that installation.
#
# Usage:  ./isaaclab.sh -p scripts/rsl_rl/train.py --task Template-Lunabotics-Direct-v0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_FILE="$SCRIPT_DIR/isaaclab_path.cfg"

if [ ! -f "$CFG_FILE" ]; then
    echo "ERROR: $CFG_FILE not found."
    echo "Copy isaaclab_path.cfg.example to isaaclab_path.cfg and set ISAACLAB_PATH."
    exit 1
fi

ISAACLAB_PATH=$(grep "^ISAACLAB_PATH=" "$CFG_FILE" | cut -d'=' -f2-)

if [ -z "$ISAACLAB_PATH" ]; then
    echo "ERROR: ISAACLAB_PATH not set in $CFG_FILE"
    exit 1
fi

if [ ! -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
    echo "ERROR: isaaclab.sh not found at $ISAACLAB_PATH"
    echo "Check that ISAACLAB_PATH in isaaclab_path.cfg is correct."
    exit 1
fi

exec "$ISAACLAB_PATH/isaaclab.sh" "$@"
