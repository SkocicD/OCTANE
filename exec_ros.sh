#!/bin/bash

# Open an interactive shell with ROS 2 and the OCTANE workspace sourced.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$SCRIPT_DIR/workspace"

ROS_SETUP=""
for distro in humble jazzy; do
    if [ -f "/opt/ros/${distro}/setup.bash" ]; then
        ROS_SETUP="/opt/ros/${distro}/setup.bash"
        break
    fi
done

if [ -z "$ROS_SETUP" ]; then
    echo "[ERROR] ROS 2 not found. Run ./build_system.sh first."
    exit 1
fi

echo "[OK] Sourcing ROS 2: $ROS_SETUP"
exec bash --rcfile <(cat ~/.bashrc; echo "source $ROS_SETUP"; echo "source ${WORKSPACE_ROOT}/install/setup.bash 2>/dev/null"; echo 'echo "[OK] OCTANE workspace ready"')
