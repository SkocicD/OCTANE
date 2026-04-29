#!/bin/bash

# Launch OCTANE system natively on Jetson.
# Sources ROS 2 and the workspace, then launches the requested subsystem.
#
# Usage:
#   ./launch_system.sh                  - Launch all subsystems
#   ./launch_system.sh supervisor       - Launch supervisor only
#   ./launch_system.sh perception       - Launch perception only
#   ./launch_system.sh mapping          - Launch mapping only
#   ./launch_system.sh network          - Launch network only

WORKSPACE_ROOT="/home/csulunabotics/OCTANE/workspace"
LAUNCH_PKG="octane"

# Source ROS 2
ROS_SETUP=""
for distro in humble jazzy; do
    if [ -f "/opt/ros/${distro}/setup.bash" ]; then
        ROS_SETUP="/opt/ros/${distro}/setup.bash"
        break
    fi
done

if [ -z "$ROS_SETUP" ]; then
    echo "[ERROR] ROS 2 not found. Run ./build_system.sh first to install dependencies."
    exit 1
fi

source "$ROS_SETUP"

# Source built workspace
WORKSPACE_SETUP="${WORKSPACE_ROOT}/install/setup.bash"
if [ ! -f "$WORKSPACE_SETUP" ]; then
    echo "[ERROR] Workspace not built. Run ./build_system.sh first."
    exit 1
fi

source "$WORKSPACE_SETUP"
echo "[OK] ROS 2 environment sourced"

# Determine which launch file to run
SUBSYSTEM="${1:-all}"

run_launch() {
    local name="$1"
    local file="${WORKSPACE_ROOT}/src/octane/octane/launch/${name}.launch.py"
    if [ ! -f "$file" ]; then
        echo "[ERROR] Launch file not found: ${name}.launch.py"
        exit 1
    fi
    echo "[LAUNCH] Starting ${name}..."
    ros2 launch "$LAUNCH_PKG" "${name}.launch.py"
}

case "$SUBSYSTEM" in
    supervisor)  run_launch supervisor ;;
    perception)  run_launch perception ;;
    mapping)     run_launch mapping ;;
    network)     run_launch network ;;
    all)
        echo "[LAUNCH] Starting all OCTANE subsystems..."
        ros2 launch "$LAUNCH_PKG" supervisor.launch.py &
        ros2 launch "$LAUNCH_PKG" perception.launch.py &
        ros2 launch "$LAUNCH_PKG" mapping.launch.py &
        ros2 launch "$LAUNCH_PKG" network.launch.py
        ;;
    *)
        echo "Unknown subsystem: $SUBSYSTEM"
        echo "Usage: $0 [supervisor|perception|mapping|network|all]"
        exit 1
        ;;
esac
