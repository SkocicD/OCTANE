#!/bin/bash

# Octane system startup script
# Main entry point for launching all packcages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"

echo "=== Starting Octane System ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

# Check if workspace is built (check for both install dir and a key package)
if [ ! -d "${WORKSPACE_ROOT}/install" ] || [ ! -d "${WORKSPACE_ROOT}/install/octane" ]; then
    echo "=== No build found, building system ==="
    "${WORKSPACE_ROOT}/build_system.sh"
    echo ""
else
    echo "[OK] Using existing build"
    echo ""
fi

# Source ROS2 and workspace
if [ -f "${WORKSPACE_ROOT}/install/setup.bash" ]; then
    source "${WORKSPACE_ROOT}/install/setup.bash"
    echo "[OK] Workspace sourced"
else
    echo "[ERROR] Failed to source workspace"
    exit 1
fi

# Launch supervisor subsystem
echo ""
echo "=== Launching Supervisor Subsystem ==="
ros2 launch octane supervisor.launch.py &
SUPERVISOR_PID=$!

# Launch perception subsystem
echo ""
echo "=== Launching Perception Subsystem ==="
ros2 launch octane perception.launch.py &
PERCEPTION_PID=$!

# Launch mapping subsystem
echo ""
echo "=== Launching Mapping Subsystem ==="
ros2 launch octane mapping.launch.py &
MAPPING_PID=$!

# Launch WiFi communication subsystem
echo ""
echo "=== Launching WiFi Communication Subsystem ==="
ros2 launch octane wifi.launch.py &
WiFi_PID=$!

# Wait for user interrupt
echo ""
echo "=== Octane System Running ==="
echo "Press Ctrl+C to shutdown"
echo ""

trap 'echo ""; echo "=== Shutting Down Octane System ==="; kill $SUPERVISOR_PID $PERCEPTION_PID $MAPPING_PID $WiFi_PID 2>/dev/null; exit 0' SIGINT SIGTERM

wait
