#!/bin/bash

# Build all Octane system packages.
# Auto-discovers packages in src/ and src/external_pkgs/ — Isaac ROS / nvblox
# sources (cloned via clone_isaac_ros.sh) are built automatically from external_pkgs/.
#
# Usage:
#   ./build_system.sh           - Smart build (skip orbbec if already built)
#   ./build_system.sh --all     - Force build everything in src/
#   ./build_system.sh --octane  - Only octane_* packages
#   ./build_system.sh --orbbec  - Only orbbec packages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Fix the workspace root to point to the actual workspace directory
WORKSPACE_ROOT="/media/csulunabotics/SSD/OCTANE/workspace"

echo "=== Building Octane System ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

cd "${WORKSPACE_ROOT}"

# Source ROS environment if available
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
    echo "[INFO] Sourced ROS 2 Humble environment"
else
    echo "[WARNING] ROS 2 environment not found"
fi

# Check if colcon is available
if ! command -v colcon &> /dev/null; then
    echo "[ERROR] colcon not found. Please install colcon: sudo apt install python3-colcon-common-extensions"
    exit 1
fi

OCTANE_PKGS="octane_msgs octane_perception octane_mapping octane_supervisor octane_network octane"
ORBBEC_PKGS="astra_camera astra_camera_msgs"

# Build packages in the correct order to handle dependencies
echo "[BUILD] Building packages in dependency order..."

# First build message packages
echo "[STEP] Building message packages..."
colcon build --base-paths ${WORKSPACE_ROOT}/src --packages-select octane_msgs --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

# Then build other core packages
CORE_PKGS="octane_supervisor octane_network octane_perception"
echo "[STEP] Building core packages..."
colcon build --base-paths ${WORKSPACE_ROOT}/src --packages-select ${CORE_PKGS} --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

# Build remaining packages based on command line args
case "$1" in
    --orbbec)
        echo "[MODE] Building orbbec packages only"
        colcon build --base-paths ${WORKSPACE_ROOT}/src --packages-select ${ORBBEC_PKGS} --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
    --octane)
        echo "[MODE] Building octane packages only"
        colcon build --base-paths ${WORKSPACE_ROOT}/src --packages-select ${OCTANE_PKGS} --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
    --all)
        echo "[MODE] Force building everything in src/"
        colcon build --base-paths ${WORKSPACE_ROOT}/src --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
    *)
        # Smart build: build everything except CUDA-heavy packages that cause issues
        echo "[MODE] Smart build (excluding problematic CUDA packages)"
        colcon build --base-paths ${WORKSPACE_ROOT}/src --packages-skip isaac_ros_common isaac_ros_nitros --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
esac

echo ""
echo "[OK] Build complete"
