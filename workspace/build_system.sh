#!/bin/bash

# Build all Octane system packages.
# Auto-discovers any packages in src/ — if Isaac ROS / nvblox sources are
# cloned (via clone_isaac_ros.sh), they get built automatically.
#
# Usage:
#   ./build_system.sh           - Smart build (skip orbbec if already built)
#   ./build_system.sh --all     - Force build everything in src/
#   ./build_system.sh --octane  - Only octane_* packages
#   ./build_system.sh --orbbec  - Only orbbec packages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"

echo "=== Building Octane System ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

cd "${WORKSPACE_ROOT}"

# Auto-clone Isaac ROS deps (nvblox + supporting packages) if missing
if [ ! -d "${WORKSPACE_ROOT}/src/isaac_ros_nvblox" ] && [ -f "${WORKSPACE_ROOT}/clone_isaac_ros.sh" ]; then
    echo "[INFO] Isaac ROS sources not found — cloning…"
    "${WORKSPACE_ROOT}/clone_isaac_ros.sh"
    echo ""
fi

# Install Python dependencies
echo "[DEPS] Installing Python dependencies..."
pip install Jetson.GPIO --quiet 2>/dev/null && echo "[DEPS] Jetson.GPIO installed" || echo "[DEPS] Jetson.GPIO unavailable (not a Jetson — skipping)"
echo ""

OCTANE_PKGS="octane_msgs octane_perception octane_mapping octane"
ORBBEC_PKGS="orbbec_camera_msgs orbbec_camera"

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF"
COLCON_FLAGS="--base-paths ${WORKSPACE_ROOT}/src --parallel-workers 1 --event-handlers console_cohesion+"

case "$1" in
    --orbbec)
        echo "[MODE] Building orbbec packages only"
        MAKEFLAGS="-j2" colcon build ${COLCON_FLAGS} --packages-select ${ORBBEC_PKGS} --cmake-args ${CMAKE_ARGS}
        ;;
    --octane)
        echo "[MODE] Building octane packages only"
        MAKEFLAGS="-j2" colcon build ${COLCON_FLAGS} --packages-select ${OCTANE_PKGS} --cmake-args ${CMAKE_ARGS}
        ;;
    --all)
        echo "[MODE] Force building everything in src/"
        MAKEFLAGS="-j2" colcon build ${COLCON_FLAGS} --cmake-args ${CMAKE_ARGS}
        ;;
    *)
        # Smart build: skip orbbec if already built, build everything else
        if [ -d "${WORKSPACE_ROOT}/install/orbbec_camera" ] && [ -d "${WORKSPACE_ROOT}/install/orbbec_camera_msgs" ]; then
            echo "[MODE] Smart build (orbbec cached, building everything else)"
            MAKEFLAGS="-j2" colcon build ${COLCON_FLAGS} --packages-skip ${ORBBEC_PKGS} --cmake-args ${CMAKE_ARGS}
        else
            echo "[MODE] Smart build (first run, building everything)"
            MAKEFLAGS="-j2" colcon build ${COLCON_FLAGS} --cmake-args ${CMAKE_ARGS}
        fi
        ;;
esac

echo ""
echo "[OK] Build complete"
