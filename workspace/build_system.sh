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
WORKSPACE_ROOT="${SCRIPT_DIR}"

echo "=== Building Octane System ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

cd "${WORKSPACE_ROOT}"

# Auto-clone Isaac ROS deps (nvblox + supporting packages) if missing
if [ ! -d "${WORKSPACE_ROOT}/src/external_pkgs/isaac_ros_nvblox" ] && [ -f "${WORKSPACE_ROOT}/clone_isaac_ros.sh" ]; then
    echo "[INFO] Isaac ROS sources not found — cloning…"
    "${WORKSPACE_ROOT}/clone_isaac_ros.sh"
    echo ""
fi

OCTANE_PKGS="octane_msgs octane_perception octane_mapping octane_supervisor octane_network octane"
ORBBEC_PKGS="astra_camera astra_camera_msgs"

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
        if [ -d "${WORKSPACE_ROOT}/install/astra_camera" ]; then
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
