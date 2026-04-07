#!/bin/bash

# Build all Octane system packages
# Usage:
#   ./build_system.sh           - Smart build (skip orbbec if already built)
#   ./build_system.sh --all     - Force build all packages
#   ./build_system.sh --orbbec  - Only build orbbec packages
#   ./build_system.sh --octane  - Only build octane packages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"

echo "=== Building Octane System ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

cd "${WORKSPACE_ROOT}"

# Parse command line arguments
BUILD_MODE="smart"
if [ "$1" == "--all" ]; then
    BUILD_MODE="all"
    echo "[MODE] Force building all packages"
elif [ "$1" == "--orbbec" ]; then
    BUILD_MODE="orbbec"
    echo "[MODE] Building orbbec packages only"
elif [ "$1" == "--octane" ]; then
    BUILD_MODE="octane"
    echo "[MODE] Building octane packages only"
else
    echo "[MODE] Smart build (skip orbbec if already built)"
fi
echo ""

# Build based on mode
if [ "$BUILD_MODE" == "orbbec" ]; then
    # Only build orbbec packages
    echo "Building orbbec packages..."
    MAKEFLAGS="-j2" colcon build \
        --base-paths "${WORKSPACE_ROOT}/src" \
        --packages-select \
        orbbec_camera_msgs orbbec_camera \
        --parallel-workers 1 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        --event-handlers console_cohesion+

elif [ "$BUILD_MODE" == "octane" ]; then
    # Only build octane packages
    echo "Building octane packages..."
    MAKEFLAGS="-j2" colcon build \
        --base-paths "${WORKSPACE_ROOT}/src" \
        --packages-select \
        octane_msgs octane_perception octane_mapping octane \
        --parallel-workers 1 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        --event-handlers console_cohesion+

elif [ "$BUILD_MODE" == "all" ]; then
    # Force build everything
    echo "Building all packages..."
    MAKEFLAGS="-j2" colcon build \
        --base-paths "${WORKSPACE_ROOT}/src" \
        --packages-select \
        orbbec_camera_msgs orbbec_camera octane_msgs octane_perception octane_mapping octane \
        --parallel-workers 1 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        --event-handlers console_cohesion+

else
    # Smart build - check if orbbec is already built
    ORBBEC_BUILT=false
    if [ -d "${WORKSPACE_ROOT}/install/orbbec_camera_msgs" ] && [ -d "${WORKSPACE_ROOT}/install/orbbec_camera" ]; then
        echo "[INFO] Orbbec packages already built, skipping..."
        ORBBEC_BUILT=true
        echo ""
    fi

    if [ "$ORBBEC_BUILT" = true ]; then
        # Skip orbbec, only build our packages (fast)
        echo "Building octane packages only..."
        MAKEFLAGS="-j2" colcon build \
            --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select \
            octane_msgs octane_perception octane_mapping octane \
            --parallel-workers 1 \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
            --event-handlers console_cohesion+
    else
        # Build everything including orbbec (slow)
        echo "Building all packages including orbbec..."
        MAKEFLAGS="-j2" colcon build \
            --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select \
            orbbec_camera_msgs orbbec_camera octane_msgs octane_perception octane_mapping octane \
            --parallel-workers 1 \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
            --event-handlers console_cohesion+
    fi
fi

echo ""
echo "[OK] Build complete"
