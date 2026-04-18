#!/bin/bash

# Clean build artifacts from workspace
# Usage:
#   ./clean_workspace.sh           - Clean only octane packages
#   ./clean_workspace.sh --all     - Clean everything including orbbec
#   ./clean_workspace.sh --orbbec  - Clean only orbbec packages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"

echo "=== Cleaning Workspace ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

cd "${WORKSPACE_ROOT}"

# Parse command line arguments
CLEAN_MODE="octane"
if [ "$1" == "--all" ]; then
    CLEAN_MODE="all"
    echo "[MODE] Cleaning all packages"
elif [ "$1" == "--orbbec" ]; then
    CLEAN_MODE="orbbec"
    echo "[MODE] Cleaning orbbec packages only"
else
    echo "[MODE] Cleaning octane packages only"
fi
echo ""

# Clean based on mode
if [ "$CLEAN_MODE" == "all" ]; then
    # Remove everything
    if [ -d "build" ]; then
        echo "Removing build/"
        rm -rf build
    fi

    if [ -d "install" ]; then
        echo "Removing install/"
        rm -rf install
    fi

    if [ -d "log" ]; then
        echo "Removing log/"
        rm -rf log
    fi

    if [ -d "src/build" ]; then
        echo "Removing src/build/"
        rm -rf src/build
    fi

elif [ "$CLEAN_MODE" == "orbbec" ]; then
    # Remove only orbbec packages
    if [ -d "build/orbbec_camera_msgs" ]; then
        echo "Removing build/orbbec_camera_msgs/"
        rm -rf build/orbbec_camera_msgs
    fi

    if [ -d "build/orbbec_camera" ]; then
        echo "Removing build/orbbec_camera/"
        rm -rf build/orbbec_camera
    fi

    if [ -d "install/orbbec_camera_msgs" ]; then
        echo "Removing install/orbbec_camera_msgs/"
        rm -rf install/orbbec_camera_msgs
    fi

    if [ -d "install/orbbec_camera" ]; then
        echo "Removing install/orbbec_camera/"
        rm -rf install/orbbec_camera
    fi

else
    # Remove only octane packages
    if [ -d "build/octane_msgs" ]; then
        echo "Removing build/octane_msgs/"
        rm -rf build/octane_msgs
    fi

    if [ -d "build/octane_perception" ]; then
        echo "Removing build/octane_perception/"
        rm -rf build/octane_perception
    fi

    if [ -d "build/octane_mapping" ]; then
        echo "Removing build/octane_mapping/"
        rm -rf build/octane_mapping
    fi

    if [ -d "build/octane_supervisor" ]; then
        echo "Removing build/octane_supervisor/"
        rm -rf build/octane_supervisor
    fi

    if [ -d "build/octane_wifi" ]; then
        echo "Removing build/octane_wifi/"
        rm -rf build/octane_wifi
    fi

    if [ -d "build/octane" ]; then
        echo "Removing build/octane/"
        rm -rf build/octane
    fi

    if [ -d "install/octane_msgs" ]; then
        echo "Removing install/octane_msgs/"
        rm -rf install/octane_msgs
    fi

    if [ -d "install/octane_perception" ]; then
        echo "Removing install/octane_perception/"
        rm -rf install/octane_perception
    fi

    if [ -d "install/octane_mapping" ]; then
        echo "Removing install/octane_mapping/"
        rm -rf install/octane_mapping
    fi

    if [ -d "install/octane_supervisor" ]; then
        echo "Removing install/octane_supervisor/"
        rm -rf install/octane_supervisor
    fi

    if [ -d "install/octane_wifi" ]; then
        echo "Removing install/octane_wifi/"
        rm -rf install/octane_wifi
    fi

    if [ -d "install/octane" ]; then
        echo "Removing install/octane/"
        rm -rf install/octane
    fi
fi

echo ""
echo "[OK] Workspace cleaned"
echo "Run './build_system.sh' to rebuild"
