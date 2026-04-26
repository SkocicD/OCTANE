#!/bin/bash
# Clone Isaac ROS dependencies (nvblox + supporting packages) into workspace/src/external_pkgs/.
# Run this once before the first build that needs nvblox.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src/external_pkgs"
REPOS_FILE="${SCRIPT_DIR}/isaac_ros.repos"

if ! command -v vcs &> /dev/null; then
    echo "[ERROR] vcs tool not found. Install with: pip install vcstool"
    exit 1
fi

mkdir -p "${SRC_DIR}"
cd "${SRC_DIR}"

echo "Cloning Isaac ROS dependencies into ${SRC_DIR}..."
vcs import < "${REPOS_FILE}"

echo ""
echo "[OK] Isaac ROS source cloned. Build with:"
echo "  ./build_system.sh --all"
