#!/bin/bash

# Build all Octane system packages.
# Ensures ROS 2, colcon, and required apt deps are installed before building.
# Clones any missing external repos automatically.
#
# Usage:
#   ./build_system.sh           - Smart build (skip CUDA-heavy isaac_ros packages)
#   ./build_system.sh --all     - Force build everything in src/
#   ./build_system.sh --octane  - Only octane_* packages
#   ./build_system.sh --orbbec  - Only orbbec packages

set -e

WORKSPACE_ROOT="/home/csulunabotics/OCTANE/workspace"
EXT_PKGS="${WORKSPACE_ROOT}/src/external_pkgs"

echo "=== Building Octane System ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

# ── 1. ROS 2 ──────────────────────────────────────────────────────────────────
# Pick the right ROS distro for this Ubuntu release
UBUNTU_CODENAME=$(lsb_release -sc 2>/dev/null || echo "jammy")
case "$UBUNTU_CODENAME" in
    noble)   ROS_DISTRO="jazzy" ;;
    jammy)   ROS_DISTRO="humble" ;;
    focal)   ROS_DISTRO="foxy" ;;
    *)       ROS_DISTRO="humble" ;;
esac

ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"

if [ ! -f "$ROS_SETUP" ]; then
    echo "[SETUP] ROS 2 ${ROS_DISTRO} not found — installing..."

    sudo apt update -qq
    sudo apt install -y software-properties-common curl

    # Add ROS 2 apt repository
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

    sudo apt update -qq
    sudo apt install -y ros-${ROS_DISTRO}-desktop
    echo "[OK] ROS 2 ${ROS_DISTRO} installed"
fi

source "$ROS_SETUP"
echo "[OK] Sourced ROS 2 ${ROS_DISTRO}: $ROS_SETUP"

# ── 2. System apt dependencies ─────────────────────────────────────────────────
APT_DEPS=(
    python3-colcon-common-extensions
    python3-rosdep
    python3-vcstool
    libusb-1.0-0-dev
    libuvc-dev
    libgoogle-glog-dev
    nlohmann-json3-dev
    libeigen3-dev
)

MISSING_APT=()
for pkg in "${APT_DEPS[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING_APT+=("$pkg")
    fi
done

if [ ${#MISSING_APT[@]} -gt 0 ]; then
    echo "[SETUP] Installing missing apt packages: ${MISSING_APT[*]}"
    sudo apt update -qq
    sudo apt install -y "${MISSING_APT[@]}"
    echo "[OK] apt dependencies installed"
else
    echo "[OK] apt dependencies already satisfied"
fi

# ── 3. rosdep (optional — network may be unavailable) ─────────────────────────
if ! [ -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    echo "[SETUP] Initialising rosdep..."
    sudo rosdep init 2>/dev/null || echo "[WARN]  rosdep init failed (network?), skipping"
fi
rosdep update --rosdistro "${ROS_DISTRO}" -q 2>/dev/null \
    && echo "[OK] rosdep updated" \
    || echo "[WARN]  rosdep update failed (network?), skipping — apt deps already handled above"

# ── 4. External repos ──────────────────────────────────────────────────────────
echo "[CHECK] Verifying external repos..."

clone_if_missing() {
    local dir="$1" url="$2" branch="${3:-main}"
    if [ ! -d "${EXT_PKGS}/${dir}/.git" ] && [ ! -d "${EXT_PKGS}/${dir}/$(ls ${EXT_PKGS}/${dir} 2>/dev/null | head -1)" ]; then
        echo "[CLONE] ${dir} missing — cloning..."
        mkdir -p "${EXT_PKGS}/${dir}"
        git clone --depth 1 -b "$branch" "$url" "${EXT_PKGS}/${dir}"
    else
        echo "[OK]    ${dir}"
    fi
}

mkdir -p "${EXT_PKGS}"

clone_if_missing ros2_astra_camera https://github.com/orbbec/ros2_astra_camera.git main
clone_if_missing depth-anything-3  https://github.com/ByteDance-Seed/Depth-Anything-3.git main

# Isaac ROS repos via vcs if repos file exists
if [ -f "${WORKSPACE_ROOT}/isaac_ros.repos" ]; then
    NEEDS_VCS_CLONE=false
    for repo in isaac_ros_common isaac_ros_nitros isaac_ros_image_pipeline isaac_ros_nvblox negotiated; do
        if [ ! -d "${EXT_PKGS}/${repo}" ]; then
            NEEDS_VCS_CLONE=true
            break
        fi
    done

    if [ "$NEEDS_VCS_CLONE" = true ]; then
        echo "[CLONE] Isaac ROS repos missing — cloning via vcs..."
        cd "${EXT_PKGS}"
        vcs import < "${WORKSPACE_ROOT}/isaac_ros.repos"
        cd "${WORKSPACE_ROOT}"
    else
        echo "[OK]    Isaac ROS repos"
    fi
fi

# ── 5. Stale cache check ───────────────────────────────────────────────────────
# If build cache references a different workspace path, wipe it to avoid broken builds
if grep -qr "OCTANE_backup\|OCTANE_old" "${WORKSPACE_ROOT}/build" 2>/dev/null; then
    echo "[WARN]  Stale build cache detected (old workspace path) — wiping build/install/log..."
    rm -rf "${WORKSPACE_ROOT}/build" "${WORKSPACE_ROOT}/install" "${WORKSPACE_ROOT}/log"
    echo "[OK]    Cache cleared"
fi

# ── 6. Build ───────────────────────────────────────────────────────────────────
cd "${WORKSPACE_ROOT}"

OCTANE_PKGS="octane_msgs octane_perception octane_mapping octane_supervisor octane_network octane"
ORBBEC_PKGS="astra_camera astra_camera_msgs"
# All external isaac_ros/nvblox packages — only built with --all or --external
EXTERNAL_PKGS_DIR="${WORKSPACE_ROOT}/src/external_pkgs"

COLCON_ARGS="--event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF"

echo ""
echo "[BUILD] Building packages..."

case "$1" in
    --orbbec)
        echo "[MODE] Orbbec packages only"
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${ORBBEC_PKGS} ${COLCON_ARGS}
        ;;
    --octane)
        echo "[MODE] Octane packages only"
        # msgs must come first
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select octane_msgs ${COLCON_ARGS}
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${OCTANE_PKGS} ${COLCON_ARGS}
        ;;
    --external)
        echo "[MODE] External packages only (isaac_ros, nvblox — requires CUDA headers)"
        # isaac_ros_common must be built before everything else that depends on it
        colcon build --base-paths "${EXTERNAL_PKGS_DIR}" \
            --packages-select isaac_ros_common ${COLCON_ARGS}
        colcon build --base-paths "${EXTERNAL_PKGS_DIR}" \
            --packages-skip isaac_ros_common ${COLCON_ARGS}
        ;;
    --all)
        echo "[MODE] Full build (octane + orbbec + external)"
        # Order: isaac_ros_common → rest of external → octane msgs → octane
        colcon build --base-paths "${EXTERNAL_PKGS_DIR}" \
            --packages-select isaac_ros_common ${COLCON_ARGS}
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-skip isaac_ros_common ${COLCON_ARGS}
        ;;
    *)
        # Default: only build OCTANE packages — skip external_pkgs entirely
        echo "[MODE] Smart build (octane packages only, skipping external_pkgs)"
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select octane_msgs ${COLCON_ARGS}
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${OCTANE_PKGS} ${COLCON_ARGS}
        ;;
esac

echo ""
echo "[OK] Build complete"
echo "Source the workspace with: source ${WORKSPACE_ROOT}/install/setup.bash"
