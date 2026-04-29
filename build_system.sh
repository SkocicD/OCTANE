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

# ── 3. rosdep ─────────────────────────────────────────────────────────────────
if ! [ -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    echo "[SETUP] Initialising rosdep..."
    sudo rosdep init
fi
echo "[SETUP] Updating rosdep..."
rosdep update --rosdistro "${ROS_DISTRO}" -q

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

# ── 5. Build ───────────────────────────────────────────────────────────────────
cd "${WORKSPACE_ROOT}"

OCTANE_PKGS="octane_msgs octane_perception octane_mapping octane_supervisor octane_network octane"
ORBBEC_PKGS="astra_camera astra_camera_msgs"

echo ""
echo "[BUILD] Building packages..."

# Always build message packages first (other packages depend on them)
echo "[STEP] Building message packages..."
colcon build \
    --base-paths "${WORKSPACE_ROOT}/src" \
    --packages-select octane_msgs \
    --event-handlers console_cohesion+ \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

echo "[STEP] Building core packages..."
colcon build \
    --base-paths "${WORKSPACE_ROOT}/src" \
    --packages-select octane_supervisor octane_network octane_perception \
    --event-handlers console_cohesion+ \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

case "$1" in
    --orbbec)
        echo "[MODE] Building orbbec packages only"
        colcon build \
            --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${ORBBEC_PKGS} \
            --event-handlers console_cohesion+ \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
    --octane)
        echo "[MODE] Building octane packages only"
        colcon build \
            --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${OCTANE_PKGS} \
            --event-handlers console_cohesion+ \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
    --all)
        echo "[MODE] Building everything in src/"
        colcon build \
            --base-paths "${WORKSPACE_ROOT}/src" \
            --event-handlers console_cohesion+ \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
    *)
        echo "[MODE] Smart build (skipping CUDA-heavy isaac_ros packages)"
        colcon build \
            --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-skip isaac_ros_common isaac_ros_nitros \
            --event-handlers console_cohesion+ \
            --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
        ;;
esac

echo ""
echo "[OK] Build complete"
echo "Source the workspace with: source ${WORKSPACE_ROOT}/install/setup.bash"
