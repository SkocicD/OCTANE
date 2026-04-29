#!/bin/bash

# Build OCTANE system packages.
# External packages (isaac_ros, nvblox, orbbec) are built automatically on
# first use and skipped on subsequent runs — no mode wipes or rebuilds them
# unless you explicitly pass --external.
#
# Usage:
#   ./build_system.sh              - Build octane (+ auto-build external if missing)
#   ./build_system.sh --octane     - Only octane_* packages (+ auto-build external if missing)
#   ./build_system.sh --orbbec     - Only orbbec packages (+ auto-build external if missing)
#   ./build_system.sh --all        - Rebuild octane + orbbec (+ auto-build external if missing)
#   ./build_system.sh --external   - Force rebuild external packages only

set -e

WORKSPACE_ROOT="/home/csulunabotics/OCTANE/workspace"
EXT_PKGS="${WORKSPACE_ROOT}/src/external_pkgs"

echo "=== Building Octane System ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

# ── Helpers ────────────────────────────────────────────────────────────────────
ext_installed() {
    [ -d "${WORKSPACE_ROOT}/install/$1" ]
}

# ── 1. ROS 2 ──────────────────────────────────────────────────────────────────
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
echo "[OK] Sourced ROS 2 ${ROS_DISTRO}"

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
    dpkg -s "$pkg" &>/dev/null || MISSING_APT+=("$pkg")
done

if [ ${#MISSING_APT[@]} -gt 0 ]; then
    echo "[SETUP] Installing missing apt packages: ${MISSING_APT[*]}"
    sudo apt update -qq && sudo apt install -y "${MISSING_APT[@]}"
else
    echo "[OK] apt dependencies satisfied"
fi

# ── 3. rosdep (optional) ──────────────────────────────────────────────────────
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || \
    sudo rosdep init 2>/dev/null || true
rosdep update --rosdistro "${ROS_DISTRO}" -q 2>/dev/null && \
    echo "[OK] rosdep updated" || \
    echo "[WARN] rosdep update failed (network?) — skipping"

# ── 4. PyTorch for Jetson ─────────────────────────────────────────────────────
if ! python3 -c "import torch" 2>/dev/null; then
    echo "[SETUP] Installing PyTorch for Jetson (JetPack 6.2 / L4T R36)..."
    pip3 install --no-cache \
        https://developer.download.nvidia.com/compute/redist/jp/v62/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
    echo "[OK] PyTorch installed"
else
    echo "[OK] PyTorch already installed"
fi

# ── 5. Depth Anything 3 Python package ────────────────────────────────────────
if ! python3 -c "from depth_anything_3.api import DepthAnything3" 2>/dev/null; then
    echo "[SETUP] Installing depth_anything_3..."
    pip3 install -e "${EXT_PKGS}/depth-anything-3" --quiet
    echo "[OK] depth_anything_3 installed"
else
    echo "[OK] depth_anything_3 already installed"
fi

# ── 6. External repos ──────────────────────────────────────────────────────────
echo "[CHECK] Verifying external repos..."

clone_if_missing() {
    local dir="$1" url="$2" branch="${3:-main}"
    if [ ! -d "${EXT_PKGS}/${dir}/.git" ]; then
        echo "[CLONE] ${dir}..."
        git clone --depth 1 -b "$branch" "$url" "${EXT_PKGS}/${dir}"
    else
        echo "[OK]    ${dir}"
    fi
}

mkdir -p "${EXT_PKGS}"
clone_if_missing ros2_astra_camera https://github.com/orbbec/ros2_astra_camera.git main
clone_if_missing depth-anything-3  https://github.com/ByteDance-Seed/Depth-Anything-3.git main

if [ -f "${WORKSPACE_ROOT}/isaac_ros.repos" ]; then
    NEEDS_VCS=false
    for repo in isaac_ros_common isaac_ros_nitros isaac_ros_image_pipeline isaac_ros_nvblox negotiated; do
        [ -d "${EXT_PKGS}/${repo}" ] || { NEEDS_VCS=true; break; }
    done
    if [ "$NEEDS_VCS" = true ]; then
        echo "[CLONE] Isaac ROS repos via vcs..."
        cd "${EXT_PKGS}" && vcs import < "${WORKSPACE_ROOT}/isaac_ros.repos" && cd "${WORKSPACE_ROOT}"
    else
        echo "[OK]    Isaac ROS repos"
    fi
fi

# Init nvblox_core submodule if missing
NVBLOX_CORE="${EXT_PKGS}/isaac_ros_nvblox/nvblox_ros/nvblox_core/CMakeLists.txt"
if [ ! -f "$NVBLOX_CORE" ] && [ -d "${EXT_PKGS}/isaac_ros_nvblox/nvblox_ros" ]; then
    echo "[SETUP] Initialising nvblox_core submodule..."
    cd "${EXT_PKGS}/isaac_ros_nvblox/nvblox_ros"
    git submodule update --init --recursive
    cd "${WORKSPACE_ROOT}"
fi

# ── 7. Stale cache check ───────────────────────────────────────────────────────
if grep -qr "OCTANE_backup\|OCTANE_old" "${WORKSPACE_ROOT}/build" 2>/dev/null; then
    echo "[WARN] Stale build cache — wiping octane build artifacts..."
    for pkg in octane octane_msgs octane_perception octane_mapping octane_supervisor octane_network; do
        rm -rf "${WORKSPACE_ROOT}/build/${pkg}" "${WORKSPACE_ROOT}/install/${pkg}"
    done
fi

# ── 8. Build ───────────────────────────────────────────────────────────────────
cd "${WORKSPACE_ROOT}"

OCTANE_PKGS="octane_msgs octane_perception octane_mapping octane_supervisor octane_network octane"
ORBBEC_PKGS="astra_camera astra_camera_msgs"

COLCON_ARGS=(--event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF)
EXT_COLCON_ARGS=(--event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DUSE_NATIVE_CUDA_ARCHITECTURE=1)

# ── External packages: build once, skip if already installed ──────────────────
build_external() {
    echo "[EXTERNAL] Building isaac_ros_common..."
    colcon build --base-paths "${EXT_PKGS}" \
        --packages-select isaac_ros_common "${EXT_COLCON_ARGS[@]}"
    source "${WORKSPACE_ROOT}/install/setup.bash" 2>/dev/null || true

    echo "[EXTERNAL] Building remaining external packages..."
    colcon build --base-paths "${EXT_PKGS}" \
        --packages-skip isaac_ros_common "${EXT_COLCON_ARGS[@]}"
    source "${WORKSPACE_ROOT}/install/setup.bash" 2>/dev/null || true
    echo "[OK] External packages built"
}

check_and_build_external() {
    local missing=false
    for pkg in isaac_ros_common nvblox_ros nvblox_msgs astra_camera; do
        ext_installed "$pkg" || { missing=true; break; }
    done
    if [ "$missing" = true ]; then
        echo "[EXTERNAL] Some external packages not yet built — building now (this takes a while)..."
        build_external
    else
        echo "[OK] External packages already built — skipping"
    fi
}

echo ""
echo "[BUILD] Building packages..."

case "$1" in
    --octane)
        echo "[MODE] Octane packages only"
        check_and_build_external
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select octane_msgs "${COLCON_ARGS[@]}"
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${OCTANE_PKGS} "${COLCON_ARGS[@]}"
        ;;
    --orbbec)
        echo "[MODE] Orbbec packages only"
        check_and_build_external
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${ORBBEC_PKGS} "${COLCON_ARGS[@]}"
        ;;
    --external)
        echo "[MODE] Force rebuild external packages"
        build_external
        ;;
    --all)
        echo "[MODE] Rebuild octane + orbbec (external auto-built if missing)"
        check_and_build_external
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${ORBBEC_PKGS} "${COLCON_ARGS[@]}"
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select octane_msgs "${COLCON_ARGS[@]}"
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${OCTANE_PKGS} "${COLCON_ARGS[@]}"
        ;;
    *)
        echo "[MODE] Default build (octane + auto-build external if missing)"
        check_and_build_external
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select octane_msgs "${COLCON_ARGS[@]}"
        colcon build --base-paths "${WORKSPACE_ROOT}/src" \
            --packages-select ${OCTANE_PKGS} "${COLCON_ARGS[@]}"
        ;;
esac

echo ""
echo "[OK] Build complete"
echo "Source the workspace: source ${WORKSPACE_ROOT}/install/setup.bash"
