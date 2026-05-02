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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_ROOT="${SCRIPT_DIR}/workspace"
EXT_PKGS="${WORKSPACE_ROOT}/src/external_pkgs"

# Ensure CUDA tools are on PATH so CMake's find_package(CUDAToolkit) works
CUDA_HOME="/usr/local/cuda"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# All caches and temp files go onto SSD2 alongside the workspace
SSD_CACHE="${SCRIPT_DIR}/.cache"

SSD_PYTHON="${SSD_CACHE}/python"
mkdir -p "${SSD_PYTHON}"
export PYTHONUSERBASE="${SSD_PYTHON}"
export PATH="${SSD_PYTHON}/bin:${PATH}"

export TMPDIR="${SSD_CACHE}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${TMPDIR}"

export _JAVA_OPTIONS="-Djava.io.tmpdir=${TMPDIR}"

export ROS_LOG_DIR="${SSD_CACHE}/ros/log"
mkdir -p "${ROS_LOG_DIR}"

export PIP_CACHE_DIR="${SSD_CACHE}/pip"
mkdir -p "${PIP_CACHE_DIR}"

export ISAAC_ROS_WS="${WORKSPACE_ROOT}/isaac_ros_assets"
ISAAC_ROS_ASSETS_DIR="${SSD_CACHE}/isaac_ros_assets"
mkdir -p "${ISAAC_ROS_ASSETS_DIR}/isaac_ros_nvblox"

export XDG_CACHE_HOME="${SSD_CACHE}/xdg"
mkdir -p "${XDG_CACHE_HOME}"

export CUDA_CACHE_PATH="${SSD_CACHE}/cuda"
mkdir -p "${CUDA_CACHE_PATH}"

export npm_config_cache="${SSD_CACHE}/npm"
mkdir -p "${npm_config_cache}"

# ── Helpers ────────────────────────────────────────────────────────────────────
ext_installed() {
    [ -d "${WORKSPACE_ROOT}/install/$1" ]
}

# Remove dist-info dirs that are missing METADATA — left behind by interrupted pip installs.
clean_corrupted_pip() {
    local site_pkgs="${SSD_PYTHON}/lib/python3.10/site-packages"
    [ -d "$site_pkgs" ] || return 0
    local removed=0
    for dist_info in "${site_pkgs}"/*.dist-info; do
        [ -d "$dist_info" ] || continue
        if [ ! -f "${dist_info}/METADATA" ]; then
            rm -rf "$dist_info"
            removed=$((removed + 1))
        fi
    done
    if [ "$removed" -gt 0 ]; then
        echo "[CLEANUP] Removed ${removed} corrupted pip dist-info dir(s)"
    fi
}
clean_corrupted_pip

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
    git-lfs
    libusb-1.0-0-dev
    libuvc-dev
    libgoogle-glog-dev
    nlohmann-json3-dev
    libeigen3-dev
    cuda-nvtx-12-6
    ros-${ROS_DISTRO}-camera-info-manager
    ros-${ROS_DISTRO}-image-transport
    ros-${ROS_DISTRO}-image-transport-plugins
    ros-${ROS_DISTRO}-image-publisher
    ros-${ROS_DISTRO}-diagnostic-updater
    ros-${ROS_DISTRO}-cv-bridge
)

MISSING_APT=()
for pkg in "${APT_DEPS[@]}"; do
    dpkg -s "$pkg" &>/dev/null || MISSING_APT+=("$pkg")
done

if [ ${#MISSING_APT[@]} -gt 0 ]; then
    echo "[SETUP] Installing missing apt packages: ${MISSING_APT[*]}"
    sudo apt update -qq && sudo apt install -y "${MISSING_APT[@]}"
    sudo apt-get clean  # free downloaded package cache from internal storage
else
    echo "[OK] apt dependencies satisfied"
fi

# ── 3. rosdep (optional) ──────────────────────────────────────────────────────
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || \
    sudo rosdep init 2>/dev/null || true
rosdep update --rosdistro "${ROS_DISTRO}" -q 2>/dev/null && \
    echo "[OK] rosdep updated" || \
    echo "[WARN] rosdep update failed (network?) — skipping"

# ── 4. External repos (clone before pip installs that depend on them) ─────────
echo "[CHECK] Verifying external repos..."

clone_if_missing() {
    local dir="$1" url="$2" branch="${3:-}"
    if [ ! -d "${EXT_PKGS}/${dir}/.git" ]; then
        echo "[CLONE] ${dir}..."
        if [ -n "$branch" ]; then
            git clone --depth 1 -b "$branch" "$url" "${EXT_PKGS}/${dir}"
        else
            git clone --depth 1 "$url" "${EXT_PKGS}/${dir}"
        fi
    else
        echo "[OK]    ${dir}"
    fi
}

mkdir -p "${EXT_PKGS}"
clone_if_missing ros2_astra_camera https://github.com/orbbec/ros2_astra_camera.git
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

# Pull Git LFS objects for all external repos (pre-built .so files are stored in LFS)
git lfs install --skip-repo 2>/dev/null || true
for repo in isaac_ros_common isaac_ros_nitros isaac_ros_image_pipeline isaac_ros_nvblox; do
    REPO_PATH="${EXT_PKGS}/${repo}"
    if [ -d "${REPO_PATH}/.git" ]; then
        LFS_STUBS=$(find "${REPO_PATH}" -name "*.so" -exec file {} \; 2>/dev/null | grep -c "ASCII text" || true)
        if [ "${LFS_STUBS}" -gt 0 ]; then
            echo "[LFS] Pulling ${LFS_STUBS} binary objects for ${repo}..."
            git -C "${REPO_PATH}" lfs pull
        else
            echo "[OK]  LFS objects already present: ${repo}"
        fi
    fi
done

# Patch VPI 2.x API → VPI 4.x: pBase renamed to data, offsetBytes removed
for f in \
    "${EXT_PKGS}/isaac_ros_nitros/isaac_ros_nitros_type/isaac_ros_nitros_image_type/src/nitros_image.cpp" \
    "${EXT_PKGS}/isaac_ros_image_pipeline/isaac_ros_gxf_extensions/gxf_isaac_sgm/gxf/gems/vpi/image_wrapper.cpp" \
    "${EXT_PKGS}/isaac_ros_image_pipeline/isaac_ros_gxf_extensions/gxf_isaac_tensorops/gxf/extensions/tensorops/core/VPITensorOperators.cpp" \
    "${EXT_PKGS}/isaac_ros_image_pipeline/isaac_ros_gxf_extensions/gxf_isaac_tensorops/gxf/extensions/tensorops/core/VPITensorOperators.h" \
    "${EXT_PKGS}/isaac_ros_image_pipeline/isaac_ros_gxf_extensions/gxf_isaac_image_flip/gxf/image_flip.cpp"; do
    if [ -f "$f" ] && grep -q "\.pBase" "$f"; then
        sed -i 's/\.pBase/.data/g' "$f"
        sed -i '/\.offsetBytes\s*=/d' "$f"
        echo "[PATCH] VPI4 API: $(basename $f)"
    fi
done

# Patch isaac_ros_nitros for rclcpp API change (add_to_wait_set takes pointer, not reference)
NITROS_PUB="${EXT_PKGS}/isaac_ros_nitros/isaac_ros_nitros/src/nitros_publisher.cpp"
if [ -f "${NITROS_PUB}" ] && grep -q "add_to_wait_set(\*wait_set)" "${NITROS_PUB}"; then
    sed -i 's/guard_condition_.add_to_wait_set(\*wait_set)/guard_condition_.add_to_wait_set(wait_set)/' "${NITROS_PUB}"
    echo "[PATCH] isaac_ros_nitros: fixed add_to_wait_set dereference for humble rclcpp"
fi

# Init nvblox_core submodule if missing
NVBLOX_CORE="${EXT_PKGS}/isaac_ros_nvblox/nvblox_ros/nvblox_core/CMakeLists.txt"
if [ ! -f "$NVBLOX_CORE" ] && [ -d "${EXT_PKGS}/isaac_ros_nvblox/nvblox_ros" ]; then
    echo "[SETUP] Initialising nvblox_core submodule..."
    cd "${EXT_PKGS}/isaac_ros_nvblox/nvblox_ros"
    git submodule update --init --recursive
    cd "${WORKSPACE_ROOT}"
fi

# ── 5. PyTorch for Jetson ─────────────────────────────────────────────────────
# Wheels are Jetson-specific (JetPack 6.x / L4T R36, cp310, aarch64).
# If the CDN wheel can't be found, the build stops — install manually and re-run.
TORCH_WHEEL_CACHE="${SSD_CACHE}/pip/torch-jetson"
mkdir -p "${TORCH_WHEEL_CACHE}"

# Check against our own site-packages directly — pip3 show can miss PYTHONUSERBASE installs
TORCH_DIST_INFO=$(find "${SSD_PYTHON}/lib" -maxdepth 4 -name "torch-*.dist-info" -type d 2>/dev/null | head -1)
if [ -z "$TORCH_DIST_INFO" ] || [ ! -f "${TORCH_DIST_INFO}/METADATA" ]; then
    echo "[SETUP] Installing PyTorch for Jetson (JetPack 6.2 / L4T R36)..."
    # Check if we have a cached wheel already
    CACHED_WHEEL=$(find "${TORCH_WHEEL_CACHE}" -name "torch-*cp310*aarch64*.whl" | sort -V | tail -1)
    if [ -n "$CACHED_WHEEL" ]; then
        echo "[CACHE] Using cached wheel: $(basename "$CACHED_WHEEL")"
        pip3 install "$CACHED_WHEEL"
    else
        # JetPack 6.2 CDN path is empty — wheels ship under v61 (L4T R36, same ABI)
        for JP_VER in v62 v61 v60; do
            TORCH_BASE="https://developer.download.nvidia.com/compute/redist/jp/${JP_VER}/pytorch"
            TORCH_WHEEL=$(curl -s "${TORCH_BASE}/" 2>/dev/null \
                | grep -o 'torch[^"<> ]*cp310[^"<> ]*aarch64\.whl' | sort -V | tail -1)
            [ -n "$TORCH_WHEEL" ] && break
        done
        if [ -z "$TORCH_WHEEL" ]; then
            echo "[ERROR] PyTorch wheel not discoverable from NVIDIA CDN."
            echo "        Install manually then re-run: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
            exit 1
        fi
        echo "[DOWNLOAD] Fetching ${TORCH_WHEEL}..."
        curl -L "${TORCH_BASE}/${TORCH_WHEEL}" -o "${TORCH_WHEEL_CACHE}/${TORCH_WHEEL}"
        pip3 install "${TORCH_WHEEL_CACHE}/${TORCH_WHEEL}"
    fi
    echo "[OK] PyTorch installed"
else
    echo "[OK] PyTorch already installed"
fi

# ── 5b. libcusparseLt ─────────────────────────────────────────────────────────
# JetPack's CUDA 12.x ships libcusparse but not libcusparseLt — PyTorch links
# against it. Install the real library from the CUDA apt repo.
if ! dpkg -s libcusparselt0 &>/dev/null 2>&1; then
    echo "[SETUP] Installing libcusparseLt for PyTorch CUDA support..."
    # Add CUDA apt keyring + repo for aarch64 if not already present
    if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2204-arm64.list ]; then
        CUDA_KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
        CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/${CUDA_KEYRING_DEB}"
        wget -q "$CUDA_KEYRING_URL" -O "/tmp/${CUDA_KEYRING_DEB}"
        sudo dpkg -i "/tmp/${CUDA_KEYRING_DEB}"
        sudo apt-get update -qq
    fi
    sudo apt-get install -y libcusparselt0 libcusparselt-dev
    echo "[OK] libcusparseLt installed"
else
    echo "[OK] libcusparseLt already installed"
fi

# ── 5c. torchvision for Jetson ────────────────────────────────────────────────
# No pre-built aarch64 wheel is ABI-compatible with the JetPack torch — must build
# from source against it. Wheel is cached so subsequent runs skip the 30-min build.
TV_WHEEL_CACHE="${SSD_CACHE}/pip/torchvision-jetson"
mkdir -p "${TV_WHEEL_CACHE}"
if ! python3 -c "import torchvision; import torch; torchvision.ops.nms" &>/dev/null 2>&1; then
    CACHED_TV=$(find "${TV_WHEEL_CACHE}" -name "torchvision-*.whl" | sort -V | tail -1)
    if [ -n "$CACHED_TV" ]; then
        echo "[CACHE] Using cached torchvision wheel: $(basename "$CACHED_TV")"
        pip3 install "$CACHED_TV" --no-deps
    else
        echo "[SETUP] Building torchvision 0.20.0 from source (SM87, ~30 min)..."
        TV_BUILD="/tmp/torchvision_build"
        rm -rf "$TV_BUILD"
        git clone --depth 1 --branch v0.20.0 https://github.com/pytorch/vision "$TV_BUILD"
        cd "$TV_BUILD"
        FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.7" python3 setup.py bdist_wheel
        TV_WHL=$(find "$TV_BUILD/dist" -name "torchvision-*.whl" | head -1)
        cp "$TV_WHL" "${TV_WHEEL_CACHE}/"
        pip3 install "$TV_WHL" --no-deps
        cd "${WORKSPACE_ROOT}"
        echo "[OK] torchvision built and installed"
    fi
else
    echo "[OK] torchvision already installed"
fi

# ── 6. Depth Anything 3 Python package ────────────────────────────────────────
# pycolmap/trimesh/gsplat have no aarch64 wheel — export/__init__.py guards their
# imports so inference works without them (only SfM/3DGS export paths are affected).
if ! pip3 show depth-anything-3 &>/dev/null; then
    echo "[SETUP] Installing depth_anything_3..."
    pip3 install -e "${EXT_PKGS}/depth-anything-3" --no-deps
    # Install inference-only deps — timm/torchvision installed with --no-deps to
    # prevent pulling in standard PyPI torch over the JetPack-specific wheel.
    pip3 install "numpy<2" pillow imageio safetensors einops omegaconf opencv-python-headless huggingface-hub addict "moviepy==1.0.3" evo e3nn pypose numba pandas prettytable
    pip3 install timm --no-deps
    echo "[OK] depth_anything_3 installed"
else
    # Ensure deps added after initial install are present
    pip3 install addict "moviepy==1.0.3" evo e3nn pypose numba pandas prettytable --quiet
    echo "[OK] depth_anything_3 already installed"
fi

# ── 7. gs-usb (CAN adapter Python library) ────────────────────────────────────
if ! python3 -c "import gs_usb" &>/dev/null; then
    echo "[SETUP] Installing gs-usb..."
    pip3 install gs-usb
    echo "[OK] gs-usb installed"
else
    echo "[OK] gs-usb already installed"
fi

# ── 8. rosdep install for external packages ───────────────────────────────────
echo "[SETUP] Installing ROS deps for external packages via rosdep..."
# Skip JetPack-native packages that rosdep can't resolve — they ship with JetPack
ROSDEP_SKIP_KEYS=(
    libnvvpi4 vpi4-dev cvcuda0-dev libucx0 tensorrt
    libnvvpi3 libnvvpi2
    posix_ipc nlohmann_json
    python3-onnxscript-pip-shim
    isaac_ros_peoplenet_models_install
    ament_python
)
rosdep install --from-paths "${EXT_PKGS}" --ignore-src -r -y \
    --rosdistro "${ROS_DISTRO}" \
    --skip-keys="${ROSDEP_SKIP_KEYS[*]}" 2>&1 | grep -v "^#" || true
echo "[OK] rosdep install done"

# ── 9. Stale cache check ───────────────────────────────────────────────────────
if grep -qr "OCTANE_backup\|OCTANE_old" "${WORKSPACE_ROOT}/build" 2>/dev/null; then
    echo "[WARN] Stale build cache — wiping octane build artifacts..."
    for pkg in octane octane_msgs octane_perception octane_mapping octane_supervisor octane_network; do
        rm -rf "${WORKSPACE_ROOT}/build/${pkg}" "${WORKSPACE_ROOT}/install/${pkg}"
    done
fi

# ── 10. Build ──────────────────────────────────────────────────────────────────
cd "${WORKSPACE_ROOT}"

OCTANE_PKGS="octane_msgs octane_perception octane_mapping octane_supervisor octane_network octane_manual_ctrl octane_serial octane"
ORBBEC_PKGS="astra_camera astra_camera_msgs"

COLCON_ARGS=(--event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF)
EXT_COLCON_ARGS=(--event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DUSE_NVTX=OFF -DCMAKE_CUDA_ARCHITECTURES=87)

# ── External packages: build once, skip if already installed ──────────────────
build_external() {
    echo "[EXTERNAL] Building isaac_ros_common..."
    colcon build --base-paths "${EXT_PKGS}" \
        --packages-select isaac_ros_common "${EXT_COLCON_ARGS[@]}"
    source "${WORKSPACE_ROOT}/install/setup.bash" 2>/dev/null || true

    echo "[EXTERNAL] Building remaining external packages..."
    colcon build --base-paths "${EXT_PKGS}" \
        --packages-skip isaac_ros_common \
        --packages-skip nvblox_image_padding nvblox_examples_bringup \
            multi_realsense_emitter_synchronizer realsense_splitter semantic_label_conversion \
            gxf_isaac_sgm gxf_isaac_image_flip gxf_isaac_tensorops gxf_isaac_camera_utils \
            isaac_ros_vpi_utils isaac_ros_stereo_image_proc isaac_ros_depth_image_proc isaac_ros_image_proc \
            custom_nitros_dnn_image_encoder isaac_ros_pynitros \
            isaac_ros_image_pipeline \
            custom_nitros_string custom_nitros_message_filter \
            isaac_ros_nitros_topic_tools isaac_ros_nitros_bridge_ros2 \
        "${EXT_COLCON_ARGS[@]}"
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

# Patch cv_bridge includes to use .h instead of .hpp
for f in \
    "${EXT_PKGS}/isaac_ros_nvblox/nvblox_ros/include/nvblox_ros/conversions/image_conversions.hpp" \
    "${EXT_PKGS}/isaac_ros_nvblox/nvblox_examples/nvblox_image_padding/include/nvblox_image_padding/image_padding_cropping_node.hpp"; do
    if [ -f "$f" ] && grep -q "cv_bridge.hpp" "$f"; then
        sed -i 's|#include <cv_bridge/cv_bridge.hpp>|#include <cv_bridge/cv_bridge.h>|g' "$f"
        echo "[PATCH] cv_bridge: $(basename $f)"
    fi
done
