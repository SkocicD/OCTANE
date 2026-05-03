#!/bin/bash

# Launch OCTANE system natively on Jetson.
# Sources ROS 2 and the workspace, then launches the requested subsystem.
#
# Usage:
#   ./launch_system.sh                  - Launch all subsystems
#   ./launch_system.sh supervisor       - Launch supervisor only
#   ./launch_system.sh perception       - Launch perception only
#   ./launch_system.sh mapping          - Launch mapping only
#   ./launch_system.sh network          - Launch network only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$SCRIPT_DIR/workspace"
LAUNCH_PKG="octane"

# Load rover config
source "${SCRIPT_DIR}/octane.conf"

# Source ROS 2
ROS_SETUP=""
for distro in humble jazzy; do
    if [ -f "/opt/ros/${distro}/setup.bash" ]; then
        ROS_SETUP="/opt/ros/${distro}/setup.bash"
        break
    fi
done

if [ -z "$ROS_SETUP" ]; then
    echo "[ERROR] ROS 2 not found. Run ./build_system.sh first to install dependencies."
    exit 1
fi

source "$ROS_SETUP"

# Source built workspace
WORKSPACE_SETUP="${WORKSPACE_ROOT}/install/setup.bash"
if [ ! -f "$WORKSPACE_SETUP" ]; then
    echo "[ERROR] Workspace not built. Run ./build_system.sh first."
    exit 1
fi

source "$WORKSPACE_SETUP"
export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/fastdds_no_shm.xml"
export PYTHONUSERBASE="${SCRIPT_DIR}/.cache/python"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
echo "[OK] ROS 2 environment sourced"

# Verify avahi is running so octane.local resolves before nodes start
if ! systemctl is-active --quiet avahi-daemon 2>/dev/null; then
    echo "[WARN] avahi-daemon is not running — octane.local will not resolve."
    echo "       Run: sudo ./setup_mdns.sh"
fi

# Determine which launch file to run
SUBSYSTEM="${1:-all}"

run_launch() {
    local name="$1"
    local file="${WORKSPACE_ROOT}/src/octane/octane/launch/${name}.launch.py"
    if [ ! -f "$file" ]; then
        echo "[ERROR] Launch file not found: ${name}.launch.py"
        exit 1
    fi
    echo "[LAUNCH] Starting ${name}..."
    ros2 launch "$LAUNCH_PKG" "${name}.launch.py"
}

kill_port() {
    local port="$1"
    local pid
    pid=$(fuser "${port}/tcp" 2>/dev/null) || true
    if [ -n "$pid" ]; then
        echo "[CLEANUP] Killing stale process on port ${port} (pid ${pid})..."
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
}

kill_cameras() {
    echo "[CLEANUP] Releasing cameras and killing stale ROS nodes..."

    # Kill any running octane / camera ROS nodes so devices aren't held across launches
    for pattern in astra_camera_node rgb_camera_node camera_frame_splitter \
                   astra_depth_node depth_estimation_node nvblox_node \
                   pc_container point_cloud_xyzrgb; do
        pkill -f "$pattern" 2>/dev/null || true
    done
    sleep 1

    # Force-release any process still holding /dev/video* (UVC colour cameras)
    for dev in /dev/video*; do
        [ -e "$dev" ] || continue
        fuser -k "$dev" 2>/dev/null || true
    done

    # USB reset for the Orbbec — OpenNI2 can leave the device locked after a crash
    ORBBEC_USB=$(lsusb | grep -i "2bc5:0403" | grep -oP 'Bus \K[0-9]+' | head -1)
    ORBBEC_DEV=$(lsusb | grep -i "2bc5:0403" | grep -oP 'Device \K[0-9]+' | head -1)
    if [ -n "$ORBBEC_USB" ] && [ -n "$ORBBEC_DEV" ]; then
        USBDEV=$(printf "/dev/bus/usb/%03d/%03d" "$ORBBEC_USB" "$ORBBEC_DEV")
        if [ -e "$USBDEV" ]; then
            python3 - "$USBDEV" <<'EOF' 2>/dev/null && echo "[CLEANUP] Orbbec USB reset OK" || true
import sys, fcntl
with open(sys.argv[1], 'wb') as f:
    fcntl.ioctl(f, 0x5514, 0)
EOF
        fi
    fi

    sleep 1
    echo "[CLEANUP] Camera cleanup done"
}

case "$SUBSYSTEM" in
    supervisor)  run_launch supervisor ;;
    perception)
        kill_cameras
        run_launch perception
        ;;
    mapping)
        kill_cameras
        run_launch mapping
        ;;
    network)
        kill_port "${OCTANE_TCP_PORT}"
        echo "[LAUNCH] Starting network (TCP :${OCTANE_TCP_PORT}, UDP :${OCTANE_UDP_PORT})..."
        ros2 launch "$LAUNCH_PKG" network.launch.py \
            tcp_port:="${OCTANE_TCP_PORT}"
        ;;
    all)
        kill_cameras
        kill_port "${OCTANE_TCP_PORT}"
        echo "[LAUNCH] Starting all OCTANE subsystems..."
        PIDS=()
        ros2 launch "$LAUNCH_PKG" supervisor.launch.py &
        PIDS+=($!)
        ros2 launch "$LAUNCH_PKG" perception.launch.py &
        PIDS+=($!)
        ros2 launch "$LAUNCH_PKG" mapping.launch.py &
        PIDS+=($!)
        ros2 launch "$LAUNCH_PKG" manual_ctrl.launch.py &
        PIDS+=($!)
        ros2 launch "$LAUNCH_PKG" network.launch.py \
            tcp_port:="${OCTANE_TCP_PORT}" &
        PIDS+=($!)

        trap 'echo ""; echo "[STOP] Shutting down all subsystems..."; kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; exit 0' SIGINT SIGTERM

        echo "[OK] All subsystems running (PIDs: ${PIDS[*]})"
        echo "     Press Ctrl+C to stop all"
        wait "${PIDS[@]}"
        ;;
    *)
        echo "Unknown subsystem: $SUBSYSTEM"
        echo "Usage: $0 [supervisor|perception|mapping|network|all]"
        exit 1
        ;;
esac
