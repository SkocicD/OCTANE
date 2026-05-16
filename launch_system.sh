#!/bin/bash

# Launch OCTANE system natively on Jetson.
# Sources ROS 2 and the workspace, then launches the requested subsystem.
#
# Usage:
#   ./launch_system.sh                  - Launch all subsystems
#   ./launch_system.sh supervisor       - Launch supervisor only
#   ./launch_system.sh sensors          - Launch sensors only (ADXL345, etc.)
#   ./launch_system.sh perception       - Launch perception only
#   ./launch_system.sh mapping          - Launch mapping only
#   ./launch_system.sh network          - Launch network only
#   ./launch_system.sh logging          - Launch camera recorder only
#
# Flags (combinable with any subsystem or 'all'):
#   --record   Also launch the camera recorder (logging subsystem)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$SCRIPT_DIR/workspace"
LAUNCH_PKG="octane"

# Parse flags
RECORD=false
POSITIONAL_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --record) RECORD=true ;;
        *) POSITIONAL_ARGS+=("$arg") ;;
    esac
done

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
SUBSYSTEM="${POSITIONAL_ARGS[0]:-all}"

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

usb_reset_orbbec() {
    local bus dev usbdev
    bus=$(lsusb | grep -i "2bc5:0403" | grep -oP 'Bus \K[0-9]+' | head -1)
    dev=$(lsusb | grep -i "2bc5:0403" | grep -oP 'Device \K[0-9]+' | head -1)
    if [ -n "$bus" ] && [ -n "$dev" ]; then
        usbdev=$(printf "/dev/bus/usb/%03d/%03d" "$bus" "$dev")
        if [ -e "$usbdev" ]; then
            python3 - "$usbdev" <<'EOF' 2>/dev/null && echo "[CLEANUP] Orbbec USB reset OK" || true
import sys, fcntl
with open(sys.argv[1], 'wb') as f:
    fcntl.ioctl(f, 0x5514, 0)
EOF
        fi
    fi
}

# Find the gs_usb-backed CAN interface (not the onboard mttcan can0/can1)
find_can_usb_iface() {
    for iface in /sys/class/net/can*/; do
        local driver
        driver=$(readlink -f "${iface}device/driver" 2>/dev/null | xargs basename 2>/dev/null)
        if [ "$driver" = "gs_usb" ]; then
            basename "$iface"
            return
        fi
    done
}

release_usb() {
    echo "[CLEANUP] Releasing all USB devices and killing stale ROS nodes..."

    # Kill all octane ROS nodes that hold USB devices
    for pattern in astra_camera_node rgb_camera_node camera_frame_splitter \
                   astra_depth_node depth_estimation_node nvblox_node \
                   pc_container point_cloud_xyzrgb \
                   rs485_drive_node rs485_debug_node \
                   can_drive_node can_debug_node \
                   serial_actuator_node manual_actuator_node \
                   adxl345_node imu_monitor_node; do
        pkill -f "$pattern" 2>/dev/null || true
    done
    sleep 1

    # Force-release the gs_usb CAN transceiver if any process is still holding it
    # (can_drive_node uses libusb directly and may survive pkill)
    for d in /sys/bus/usb/devices/*/; do
        [ "$(cat ${d}idVendor 2>/dev/null)" = "1d50" ] || continue
        [ "$(cat ${d}idProduct 2>/dev/null)" = "606f" ] || continue
        BUS=$(cat ${d}busnum 2>/dev/null); DEV=$(cat ${d}devnum 2>/dev/null)
        DEVPATH=$(printf "/dev/bus/usb/%03d/%03d" "$BUS" "$DEV")
        fuser -k "$DEVPATH" 2>/dev/null || true
    done

    # Release all serial USB devices (RS485, Arduino, ADXL345, etc.)
    for dev in /dev/ttyUSB* /dev/ttyACM* /dev/rs485_drive; do
        [ -e "$dev" ] || continue
        fuser -k "$dev" 2>/dev/null || true
    done

    # Release all UVC video devices (RGB cameras)
    for dev in /dev/video*; do
        [ -e "$dev" ] || continue
        fuser -k "$dev" 2>/dev/null || true
    done

    # USB reset for the Orbbec depth camera (OpenNI2 can leave it locked after a crash)
    usb_reset_orbbec

    # Bring the gs_usb CAN interface down — that's all that's needed.
    # Never USB-reset the CAN transceiver; USBDEVFS_RESET crashes CANable firmware.
    CAN_IFACE=$(find_can_usb_iface)
    if [ -n "$CAN_IFACE" ]; then
        sudo ip link set "$CAN_IFACE" down 2>/dev/null || true
        echo "[CLEANUP] $CAN_IFACE down"
    else
        echo "[WARN] No gs_usb CAN interface found — transceiver may not be plugged in"
    fi

    echo "[CLEANUP] USB release done"
}

bring_up_can() {
    local CAN_IFACE
    CAN_IFACE=$(find_can_usb_iface)
    if [ -n "$CAN_IFACE" ]; then
        sudo ip link set "$CAN_IFACE" type can bitrate 1000000 2>/dev/null || true
        sudo ip link set "$CAN_IFACE" up 2>/dev/null || true
        echo "[OK] $CAN_IFACE up"
    else
        echo "[WARN] No gs_usb CAN interface — transceiver not plugged in"
    fi
}

case "$SUBSYSTEM" in
    supervisor)
        release_usb
        bring_up_can
        run_launch supervisor
        ;;
    sensors)
        release_usb
        bring_up_can
        run_launch sensors
        ;;
    perception)
        release_usb
        bring_up_can
        run_launch perception
        ;;
    mapping)
        release_usb
        bring_up_can
        run_launch mapping
        ;;
    network)
        kill_port "${OCTANE_TCP_PORT}"
        echo "[LAUNCH] Starting network (TCP :${OCTANE_TCP_PORT}, UDP :${OCTANE_UDP_PORT})..."
        ros2 launch "$LAUNCH_PKG" network.launch.py \
            tcp_port:="${OCTANE_TCP_PORT}"
        ;;
    logging)
        run_launch logging
        ;;
    all)
        release_usb
        bring_up_can
        kill_port "${OCTANE_TCP_PORT}"
        echo "[LAUNCH] Starting all OCTANE subsystems..."
        PIDS=()
        ros2 launch "$LAUNCH_PKG" supervisor.launch.py &
        PIDS+=($!)
        ros2 launch "$LAUNCH_PKG" sensors.launch.py &
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
        if [ "$RECORD" = true ]; then
            ros2 launch "$LAUNCH_PKG" logging.launch.py &
            PIDS+=($!)
        fi

        trap 'echo ""; echo "[STOP] Shutting down all subsystems..."; kill "${PIDS[@]}" 2>/dev/null; wait "${PIDS[@]}" 2>/dev/null; exit 0' SIGINT SIGTERM

        echo "[OK] All subsystems running (PIDs: ${PIDS[*]})"
        echo "     Press Ctrl+C to stop all"
        wait "${PIDS[@]}"
        ;;
    *)
        echo "Unknown subsystem: $SUBSYSTEM"
        echo "Usage: $0 [supervisor|sensors|perception|mapping|network|logging|all]"
        exit 1
        ;;
esac
