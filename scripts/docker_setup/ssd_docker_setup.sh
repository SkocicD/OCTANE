#!/bin/bash

# Configure or verify Docker data-root location.
# Usage:
#   ./ssd_docker_setup.sh           # interactive: pick a disk, apply config
#   ./ssd_docker_setup.sh --check   # show current Docker data-root and disk info, no changes

DAEMON_JSON="/etc/docker/daemon.json"

check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "ERROR: This script must be run as root (use sudo)."
        exit 1
    fi
}

show_status() {
    echo "=== Current Docker Storage ==="
    local root
    root=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $NF}')
    if [ -z "$root" ]; then
        echo "  Docker is not running or not installed."
    else
        echo "  Docker Root Dir : $root"
        df -h "$root" 2>/dev/null | tail -1 | awk '{printf "  Filesystem      : %s\n  Size            : %s\n  Used            : %s\n  Available       : %s\n", $1, $2, $3, $4}'
    fi
    echo ""
}

list_mounts() {
    echo "=== Available Mount Points ==="
    local i=1
    MOUNT_LIST=()
    while IFS= read -r line; do
        MOUNT_LIST+=("$line")
        local mp size
        mp=$(echo "$line" | awk '{print $1}')
        size=$(echo "$line" | awk '{print $2}')
        avail=$(echo "$line" | awk '{print $4}')
        echo "  [$i] $mp  (size: $size, avail: $avail)"
        i=$((i + 1))
    done < <(df -h --output=target,size,used,avail,pcent | tail -n +2 | grep -v "^/boot\|^/sys\|^/dev\|tmpfs\|udev\|SWAP" | sort)
    echo ""
}

pick_mount() {
    list_mounts
    local count=${#MOUNT_LIST[@]}
    while true; do
        read -rp "Select mount point [1-$count]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$count" ]; then
            SELECTED_MOUNT=$(echo "${MOUNT_LIST[$((choice-1))]}" | awk '{print $1}')
            echo "Selected: $SELECTED_MOUNT"
            return
        fi
        echo "Invalid selection, try again."
    done
}

apply_config() {
    local target_dir="$1"

    echo "=== Applying Docker Config ==="
    echo "  data-root: $target_dir"

    # Create the target directory
    mkdir -p "$target_dir"
    chmod 711 "$target_dir"

    # Stop Docker cleanly (socket too, to prevent auto-restart)
    echo "  Stopping Docker..."
    systemctl stop docker.socket 2>/dev/null || true
    systemctl stop docker

    # Update daemon.json preserving all existing keys
    if [ -f "$DAEMON_JSON" ]; then
        # Replace data-root if present, otherwise inject it
        if grep -q '"data-root"' "$DAEMON_JSON"; then
            python3 -c "
import json, sys
with open('$DAEMON_JSON') as f:
    d = json.load(f)
d['data-root'] = '$target_dir'
with open('$DAEMON_JSON', 'w') as f:
    json.dump(d, f, indent=2)
print('  Updated existing daemon.json')
"
        else
            python3 -c "
import json
with open('$DAEMON_JSON') as f:
    d = json.load(f)
d['data-root'] = '$target_dir'
with open('$DAEMON_JSON', 'w') as f:
    json.dump(d, f, indent=2)
print('  Injected data-root into daemon.json')
"
        fi
    else
        echo '{"data-root": "'"$target_dir"'"}' > "$DAEMON_JSON"
        echo "  Created new daemon.json"
    fi

    # Start Docker
    echo "  Starting Docker..."
    systemctl start docker

    echo ""
    show_status
    echo "Done. Docker is now storing data on: $target_dir"
}

# ── main ──────────────────────────────────────────────────────────────────────

if [ "$1" = "--check" ]; then
    show_status
    exit 0
fi

check_root

echo "=== Docker Storage Setup ==="
echo ""
show_status

pick_mount

DOCKER_DIR="$SELECTED_MOUNT/docker"
echo ""
echo "Docker data-root will be set to: $DOCKER_DIR"
read -rp "Proceed? [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

apply_config "$DOCKER_DIR"
