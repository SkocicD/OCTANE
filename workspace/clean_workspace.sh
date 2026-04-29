#!/bin/bash

# Clean build artifacts from the OCTANE workspace.
# External packages (isaac_ros, nvblox, orbbec) are NEVER wiped unless you
# explicitly pass --external and confirm.
#
# Usage:
#   ./clean_workspace.sh              - Clean only octane_* packages
#   ./clean_workspace.sh --orbbec     - Clean only orbbec packages
#   ./clean_workspace.sh --all        - Clean octane + orbbec (keeps external)
#   ./clean_workspace.sh --external   - Also wipe external packages (asks confirmation)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"

echo "=== Cleaning Workspace ==="
echo "Workspace: ${WORKSPACE_ROOT}"
echo ""

cd "${WORKSPACE_ROOT}"

OCTANE_PKGS=(octane_msgs octane_perception octane_mapping octane_supervisor octane_network octane)
ORBBEC_PKGS=(astra_camera astra_camera_msgs)

remove_pkg() {
    local name="$1"
    for dir in "build/${name}" "install/${name}"; do
        if [ -d "${dir}" ]; then
            echo "  rm -rf ${dir}"
            rm -rf "${dir}"
        fi
    done
}

clean_octane() {
    echo "[CLEAN] Octane packages"
    for pkg in "${OCTANE_PKGS[@]}"; do remove_pkg "$pkg"; done
}

clean_orbbec() {
    echo "[CLEAN] Orbbec packages"
    for pkg in "${ORBBEC_PKGS[@]}"; do remove_pkg "$pkg"; done
}

clean_external() {
    echo "[CLEAN] External packages (build/ install/ log/ minus octane+orbbec)"

    # Determine which dirs to keep
    local keep=("${OCTANE_PKGS[@]}" "${ORBBEC_PKGS[@]}")

    for top in build install; do
        [ -d "${top}" ] || continue
        for entry in "${top}"/*/; do
            pkg=$(basename "$entry")
            protected=false
            for k in "${keep[@]}"; do
                [ "$pkg" = "$k" ] && { protected=true; break; }
            done
            if ! $protected; then
                echo "  rm -rf ${top}/${pkg}"
                rm -rf "${top}/${pkg}"
            fi
        done
    done

    # log is always safe to wipe
    [ -d log ] && { echo "  rm -rf log/"; rm -rf log; }
}

case "${1:-}" in
    --octane|"")
        echo "[MODE] Octane packages only (external preserved)"
        echo ""
        clean_octane
        ;;
    --orbbec)
        echo "[MODE] Orbbec packages only (external preserved)"
        echo ""
        clean_orbbec
        ;;
    --all)
        echo "[MODE] Octane + orbbec (external preserved)"
        echo ""
        clean_octane
        clean_orbbec
        ;;
    --external)
        echo "[MODE] Full clean including external packages"
        echo ""
        echo "WARNING: This will wipe isaac_ros, nvblox, orbbec, and all other"
        echo "         external packages. They will need to be rebuilt (takes ~30 min)."
        echo ""
        read -rp "Are you sure? [y/N] " confirm
        if [[ "${confirm,,}" != "y" ]]; then
            echo "Aborted."
            exit 0
        fi
        echo ""
        clean_octane
        clean_orbbec
        clean_external
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--octane|--orbbec|--all|--external]"
        exit 1
        ;;
esac

# Always clean log for octane/orbbec modes too (small, safe)
[ -d log ] && rm -rf log

echo ""
echo "[OK] Workspace cleaned"
echo "Run './build_system.sh' to rebuild"
