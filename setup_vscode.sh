#!/bin/bash
# Adds or removes a GNOME autostart entry that opens VS Code in the OCTANE
# directory on login.  Safe to re-run — idempotent in both directions.
#
# Usage (can be run with or without sudo):
#   ./setup_vscode.sh          # uses VSCODE_AUTOSTART value below
#   sudo ./setup_vscode.sh     # same, but resolves home dir from SUDO_USER

# ── Toggle here ───────────────────────────────────────────────────────────────
VSCODE_AUTOSTART=true
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve the real user's home directory whether run with or without sudo
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
else
    USER_HOME="$HOME"
fi

AUTOSTART_DIR="${USER_HOME}/.config/autostart"
DESKTOP_FILE="${AUTOSTART_DIR}/octane-vscode.desktop"

if [ "$VSCODE_AUTOSTART" = true ]; then
    mkdir -p "$AUTOSTART_DIR"
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Type=Application
Name=OCTANE VS Code
Exec=/usr/bin/code ${SCRIPT_DIR}
X-GNOME-Autostart-enabled=true
EOF
    echo "[OK] VS Code autostart enabled — will open ${SCRIPT_DIR} on login"
else
    if [ -f "$DESKTOP_FILE" ]; then
        rm "$DESKTOP_FILE"
        echo "[OK] VS Code autostart disabled — removed ${DESKTOP_FILE}"
    else
        echo "[OK] VS Code autostart already disabled"
    fi
fi
