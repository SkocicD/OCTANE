#!/bin/bash
# Ephemeral workspace manager for OCTANE Docker containers

set -e

WORKSPACE_BASE="/media/csulunabotics/SSD/OCTANE/workspace"
EPHEMERAL_WORKSPACE="/tmp/octane_workspace_$(date +%s)_$$"

# Create ephemeral workspace
create_ephemeral_workspace() {
    mkdir -p "$EPHEMERAL_WORKSPACE"
    echo "Created ephemeral workspace at: $EPHEMERAL_WORKSPACE"

    # Copy source files to the ephemeral workspace
    if [ -d "$WORKSPACE_BASE/src" ]; then
        cp -r "$WORKSPACE_BASE/src" "$EPHEMERAL_WORKSPACE/"
        echo "Copied source files to ephemeral workspace"
    fi

    echo "$EPHEMERAL_WORKSPACE"
}

# Clean up ephemeral workspace
cleanup_ephemeral_workspace() {
    if [ -n "$EPHEMERAL_WORKSPACE" ] && [ -d "$EPHEMERAL_WORKSPACE" ]; then
        rm -rf "$EPHEMERAL_WORKSPACE"
        echo "Cleaned up ephemeral workspace: $EPHEMERAL_WORKSPACE"
    fi
}

# Show help
show_help() {
    echo "Usage: $0 [command]"
    echo "Commands:"
    echo "  create    Create a new ephemeral workspace"
    echo "  cleanup   Clean up the ephemeral workspace"
    echo "  help      Show this help message"
}

# Set up cleanup trap
trap cleanup_ephemeral_workspace EXIT

# Main
case "${1:-create}" in
    create)
        create_ephemeral_workspace
        ;;
    cleanup)
        cleanup_ephemeral_workspace
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac