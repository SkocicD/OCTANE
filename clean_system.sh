#!/bin/bash

# Clean previous build artifacts from the OCTANE workspace.
# Usage:
#   ./clean_system.sh           - Clean only octane packages
#   ./clean_system.sh --all     - Clean everything (build, install, log)
#   ./clean_system.sh --orbbec  - Clean only orbbec packages

WORKSPACE_ROOT="/home/csulunabotics/OCTANE/workspace"

exec "${WORKSPACE_ROOT}/clean_workspace.sh" "$@"
