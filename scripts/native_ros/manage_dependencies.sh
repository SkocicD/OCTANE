#!/bin/bash

# Dependency management script for OCTANE ROS system
# Usage: ./manage_dependencies.sh [--install] [--remove]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default dependencies to install
ROS_PACKAGES="python3-colcon-common-extensions ros-humble-desktop"

show_help() {
    echo "Usage: $0 [--install] [--remove]"
    echo "  --install    Install all dependencies"
    echo "  --remove     Remove all dependencies (with confirmation)"
    echo "  --list       List current dependencies"
    echo ""
    echo "This script manages the dependencies for the OCTANE ROS system."
}

install_dependencies() {
    echo "Installing dependencies: ${ROS_PACKAGES}"

    # Update package list
    sudo apt update

    # Install ROS Humble
    echo "Installing ROS 2 Humble..."
    sudo apt install -y ${ROS_PACKAGES}

    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip3 install --user python-dateutil numpy
}

remove_dependencies() {
    echo "WARNING: This will remove ROS packages from your system!"
    read -p "Are you sure you want to remove all ROS dependencies? (yes/no): " confirm
    if [[ $confirm == "yes" ]]; then
        echo "Removing ROS packages..."
        sudo apt remove -y ${ROS_PACKAGES} || echo "Some packages might not be removable"
    else
        echo "Aborted."
    fi
}

list_dependencies() {
    echo "Current system dependencies:"
    echo "ROS packages: ${ROS_PACKAGES}"
    echo ""
    echo "External packages in src/external_pkgs/:"
    ls -la /media/csulunabotics/SSD/OCTANE/workspace/src/external_pkgs/
}

# Parse command line arguments
case "$1" in
    --install)
        install_dependencies
        ;;
    --remove)
        remove_dependencies
        ;;
    --list)
        list_dependencies
        ;;
    *)
        show_help
        ;;
esac