#!/bin/bash

# Setup script for native ROS development environment

echo "Setting up native ROS development environment..."

# Check if ROS is installed
if ! command -v ros2 &> /dev/null; then
    echo "ROS 2 not found. Installing ROS 2 Humble..."
    # Add ROS 2 repository
    sudo apt update && sudo apt install -y software-properties-common
    sudo add-apt-repository universe
    sudo apt update && sudo apt install -y curl
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /tmp/ros.key
    sudo apt-key add /tmp/ros.key
    sudo apt update
    sudo apt install -y ros-humble-desktop python3-argcomplete python3-colcon-common-extensions
fi

echo "ROS environment is ready!"
echo "To build OCTANE natively, run: ./scripts/native/run_native.sh"