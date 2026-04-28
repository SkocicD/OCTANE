#!/bin/bash

# Native ROS setup and build script for OCTANE

# Source ROS environment
source /opt/ros/humble/setup.bash

# Go to workspace
cd /media/csulunabotics/SSD/OCTANE/workspace

# Build the system
echo "Building OCTANE system..."
./build_system.sh