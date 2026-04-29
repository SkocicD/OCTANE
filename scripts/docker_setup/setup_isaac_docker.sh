#!/bin/bash

# Complete Docker Setup and Build Script
# This script sets up the Docker environment with layered approach using NVIDIA Isaac ROS base image

echo "=== OCTANE Docker Setup with Layered Approach ==="

# Step 1: Authenticate with NVIDIA NGC
echo "Step 1: Authenticating with NVIDIA NGC..."
/home/csulunabotics/OCTANE/scripts/docker_setup/auth_ngc.sh

# Step 2: Pull and preserve base image
echo "Step 2: Preserving NVIDIA Isaac ROS base image..."
/home/csulunabotics/OCTANE/scripts/docker_setup/manage_base_image.sh --preserve

# Step 3: Build the application layer
echo "Step 3: Building application image..."
/home/csulunabotics/OCTANE/scripts/docker_setup/build_layered.sh --layered

echo "=== Setup Complete ==="
echo "Docker environment is now configured with layered approach"
echo "Base Isaac ROS image is preserved and will not be deleted"
echo "You can now use the application normally"