#!/bin/bash

# Script to build Docker images with proper layering strategy
echo "Building Docker images with layering strategy..."

# Retry a docker build command up to MAX_RETRIES times on failure
MAX_RETRIES=3
retry_build() {
    local attempt=1
    until "$@"; do
        if [ $attempt -ge $MAX_RETRIES ]; then
            echo "ERROR: Command failed after $MAX_RETRIES attempts: $*"
            return 1
        fi
        echo "Build failed (attempt $attempt/$MAX_RETRIES). Retrying in 10s..."
        attempt=$((attempt + 1))
        sleep 10
    done
}

# Function to check if base image exists
check_base_image() {
    echo "Checking if NVIDIA Isaac ROS base image exists locally..."
    if docker image inspect nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos >/dev/null 2>&1; then
        echo "NVIDIA Isaac ROS base image found locally"
        return 0
    else
        echo "NVIDIA Isaac ROS base image not found locally"
        return 1
    fi
}

# Function to pull base image if needed
pull_base_image() {
    echo "Authenticating with NVIDIA NGC..."
    /home/csulunabotics/OCTANE/scripts/docker_setup/auth_ngc.sh

    echo "Pulling NVIDIA Isaac ROS base image..."
    docker pull nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos

    if [ $? -eq 0 ]; then
        echo "Successfully pulled NVIDIA Isaac ROS base image"
        return 0
    else
        echo "Failed to pull NVIDIA Isaac ROS base image"
        return 1
    fi
}

# Function to build layered images
build_layered_images() {
    echo "Building layered Docker images..."

    # Build base layer (if not already built)
    if ! check_base_image; then
        echo "Base image not found, pulling it first..."
        pull_base_image
    fi

    # Build the dependencies layer first
    echo "Building dependencies layer..."
    retry_build docker build -t octane-deps:latest -f Dockerfile.layers --target dependencies .

    if [ $? -eq 0 ]; then
        echo "Successfully built dependencies layer"
    else
        echo "Failed to build dependencies layer"
        return 1
    fi

    # Build the application layer
    echo "Building application layer..."
    retry_build docker build -t octane-app:latest -f Dockerfile.layers --target application .

    if [ $? -eq 0 ]; then
        echo "Successfully built application layer"
        return 0
    else
        echo "Failed to build application layer"
        return 1
    fi
}

# Main execution
echo "Starting Docker image build process..."

# Check if we're using the layered approach
if [ "$1" = "--layered" ]; then
    echo "Using layered build approach"
    build_layered_images
else
    echo "Usage: $0 --layered"
    echo "This script builds Docker images using the layered approach with NVIDIA Isaac ROS base image"
    exit 1
fi