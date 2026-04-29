#!/bin/bash

# Script to manage Docker base images and ensure they're preserved
echo "Managing Docker base images..."

# Function to tag and preserve base image
preserve_base_image() {
    echo "Tagging NVIDIA Isaac ROS base image for preservation..."

    # Pull the base image if not present
    if ! docker image inspect nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos >/dev/null 2>&1; then
        echo "Pulling NVIDIA Isaac ROS base image..."
        /home/csulunabotics/OCTANE/scripts/docker_setup/auth_ngc.sh
        docker pull nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos
    fi

    # Tag the base image with a preservation tag
    echo "Tagging base image for preservation..."
    docker tag nvcr.io/nvidia/isaac/ros:isaac_ros_054e16b5c3a328b621af47d26009c348-arm64-fastos isaac-ros:base-preserved

    if [ $? -eq 0 ]; then
        echo "Successfully tagged base image for preservation"
    else
        echo "Failed to tag base image"
        return 1
    fi

    # List images to confirm
    echo "Current Docker images:"
    docker images | grep -E "(isaac|nvcr)"

    return 0
}

# Function to check if base image is preserved
check_preserved_base() {
    echo "Checking preserved base image..."
    if docker image inspect isaac-ros:base-preserved >/dev/null 2>&1; then
        echo "Base image is preserved with tag: isaac-ros:base-preserved"
        return 0
    else
        echo "Base image not found with preservation tag"
        return 1
    fi
}

# Main execution
case "$1" in
    --preserve)
        preserve_base_image
        ;;
    --check)
        check_preserved_base
        ;;
    *)
        echo "Usage: $0 [--preserve|--check]"
        echo "  --preserve: Pull and tag NVIDIA Isaac ROS base image for preservation"
        echo "  --check: Check if base image is preserved"
        ;;
esac