#!/bin/bash

# Docker cleanup script - removes dangling images, stopped containers, and build cache
# Preserves base Isaac ROS image
echo "Starting Docker cleanup..."

# Remove dangling images (untagged), but preserve base image
echo "Removing dangling images (preserving base Isaac ROS image)..."
docker image prune -f --filter "label!=base"

# Remove unused containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused networks
echo "Removing unused networks..."
docker network prune -f

# Remove unused volumes
echo "Removing unused volumes..."
docker volume prune -f

# Remove build cache
echo "Cleaning build cache..."
docker builder prune -f --all

echo "Docker cleanup completed!"
echo "Note: Base Isaac ROS image has been preserved."