#!/bin/bash

# Comprehensive clean script for OCTANE workspace
# This script cleans both Docker images and workspace build cache

echo "=== OCTANE Clean System ==="

# Clean Docker images
echo "Cleaning Docker images..."
docker rmi octane-deps:latest octane-app:latest 2>/dev/null || true
echo "Docker images cleaned."

# Clean Docker build cache
echo "Cleaning Docker build cache..."
docker builder prune -f --all 2>/dev/null || true
echo "Docker build cache cleaned."

# Clean workspace build cache (requires sudo for root-owned files)
echo "Cleaning workspace build cache..."
if [ -d "workspace/build" ]; then
    sudo rm -rf workspace/build/*
    echo "Workspace build cache cleaned."
else
    echo "No build directory found."
fi

if [ -d "workspace/install" ]; then
    echo "Cleaning install directory..."
    sudo rm -rf workspace/install/*
    echo "Install directory cleaned."
else
    echo "No install directory found."
fi

echo "Clean complete!"