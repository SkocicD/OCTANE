#!/bin/bash

# Flexible Docker build script with dependency and application layer options
# Usage: ./build_docker.sh [--deps] [--app] [--clean]
#
# Examples:
#   ./build_docker.sh --deps        # Build only dependencies layer
#   ./build_docker.sh --app        # Build only application layer
#   ./build_docker.sh --deps --app # Build both layers (default)
#   ./build_docker.sh --clean      # Clean previous builds
#
# NOTE: This script must be run from the project root directory!

set -e

# Check if we're in the correct directory (should be project root)
if [ ! -d "scripts" ] || [ ! -d "workspace" ]; then
    echo "ERROR: This script must be run from the project root directory!"
    echo "Current directory: $(pwd)"
    echo "Please run from /media/csulunabotics/SSD/OCTANE"
    exit 1
fi

# Parse command line arguments
BUILD_DEPS=false
BUILD_APP=false
CLEAN_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --deps)
            BUILD_DEPS=true
            shift
            ;;
        --app)
            BUILD_APP=true
            shift
            ;;
        --clean)
            CLEAN_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--deps] [--app] [--clean]"
            echo "  --deps   Build dependencies layer"
            echo "  --app    Build application layer"
            echo "  --clean  Clean previous builds"
            exit 1
            ;;
    esac
done

# Set defaults if no options specified
if [ "$BUILD_DEPS" = false ] && [ "$BUILD_APP" = false ]; then
    BUILD_DEPS=true
    BUILD_APP=true
fi

echo "=== OCTANE Docker Build ==="
[ "$BUILD_DEPS" = true ] && echo "  - Building dependencies layer"
[ "$BUILD_APP" = true ] && echo "  - Building application layer"
[ "$CLEAN_MODE" = true ] && echo "  - Cleaning previous builds"

# Clean mode - remove old images and workspace build cache
if [ "$CLEAN_MODE" = true ]; then
    echo "Cleaning previous builds..."
    # Remove old build images to prevent clutter
    docker rmi octane-deps:latest octane-app:latest 2>/dev/null || true
    # Clean workspace build cache
    /media/csulunabotics/SSD/OCTANE/scripts/docker_setup/clean_workspace_build.sh
    # Remove any existing containers with the same name
    docker rm -f octane_container 2>/dev/null || true
fi

# Build dependencies layer if requested
if [ "$BUILD_DEPS" = true ]; then
    echo "Building dependencies layer..."
    docker build -t octane-deps:latest -f scripts/docker_setup/Dockerfile.layers --target dependencies .
    echo "Dependencies layer built successfully!"
fi

# Build application layer if requested
if [ "$BUILD_APP" = true ]; then
    echo "Building application layer..."
    docker build -t octane-app:latest -f scripts/docker_setup/Dockerfile.layers --target application .
    echo "Application layer built successfully!"
fi

# If building both layers, build the full application
if [ "$BUILD_DEPS" = true ] && [ "$BUILD_APP" = true ]; then
    echo "Building full application..."
    docker build -t octane-app:latest -f scripts/docker_setup/Dockerfile.layers .
    echo "Full application built successfully!"
fi

echo "=== Build Complete ==="
echo "Run container with: docker run -it octane-app:latest"