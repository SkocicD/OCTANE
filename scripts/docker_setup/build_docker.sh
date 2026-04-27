#!/bin/bash

# Flexible Docker build script with dependency and application layer options
# Usage: ./build_docker.sh [--deps] [--app] [--clean]
#
# Examples:
#   ./build_docker.sh --deps        # Build only dependencies layer
#   ./build_docker.sh --app        # Build only application layer
#   ./build_docker.sh --deps --app # Build both layers (default)
#   ./build_docker.sh --clean      # Clean previous builds

set -e

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

# Clean mode - remove old images
if [ "$CLEAN_MODE" = true ]; then
    echo "Cleaning previous builds..."
    # Remove old build images to prevent clutter
    docker rmi octane-deps:latest octane-app:latest 2>/dev/null || true
fi

# Build dependencies layer if requested
if [ "$BUILD_DEPS" = true ]; then
    echo "Building dependencies layer..."
    docker build --target dependencies -t octane-deps:latest -f Dockerfile.layers .
    echo "Dependencies layer built successfully!"
fi

# Build application layer if requested
if [ "$BUILD_APP" = true ]; then
    echo "Building application layer..."
    docker build --target application -t octane-app:latest -f Dockerfile.layers .
    echo "Application layer built successfully!"
fi

echo "=== Build Complete ==="
echo "Run container with: docker run -it octane-app:latest"