#!/bin/bash

# Intuitive Docker build script with configuration options
# Usage: ./build_config.sh [minimal|full|dev]

set -e

CONFIG=${1:-full}  # Default to full configuration
echo "Building OCTANE with $CONFIG configuration..."

case "$CONFIG" in
    minimal)
        echo "Using minimal configuration (core dependencies only)"
        docker build -t octane-minimal:latest -f Dockerfile .
        ;;
    full)
        echo "Using full configuration (all dependencies)"
        docker build -t octane-full:latest -f Dockerfile .
        ;;
    dev)
        echo "Using development configuration"
        docker build -t octane-dev:latest -f Dockerfile.dev .
        ;;
    *)
        echo "Usage: $0 [minimal|full|dev]"
        echo "  minimal - Core dependencies only (smaller, faster)"
        echo "  full    - All dependencies (default)"
        echo "  dev     - Development environment"
        exit 1
        ;;
esac