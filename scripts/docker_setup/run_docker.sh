#!/bin/bash

# Run OCTANE container
# Usage: ./run_docker.sh [options]
# Options:
#   (no args)     - Run interactive bash shell
#   --ros         - Run ROS environment
#   --command CMD - Run specific command

set -e

IMAGE_NAME="octane-app:latest"

# Check if image exists
if ! docker image inspect $IMAGE_NAME >/dev/null 2>&1; then
    echo "Error: $IMAGE_NAME not found. Please build the image first:"
    echo "  ./scripts/docker_setup/build_docker.sh --app"
    exit 1
fi

echo "Starting OCTANE container..."

# Default command
CMD="bash"

# Parse arguments
if [ "$#" -gt 0 ]; then
    case "$1" in
        --ros)
            CMD="bash -c 'source /opt/ros/jazzy/setup.bash && exec bash'"
            ;;
        --command)
            shift
            CMD="$*"
            ;;
        *)
            echo "Usage: $0 [--ros] [--command CMD]"
            echo "  (no args)     - Run interactive bash shell"
            echo "  --ros         - Run ROS environment"
            echo "  --command CMD - Run specific command"
            exit 1
            ;;
    esac
fi

# Run container
echo "Running: $CMD"
docker run -it --rm \
    --name octane_container \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    $IMAGE_NAME \
    $CMD