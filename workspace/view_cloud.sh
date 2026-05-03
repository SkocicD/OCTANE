#!/bin/bash
# View the combined point cloud from point_cloud_mux_node in RViz2.
# Run this after the mapping subsystem is up:
#   ./launch_system.sh mapping   (or all)
#   ./view_cloud.sh

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="$WORKSPACE_DIR/src/octane/octane/rviz/combined_cloud.rviz"

source "$WORKSPACE_DIR/install/setup.bash"

echo "Opening combined point cloud viewer — topic: /mapping/point_cloud/combined"
rviz2 -d "$RVIZ_CONFIG"
