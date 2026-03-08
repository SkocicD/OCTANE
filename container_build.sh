#!/bin/bash

# Build Docker container for Isaac Sim ROS 2 bridge

echo "=== Building Isaac Sim Bridge Container ==="
echo ""
echo "This will build a Docker image with:"
echo "  - ROS2 Humble"
echo "  - FastDDS (rmw_fastrtps_cpp) configured for Windows-WSL2 discovery"
echo "  - XFCE4 Desktop Environment + TigerVNC (for debugging)"
echo "  - ROS Visualization Tools (rqt, rviz2)"
echo ""
echo "This may take 15-30 minutes depending on your system..."
echo ""

# Build with progress output
docker compose build --progress=plain

if [ $? -eq 0 ]; then
    echo ""
    echo "=== BUILD SUCCESSFUL ==="
    echo ""
    echo "Next steps:"
    echo "  1. Copy .env.example to .env and set ISAAC_SIM_HOST"
    echo "  2. Start bridge: ./run_bridge.sh"
    echo "  3. (Optional) Start VNC: ./start_vnc.sh  →  vnc://localhost:5901 (password: octane)"
    echo ""
else
    echo ""
    echo "=== BUILD FAILED ==="
    echo "Check the output above for errors."
    echo ""
    exit 1
fi
