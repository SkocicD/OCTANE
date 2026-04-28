#!/bin/bash

# Start VNC server in Docker container
# This runs in the foreground - press Ctrl+C to stop

echo "=== Starting VNC Server ==="
echo "Password: octane"
echo ""
echo "Connect with VNC viewer:"
echo "  Address: vnc://localhost:5901"
echo "  Password: octane"
echo ""
echo "VNC server will run in the foreground."
echo "Press Ctrl+C to stop the VNC server."
echo ""
echo "========================================="
echo ""

# Kill any existing VNC server first
docker exec ros2 bash -c "vncserver -kill :1 2>/dev/null || true"

# Start VNC server with localhost disabled (allows external connections)
docker exec -it ros2 bash -c "vncserver :1 -geometry 1920x1080 -depth 24 -localhost=0 -fg"
