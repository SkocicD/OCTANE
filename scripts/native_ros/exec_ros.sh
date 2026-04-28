#!/bin/bash
docker exec -it ros2 bash -c "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash 2>/dev/null; exec bash"
