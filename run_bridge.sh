#!/bin/bash

if [ ! -f .env ]; then
    echo "ERROR: .env not found."
    echo "Copy .env.example to .env and set ISAAC_SIM_HOST to your Windows host IP."
    echo "  From WSL2: ip route show default | awk '{print \$3}'"
    exit 1
fi

# Load .env so ISAAC_SIM_HOST is available for substitution
set -a; source .env; set +a

# Generate fastdds_bridge.xml with the real IP substituted in.
# $ENV{VAR} is not supported in <address> fields in FastDDS 2.6.x (ROS 2 Humble),
# so we do the substitution here using sed before the container starts.
sed "s|\${ISAAC_SIM_HOST}|${ISAAC_SIM_HOST}|g" \
    docker_setup/fastdds_bridge.xml.template > docker_setup/fastdds_bridge.xml

echo "FastDDS profile generated with ISAAC_SIM_HOST=${ISAAC_SIM_HOST}"

docker compose up -d && ./exec_ros.sh
