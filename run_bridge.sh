#!/bin/bash

if [ ! -f .env ]; then
    echo "ERROR: .env not found."
    echo "Copy .env.example to .env and set ISAAC_SIM_HOST to your Windows host IP."
    echo "  From WSL2: ip route show default | awk '{print \$3}'"
    exit 1
fi

# ISAAC_SIM_HOST is passed to the container via .env / docker-compose.
# envsubst runs inside the container at startup to resolve it into the FastDDS profile.
docker compose up -d && ./exec_ros.sh
