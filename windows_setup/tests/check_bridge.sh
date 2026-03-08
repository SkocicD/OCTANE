#!/bin/bash
# Isaac Sim Bridge — End-to-end Verification
#
# Run this from WSL2 (not inside the container) after:
#   1. setup_bridge.ps1 has been run on Windows
#   2. Isaac Sim is open with omni.isaac.ros2_bridge extension enabled
#   3. The container is running: ./run_bridge.sh  (from repo root)
#
# Uses docker exec so nothing needs to be copied into the container.

CONTAINER="ros2_bridge"
ROS_SETUP="source /opt/ros/humble/setup.bash"
PASS=0
FAIL=0

check() {
    local label=$1
    local code=$2
    local hint=${3:-""}
    if [ "$code" -eq 0 ]; then
        echo "  [PASS] $label"
        ((PASS++))
    else
        echo "  [FAIL] $label"
        [ -n "$hint" ] && echo "         -> $hint"
        ((FAIL++))
    fi
}

echo ""
echo "=== Isaac Sim Bridge Verification ==="
echo ""

# ── Container health ──────────────────────────────────────────────────────────
echo "Container:"
docker ps --filter "name=${CONTAINER}" --filter "status=running" -q | grep -q .
check "ros2_bridge container is running" $? "Run ./run_bridge.sh from the repo root"

# Bail early — remaining checks all require the container
if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Container is not running. Aborting further checks."
    exit 1
fi

# ── Container environment ─────────────────────────────────────────────────────
echo ""
echo "Container environment:"

ISAAC_HOST=$(docker exec "$CONTAINER" printenv ISAAC_SIM_HOST 2>/dev/null)
DOMAIN_ID=$(docker exec  "$CONTAINER" printenv ROS_DOMAIN_ID 2>/dev/null)
RMW=$(docker exec        "$CONTAINER" printenv RMW_IMPLEMENTATION 2>/dev/null)
PROFILE=$(docker exec    "$CONTAINER" printenv FASTRTPS_DEFAULT_PROFILES_FILE 2>/dev/null)

[ -n "$ISAAC_HOST" ]
check "ISAAC_SIM_HOST is set ($ISAAC_HOST)"          $? "Add ISAAC_SIM_HOST to your .env file"
[ "$DOMAIN_ID" = "0" ]
check "ROS_DOMAIN_ID = 0"                            $? "Check .env or docker-compose.yaml"
[ "$RMW" = "rmw_fastrtps_cpp" ]
check "RMW_IMPLEMENTATION = rmw_fastrtps_cpp"        $? "Check docker-compose.yaml"
[ -n "$PROFILE" ]
check "FASTRTPS_DEFAULT_PROFILES_FILE is set"        $? "Check docker-compose.yaml volume mount"

if [ -n "$PROFILE" ]; then
    docker exec "$CONTAINER" test -f "$PROFILE" 2>/dev/null
    check "FastDDS XML profile is mounted in container" $? "Volume mount may be missing — check docker-compose.yaml"
fi

# ── Network ───────────────────────────────────────────────────────────────────
echo ""
echo "Network:"

if [ -n "$ISAAC_HOST" ]; then
    docker exec "$CONTAINER" ping -c 1 -W 2 "$ISAAC_HOST" > /dev/null 2>&1
    check "Container can ping Windows host ($ISAAC_HOST)" $? \
        "Check ISAAC_SIM_HOST in .env. Is the Windows firewall allowing ICMP?"
fi

# ── ROS 2 bridge ──────────────────────────────────────────────────────────────
echo ""
echo "ROS 2 bridge:"
echo "  (Waiting up to 5s for DDS discovery...)"

# Give discovery a moment — DDS unicast handshake isn't instant
sleep 5

TOPIC_RAW=$(docker exec "$CONTAINER" bash -c "$ROS_SETUP && ros2 topic list 2>/dev/null")

# Strip ANSI escape codes, then keep only lines that start with '/' (actual topic names).
# Without this, FastDDS XML parse errors (which mention file paths) skew the count.
TOPIC_LIST=$(echo "$TOPIC_RAW" | sed 's/\x1b\[[0-9;]*[mGKH]//g' | grep '^/')
TOPIC_COUNT=$(echo "$TOPIC_LIST" | grep -c '^/' || true)

# ros2 topic list always exits 0; count > 2 means something beyond the default
# /parameter_events and /rosout that any ROS 2 node publishes
[ "$TOPIC_COUNT" -gt 2 ]
check "ros2 topic list shows Isaac Sim topics ($TOPIC_COUNT topics)" $? \
    "Isaac Sim may not be publishing yet. Confirm omni.isaac.ros2_bridge is enabled and an Action Graph is running."

if [ "$TOPIC_COUNT" -gt 0 ]; then
    echo ""
    echo "  Topics visible from container:"
    echo "$TOPIC_LIST" | sed 's/^/    /'

    # Spot-check for /clock — easiest Isaac Sim topic to confirm bridge is live
    echo ""
    echo "$TOPIC_LIST" | grep -q "^/clock$"
    check "/clock topic present" $? \
        "Add a ROS2 Clock node to your Isaac Sim Action Graph and hit Play"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────"
if [ "$FAIL" -eq 0 ]; then
    echo "All $PASS checks passed. Bridge is live."
else
    echo "$FAIL of $((PASS + FAIL)) check(s) failed. See hints above."
fi
echo ""
