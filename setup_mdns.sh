#!/bin/bash

# Configure mDNS (avahi) so the rover advertises itself as <hostname>.local.
# Reads all settings from octane.conf — edit that file, not this one.
#
# Run once after flashing or changing octane.conf:
#   sudo ./setup_mdns.sh
#
# After this runs, any device on the same network can reach the rover at
# octane.local (or whatever OCTANE_HOSTNAME is set to in octane.conf).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] Run with sudo: sudo ./setup_mdns.sh"
    exit 1
fi

# Load config
source "${SCRIPT_DIR}/octane.conf"

echo "[CONFIG] Hostname   : ${OCTANE_HOSTNAME}"
echo "[CONFIG] Service    : ${OCTANE_SERVICE_NAME}"
echo "[CONFIG] TCP port   : ${OCTANE_TCP_PORT}"
echo "[CONFIG] Interfaces : ${OCTANE_INTERFACES:-all}"
echo ""

# ── 1. Hostname ───────────────────────────────────────────────────────────────
echo "[1/4] Setting hostname to '${OCTANE_HOSTNAME}'..."
hostnamectl set-hostname "${OCTANE_HOSTNAME}"
echo "      $(hostname)"

# ── 2. Install avahi ──────────────────────────────────────────────────────────
echo "[2/4] Ensuring avahi-daemon is installed..."
if ! dpkg -s avahi-daemon &>/dev/null; then
    apt-get install -y avahi-daemon avahi-utils
else
    echo "      already installed"
fi

# ── 3. Configure avahi-daemon.conf ────────────────────────────────────────────
echo "[3/4] Writing /etc/avahi/avahi-daemon.conf..."

# Build allow-interfaces line only if specific interfaces are set
if [ -n "${OCTANE_INTERFACES}" ]; then
    IFACE_LINE="allow-interfaces=${OCTANE_INTERFACES}"
else
    IFACE_LINE="# allow-interfaces=   (all interfaces)"
fi

cat > /etc/avahi/avahi-daemon.conf << EOF
[server]
host-name=${OCTANE_HOSTNAME}
use-ipv4=${OCTANE_USE_IPV4}
use-ipv6=${OCTANE_USE_IPV6}
${IFACE_LINE}
ratelimit-interval-usec=1000000
ratelimit-burst=1000

[wide-area]
enable-wide-area=yes

[publish]
publish-addresses=yes
publish-hinfo=yes
publish-workstation=yes
publish-domain=yes

[reflector]
enable-reflector=no

[rlimits]
rlimit-core=0
rlimit-data=4194304
rlimit-fsize=0
rlimit-nofile=768
rlimit-stack=4194304
rlimit-nproc=3
EOF

# ── 4. Register OCTANE TCP service ────────────────────────────────────────────
echo "[4/4] Writing /etc/avahi/services/octane.service..."
mkdir -p /etc/avahi/services

cat > /etc/avahi/services/octane.service << EOF
<?xml version="1.0" standalone='no'?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group>
  <name>${OCTANE_SERVICE_NAME}</name>
  <service>
    <type>_octane._tcp</type>
    <port>${OCTANE_TCP_PORT}</port>
  </service>
</service-group>
EOF

# ── Enable and restart ────────────────────────────────────────────────────────
systemctl enable avahi-daemon
systemctl restart avahi-daemon

echo ""
echo "[OK] mDNS configured. Rover is advertising as '${OCTANE_HOSTNAME}.local'"
echo ""
echo "     Verify from another machine on the same network:"
echo "       avahi-resolve -n ${OCTANE_HOSTNAME}.local"
echo "       ping ${OCTANE_HOSTNAME}.local"
echo "       nc -zv ${OCTANE_HOSTNAME}.local ${OCTANE_TCP_PORT}"
