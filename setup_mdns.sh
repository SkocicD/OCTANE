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
echo "[1/5] Setting hostname to '${OCTANE_HOSTNAME}'..."
hostnamectl set-hostname "${OCTANE_HOSTNAME}"
echo "      $(hostname)"

# ── 2. /etc/hosts ─────────────────────────────────────────────────────────────
echo "[2/5] Updating /etc/hosts for local hostname resolution..."
# Remove any existing 127.0.1.1 line and re-add with current hostname
sed -i '/^127\.0\.1\.1/d' /etc/hosts
echo "127.0.1.1	${OCTANE_HOSTNAME}" >> /etc/hosts
echo "      done"

# ── 3. Install avahi ──────────────────────────────────────────────────────────
echo "[3/5] Ensuring avahi-daemon is installed..."
if ! dpkg -s avahi-daemon &>/dev/null; then
    apt-get install -y avahi-daemon avahi-utils
else
    echo "      already installed"
fi

# ── 4. Configure avahi-daemon.conf ────────────────────────────────────────────
echo "[4/5] Writing /etc/avahi/avahi-daemon.conf..."

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

# ── 5. Register OCTANE TCP service ────────────────────────────────────────────
echo "[5/5] Writing /etc/avahi/services/octane.service..."
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

# ── Install boot-time hosts-fix service ───────────────────────────────────────
cat > /usr/local/bin/octane-fix-hosts.sh << 'FIXEOF'
#!/bin/bash
# Ensure the current hostname is resolvable locally via /etc/hosts.
HOSTNAME="$(hostname)"
sed -i '/^127\.0\.1\.1/d' /etc/hosts
echo "127.0.1.1	${HOSTNAME}" >> /etc/hosts
FIXEOF
chmod +x /usr/local/bin/octane-fix-hosts.sh

cat > /etc/systemd/system/octane-fix-hosts.service << 'SVCEOF'
[Unit]
Description=Fix /etc/hosts local hostname entry
DefaultDependencies=no
Before=network-pre.target avahi-daemon.service
After=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/octane-fix-hosts.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable octane-fix-hosts.service

# ── Enable and restart avahi ──────────────────────────────────────────────────
systemctl enable avahi-daemon
systemctl restart avahi-daemon

echo ""
echo "[OK] mDNS configured. Rover is advertising as '${OCTANE_HOSTNAME}.local'"
echo ""
echo "     Verify from another machine on the same network:"
echo "       avahi-resolve -n ${OCTANE_HOSTNAME}.local"
echo "       ping ${OCTANE_HOSTNAME}.local"
echo "       nc -zv ${OCTANE_HOSTNAME}.local ${OCTANE_TCP_PORT}"
