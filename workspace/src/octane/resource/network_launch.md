# Network Launch - network.launch.py

**Ground station communication layer with TCP command/control and UDP heartbeat monitoring.**

## Overview

The network subsystem provides bi-directional communication between the rover and ground station. It implements a lean binary protocol for efficient data transfer over the network.

**Key Features:**
- TCP-based mode commands with ACK responses
- Telemetry broadcasting at configurable rate
- UDP heartbeat for connection health monitoring
- CRC-8 validation for data integrity

## Launch Location

`workspace/src/octane/octane/launch/network.launch.py`

## Nodes Launched

### 1. `network_comm_node`

**Purpose:** TCP server for mode commands and telemetry broadcasting

**Functionality:**

**Incoming Commands (Ground → Rover):**
- Listens on TCP port 5000 (configurable)
- Receives binary frames from ground station
- Decodes mode commands using protocol spec
- Publishes to `/supervisor/mode_command` ROS2 topic

**Outgoing Telemetry (Rover → Ground):**
- Subscribes to `/supervisor/state` for current mode
- Broadcasts telemetry at specified rate (10Hz default)
- Binary frame format: `[MAGIC][TYPE][LEN][STATE][CRC]`
- Includes fault status when active

**Binary Protocol:**
```
Command Frame (6 bytes):
[0x4F][0x43][0x02][MODE][ESTOP][CRC]
 O    C    LEN  mode  e    crc

Telemetry Frame (5 bytes):
[0x4F][0x54][0x01][STATE][CRC]
 O    T    LEN  state  crc
```

**Protocol Details:**
- MAGIC: `0x4F` ('O') - Frame start marker
- TYPE: `0x43` ('C') = Command, `0x54` ('T') = Telemetry
- LEN: Payload length in bytes
- CRC: CRC-8 checksum for integrity

**Topics:**
- **Subscribes:** `/supervisor/state` (String)
- **Publishes:** `/supervisor/mode_command` (String)

**Parameters:**
- `host` (string) - TCP bind address (default: 0.0.0.0)
- `port` (int) - TCP port number (default: 5000)
- `telemetry_rate` (float) - Broadcast rate in Hz (default: 10.0)

---

### 2. `heartbeat_sender`

**Purpose:** UDP heartbeat for connection health monitoring

**Functionality:**
- Sends UDP broadcast every 300ms (0.33 Hz default)
- Includes current mode state in heartbeat
- 5-byte minimal frame: `[MAGIC][STATE][SEQ_HI][SEQ_LO][CRC]`
- Ground station uses heartbeat to detect connection loss

**Heartbeat Frame:**
```
[0x4F][STATE][SEQ_HI][SEQ_LO][CRC]
 O     state  seq       crc
5 bytes total

STATE: '0'=STANDBY, '1'=MANUAL, '2'=AUTONOMOUS, '3'=FAULT
SEQ: 16-bit sequence number (wraps at 65535)
CRC: CRC-8 of first 4 bytes
```

**Ground Station Usage:**
- If no heartbeat received for 6+ seconds → connection lost
- Heartbeat ring pulses orange on each received packet
- Network selector shows active interface traffic

**Topics:**
- **Subscribes:** `/supervisor/state` (String)

**Parameters:**
- `host` (string) - UDP broadcast address (default: 255.255.255.255)
- `port` (int) - UDP port number (default: 5001)
- `rate_hz` (float) - Heartbeat frequency (default: 0.33)

---

## Network Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   ROVER SIDE                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │  network_comm_node.py                            │   │
│  │  ├── TCP Server (port 5000)                      │   │
│  │  │   ↘ Receive mode commands                     │   │
│  │  │   ↗ Broadcast telemetry                       │   │
│  │  └── Subscribes: /supervisor/state               │   │
│  └──────────────────────────────────────────────────┘   │
│                    ↕                                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │  heartbeat_sender.py                             │   │
│  │  ├── UDP Broadcast (port 5001, 300ms interval)   │   │
│  │  └── Subscribes: /supervisor/state               │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────┘
                   │ WiFi / Ethernet
                   ▼
┌──────────────────┴──────────────────────────────────────┐
│               GROUND STATION                             │
│  ┌──────────────────────────────────────────────────┐   │
│  │  NetworkModeClient.cs (GUI)                      │   │
│  │  ├── TCP Client → rover:5000                     │   │
│  │  │   ↘ Send mode commands                        │   │
│  │  │   ↗ Receive telemetry                         │   │
│  │  └── UDP Listener ← rover:5001                   │   │
│  │       ↕                                           │   │
│  │       Heartbeat ring pulses on each packet       │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## Protocol Specification

**Frame Structure:**
```
Header (3 bytes):     [MAGIC][TYPE][LENGTH]
Payload (N bytes):    Mode/EStop for commands, State for telemetry
Checksum (1 byte):    CRC-8 of header + payload
```

**Mode Codes:**
- `0` = STANDBY
- `1` = MANUAL
- `2` = AUTONOMOUS
- `3` = FAULT_RESET

**State Codes:**
- `'0'` = STANDBY
- `'1'` = MANUAL
- `'2'` = AUTONOMOUS
- `'3'` = FAULT

**CRC-8 Polynomial:** `0x07`

---

## Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `tcp_host` | 0.0.0.0 | TCP bind address |
| `tcp_port` | 5000 | TCP port for commands |
| `telemetry_rate` | 10.0 | Telemetry broadcast rate (Hz) |
| `udp_host` | 255.255.255.255 | UDP broadcast address |
| `udp_port` | 5001 | UDP port for heartbeat |
| `heartbeat_rate` | 0.33 | Heartbeat frequency (Hz) |

---

## Usage Examples

**Launch with default settings:**
```bash
ros2 launch octane network.launch.py
```

**Custom port and rate:**
```bash
ros2 launch octane network.launch.py tcp_port:=5005 telemetry_rate:=20.0 heartbeat_rate:=0.5
```

**Listen on specific interface:**
```bash
ros2 launch octane network.launch.py tcp_host:=192.168.1.100
```

---

## Ground Station Integration

**Network Selector in GUI:**
- Shows available network interfaces (WiFi/Ethernet)
- Displays actual network names/SSIDs via netsh API
- Select interface to monitor traffic on that adapter
- Data usage graph shows real-time upload/download rates

**Connection Status Display:**
- **Green** = Connected, heartbeat received
- **Red** = Offline or heartbeat timeout
- **Orange ring pulse** = Each heartbeat packet received

---

## Troubleshooting

**Heartbeat not working:**
```bash
# Check if UDP port 5001 is receiving packets
netstat -an | findstr :5001

# Verify firewall allows UDP broadcast
# Windows: Allow UDP 5001 in Windows Firewall
```

**TCP connection refused:**
```bash
# Check if TCP port 5000 is listening
netstat -an | findstr :5000

# Verify network_comm_node is running
ros2 node list | grep network
```

**CRC errors in logs:**
```bash
# CRC mismatch indicates corrupted data
# Check network cable/WiFi signal quality
# Verify both sides use same protocol version
```

---

## Data Flow

See `../../octane_network/resource/data_flow.md` for complete message flow documentation.

---

## See Also

- `../../octane_network/resource/protocol.md` - Wire format specification
- `../../octane_network/resource/messages.md` - Message payload reference
- `../../octane_network/resource/nodes.md` - Node API documentation
- `../octane.md` - Main octane package documentation
