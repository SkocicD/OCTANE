# octane_network

Wireless communication package for OCTANE rover ground station link.

Provides TCP command/control and UDP heartbeat communication between the rover and ground station.

## Components

### Nodes

- **network_comm_node** - TCP server for bidirectional mode commands and telemetry
- **heartbeat_sender** - UDP broadcast for connection health monitoring

### Protocol

Lean binary protocol optimized for minimal bandwidth usage:

- **TCP (port 5000)**: Mode commands, state updates, fault alerts
  - 83% smaller than JSON (5-6 bytes vs 23-35 bytes per message)
  - CRC-8 error detection
  
- **UDP (port 5001)**: Connection heartbeat
  - 5 bytes per packet
  - Configurable rate (default: 300ms interval)
  - GUI timeout: 6.5 seconds

## Documentation

- **messages.md** - Message payload reference (data structures)
- **protocol.md** - Wire format specification (full frames with CRC)
- **data_flow.md** - Complete message flow diagrams
- **nodes.md** - Node descriptions and ROS2 topics

## Usage

Launch both nodes together:

```bash
ros2 launch octane network.launch.py
```

Parameters:
- `tcp_host` (default: 0.0.0.0) - TCP bind address
- `tcp_port` (default: 5000) - TCP port for commands
- `telemetry_rate` (default: 10.0) - Telemetry broadcast rate (Hz)
- `udp_host` (default: 255.255.255.255) - UDP broadcast address
- `udp_port` (default: 5001) - UDP port for heartbeat
- `heartbeat_rate` (default: 0.33) - Heartbeat frequency in Hz

## Architecture

```
┌─────────────────┐          TCP 5000          ┌─────────────────┐
│  Ground Station │◄────Mode Commands─────────►│      Rover      │
│    (C# GUI)     │◄─────Telemetry────────────►│  (ROS2 Node)    │
│                 │◄─────Heartbeat (UDP)───────│                 │
└─────────────────┘          5001              └─────────────────┘
```

- **Commands**: GUI → TCP → network_comm_node → /supervisor/mode_command → mode_manager_node
- **Telemetry**: mode_manager_node → /supervisor/state → network_comm_node → TCP → GUI
- **Heartbeat**: heartbeat_sender → UDP broadcast → GUI connection monitor
