# OCTANE network - Nodes

## `network_comm_node`

TCP server bridging ROS2 topics to ground station communication.

### Parameters
- `host` (default: "0.0.0.0") - TCP bind address
- `port` (default: 5000) - TCP port
- `telemetry_rate` (default: 10.0) - Telemetry broadcast rate (Hz)
- `timeout_seconds` (default: 2.0) - Connection timeout

### ROS2 Subscribers
| Topic | Type | Description |
|-------|------|-------------|
| `/supervisor/state` | std_msgs/String | Current state (STANDBY/MANUAL/AUTONOMOUS/FAULT) |
| `/supervisor/fault_signal` | std_msgs/String | Fault triggered event |

### ROS2 Publishers
| Topic | Type | Description |
|-------|------|-------------|
| `/supervisor/mode_command` | std_msgs/String | Mode command from ground station |
| `/supervisor/fault_reset` | std_msgs/Empty | Fault reset command from ground station |

### TCP Behavior
**Server Mode:**
- Listens on configured host:port
- Accepts single ground station connection
- Broadcasts telemetry at `telemetry_rate` Hz
- Sends immediate fault alerts on reception

**Command Handling:**
- Decodes Command messages from ground station
- Validates mode (standby, manual, autonomous, fault_reset)
- Publishes to ROS2 topic if valid
- Sends Command ACK with success/failure status

### Thread Model
- **Main thread:** ROS2 spin, timer callbacks
- **Accept thread:** TCP accept loop (1s timeout)
- **Receive thread:** TCP recv loop, message decode, ROS2 publish

### Error Handling
- Connection loss: Logs warning, attempts re-accept
- Invalid message: Logs error, requests continue
- CRC failure: Discards message, logs warning
- ROS2 topic publish failure: Logs error, continues

## Files
---
- `nodes/heartbeat_sender.py` - UDP heartbeat sender node
- `nodes/network_comm_node.py` - Main node implementation
- `classes/protocol.py` - Binary protocol encoder/decoder (shared utilities)
- `config/network_params.yaml` - Node parameters
- `launch/network.launch.py` - Launch file with parameter arguments
