# octane_wifi

Wireless communication package for OCTANE rover ground station link.

## TCP Protocol Design

### Message Frame Format
```
+--------+--------+--------+----------+
| Magic  | Type   | Seq    | Payload  |
| 2B     | 1B     | 2B     | N bytes  |
+--------+--------+--------+----------+
| Length (4B, big-endian, includes payload only) |
+------------------------------------------------+
| Payload (JSON)                                 |
+------------------------------------------------+
| CRC32 (4B, optional, for critical messages)    |
+------------------------------------------------+
```

**Header (7 bytes + 4B length):**
- Magic: `0x4F 0x54` ("OT" for OCTANE)
- Type: `0x01`=Telemetry, `0x02`=Command, `0x03`=Command ACK, `0x04`=Fault Alert
- Seq: Sequence number (wrap at 65535)
- Length: Payload size in bytes

**Payload Format (JSON):**
```json
// Telemetry (0x01)
{"t": "<timestamp>", "m": "<state>", "f": "<fault>", "b": <battery_v>}

// Command (0x02)
{"m": "<mode>", "e": <estop_flag>}

// Command ACK (0x03)
{"s": <success>, "seq": <seq_num>}

// Fault Alert (0x04)
{"f": "<fault_type>", "s": "<severity>"}
```

**Compact field codes:**
- `t` = timestamp (float, unix epoch)
- `m` = mode/state
- `f` = fault
- `b` = battery voltage
- `s` = success status

## Package Structure
```
octane_wifi/
├── launch/
│   └── wifi.launch.py
├── octane_wifi/
│   ├── nodes/
│   │   └── wifi_comm_node.py
│   └── protocol.py
├── config/
│   └── wifi_params.yaml
├── setup.py
└── package.xml
```

## Nodes

### wifi_comm_node
**Single node handling both directions:**

**TCP Server (port 5000):**
- Accepts ground station connections
- Listens for Command messages
- Sends Telemetry and Fault Alert messages

**ROS2 Subscribers:**
- `/supervisor/state` → encode as Telemetry
- `/supervisor/fault_signal` → encode as Fault Alert
- Battery/motor topics (TBD) → include in Telemetry

**ROS2 Publishers:**
- `/supervisor/mode_command` ← decode from Command messages
- `/supervisor/fault_reset` ← when Command contains reset flag

## Configuration (wifi_params.yaml)
```yaml
wifi_comm_node:
  ros__parameters:
    host: "0.0.0.0"  # Listen on all interfaces
    port: 5000
    telemetry_rate: 10.0  # Hz
    timeout_seconds: 2.0  # Connection timeout
```

## Implementation Sequence
1. Create package structure (setup.py, package.xml)
2. Implement protocol.py (encoding/decoding)
3. Implement wifi_comm_node.py (TCP server + ROS2 bridge)
4. Create wifi_params.yaml
5. Create wifi.launch.py
6. Test with mock ground station
