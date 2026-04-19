# OCTANE Network - Data Flow

Complete message flow from ROS2 topics to network transmission.

---

## Ground Station → Rover (Mode Command)

```
┌─────────────────┐
│ Ground Station  │
│   (C# GUI)      │
└────────┬────────┘
         │
         │ 1. User clicks "Manual" button
         │ 2. Send binary frame: [O][C][2][1][0][crc]
         │    - MAGIC=0x4F, TYPE=0x43 (C), LEN=2
         │    - MODE='1' (manual), ESTOP='0'
         │    - CRC8 checksum
         ▼
┌─────────────────────────────────────────┐
│     TCP Socket (port 5000)              │
└────────┬────────────────────────────────┘
         │
         │ 3. Raw bytes arrive at rover
         ▼
┌─────────────────────────────────────────────────┐
│  network_comm_node.py (ROS2 Node)               │
│                                                 │
│  recv_loop() receives bytes → adds to buffer    │
│  process_buffer() accumulates until full frame  │
│  decode_message() validates CRC, extracts:      │
│    → {'type': 'command', 'mode': 'manual'}      │
└────────┬────────────────────────────────────────┘
         │
         │ 4. Publish to ROS2 topic
         │    Topic: /supervisor/mode_command
         │    Type: std_msgs/String
         │    Data: "MANUAL"
         ▼
┌─────────────────────────────────────────────────┐
│  mode_manager_node.py (ROS2 Node)               │
│                                                 │
│  mode_command_callback() receives "MANUAL"      │
│  StateMachine.transition(Mode.MANUAL)           │
│  Validates transition (e.g., STANDBY→MANUAL)    │
└────────┬────────────────────────────────────────┘
         │
         │ 5. State change confirmed
         ▼
┌─────────────────────────────────────────────────┐
│  ROS2 Topic: /supervisor/state                  │
│  Data: "MANUAL"                                 │
└────────┬────────────────────────────────────────┘
         │
         │ 6. State broadcast to all subscribers
         ▼
┌─────────────────────────────────────────────────┐
│  network_comm_node.py (still running)           │
│  state_callback() receives "MANUAL"             │
│  Updates current_state = "MANUAL"               │
└─────────────────────────────────────────────────┘
```

---

## Rover → Ground Station (Telemetry)

```
┌─────────────────────────────────────────────────┐
│  mode_manager_node.py                           │
│  State transitions to MANUAL                    │
│  Publishes to /supervisor/state: "MANUAL"       │
└────────┬────────────────────────────────────────┘
         │
         │ 1. State change event
         ▼
┌─────────────────────────────────────────────────┐
│  network_comm_node.py                           │
│  state_callback() stores: current_state="MANUAL"│
└────────┬────────────────────────────────────────┘
         │
         │ 2. Timer fires (10Hz = every 100ms)
         ▼
┌─────────────────────────────────────────────────┐
│  send_telemetry()                               │
│                                                 │
│  encode_telemetry('MANUAL', fault=None)         │
│  → Returns: b'OT\x011D' (5 bytes)               │
│     - 0x4F (O) = MAGIC                          │
│     - 0x54 (T) = TYPE_TELEMETRY                 │
│     - 0x01 = LENGTH (1 byte payload)            │
│     - 0x31 ('1') = STATE_MANUAL                 │
│     - 0x44 = CRC8                               │
└────────┬────────────────────────────────────────┘
         │
         │ 3. Send binary frame over TCP
         ▼
┌─────────────────────────────────────────┐
│  TCP Socket                             │
└────────┬────────────────────────────────┘
         │
         │ 4. Bytes transmitted to ground station
         ▼
┌─────────────────────────────────────────────────┐
│  NetworkModeClient.cs (Ground Station)          │
│  ReceiveLoop() accumulates bytes in buffer      │
│  NetworkProtocol.DecodeFrame() validates CRC    │
│    → DecodedMessage { Type="telemetry", State='1'} │
└────────┬────────────────────────────────────────┘
         │
         │ 5. Map state code to string
         │    '1' → "MANUAL"
         ▼
┌─────────────────────────────────────────────────┐
│  MainView.axaml.cs                              │
│  OnRosStateReceived("MANUAL")                   │
│  Updates UI: LED turns green, shows "MANUAL"    │
└─────────────────────────────────────────────────┘
```

---

## Key Design Principles

1. **ROS2 topics are local only**: 
   `/supervisor/mode_command`, `/supervisor/state` etc. are just in-memory 
   message buses within the rover's ROS2 system.

2. **`network_comm_node` is the bridge**:
   - **Outbound**: Subscribes to ROS2 topics → encodes to binary → sends over TCP
   - **Inbound**: Receives TCP bytes → decodes binary → publishes to ROS2 topics

3. **Binary protocol efficiency**:
   - JSON: `{"type":"mode_command","mode":"manual"}` = ~35 bytes
   - Binary: `[O][C][2][1][0][crc]` = **6 bytes** (83% reduction)

4. **CRC ensures integrity**: 
   Every frame has CRC8 check - corrupted packets are discarded.

5. **No JSON on the wire**: 
   The TCP connection between rover and ground station uses **only** the lean 
   binary protocol from `protocol.py`/`NetworkProtocol.cs`.

---

## Transport Layer Considerations

### Current Implementation: All TCP

The current design uses TCP for **all** communication:
- Mode commands (Ground → Rover)
- State acknowledgments (Rover → Ground)
- Telemetry updates (Rover → Ground)
- Fault alerts (Rover → Ground)

### Potential UDP for Passive Telemetry

**Question**: Should passive telemetry (sensor data, camera streams, 3D map updates) 
use UDP instead of TCP?

**Rationale for UDP**:
- **No retransmission delay**: Lost sensor packets are stale anyway
- **Lower latency**: No TCP handshake or congestion control
- **Better for high-frequency data**: 30+ Hz sensor updates, camera frames
- **Graceful degradation**: Missing a few telemetry packets is acceptable

**Recommendation for future**:
- **TCP**: Mode commands, state changes, fault alerts (reliability critical)
- **UDP**: Sensor telemetry, camera frames, mapping data (real-time visualization)

This hybrid approach would give the best of both worlds: reliable command/control 
with low-latency streaming for visualization data.

---

## Heartbeat System (Connection Health Monitoring)

To detect connection loss without relying on TCP state, the rover sends periodic
UDP heartbeat packets.

### Implementation

**Rover side**: `heartbeat_sender.py` node
- Sends UDP broadcast every 300ms (3.33 Hz)
- Minimal frame format: `[MAGIC][STATE][SEQ][CRC]` = 5 bytes
- Subscribes to `/supervisor/state` to include current mode

**Heartbeat frame format**:
```
Byte 0: MAGIC = 0x4F ('O')
Byte 1: STATE = '0'(STANDBY), '1'(MANUAL), '2'(AUTONOMOUS), '3'(FAULT)
Byte 2-3: SEQ = 16-bit sequence number (wraps at 65535)
Byte 4: CRC = CRC-8 of first 4 bytes

Total: 5 bytes
```

**GUI side**: `NetworkModeClient.cs` should listen for heartbeats
- Track last received heartbeat timestamp
- If no heartbeat for 6.5+ seconds, set `IsConnected = false`
- When heartbeat resumes, set `IsConnected = true`

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `heartbeat_rate_hz` | 0.33 | Heartbeat frequency (1/3 Hz = 300ms interval) |
| `udp_port` | 5001 | UDP port for heartbeat (separate from TCP port 5000) |
| `timeout_seconds` | 6.5 | GUI timeout before marking connection lost |

**Recommendation**: GUI timeout should be at least 2x the heartbeat interval
to account for occasional packet loss. With 300ms interval and 6.5s timeout,
the GUI can miss ~21 consecutive heartbeats before declaring disconnection.

---

## File References

- `protocol.py` - Binary frame encoding/decoding (TCP messages)
- `heartbeat_sender.py` - UDP heartbeat sender node
- `NetworkProtocol.cs` - Binary frame encoding/decoding (C#)
- `network_comm_node.py` - ROS2 TCP bridge node
- `NetworkModeClient.cs` - Ground station TCP client
- `messages.md` - Message payload reference
- `protocol.md` - Wire format specification
