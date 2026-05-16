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
| `/manual_ctrl/key_state` | std_msgs/UInt8 | Key bitfield from M frame (bits 0-7) |
| `/manual_ctrl/speed_modifier` | std_msgs/UInt16 | Speed dial from M frame (0-500, default 100) |
| `/network/client_ip` | std_msgs/String | GUI IP on connect, empty string on disconnect |
| `/network/stream_request` | std_msgs/String | Video request forwarded to video_stream_node |

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

---

## `video_stream_node`

On-demand JPEG video streamer. Subscribes to camera topics, compresses frames,
and sends UDP chunks to the GUI. Activated by `/network/stream_request`.

### Parameters
| Parameter | Default | Description |
|---|---|---|
| `udp_port` | 5002 | UDP destination port on GUI |
| `default_scale` | 50 | % scale used when GUI sends scale=0 |
| `jpeg_quality` | 70 | JPEG encode quality (1–100) |
| `depth_max_m` | 8.0 | DA3 depth clip distance (m) |
| `orbbec_depth_max_m` | 5.0 | Orbbec depth clip distance (m) |

### ROS2 Subscribers
| Topic | Type | Description |
|---|---|---|
| `/network/client_ip` | std_msgs/String | GUI IP from network_comm_node |
| `/network/stream_request` | std_msgs/String | `"source_id,variant,scale,fps"` |
| *(camera topics)* | octane_msgs/CameraFrame | Subscribed lazily on request |

See `resource/video_streaming.md` for full protocol and source ID reference.

---

## Files

- `nodes/heartbeat_sender.py` - UDP heartbeat sender node
- `nodes/network_comm_node.py` - TCP bridge node
- `nodes/video_stream_node.py` - UDP video/map streamer
- `classes/protocol.py` - Binary protocol encoder/decoder (shared utilities)
- `config/network_params.yaml` - Node parameters
- `resource/video_streaming.md` - Video streaming protocol reference
