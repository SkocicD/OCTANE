# OCTANE Manual Control - Nodes

## `manual_drive_node`
**Interprets WASD keypresses into tank drive velocity commands.**

**Subscribes:**
- `/manual_ctrl/key_state` (std_msgs/UInt8) - raw key bitfield from network node
- `/supervisor/state` (std_msgs/String) - supervisor state gate

**Publishes:**
- `/drive/command` (octane_msgs/DriveCommand) - left/right normalized velocity

**Key bitfield (from GUI):**
| Bit | Key | Action        |
|-----|-----|---------------|
| 0   | W   | Forward       |
| 1   | A   | Turn left     |
| 2   | S   | Backward      |
| 3   | D   | Turn right    |

**Tank drive mixing:**
- W/S sets base throttle on both sides (+1.0 / -1.0)
- A/D applies differential: reduces inside wheel, increases outside
- Combined (e.g. W+D): forward right turn
- Output clamped to [-1.0, 1.0]

**Parameters:**
- `throttle_scale` (default: 1.0) - max throttle output
- `turn_scale` (default: 0.6) - differential turn strength

**Note:** `speed_modifier` is not set by this node — it publishes with the message default (100).
Speed dial control comes from the GUI publishing its own DriveCommand or a separate topic.

**Gate:** Only publishes when `/supervisor/state == "MANUAL"`.
On exit from MANUAL, immediately publishes a zero-velocity stop command.

---

## `can_drive_node`
**Translates DriveCommand into CANOpen PDO commands for 6 BLDC motors.**

**Subscribes:**
- `/drive/command` (octane_msgs/DriveCommand) - velocity target + speed modifier
- `/supervisor/state` (std_msgs/String) - supervisor state gate

**Publishes:**
- `/manual_ctrl/can_status` (std_msgs/String, latched) - transceiver status
- `/manual_ctrl/can_tx` (std_msgs/String) - per-frame TX log

**Motor layout (CANOpen node IDs 1–6):**
| Node ID | Position    | Side  |
|---------|-------------|-------|
| 1       | Front-left  | Left  |
| 2       | Mid-left    | Left  |
| 3       | Back-left   | Left  |
| 4       | Back-right  | Right |
| 5       | Mid-right   | Right |
| 6       | Front-right | Right |

Right-side motors run reversed (mirrored mount). Node 4 has an additional flip.

**Parameters** (loaded from `octane/config/robot_params.yaml` via launch file):
- `ramp_time_up` (default: 0.33) - seconds to ramp from 0 to 100% throttle
- `ramp_time_down` (default: 0.33) - seconds to ramp from 100% to 0 throttle
- `dead_band` (default: 0.02) - min speed change before sending a new PDO
- `speed_scale` (default: 0.2) - global speed cap multiplier (1.0 = full motor RPM)
- `bitrate` (default: 1000000) - CAN bus bitrate in bps

**Speed calculation:**
```
effective = velocity × speed_scale × (speed_modifier / 100.0)   [clamped 0–1]
```

**Gate:** Only drives motors when `/supervisor/state == "MANUAL"`.
Ramps to zero on exit from MANUAL rather than cutting immediately.

---

## `rs485_drive_node`
**Sends Modbus RTU commands to a BLD-510B BLDC driver over RS485 (replaces broken CAN motor #2).**

**Subscribes:**
- `/drive/command` (octane_msgs/DriveCommand) - uses `left_velocity` + `speed_modifier`
- `/supervisor/state` (std_msgs/String) - supervisor state gate

**Publishes:**
- `/manual_ctrl/rs485_status` (std_msgs/String, latched) - transceiver status
- `/manual_ctrl/rs485_tx` (std_msgs/String) - per-command TX log

**Hardware:** BLD-510B on `/dev/rs485_drive` (udev symlink for CH340 VID `1a86:7523`).
Modbus RTU 8N1, CRC16 (poly 0xA001). Key registers:

| Register | Address | Description                     |
|----------|---------|---------------------------------|
| Control  | 0x8000  | High byte: ctrl flags; low byte: pole pairs |
| Speed    | 0x8005  | Target RPM (0–65535)            |
| Actual   | 0x8018  | Actual speed (read-only)        |
| Fault    | 0x801B  | Fault state (read-only)         |

**Parameters** (loaded from `octane/config/robot_params.yaml` via launch file):
- `ramp_time_up` (default: 0.33) - seconds to ramp from 0 to 100% throttle
- `ramp_time_down` (default: 0.33) - seconds to ramp from 100% to 0 throttle
- `dead_band` (default: 0.02) - min speed change before re-sending a command
- `speed_scale` (default: 0.2) - global speed cap multiplier
- `port` (default: `/dev/rs485_drive`) - serial device
- `baud_rate` (default: 9600)
- `modbus_address` (default: 1)
- `max_rpm` (default: 3000)
- `pole_pairs` (default: 4)
- `reverse` (default: false) - flip direction if motor is wired backwards

**Gate:** Only drives motor when `/supervisor/state == "MANUAL"`.
Ramps to zero on exit from MANUAL.

---

## `manual_actuator_node`
**Interprets arrow keypresses into arm and bucket relay commands.**

**Subscribes:**
- `/manual_ctrl/key_state` (std_msgs/UInt8) - raw key bitfield from network node
- `/supervisor/state` (std_msgs/String) - supervisor state gate

**Publishes:**
- `/actuator/command` (octane_msgs/ActuatorCommand) - arm and bucket direction

**Key bitfield (from GUI):**
| Bit | Key    | Action          |
|-----|--------|-----------------|
| 4   | ↑      | Arm up          |
| 5   | ↓      | Arm down        |
| 6   | ←      | Bucket dir-A    |
| 7   | →      | Bucket dir-B    |

**Command values per axis:** -1 (reverse), 0 (stop), 1 (forward)

Conflicting inputs (both directions held): resolve to 0 (stop).

**Gate:** Only publishes when `/supervisor/state == "MANUAL"`.
On exit from MANUAL, immediately publishes a full-stop command (arm=0, bucket=0).
