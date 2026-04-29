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

**Gate:** Only publishes when `/supervisor/state == "MANUAL"`.
On exit from MANUAL, immediately publishes a zero-velocity stop command.

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
