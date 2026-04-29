# OCTANE network - Message Reference

Quick reference for all message formats used in rover-ground communication.

---

## Telemetry (Rover → Ground)

**Purpose:** Broadcast current state, battery, and fault status.

**Trigger:** 10Hz timer (every 100ms)

**Type code:** `T` (0x54)

**Wire format:** `T + state_byte + [optional fields]`

### State Codes
| Byte | Char | State | Description |
|------|------|-------|-------------|
| 0x30 | `0` | STANDBY | System initialized, no control active |
| 0x31 | `1` | MANUAL | Remote joystick control enabled |
| 0x32 | `2` | AUTONOMOUS | RL policy navigation active |
| 0x33 | `3` | FAULT | E-suggestion active, power cut |

### Optional Field Markers

**Battery Voltage:**
- Marker: `B` (0x42)
- Data: 4 bytes (little-endian float)
- Example: `11.4V` → `B` + `\x00\xe6\x44\x41`

**Active Fault Indicator:**
- Marker: `F` (0x46)
- Data: 1 byte (first char of fault type)
- Fault type codes:
  - `b` = Battery (undervoltage/overcurrent)
  - `m` = Motor (overcurrent/driver failure)
  - `c` = CAN bus (communication lost)
  - `w` = network (connection timeout)
  - `a` = Actuator (position/heartbeat fault)
  - `d` = Depth camera (sensor error)

### Examples

**Minimal telemetry (state only):**
```
Bytes: [T][1]
Decoded: State = MANUAL
Size: 2 bytes
```

**State + battery:**
```
Bytes: [T][2][B][\x00][\xe6][\x44][\x41]
Decoded: State = AUTONOMOUS, Battery = 11.4V
Size: 7 bytes
```

**State + fault indicator:**
```
Bytes: [T][1][F][b]
Decoded: State = MANUAL, Active fault = Battery
Size: 4 bytes
```

---

## Command (Ground → Rover)

**Purpose:** Request mode change from ground station.

**Trigger:** Operator button press or automated script

**Type code:** `C` (0x43)

**Wire format:** `C + mode_byte + estop_byte`

### Mode Codes
| Byte | Char | Mode | Effect |
|------|------|------|--------|
| 0x30 | `0` | standby | Disable all control, return to idle |
| 0x31 | `1` | manual | Enable joystick/remote control |
| 0x32 | `2` | autonomous | Enable RL policy navigation |
| 0x33 | `3` | fault_reset | Clear faults, return to standby |

### E-Stop Flag
| Byte | Char | Meaning |
|------|------|---------|
| 0x30 | `0` | Normal operation |
| 0x31 | `1` | Emergency stop (cut power immediately) |

### Examples

**Switch to manual:**
```
Bytes: [C][1][0]
Decoded: Mode = manual, E-Stop = false
Size: 3 bytes
```

**Switch to autonomous:**
```
Bytes: [C][2][0]
Decoded: Mode = autonomous, E-Stop = false
Size: 3 bytes
```

**Emergency stop:**
```
Bytes: [C][0][1]
Decoded: Mode = standby (to stop), E-Stop = true
Size: 3 bytes
```

---

## Manipulator Command (Ground → Rover)

**Purpose:** Relay the current held-key state from the ground station to the rover for real-time drive and arm control.

**Trigger:** Sent continuously at 20 Hz while the rover is in Manual mode. Every frame reflects the full current hold state — downstream nodes handle velocity conversion.

**Type code:** `M` (0x4D)

**Wire format:** `M + bitfield`

### Bitfield Layout

| Bit | Key | Action |
|-----|-----|--------|
| 0 | W | Drive forward |
| 1 | A | Drive left |
| 2 | S | Drive backward |
| 3 | D | Drive right |
| 4 | ↑ | Arm up |
| 5 | ↓ | Arm down |
| 6 | ← | Bucket rotate left |
| 7 | → | Bucket rotate right |

Multiple bits may be set simultaneously. `0x00` means all keys released.

### Examples

**Drive forward + right:**
```
Bytes: [M][0x09]
Decoded: keys = 0b00001001 (W + D)
Size: 2 bytes payload
```

**Arm up only:**
```
Bytes: [M][0x10]
Decoded: keys = 0b00010000 (↑)
```

**All released:**
```
Bytes: [M][0x00]
Decoded: keys = 0x00 — no inputs held
```

**ROS topic:** `/manual_control/keys` (`std_msgs/UInt8`) — raw bitfield, published every received frame.

---

## Acknowledgment (Rover → Ground)

**Purpose:** Confirm command received and executed.

**Trigger:** Automatic after receiving Command message

**Type code:** `A` (0x41)

**Wire format:** `A + success_byte`

### Success Codes
| Byte | Char | Status |
|------|------|--------|
| 0x31 | `1` | Command accepted and executed |
| 0x30 | `0` | Command rejected (invalid mode, state conflict) |

### Examples

**ACK success:**
```
Bytes: [A][1]
Decoded: Command accepted
Size: 2 bytes
```

**ACK rejection:**
```
Bytes: [A][0]
Decoded: Command rejected (e.g., tried manual → autonomous directly)
Size: 2 bytes
```

---

## Fault Alert (Rover → Ground)

**Purpose:** Immediate notification of critical fault (bypasses telemetry timer).

**Trigger:** Any fault detector publishes to `/supervisor/fault_signal`

**Type code:** `F` (0x46)

**Wire format:** `F + severity_byte + fault_char`

### Severity Codes
| Byte | Char | Level | Action Required |
|------|------|-------|-----------------|
| 0x30 | `0` | Info | Log only, no immediate action |
| 0x31 | `1` | Warning | Investigate soon, system still operational |
| 0x32 | `2` | Critical | E-suggestion triggered, immediate attention required |

### Fault Type Codes
| Char | Fault Category | Specific Faults |
|------|----------------|-----------------|
| `b` | Battery | Undervoltage (<10.5V), Overcurrent (>30A) |
| `m` | Motor | Overcurrent, Driver failure, Thermal shutdown |
| `c` | CAN Bus | Communication lost, Heartbeat timeout, ACK failure |
| `w` | network | Connection lost, Timeout >2s |
| `a` | Actuator | Position mismatch, Driver failure, Timeout |
| `d` | Depth Camera | Sensor error, Timeout, Calibration loss |
| `l` | Localization | AprilTag not detected, Loss of confidence |
| `s` | Software | ML inference crash, State machine deadlock |

### Examples

**Critical battery undervoltage:**
```
Bytes: [F][2][b]
Decoded: Severity = critical, Fault = Battery
Action: E-suggestion engaged, operator must investigate
Size: 3 bytes
```

**Warning motor overcurrent:**
```
Bytes: [F][1][m]
Decoded: Severity = warning, Fault = Motor
Action: Monitor, system still operational
Size: 3 bytes
```

**Info network timeout:**
```
Bytes: [F][0][w]
Decoded: Severity = info, Fault = network
Action: Check connection, no critical action needed
Size: 3 bytes
```

---

## Complete Frame Layout (All Types)

```
+--------+--------+--------+------------+--------+
| Magic  | Type   | Length |  Payload   | CRC8   |
| 1B     | 1B     | 1B     | N bytes    | 1B     |
+--------+--------+--------+------------+--------+
```

**Total overhead:** 4 bytes per message

**Min message size:** 4 bytes (magic + type + length=0 + CRC)

**Max practical size:** ~30 bytes (before fragmentation concerns)

---

## Field Legend

| Symbol | Type | Size | Description |
|--------|------|------|-------------|
| Magic | uint8 | 1B | Always 0x4F ('O') |
| Type | uint8 | 1B | Message type (T/C/A/F) |
| Length | uint8 | 1B | Payload byte count |
| CRC8 | uint8 | 1B | CRC-8 checksum |
| state_byte | uint8 | 1B | State code (0-3) |
| mode_byte | uint8 | 1B | Mode code (0-3) |
| estop_byte | uint8 | 1B | E-stop flag (0/1) |
| severity_byte | uint8 | 1B | Fault severity (0-2) |
| fault_char | uint8 | 1B | Fault type code (a-w) |
| battery | float32 | 4B | Voltage in volts (little-endian) |

---

## Quick Lookup: State Machine Valid Transitions

| From → To | Standby | Manual | Autonomous | Fault |
|-----------|---------|--------|------------|-------|
| **Standby** | ✓ | ✓ | ✓ | ✓ |
| **Manual** | ✓ | ✗ | ✗ | ✓ |
| **Autonomous** | ✓ | ✗ | ✗ | ✓ |
| **Fault** | ✓ (reset) | ✗ | ✗ | ✗ |

Commands that violate this table will be ACK'd with `success=false`.
