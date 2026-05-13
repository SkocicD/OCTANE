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

**IMU Accelerometer:**
- Marker: `I` (0x49)
- Data: 12 bytes (3 × little-endian float32: x, y, z in m/s²)
- Source: ADXL345 on `sensors/imu/accel`
- Example: 0g idle → `I` + `\x00\x00\x00\x00` + `\x00\x00\x00\x00` + `\x1e\x85\x1c\x41` (z ≈ 9.81)

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

## Manipulator (Ground → Rover)

**Purpose:** Deliver key state bitfield and GUI speed dial value on every manual control tick.

**Trigger:** GUI manual control timer (e.g. 50ms tick) while in Manual mode.

**Type code:** `M` (0x4D)

**Wire format:** `M + bitfield + speed_hi + speed_lo`

### Key Bitfield
| Bit | Mask | Key | Action       |
|-----|------|-----|--------------|
| 0   | 0x01 | W   | Forward      |
| 1   | 0x02 | A   | Turn left    |
| 2   | 0x04 | S   | Backward     |
| 3   | 0x08 | D   | Turn right   |
| 4   | 0x10 | ↑   | Arm up       |
| 5   | 0x20 | ↓   | Arm down     |
| 6   | 0x40 | ←   | Bucket dir-A |
| 7   | 0x80 | →   | Bucket dir-B |

### Speed Modifier
`speed_modifier` is a big-endian uint16 (2 bytes). Range 0–500 (integer percentage).
`100` = 1.0× (baseline). `500` = 5.0× (full motor RPM). GUI should cap at 100 for safe operation.

Effective speed = `velocity × speed_scale × (speed_modifier / 100.0)`, clamped to [0.0, 1.0].

### Examples

**W+D held, speed dial at 100%:**
```
Bytes: [M][0x09][0x00][0x64]
Decoded: W=1 D=1, speed_modifier=100
```

**No keys, speed dial at 50%:**
```
Bytes: [M][0x00][0x00][0x32]
Decoded: (no keys), speed_modifier=50
```

**Fixed size:** Always 3 bytes payload, 7 bytes total on wire.

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
| accel_x/y/z | float32 | 4B each | Acceleration in m/s² (little-endian) |

---

## Quick Lookup: State Machine Valid Transitions

| From → To | Standby | Manual | Autonomous | Fault |
|-----------|---------|--------|------------|-------|
| **Standby** | ✓ | ✓ | ✓ | ✓ |
| **Manual** | ✓ | ✗ | ✗ | ✓ |
| **Autonomous** | ✓ | ✗ | ✗ | ✓ |
| **Fault** | ✓ (reset) | ✗ | ✗ | ✗ |

Commands that violate this table will be ACK'd with `success=false`.
