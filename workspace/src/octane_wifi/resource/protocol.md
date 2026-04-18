# OCTANE WiFi - Protocol Specification

## Design Goal
**Minimal bandwidth** - every byte counts over wireless.

## Frame Format
```
+--------+--------+--------+------------+--------+
| Magic  | Type   | Length |  Payload   | CRC8   |
| 1B     | 1B     | 1B     | N bytes    | 1B     |
+--------+--------+--------+------------+--------+
```

**Total overhead:** Only 4 bytes (magic + type + length + CRC)

**Types (ASCII for debugging):**
- `T` (0x54) = Telemetry
- `C` (0x43) = Command
- `A` (0x41) = Acknowledgment
- `F` (0x46) = Fault Alert

---

## Message Types

### 1. Telemetry (T) - Rover → Ground

**Purpose:** Periodic state broadcast (10Hz default).

**Wire format:** `T + state_char + [optional fields]`

**State codes (single byte):**
| Char | Code | State |
|------|------|-------|
| `0` | 0x30 | STANDBY |
| `1` | 0x31 | MANUAL |
| `2` | 0x32 | AUTONOMOUS |
| `3` | 0x33 | FAULT |

**Optional fields (prefixed with marker):**
- `B` + 4 bytes (little-endian float) = Battery voltage
- `F` + 1 byte (fault char) = Active fault indicator

**Examples:**
```
Wire: [O][T][1][1]              = Manual mode, 4 bytes total
Wire: [O][T][5][2][B][<float>]  = Autonomous + battery, 7 bytes total
Wire: [O][T][3][1][F][b]        = Manual + battery fault, 5 bytes total
```

**Actual byte sequences:**
```
"Manual mode"       → 4F 54 01 31                    (4 bytes!)
"Auto + 11.4V battery" → 4F 54 05 32 42 00 E6 44 41 (9 bytes)
```

---

### 2. Command (C) - Ground → Rover

**Purpose:** Switch operation modes.

**Wire format:** `C + mode_char + estop_flag`

**Mode codes:**
| Char | Mode | Description |
|------|------|-------------|
| `0` | standby | No control active |
| `1` | manual | Remote joystick control |
| `2` | autonomous | RL policy navigation |
| `3` | fault_reset | Clear faults, return to standby |

**E-Stop flag:**
- `0` = Normal operation
- `1` = Emergency stop (cut power immediately)

**Examples:**
```
Wire: [O][C][2][1][0]  → Switch to manual mode
Wire: [O][C][2][2][1]  → Switch to autonomous + e-stop flag
Wire: [O][C][2][0][0]  → Return to standby
```

**Fixed size:** Always 5 bytes on wire.

---

### 3. Acknowledgment (A) - Rover → Ground

**Purpose:** Confirm command received.

**Wire format:** `A + success_flag`

**Success flag:**
- `1` = Command executed
- `0` = Invalid command / rejected

**Examples:**
```
Wire: [O][A][1][1]  → Success
Wire: [O][A][1][0]  → Rejected
```

**Fixed size:** Always 4 bytes on wire.

---

### 4. Fault Alert (F) - Rover → Ground

**Purpose:** Immediate critical fault notification (bypasses telemetry timer).

**Wire format:** `F + severity_char + fault_char`

**Severity codes:**
| Char | Level | Description |
|------|-------|-------------|
| `0` | Info | Advisory |
| `1` | Warning | Non-critical |
| `2` | Critical | E-suggestion / emergency stop |

**Fault char:** First letter of fault type (expandable in future):
- `b` = Battery fault (undervoltage/overcurrent)
- `m` = Motor fault (overcurrent/driver failure)
- `c` = CAN bus fault
- `w` = WiFi connection loss
- `a` = Actuator fault
- `d` = Depth camera error

**Examples:**
```
Wire: [O][F][3][2][b] → Critical battery fault
Wire: [O][F][3][1][m] → Warning: motor overcurrent
Wire: [O][F][3][2][c] → Critical: CAN bus dead
```

**Fixed size:** Always 5 bytes on wire.

---

## Size Comparison

| Message Type | Old Design (JSON) | New Design | Savings |
|--------------|-------------------|------------|---------|
| Telemetry (state only) | 23 bytes | **4 bytes** | 83% |
| Command | 21 bytes | **5 bytes** | 76% |
| ACK | 15 bytes | **4 bytes** | 73% |
| Fault Alert | 35 bytes | **5 bytes** | 86% |
| Telemetry + battery | 31 bytes | **9 bytes** | 71% |

**Typical telemetry loop (10Hz):**
- Old: 230 bytes/second
- New: 40 bytes/second
- **83% bandwidth reduction**

---

## CRC8 Algorithm

**Polynomial:** 0x07 (standard CRC-8)

**Purpose:** Detect corrupted packets over unreliable wireless.

**Coverage:** Header + payload (not including CRC itself).

---

## Extensibility

**Adding new fields:** Use marker-byte prefix pattern.

Example: Add odometer position:
```
Current telemetry: [T][1]                  (Manual)
Add position:      [T][1][X][<4-byte x>][Y][<4-byte y>]
```

**Adding new fault types:** Single char may run out. Future expansion:
```
Current: [F][2][b]            (Critical battery)
Future:  [F][2][b][0x01]      (Battery sub-type 01 = undervoltage)
```

---

## Implementation Notes

**Python encoding:**
```python
from octane_wifi.protocol import encode_telemetry, encode_command

# Send "Manual mode" with 11.4V battery
packet = encode_telemetry('MANUAL', battery=11.4)
# Returns: b'OT\x012B\xe6\x44\x00\x...' (9 bytes)

# Send "Switch to autonomous"
packet = encode_command('autonomous', estop=False)
# Returns: b'OC\x022\x00\x...' (5 bytes)
```

**Parsing incoming:**
```python
from octane_wifi.protocol import decode_message

msg = decode_message(rx_buffer)
if msg:
    if msg['type'] == 'command':
        print(f"Mode: {msg['mode']}, E-Stop: {msg['estop']}")
    elif msg['type'] == 'fault':
        print(f"CRITICAL: {msg['fault']} ({msg['severity']})")
```
