# OCTANE network - Protocol Specification

## Design Goal
**Minimal bandwidth** - every byte counts over wireless.

## Frame Format

Every message on the wire has this structure:

```
+--------+--------+--------+------------+--------+
| Magic  | Type   | Length |  Payload   | CRC8   |
| 1B     | 1B     | 1B     | N bytes    | 1B     |
+--------+--------+--------+------------+--------+
```

**The 5 fields:**

| Field | Size | Purpose |
|-------|------|---------|
| **Magic** | 1 byte | Always `0x4F` ('O') - validates we're reading OCTANE protocol, not random noise |
| **Type** | 1 byte | Message type: `T`=Telemetry, `C`=Command, `A`=ACK, `F`=Fault Alert |
| **Length** | 1 byte | How many bytes in the Payload field (0-255) |
| **Payload** | N bytes | The actual data (state, mode, fault code, etc.) |
| **CRC8** | 1 byte | Error detection - if this doesn't match, packet is corrupted |

**Total overhead:** Only 4 bytes per message (magic + type + length + CRC)

**Types (ASCII for debugging):**
- `T` (0x54) = **Telemetry** - Rover broadcasts state to ground
- `C` (0x43) = **Command** - Ground sends mode change to rover
- `A` (0x41) = **ACK** - Rover confirms command received
- `F` (0x46) = **Fault Alert** - Rover immediately reports critical fault
- `V` (0x56) = **Video Request** - Ground requests camera/map stream (see `video_streaming.md`)

---

## Message Types

### 1. Telemetry (T) - Rover → Ground

**Purpose:** Periodic state broadcast

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
Wire: [O][T][1][1][crc] = Manual mode, 5 bytes total
Wire: [O][T][6][2][B][<float>][crc] = Autonomous + battery, 10 bytes total
Wire: [O][T][3][1][F][b][crc] = Manual + fault, 6 bytes total
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
Wire: [O][C][2][1][0][crc] = Switch to manual
Wire: [O][C][2][2][1][crc] = Switch to autonomous + e-stop
Wire: [O][C][2][0][0][crc] = Return to standby
```

**Fixed size:** Always 6 bytes on wire (5 + CRC).

---

### 3. Acknowledgment (A) - Rover → Ground

**Purpose:** Confirm command received.

**Wire format:** `A + success_flag`

**Success flag:**
- `1` = Command executed
- `0` = Invalid command / rejected

**Examples:**
```
Wire: [O][A][1][1][crc] = Success
Wire: [O][A][1][0][crc] = Rejected
```

**Fixed size:** Always 5 bytes on wire (4 + CRC).

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
- `w` = network connection loss
- `a` = Actuator fault
- `d` = Depth camera error

**Examples:**
```
Wire: [O][F][2][2][b][crc] = Critical battery fault
Wire: [O][F][2][1][m][crc] = Warning motor overcurrent
Wire: [O][F][2][2][c][crc] = Critical CAN bus dead
```

**Fixed size:** Always 6 bytes on wire (5 + CRC).

---

## Size Comparison (with full frame including CRC)

| Message Type | Old Design (JSON) | New Design (binary + CRC) | Savings |
|--------------|-------------------|---------------------------|---------|
| Telemetry (state only) | 23 bytes | **5 bytes** | 78% |
| Command | 21 bytes | **6 bytes** | 71% |
| ACK | 15 bytes | **5 bytes** | 67% |
| Fault Alert | 35 bytes | **6 bytes** | 83% |
| Telemetry + battery | 31 bytes | **10 bytes** | 68% |

**Typical telemetry loop (10Hz):**
- Old: 230 bytes/second
- New: 50 bytes/second (state only)
- **78% bandwidth reduction**

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
from octane_network.protocol import encode_telemetry, encode_command

# Send "Manual mode" (state only)
packet = encode_telemetry('MANUAL')
# Returns: b'OT\x011D' (5 bytes: MAGIC+TYPE+LEN+STATE+CRC)

# Send "Manual mode" with 11.4V battery
packet = encode_telemetry('MANUAL', battery=11.4)
# Returns: b'OT\x051B\x66\x66\x36\x41\xfd' (10 bytes)

# Send "Switch to autonomous"
packet = encode_command('autonomous', estop=False)
# Returns: b'OC\x022\x00\xf3' (6 bytes: MAGIC+TYPE+LEN+MODE+ESTOP+CRC)
```

**Parsing incoming:**
```python
from octane_network.protocol import decode_message

msg = decode_message(rx_buffer)
if msg:
    if msg['type'] == 'command':
        print(f"Mode: {msg['mode']}, E-Stop: {msg['estop']}")
    elif msg['type'] == 'fault':
        print(f"CRITICAL: {msg['fault']} ({msg['severity']})")
```
