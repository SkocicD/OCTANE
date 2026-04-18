# OCTANE WiFi - TCP Protocol

## Message Frame Format
```
+--------+--------+--------+----------+----------+
| Magic  | Type   | Seq    | Length   | Payload  |
| 2B     | 1B     | 2B     | 4B       | N bytes  |
+--------+--------+--------+----------+----------+
| CRC32  |
| 4B     |
+--------+
```

**Header (9 bytes):**
- Magic: `0x4F 0x54` ("OT" for OCTANE)
- Type: 1 byte (see below)
- Seq: 2 bytes big-endian sequence number
- Length: 4 bytes big-endian payload size

**Payload:** JSON with compact formatting (no spaces)

**CRC32:** 4 bytes, calculated over header + payload

## Message Types
| Type | Value | Direction | Description |
|------|-------|-----------|-------------|
| Telemetry | 0x01 | Rover → Ground | Periodic state broadcast |
| Command | 0x02 | Ground → Rover | Mode command |
| Command ACK | 0x03 | Rover → Ground | Command acknowledgment |
| Fault Alert | 0x04 | Rover → Ground | Immediate fault notification |

## Payload Formats

### Telemetry (0x01)
```json
{"t":"STANDBY","f":"battery_undervoltage","b":11.4}
```
- `t` = state (STANDBY, MANUAL, AUTONOMOUS, FAULT)
- `f` = active fault (optional)
- `b` = battery voltage (optional)

### Command (0x02)
```json
{"m":"manual","e":false}
```
- `m` = mode (standby, manual, autonomous, fault_reset)
- `e` = emergency stop flag

### Command ACK (0x03)
```json
{"s":true,"seq":42}
```
- `s` = success (true/false)
- `seq` = original command sequence number

### Fault Alert (0x04)
```json
{"f":"battery_undervoltage","s":"critical"}
```
- `f` = fault type
- `s` = severity (critical, warning, info)

## Compact Field Codes
| Code | Field | Type | Description |
|------|-------|------|-------------|
| t | state/mode | string | Current state or target mode |
| m | mode | string | Command mode |
| f | fault | string | Fault type |
| b | battery | float | Battery voltage |
| s | severity/success | string/bool | Severity or success flag |
| e | estop | bool | Emergency stop flag |
| seq | sequence | int | Sequence number |
