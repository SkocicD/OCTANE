# OCTANE Manual Control - Messages

## Key State Bitfield (UInt8)

Published by `network_comm_node` on `/manual_ctrl/key_state`.
Originates from the GUI's `GetKeyBitfield()` method.

| Bit | Mask | Key    | Meaning        |
|-----|------|--------|----------------|
| 0   | 0x01 | W      | Forward        |
| 1   | 0x02 | A      | Turn left      |
| 2   | 0x04 | S      | Backward       |
| 3   | 0x08 | D      | Turn right     |
| 4   | 0x10 | ↑      | Arm up         |
| 5   | 0x20 | ↓      | Arm down       |
| 6   | 0x40 | ←      | Bucket dir-A   |
| 7   | 0x80 | →      | Bucket dir-B   |

Multiple keys can be held simultaneously (bits OR'd together).

---

## DriveCommand (octane_msgs/DriveCommand)

Published on `/drive/command`. Subscribed by `can_drive_node` and `rs485_drive_node`.

```
float32 left_velocity    # -1.0 (full reverse) to 1.0 (full forward)
float32 right_velocity   # -1.0 (full reverse) to 1.0 (full forward)
uint16  speed_modifier   # GUI speed dial: 0–500 (percentage), default 100
```

**`speed_modifier` semantics:**

An integer percentage applied on top of the hardware `speed_scale` config value (default 0.2).
`100` = no change from baseline. Field defaults to `100` so keyboard-only operation is unaffected.

```
effective_speed = velocity × speed_scale × (speed_modifier / 100.0)
                                                      clamped to [0.0, 1.0]
```

| `speed_modifier` | multiplier | effective max speed (speed_scale=0.2) |
|-----------------|------------|---------------------------------------|
| 0               | 0.0×       | motors stopped                        |
| 50              | 0.5×       | 10% of full motor RPM                 |
| 100             | 1.0×       | 20% of full motor RPM (normal)        |
| 250             | 2.5×       | 50% of full motor RPM                 |
| 500             | 5.0×       | 100% of full motor RPM (hard cap)     |

GUI should keep `speed_modifier` in range `0–100` for safe operation. Values above 100 up to 500 are supported for testing or special use cases.

**Drive command examples:**
| Scenario       | left_velocity | right_velocity | speed_modifier |
|----------------|---------------|----------------|----------------|
| Full forward   | 1.0           | 1.0            | 100            |
| Full reverse   | -1.0          | -1.0           | 100            |
| Spin left      | -0.6          | 0.6            | 100            |
| Half speed     | 1.0           | 1.0            | 50             |
| Stop           | 0.0           | 0.0            | 100            |

The hardware nodes scale velocity by `speed_scale × (speed_modifier / 100.0)` before sending to motor hardware.

---

## ActuatorCommand (octane_msgs/ActuatorCommand)

Published on `/actuator/command`. Subscribed by `gpio_actuator_node`.

```
int8 arm      # -1 (down), 0 (stop), 1 (up)
int8 bucket   # -1 (dir-A), 0 (stop), 1 (dir-B)
```

Maps directly to two-channel relay states:

| Value | Arm relay state          | Bucket relay state       |
|-------|--------------------------|--------------------------|
| 1     | up-relay ON, down OFF    | dir-B relay ON, A OFF    |
| 0     | both OFF                 | both OFF                 |
| -1    | down-relay ON, up OFF    | dir-A relay ON, B OFF    |
