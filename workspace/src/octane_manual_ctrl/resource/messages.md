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

Published on `/drive/command`. Subscribed by `can_drive_node`.

```
float32 left_velocity    # -1.0 (full reverse) to 1.0 (full forward)
float32 right_velocity   # -1.0 (full reverse) to 1.0 (full forward)
```

**Examples:**
| Scenario       | left_velocity | right_velocity |
|----------------|---------------|----------------|
| Full forward   | 1.0           | 1.0            |
| Full reverse   | -1.0          | -1.0           |
| Spin left      | -0.6          | 0.6            |
| Forward right  | 0.4           | 1.0            |
| Stop           | 0.0           | 0.0            |

The CAN hardware node is responsible for scaling these to actual motor units (RPM, PWM, etc.).

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
