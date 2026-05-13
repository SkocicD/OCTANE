# OCTANE Manual Control - Data Flow

## Full Pipeline (GUI keypress → hardware command)

```
┌─────────────────────────────────────────────────────────────────┐
│ Ground Station GUI (50ms tick, Manual mode active)              │
│ Operator holds W + D, speed dial at 80%                        │
│ GetKeyBitfield() → 0b00001001 (bits 0,3 = W,D)                 │
│ speed_modifier → 80                                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │  M frame [4F][4D][01][bitfield][CRC]
                       │  TCP → octane.local:5000
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ network_comm_node                                               │
│ Decodes M frame → {'type': 'manipulator', 'bitfield': 0x09}    │
│ Publishes → /manual_ctrl/key_state (UInt8: 9)                  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌──────────────────┐    ┌──────────────────────────┐
│ manual_drive_node│    │ manual_actuator_node      │
│ state == MANUAL? │    │ state == MANUAL?          │
│ ✓ yes            │    │ ✓ yes                     │
│                  │    │                           │
│ bit0(W)=1 → +1.0 │    │ bits 4-7 all 0            │
│ bit3(D)=1 → turn │    │ arm=0, bucket=0           │
│                  │    │                           │
│ left  = 1.0-0.6  │    │ Publishes stop (no input) │
│       = 0.4      │    └──────────────────────────┘
│ right = clamped  │
│       = 1.0      │
│ speed_modifier   │
│       = 80       │  ← GUI sets this (default 100)
│                  │
│ Publishes:       │
│ left_vel=0.4     │
│ right_vel=1.0    │
│ speed_mod=80     │
└──────────┬───────┘
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ /drive/command (DriveCommand)                                   │
│ left_velocity: 0.4   right_velocity: 1.0   speed_modifier: 80  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────────────┐
│ can_drive_node       │  │ rs485_drive_node                     │
│ Ramps to target      │  │ Ramps left_velocity to target        │
│ effective =          │  │ effective =                          │
│  vel × 0.2 × 0.80   │  │  vel × 0.2 × 0.80                   │
│ Sends PDO to 5 CAN  │  │ Sends Modbus RTU to BLD-510B         │
│ motors (1,3,4,5,6)  │  │ motor #2 (mid-left, /dev/rs485_drive)│
└──────────────────────┘  └──────────────────────────────────────┘
```

## Supervisor Gate

```
┌─────────────────────────────────────────────────────────────────┐
│ /supervisor/state                                               │
│ data: "MANUAL"                                                  │
└────────┬────────────────────────────────────────────────────────┘
         │  subscribed by both manual ctrl nodes
         ▼
  manual_active = (state == "MANUAL")
         │
  True ──┼──→ publish commands normally
         │
  False ─┼──→ ignore all key state messages
         │    (on transition: publish stop immediately)
         ▼
```

## State Transition: Exit from Manual

```
Operator clicks Standby
  → GUI sends mode command
  → supervisor transitions MANUAL → STANDBY
  → publishes /supervisor/state = "STANDBY"
  → manual_drive_node: was_active=True, now False → publish stop (0.0, 0.0)
  → manual_actuator_node: was_active=True, now False → publish stop (0, 0)
  → CAN/RS485/GPIO nodes receive stop before any further key state arrives
```

## Topic Map

```
/manual_ctrl/key_state  ──→  manual_drive_node  ──→  /drive/command
                         └─→  manual_actuator_node ─→  /actuator/command

/supervisor/state        ──→  manual_drive_node  (gate)
                         └─→  manual_actuator_node (gate)

/drive/command           ──→  can_drive_node     → CAN motors 1,3,4,5,6
                         └─→  rs485_drive_node   → motor 2 (BLD-510B, RS485)

/actuator/command        ──→  serial_actuator_node → Arduino relay board
```

## Drive Config (robot_params.yaml)

Shared parameters loaded by both `can_drive_node` and `rs485_drive_node`:

| Parameter        | Default | Description                               |
|------------------|---------|-------------------------------------------|
| `ramp_time_up`   | 0.33 s  | Time to ramp from 0% to 100% throttle    |
| `ramp_time_down` | 0.33 s  | Time to ramp from 100% to 0% throttle    |
| `dead_band`      | 0.02    | Min speed change before re-sending        |
| `speed_scale`    | 0.2     | Global speed cap (1.0 = full motor RPM)  |
