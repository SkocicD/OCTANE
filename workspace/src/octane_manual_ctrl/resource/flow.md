# OCTANE Manual Control - Data Flow

## Full Pipeline (GUI keypress → hardware command)

```
┌─────────────────────────────────────────────────────────────────┐
│ Ground Station GUI (50ms tick, Manual mode active)              │
│ Operator holds W + D                                            │
│ GetKeyBitfield() → 0b00001001 (bits 0,3 = W,D)                 │
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
│ right = 1.0+0.6  │
│       = clamped  │
│       = 1.0      │
│                  │
│ Publishes:       │
│ left_vel=0.4     │
│ right_vel=1.0    │
└──────────┬───────┘
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ /drive/command (DriveCommand)                                   │
│ left_velocity: 0.4   right_velocity: 1.0                       │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ can_drive_node  [TODO: implement]                               │
│ Scales to motor units, sends CAN frames to 6 motors            │
│ left side (FL, ML, RL): 0.4 * MAX_RPM                          │
│ right side (FR, MR, RR): 1.0 * MAX_RPM                         │
└─────────────────────────────────────────────────────────────────┘
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
  → CAN/GPIO nodes receive stop before any further key state arrives
```

## Topic Map

```
/manual_ctrl/key_state  ──→  manual_drive_node  ──→  /drive/command
                         └─→  manual_actuator_node ─→  /actuator/command

/supervisor/state        ──→  manual_drive_node  (gate)
                         └─→  manual_actuator_node (gate)

/drive/command           ──→  can_drive_node     [TODO]
/actuator/command        ──→  gpio_actuator_node [TODO]
```
