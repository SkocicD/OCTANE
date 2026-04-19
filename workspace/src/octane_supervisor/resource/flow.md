# OCTANE Supervisor - Data Flow

## Overview Flow
```
┌─────────────────────────────────────────────────────────────────┐
│ Ground Station (WiFi)                                           │
│ Sends: mode_command (manual, autonomous, standby, fault_reset) │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ /supervisor/mode_command (std_msgs/String)                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ mode_manager_node                                               │
│ 1. Parse mode command                                           │
│ 2. Call StateMachine.transition(target_mode)                    │
│ 3. Validate transition permissions                              │
│ 4. Update state if allowed                                      │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ StateMachine                                                    │
│ States: STANDBY ↔ MANUAL                                        │
│         ↓ ↑ ↓ ↑                                                 │
│         AUTONOMOUS  FAULT                                       │
│ Valid transitions:                                              │
│ - STANDBY → MANUAL, AUTONOMOUS, FAULT                          │
│ - MANUAL → STANDBY, FAULT (NOT → AUTONOMOUS directly)          │
│ - AUTONOMOUS → STANDBY, FAULT                                  │
│ - FAULT → STANDBY (only after reset + cleared)                 │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Publishers (10Hz)                                               │
│ /supervisor/state → "MANUAL"                                    │
│ /supervisor/navigation_enabled → false                          │
│ /supervisor/manual_enabled → true                               │
│ /supervisor/e_suggestion → false                                │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Motor Control / Navigation Nodes                                │
│ Check flags before accepting commands                           │
│ If e_suggestion=true → emergency stop, ignore all inputs        │
└─────────────────────────────────────────────────────────────────┘
```

## Fault Detection Flow (Parallel Path)
```
┌─────────────────────────────────────────────────────────────────┐
│ Sensor Topics                                                   │
│ /sensors/battery, /sensors/motor, /sensors/cpu, etc.           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ fault_checker_node                                              │
│ 1. Load faults.yaml (27 fault definitions)                     │
│ 2. Subscribe to all source topics listed in config              │
│ 3. Every 10Hz, evaluate conditions:                            │
│    Example: "battery_voltage < 10.5"                            │
│    When sensor data: battery_voltage = 10.2                     │
│    Evaluated: 10.2 < 10.5 → TRUE → FAULT TRIGGERED             │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ /supervisor/fault_signal (std_msgs/String)                     │
│ data: "battery_undervoltage"                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ mode_manager_node (fault_signal_callback)                       │
│ 1. Receive fault type name                                      │
│ 2. Call StateMachine.transition(Mode.STANDBY, fault_type=...)   │
│ 3. Always allow transition to FAULT state                       │
│ 4. Record active fault type                                     │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ FAULT State Active                                              │
│ state = "FAULT"                                                 │
│ navigation_enabled = false                                      │
│ manual_enabled = false                                          │
│ e_suggestion = true  ← E-suggestion activated                   │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Motor Control / Navigation Nodes                                │
│ See e_suggestion = true → CUT POWER, emergency stop             │
└─────────────────────────────────────────────────────────────────┘
```

## Fault Recovery Flow
```
┌─────────────────────────────────────────────────────────────────┐
│ Operator detects fault condition resolved (e.g., battery fixed) │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ /supervisor/mode_command                                        │
│ data: "fault_reset"                                             │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ mode_manager_node                                               │
│ 1. Validate FAULT → STANDBY transition                          │
│ 2. Check if faults are cleared (optional TODO)                  │
│ 3. Allow transition, clear fault state                          │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STANDBY State Active                                            │
│ state = "STANDBY"                                               │
│ navigation_enabled = false                                      │
│ manual_enabled = false                                          │
│ e_suggestion = false  ← E-suggestion cleared                    │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Ready for new mode command (manual/autonomous)                  │
└─────────────────────────────────────────────────────────────────┘
```

## Concurrent Operation Diagram
```
Time →
┌────────────────────────────────────────────────────────────────────┐
│ mode_manager_node                                                  │
│ State: STANDBY → MANUAL → FAULT → STANDBY                         │
│                                                                    │
│ fault_checker_node                                                 │
│ Monitoring sensors ──────────────────────────────[FAULT DETECTED]→│
│                                                                    │
│ Publishers (10Hz)                                                  │
│ e_suggestion:  F ────────→ F ──────────────────→ T ───→ F         │
│ manual_enabled: F ────────→ T ─────────────────→ F ───→ F         │
│ navigation:     F ────────→ F ─────────────────→ F ───→ F         │
│                                                                    │
│ Motor Control Node                                                 │
│ Action:         Idle ──────→ Accept Joystick ──→ STOP ─→ Idle     │
└────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Separation of concerns:** `state_machine.py` is pure Python, no ROS2 dependencies
2. **YAML-driven config:** Fault thresholds externalized, no recompilation needed
3. **Safe transitions:** Cannot go MANUAL → AUTONOMOUS directly (must go through STANDBY)
4. **Fault priority:** FAULT state is always reachable from any state
5. **E-suggestion naming:** Distinguishes software faults from hardware E-Stop button
