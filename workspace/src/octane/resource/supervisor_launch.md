# Supervisor Launch - supervisor.launch.py

**State machine and fault management system for the OCTANE rover.**

## Overview

The supervisor subsystem manages all rover operational states and monitors for fault conditions. It is the "brain" that coordinates mode transitions and ensures safe operation.

## Launch Location

`workspace/src/octane/octane/launch/supervisor.launch.py`

## Nodes Launched

### 1. `mode_manager_node`

**Purpose:** Handles mode commands and state transitions

**Functionality:**
- Receives mode commands from ground station via TCP
- Validates state transitions (e.g., STANDBY → MANUAL is valid, STANDBY → FAULT is always allowed)
- Publishes current state to `/supervisor/state`
- Manages `navigation_enabled` and `manual_enabled` flags

**State Machine:**
```
STANDBY ←→ MANUAL
   ↑ ↓       ↑
   └─FAULT←──┘
```

**Transitions:**
- `STANDBY → MANUAL` - Enable remote driving
- `STANDBY → AUTONOMOUS` - Enable RL policy navigation
- `STANDBY → FAULT` - Emergency stop (any state)
- `MANUAL → STANDBY` - Disable remote control
- `AUTONOMOUS → STANDBY` - Disable navigation
- `FAULT → STANDBY` - Reset after fault cleared

**Topics:**
- **Subscribes:** `/supervisor/mode_command` (String)
- **Publishes:** `/supervisor/state` (String)

**Parameters:**
- `check_rate` (Hz) - State check frequency (default: 10.0)

---

### 2. `fault_manager_node`

**Purpose:** Aggregates fault signals and provides reset functionality

**Functionality:**
- Maintains registry of all known faults (name, severity, active status, timestamp)
- Receives fault signals from `fault_checker_node` and other detectors
- Handles fault reset commands from ground station
- Only clears auto-recover faults on reset; blocks if critical faults remain active

**Fault Severities:**
- `info` - Minor issue, logs but doesn't trigger fault state
- `warning` - Noticeable issue, may escalate if unaddressed
- `critical` - Immediate fault, triggers emergency stop

**Topics:**
- **Subscribes:** `/supervisor/fault_signal` (String)
- **Publishes:** `/supervisor/fault_status` (String)

**Parameters:**
- `check_rate` (Hz) - Status update frequency (default: 10.0)

---

### 3. `fault_checker_node`

**Purpose:** Evaluates fault conditions from sensor data

**Functionality:**
- Loads `faults.yaml` configuration on startup
- Subscribes dynamically to all sensor topics listed in config
- Every `fault_check_rate` Hz:
  - Gathers latest sensor values into evaluation context
  - Evaluates each fault's condition expression
  - If true, publishes fault name to `/supervisor/fault_signal`
- For auto-recover faults, automatically clears when condition becomes false

**Example Fault Evaluation:**
```yaml
# Configuration
battery_undervoltage:
  severity: critical
  auto_recover: false
  condition: "battery_voltage < 10.5"
  source: "/sensors/battery/voltage"

# Runtime
Sensor data: battery_voltage = 10.2
Evaluated: 10.2 < 10.5 → TRUE
Result: Publish "battery_undervoltage" to /supervisor/fault_signal
```

**Topics:**
- **Subscribes:** Dynamically based on `faults.yaml` `source` fields
- **Publishes:** `/supervisor/fault_signal` (String)

**Parameters:**
- `faults_config` (string) - Path to faults.yaml
- `fault_check_rate` (Hz) - Condition check frequency (default: 10.0)

---

### 4. `state_monitor_node` ⭐ NEW

**Purpose:** Displays state changes in a clean, readable format

**Functionality:**
- Subscribes to `/supervisor/state` and prints formatted transitions
- Shows timestamped FROM → TO state changes
- Displays visual indicators for each state
- **Automatically opens in separate terminal window** when supervisor.launch.py runs

**Example Output:**
```
==================================================
    OCTANE ROVER STATE MONITOR (Supervisor)
==================================================
    Watching: /supervisor/state
    Press Ctrl+C to stop

[1234567.890] STATE CHANGE:
    FROM: UNKNOWN        
    TO:   STANDBY       
--------------------------------------------------
    ⏸️  Status: READY - Awaiting command

[1234568.120] STATE CHANGE:
    FROM: STANDBY        
    TO:   MANUAL        
--------------------------------------------------
    🎮 Status: MANUAL CONTROL ACTIVE
```

**Topics:**
- **Subscribes:** `/supervisor/state` (String)

**Auto-Launch:**
```bash
# Automatically opens in new terminal
ros2 launch octane supervisor.launch.py
```

**Manual Launch:**
```bash
# Opens in new terminal window (Windows)
start cmd /k ros2 run octane_supervisor state_monitor_node

# Or same terminal
ros2 run octane_supervisor state_monitor_node
```

---

## Fault Configuration

Located at: `workspace/src/octane_supervisor/octane_supervisor/config/faults.yaml`

**Supported Fault Types:**
- `battery_undervoltage` - Critical, manual recovery
- `battery_overvoltage` - Critical, manual recovery
- `motor_overcurrent` - Critical, manual recovery
- `sensor_timeout` - Warning, auto-recover
- `communication_lost` - Warning, auto-recover

**Fault Condition Syntax:**
- Python expressions using sensor variables
- Example: `battery_voltage < 10.5 or current > 30.0`

---

## Launch Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `check_rate` | 10.0 | Node check rate in Hz |
| `fault_check_rate` | 10.0 | Fault detection check rate in Hz |

---

## Usage Examples

**Launch supervisor with default settings:**
```bash
ros2 launch octane supervisor.launch.py
```

**Launch with custom check rate:**
```bash
ros2 launch octane supervisor.launch.py check_rate:=20.0 fault_check_rate:=50.0
```

**Watch state changes in real-time:**
```bash
# New terminal automatically opens with state monitor
ros2 launch octane supervisor.launch.py

# Or manually attach state monitor
ros2 run octane_supervisor state_monitor_node
```

---

## State Transition Flow

```
┌─────────────────────────────────────────────────┐
│           Ground Station Command                │
│              (click "Manual")                   │
└───────────────┬─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────┐
│         network_comm_node (TCP)                 │
│    Receives: [O][C][2][1][0][crc]               │
│    Decodes: {mode: 'manual'}                    │
└───────────────┬─────────────────────────────────┘
                │ Publishes to ROS2
                ▼
┌─────────────────────────────────────────────────┐
│        mode_manager_node                        │
│    1. Receive "MANUAL" on /mode_command         │
│    2. Validate: STANDBY → MANUAL ✓             │
│    3. StateMachine.transition(Mode.MANUAL)      │
│    4. Publish "MANUAL" to /supervisor/state     │
└───────────────┬─────────────────────────────────┘
                │ State change
                ▼
┌─────────────────────────────────────────────────┐
│        state_monitor_node                       │
│    [1234568.120] STATE CHANGE:                  │
│        FROM: STANDBY                            │
│        TO:   MANUAL                             │
│    🎮 Status: MANUAL CONTROL ACTIVE             │
└─────────────────────────────────────────────────┘
```

---

## See Also

- `../octane.md` - Main octane package documentation
- `../../octane_supervisor/resource/nodes.md` - Detailed node API reference
- `../../octane_supervisor/resource/faults.md` - Fault type definitions
