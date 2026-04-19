# OCTANE Supervisor - Nodes & Files

## Core Files

### `state_machine.py`
**Pure Python state machine** - ROS2-agnostic logic layer.

**States:**
- `STANDBY` - Initialized, no control active
- `MANUAL` - Remote driving via ground station
- `AUTONOMOUS` - RL policy navigation active
- `FAULT` - E-suggestion (software fault) active

**Properties:**
- `navigation_enabled` - True when in AUTONOMOUS state
- `manual_enabled` - True when in MANUAL state
- `e_suggestion` - True when in FAULT state

**Usage:** Called by `mode_manager_node` to validate state transitions.

---

### `config/faults.yaml`
**Central fault configuration** - thresholds and conditions.

**Example:**
```yaml
battery_undervoltage:
  severity: critical
  auto_recover: false
  condition: "battery_voltage < 10.5"
  source: "/sensors/battery"
  threshold: 10.5
```

**Fields:**
- `severity` - critical, warning, info
- `auto_recover` - clears automatically when condition becomes false
- `condition` - Python expression evaluated by fault_checker_node
- `source` - ROS2 topic where sensor data arrives
- `threshold` - reference value for fault triggering

---

## Nodes

### `mode_manager_node`
**Handles mode commands and state transitions.**

**Subscribes:**
- `/supervisor/mode_command` (String) - manual, autonomous, standby, fault_reset
- `/supervisor/fault_signal` (String) - fault type names

**Publishes:**
- `/supervisor/state` (String) - current state name
- `/supervisor/navigation_enabled` (Bool)
- `/supervisor/manual_enabled` (Bool)
- `/supervisor/e_suggestion` (Bool)

**Logic:**
1. Receive mode command or fault signal
2. Call `StateMachine.transition()` with target mode
3. If success, update internal state and publish
4. If fault, always allow transition to FAULT

---

### `fault_checker_node`
**Evaluates fault conditions from sensor data.**

**Subscribes:**
- Dynamically subscribes to all topics listed in `faults.yaml` `source` field

**Publishes:**
- `/supervisor/fault_signal` (String) - fault name when condition triggers

**Logic:**
1. Load `faults.yaml` on startup
2. Subscribe to all sensor topics listed as sources
3. Every `fault_check_rate` Hz (default 10Hz):
   - Gather latest sensor values into eval context
   - Evaluate each fault's `condition` expression
   - If true, publish fault name to `/supervisor/fault_signal`
4. For auto-recover faults, automatically clear when condition becomes false

**Example:**
```
Fault: battery_undervoltage
Condition: "battery_voltage < 10.5"
Sensor data arrives: battery_voltage = 10.2
Evaluated: 10.2 < 10.5 → True → Publish "battery_undervoltage"
```

---

### `fault_manager_node`
**Aggregates fault signals and provides reset functionality.**

**Subscribes:**
- `/supervisor/fault_signal` (String) - from fault_checker_node and other detectors
- `/supervisor/fault_reset` (Empty) - reset command from ground station

**Publishes:**
- `/supervisor/fault_status` (String) - summary like "FAULTS_ACTIVE [critical]: battery_undervoltage"

**Logic:**
1. Maintain registry of all known faults (name, severity, active status, timestamp)
2. On fault received: mark as active, record timestamp
3. On reset command: only clear auto-recover faults, block if critical faults active
4. Periodically publish summary status
