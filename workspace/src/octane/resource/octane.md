# OCTANE Package

**Top-level orchestration package for the CSU Lunabotics rover.**

The `octane` package serves as the central entry point for launching and managing all rover subsystems. It contains launch files that coordinate multiple ROS2 packages working together.

## Purpose

- **Centralized launch management** - Single entry point for all subsystems
- **Package orchestration** - Coordinates supervisor, perception, mapping, and network communication
- **Configuration hub** - Shared parameters and settings for all subsystems

## Structure

```
octane/
├── launch/           # Launch files for all subsystems
├── config/           # Shared configuration files
└── resource/         # Documentation (this folder)
```

## Launch Files

### `supervisor.launch.py`
**State machine and fault management system**

Launches:
- `mode_manager_node` - Handles mode transitions (standby/manual/autonomous/fault)
- `fault_manager_node` - Aggregates fault signals and provides reset functionality  
- `fault_checker_node` - Evaluates fault conditions from sensor data
- `state_monitor_node` - **Auto-opens in new terminal** displaying real-time state changes

**Topics:**
- `/supervisor/state` - Current mode state (STANDBY, MANUAL, AUTONOMOUS, FAULT)
- `/supervisor/mode_command` - Mode commands from ground station
- `/supervisor/fault_signal` - Fault alerts from detectors
- `/supervisor/fault_status` - Active fault summary

**Usage:**
```bash
ros2 launch octane supervisor.launch.py
```

---

### `perception.launch.py`
**Camera drivers and depth estimation**

Launches:
- `rgb_camera_node` - RGB camera stream
- `astra_depth_node` - Orbbec Astra depth sensor
- `depth_estimation_node` - DA3 depth estimation algorithm

**Topics:**
- `/camera/rgb/image_raw` - RGB image stream
- `/camera/depth/image_raw` - Depth image stream
- `/camera/depth/camera_info` - Camera calibration

**Usage:**
```bash
ros2 launch octane perception.launch.py
```

---

### `mapping.launch.py`
**3D terrain mapping with nvblox**

Launches:
- `nvblox_node` - TSDF volumetric mapping
- `camera_frame_splitter` - Splits CameraFrame messages for nvblox compatibility

**Topics:**
- `/mapping/tsdf_map` - 3D terrain map
- `/mapping/mesh` - Extracted mesh output

**Usage:**
```bash
ros2 launch octane mapping.launch.py
```

---

### `network.launch.py`
**Ground station communication (TCP + UDP heartbeat)**

Launches:
- `network_comm_node` - TCP server for mode commands and telemetry
- `heartbeat_sender` - UDP heartbeat for connection monitoring

**Protocol:**
- Binary framing with CRC-8 validation
- TCP port 5000 - Mode commands and telemetry
- UDP port 5001 - Heartbeat packets (300ms interval)

**Topics:**
- `/supervisor/state` - Published to ground station
- `/supervisor/mode_command` - Receives from ground station

**Usage:**
```bash
ros2 launch octane network.launch.py
```

---

## Configuration Files

### `config/cameras.yaml`
Camera parameters and calibration settings

### `config/system.yaml`
System-wide configuration and subsystem flags

### `config/supervisor_params.yaml`
Fault thresholds and state machine parameters

---

## Quick Start

**Launch entire rover stack:**
```bash
ros2 launch octane full_stack.launch.py  # When created
```

**Launch individual subsystems:**
```bash
ros2 launch octane supervisor.launch.py
ros2 launch octane perception.launch.py
ros2 launch octane mapping.launch.py
ros2 launch octane network.launch.py
```

---

## Dependencies

- `octane_supervisor` - State machine and fault management
- `octane_perception` - Camera drivers and depth estimation
- `octane_mapping` - nvblox 3D mapping
- `octane_network` - Ground station communication
- `octane_msgs` - Custom ROS2 message types
