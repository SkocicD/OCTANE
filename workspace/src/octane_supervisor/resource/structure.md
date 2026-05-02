# OCTANE Supervisor - Package Structure

## Directory Layout
```
octane_supervisor/
├── launch/
│   └── supervisor.launch.py      # Launches all 3 nodes
├── config/
│   └── faults.yaml               # 27 fault definitions
├── octane_supervisor/
│   ├── __init__.py
│   ├── state_machine.py          # Pure Python state machine class
│   └── nodes/
│       ├── __init__.py
│       ├── mode_manager_node.py  # State transitions
│       ├── fault_manager_node.py # Fault aggregation
│       └── fault_checker_node.py # Condition evaluation
├── resource/
│   ├── structure.md              # This file
│   ├── nodes.md                  # Node/file descriptions
│   └── flow.md                   # Data flow diagrams
├── setup.py
└── package.xml
```

## Build System
- **Build type:** ament_python (Python package)
- **Dependencies:** rclpy, std_msgs, geometry_msgs, sensor_msgs, octane_msgs
