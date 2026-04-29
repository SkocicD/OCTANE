# OCTANE Manual Control

Manual control interpretation package for the OCTANE Lunabotics rover.

Bridges raw keyboard input from the ground station GUI into structured
drive and actuator commands. Sits between the network layer and the
hardware interface nodes — it knows nothing about hardware, only about
what the operator intends.

See also:
- `nodes.md` - Detailed node descriptions and topic reference
- `flow.md` - Data flow from keypress to hardware command
- `messages.md` - DriveCommand and ActuatorCommand message definitions
