# octane_mapping — Architecture & Data Flow

## Overview

The mapping package is intentionally lightweight. Rather than building a live
TSDF map at runtime, it loads a pre-built static arena map and runs nav2 on top
of it. The AI policy handles all reactive obstacle avoidance from live camera
input; nav2's only job is to publish a continuously-updated path hint that
tells the AI which direction to head toward the current goal zone.

```
                   cameras.yaml (intrinsics + mount offsets)
                          |
  octane_perception       |
  ─────────────────       |
  rgb_camera_node  ──►  CameraFrame (RGB)   ─►  CameraFrameSplitter ─► Image + CameraInfo + TF
  depth_est_node   ──►  CameraFrame (depth) ─►  CameraFrameSplitter ─► Image + CameraInfo + TF
                                                        │
                                               TF tree (base_link → camera_N_frame)
                                               fed directly into AI input stack

  octane_mapping
  ──────────────
  map_server  ──►  static arena map (.pgm/.yaml)
                          │
                        nav2  ──►  /plan  (nav_msgs/Path, replanned continuously)
                          │               │
                    ESDF costmap          └──► AI policy conditioning input
                    (from static map)

  octane_ai (future)
  ──────────────────
  AI policy:  [live depth frames]  +  [/plan path hint]  ──►  DriveCommand
```

## Why static map + nav2 instead of live TSDF

- The Lunabotics arena layout is published by NASA ahead of time — zone
  boundaries and berm positions are known before the run starts.
- The AI policy (trained in Isaac Sim with domain-randomized terrain) handles
  reactive obstacle avoidance from live depth input. It does not need a
  persistent 3D map.
- nav2 path replanning on a static costmap is fast and reliable. The AI treats
  the `/plan` output as a soft goal-direction hint, not a hard constraint.
- This avoids the odom drift / TSDF accumulation complexity for a task where
  the field geometry is already known.

## Arena zones

Defined in `mapping.launch.py` as corner-to-corner rectangles in the arena
coordinate frame. Origin (0, 0) is the SW corner of the arena; X points east
(away from berm), Y points north.

Update these from the NASA field specification PDF before each competition.

| Zone | Description |
|------|-------------|
| `start` | Robot starting position, also where it returns to deposit |
| `excavation_nav` | Transition corridor between start and excavation area |
| `excavation` | Active digging area — regolith is here |
| `deposition` | Area directly in front of the berm where bucket is dumped |
| `berm` | The physical berm structure itself |

Zone parameters are loaded by the map_server and will be consumed by the
supervisor (mission sequencer) to determine which nav2 goal to send next.

## Nav2 path hint

nav2 publishes its current planned path on `/plan` (`nav_msgs/Path`).
The AI policy receives the next N waypoints from `/plan` transformed into
robot-relative coordinates as a conditioning input alongside camera frames.
If the AI deviates to avoid an obstacle, nav2 replans around the new robot
position automatically.

## Launch order

```
1. perception.launch.py   — cameras + DA3 depth estimation
2. mapping.launch.py      — map server + nav2 + zone params
3. supervisor.launch.py   — mission sequencer (sends goals to nav2)
4. octane_ai (future)     — AI policy node subscribing to /plan + camera frames
```

## Localization (future — octane_localization)

The static `odom → base_link` stub in the mapping launch will be replaced by
the localization package, which fuses:
- April tag detections (absolute position fixes)
- IMU (Jetson AGX Orin onboard BMI088, orientation + short-term dead reckoning)

April tags will be placed at known arena positions so the robot can correct
accumulated drift whenever one comes into camera view.
