# octane_mapping — Architecture & Data Flow

## Overview

The mapping package runs nvblox to produce a live 3D TSDF/ESDF view of what is
immediately around the robot from all 6 cameras.  It is not a persistent global
map — TSDF decay and radius clearing keep it as a rolling window centered on
the robot.  The AI policy uses the live depth frames directly for reactive
driving; nvblox's ESDF output is available for any obstacle distance queries.

```
  octane_perception
  ─────────────────
  rgb_camera_node ×5  ──►  CameraFrame (RGB)   ─┐
  astra_depth_node    ──►  CameraFrame (RGB)   ─┤
                                                 │  CameraFrameSplitter ─► sensor_msgs/Image
  depth_est_node  ×5  ──►  CameraFrame (depth) ─┤  (one per camera,      ► sensor_msgs/CameraInfo
  astra_depth_node    ──►  CameraFrame (depth) ─┘   per stream)          ► TF: base_link→cam_frame

  octane_mapping
  ──────────────
  nvblox_node  ◄──  Image + CameraInfo (×6 depth, ×6 color)
               ◄──  TF tree (base_link → each camera frame)
               ◄──  TF: odom → base_link  (stub now, real from octane_localization)
                │
                ├──► /nvblox/mesh          (live 3D mesh of surroundings)
                ├──► /nvblox/esdf_slice    (2D obstacle distance field)
                └──► /nvblox/map_slice     (2D occupancy slice)

  octane_localization (future)
  ────────────────────────────
  April tag detections + IMU ──► odom → base_link TF  (replaces stub)
  Zone manager               ──► current goal zone center (for AI conditioning)

  octane_ai (future)
  ──────────────────
  [live depth CameraFrames]  +  [goal zone coordinate]  ──►  DriveCommand
```

## Why nvblox as a rolling live view (not global map)

- The arena terrain (rocks, craters, regolith mounds) is unknown until seen.
- The AI handles reactive navigation; it does not consume the nvblox map directly.
- nvblox provides an ESDF that can be queried for obstacle proximity if needed.
- TSDF decay + map radius clearing keeps memory bounded and the view current as
  the robot moves through the arena.

## Arena zones

Defined in `mapping.launch.py` as corner-to-corner rectangles.  These are used
by `octane_localization`'s zone manager to determine which zone the robot is
currently in and what the next goal coordinate should be.

Origin (0, 0) is the SW corner of the arena; X east (away from berm), Y north.
Update from the NASA field spec PDF before each competition.

| Zone | Description |
|------|-------------|
| `start` | Robot starting position, also where it returns to deposit |
| `nav` | Transition corridor between start and excavation area |
| `excavation` | Active digging area — regolith is here |
| `deposition` | Area directly in front of the berm where bucket is dumped |
| `berm` | The physical berm structure itself |

## nvblox configuration

Key parameters tuned for a rolling live view on the Jetson AGX Orin:

| Parameter | Value | Why |
|-----------|-------|-----|
| `voxel_size` | 0.05 m | 5 cm resolution — enough detail for obstacles |
| `num_cameras` | 6 | All 5 near + 1 Orbbec |
| `map_clearing_radius_m` | 4.0 m | Discard voxels beyond 4 m from robot |
| `tsdf_decay_factor` | 0.95 | Voxels fade when not re-observed |
| `esdf_mode` | 2d | Ground robot, 2D slice is sufficient |
| `max_integration_distance_m` | 3.5 m | Matches camera range at process_res=392 |

## Launch order

```
1. perception.launch.py   — cameras + DA3 depth estimation
2. mapping.launch.py      — camera splitters + nvblox live view
3. localization (future)  — April tags + IMU → odom TF + zone manager
4. octane_ai (future)     — AI policy node
```
