# CSU Lunabotics — Isaac Lab RL Training Reference

## Quick Start

```bat
cd E:\IsaacLab
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lunabotics-Direct-v0 --num_envs 1024
```

Play back a trained policy (16 envs, with viewer):
```bat
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Lunabotics-Direct-v0 --num_envs 16 --load_run <run_folder_name>
```

Trained policies are saved to:
```
E:\IsaacLab\logs\rsl_rl\lunabotics_direct\
```

The Isaac Lab training viewport shows a ground grid — **each square is 1 meter**. Use the robot spacing (12 m) as a cross-check.

---

## Config Files

### 1. Robot Asset
`E:\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\lunabotics.py`

| Parameter | Value | What it does |
|---|---|---|
| `usd_path` | path to this USD folder | Robot USD to load |
| `scale` | `(0.1, 0.1, 0.1)` | See **USD Scale** section below |
| `pos` (init_state) | `(0.0, 0.0, 0.75)` | Spawn height — re-run `find_resting_height.py` if USD changes |
| `saturation_effort` | `20,000` Nm | Motor torque cap — high because density-based mass scales with volume at scale=0.1 |
| `effort_limit` | `20,000` Nm | Isaac Lab clamp — keep equal to saturation_effort |
| `velocity_limit` | `36.7` rad/s | Software limit (10× operating max) |
| `damping` | `6,000` | DCMotor gain: torque = 6000 × (vel_target − vel_actual), must exceed saturation/vel_error |

### 2. Environment Config
`E:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\lunabotics\lunabotics_env_cfg.py`

| Parameter | Value | What it does |
|---|---|---|
| `episode_length_s` | `20.0` s | Max episode length before reset |
| `decimation` | `4` | Physics steps per RL step (sim dt=0.005 s → policy dt=0.02 s) |
| `action_scale` | `3.67` rad/s | Maps policy output [-1,1] to wheel velocity targets. 35 RPM = 3.67 rad/s |
| `num_envs` | `1024` | Parallel training environments. Reduce if CUDA OOM |
| `env_spacing` | `12.0` m | Wider spacing required because robot is physically large at scale=0.1 |
| `wheel_radius` | `0.91` m | Used for physics-grounded slip detection (see Termination below) |
| `solver_type` | `0` (PGS) | PGS required for skid-steer — TGS has a confirmed velocity-reporting bug |
| `restitution` | `0.0` | Ground bounciness — zero, no bounce |

**Reward scales** (positive = reward, negative = penalty):

| Scale | Value | Effect |
|---|---|---|
| `lin_vel_reward_scale` | `1.5` | Reward for matching commanded forward speed |
| `yaw_rate_reward_scale` | `1.5` | Reward for matching commanded turn rate |
| `z_vel_reward_scale` | `-2.0` | Penalize vertical bouncing |
| `ang_vel_reward_scale` | `-0.05` | Penalize roll/pitch oscillation |
| `flat_orientation_reward_scale` | `-1.0` | Penalize tipping/tilting |
| `action_rate_reward_scale` | `-0.2` | Penalize jerky commands — enforces smooth accel/decel |
| `excessive_yaw_reward_scale` | `-1.0` | Penalize yaw rate > 1.5 rad/s — prevents spin-bounce |
| `lateral_vel_reward_scale` | `-3.0` | Penalize body-Y (sideways) drift — skid-steer cannot strafe |
| `overspeed_reward_scale` | `-3.0` | Penalize XY speed above 0.65 m/s |

**Termination conditions** (episode resets immediately):
- Robot tips beyond 0.9 rad lean
- Body XY speed exceeds `max_wheel_tangential × 2 + 0.05` — catches physics-launched robots (arm swing, collision explosion)

**Curriculum** (command complexity grows automatically):
- Phase 0 (~0–333 iters): straight only, ±0.3 m/s — learn heading-locked movement
- Phase 1 (~333–750 iters): 80% straight / 20% pure pivot, ±0.4 m/s
- Phase 2 (750+ iters): full mix — straight, pivots, arcs, ±0.5 m/s / ±1.0 rad/s

### 3. Environment Implementation
`E:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\lunabotics\lunabotics_env.py`

Key things to know:
- **Left wheels are negated**: `action=(+1,+1)` → left_vel=−3.67, right_vel=+3.67 → forward. Empirically confirmed sign convention for this robot.
- **Joint order is NOT [L,L,L,R,R,R]**: USD traversal order is `[L_Rear, R_Rear, R_Center, R_Front, L_Center, L_Front]`. The code builds a left/right mask from joint names at startup — do not assume index-based splitting.
- **Exponential smoothing** (α=0.4) on wheel velocity targets prevents policy from exploiting physics ratcheting via rapid oscillations.
- **Wheel synchrony** is enforced in software: all 3 left wheels always receive the same target, all 3 right wheels always receive the same target.

### 4. PPO Hyperparameters
`E:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\lunabotics\agents\rsl_rl_ppo_cfg.py`

| Parameter | Value | What it does |
|---|---|---|
| `num_steps_per_env` | `48` | Steps per rollout (~1 s of rover motion) |
| `max_iterations` | `1500` | Total training iterations |
| `save_interval` | `100` | Save checkpoint every N iterations |
| `entropy_coef` | `0.005` | Exploration noise — lower = more exploitation |
| `learning_rate` | `1e-3` | Adam LR (adaptive schedule) |
| `num_learning_epochs` | `5` | PPO update passes per rollout |

---

## USD Scale

Isaac Lab does **not** auto-apply metersPerUnit conversion. The `scale` parameter in `UsdFileCfg` is the only scaling mechanism — it applies a raw Xform transform at spawn time.

**This USD has metersPerUnit=0.01 but uses `scale=(0.1, 0.1, 0.1)`.** This is intentional:

The CAD was modeled in millimeters, then imported into Isaac Sim whose stage is in centimeters. Isaac Sim stamped metersPerUnit=0.01 (cm) on export, but the actual coordinate values are millimeter-scale — making every dimension appear 10× too small when interpreted as centimeters. The scale=0.1 correction compensates for this mislabeling. Two wrongs that cancel: wrong unit label on export + scale=0.1 in Isaac Lab = correct physical size.

**If you re-export the USD:**
- Model and export in centimeters (not mm) from your CAD tool, OR
- Set Isaac Sim's stage to metersPerUnit=0.001 (mm) before exporting
- Then change scale back to `(0.01, 0.01, 0.01)` and re-run `find_resting_height.py`

**After any scale change**, always re-run:
```bat
isaaclab.bat -p find_resting_height.py
```
Then update `pos=(0.0, 0.0, <stable_Z + margin>)` in `lunabotics.py`. Margin = ~20% of stable Z.

---

## USD Requirements

For a USD to work correctly with this training setup:

- **metersPerUnit = 0.01** stamped in file (actual geometry in mm-scale — see USD Scale above)
- **Articulation root** on `base_link`, `fixedBase = False`, self-collisions disabled
- **Wheel joint names** must exactly match: `Left_Front_Wheel`, `Left_Center_Wheel`, `Left_Rear_Wheel`, `Right_Front_Wheel`, `Right_Center_Wheel`, `Right_Rear_Wheel`
- **Arm/non-drive collision disabled** on all bodies except chassis + 6 wheels
- **No OmniGraph / Action Graph nodes** — delete all ROS2 graphs before export
- **Passive arm joints** (revolute linkages with no drives) are fine — they are free-moving by design

---

## Tools

`C:\Users\adam.carbone\source\repos\ros_intro\NVIDIA Isaac\Isaac Sim\Tools\`

| Script | What it does |
|---|---|
| `check_usd_compliance.py` | Read-only audit: units, joints, drives, collision, mass |
| `disable_nonwheel_collision.py` | Disables collision on all bodies except chassis + wheels |
| `lock_arm_joints.py` | Adds position-hold drives to arm joints |
| `deactivate_omnigraphs.py` | Deactivates all OmniGraph/ROS2 prims |
| `run_usd_tools.bat` / `.sh` | Runs all 4 tools in sequence |
| `inspect_usd_structure.py` | Dumps articulation roots, joints, rigid bodies, OmniGraph prims to file |
| `find_resting_height.py` | Drops robot and reports stable Z — use after any scale change |

```bat
run_usd_tools.bat "C:\path\to\robot.usd"
```
