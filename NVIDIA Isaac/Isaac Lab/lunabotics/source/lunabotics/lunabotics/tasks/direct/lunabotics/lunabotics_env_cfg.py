"""Direct RL environment config for the CSU Lunabotics 6-wheel skid-steer rover.

Clean research-grounded implementation based on legged_gym / WheeledLab / ANYmal standards.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from lunabotics.assets.lunabotics import LUNABOTICS_DIRECT_CFG  # isort: skip


@configclass
class LunaboticsDirectEnvCfg(DirectRLEnvCfg):
    # ── env ───────────────────────────────────────────────────────────────────
    episode_length_s: float = 20.0
    episode_length_max_s: float = 300.0            # 5 min cap
    # Episode length scales with how well the robot drives.  As the EMA of
    # forward speed approaches target_speed, episode_length_s ramps linearly
    # from 20s → 300s.  Early on (robot barely moves) short episodes avoid
    # wasted compute; once it drives well it gets longer runs to practice
    # sustained navigation.
    episode_length_target_speed: float = 0.45  # m/s — full episode length at this speed
    decimation: int = 4
    action_scale: float = 3.665     # 35 RPM = 3.665 rad/s — Isaac Lab velocity targets are in rad/s
    action_space: int = 2           # [forward, turn_rate]
    observation_space: int = 15     # +2 for heading (cos, sin)
    state_space: int = 0

    # ── simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physx=PhysxCfg(
            solver_type=0,                    # PGS — TGS has confirmed velocity-reporting bug
            gpu_collision_stack_size=2**28,   # 256 MB — sufficient at 0.25m terrain resolution (~16K tris/mesh)
            gpu_max_rigid_patch_count=2**18,  # 262144 — default overflows at high env counts (reported need: 203K)
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── infinite flat ground plane (visual backdrop + physics fallback) ───────
    # Plane mesh is at {prim_path}/terrain; translated to z=-0.8 in _setup_scene
    # to sit just below the arena foundation geometry.
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ── per-env regolith terrain mesh ─────────────────────────────────────────
    # A height-field mesh is generated once per env and spawned under each
    # env's namespace (/World/envs/env_N/Ground) BEFORE clone_environments().
    # All terrain is per-env — no shared global ground plane.
    # Tune these to change surface roughness.
    # size = (X_extent, Y_extent).  Arena long axis is along world Y, so size[0] < size[1].
    regolith_size: tuple = (27.0, 38.0)             # m — long axis along world Y
    regolith_horizontal_scale: float = 0.25         # 25 cm/cell — ~16K triangles/mesh vs 263K at 0.0625; fine detail handled by PBR texture
    regolith_vertical_scale: float = 0.001         # 1 mm/count
    regolith_particle_density_range: tuple = (0.5, 2.5)   # mounds/m²
    regolith_particle_radius_range: tuple = (0.40, 2.0)   # m
    regolith_crest_height_range: tuple = (0.02, 0.15)     # m

    # ── spawn zone (2×2 m start zone, bottom-left of arena) ─────────────────
    # Robot spawns at random position + random yaw within this rectangle each
    # episode.  Coordinates are relative to env origin (arena center).
    # Set zone center/size to match your arena layout.  The margin insets the
    # actual spawn area so the robot doesn't clip the zone boundary walls.
    #
    # NASA Lunabotics arena ≈ 5 m × 7.5 m.  Bottom-left = -X, -Y corner.
    # Adjust these if your sim arena has different dimensions.
    # Arena footprint at sim scale: ~26.75 m (X) × ~36.1 m (Y)
    # (real 5.35×7.22 m, scale=5.0, rotated 90° around Z)
    # Walls at approximately X = ±13.4 m, Y = ±18.0 m
    # Real 2×2 m start zone → 10×10 m at 5× scale, bottom-left corner
    spawn_zone_center_x: float = 8.0      # m — center of 10m zone, ~5m from +X wall
    spawn_zone_center_y: float = -12.5    # m — center of 10m zone, ~5m from -Y wall
    spawn_zone_size_x: float = 10.0       # m — 2m real × 5 scale
    spawn_zone_size_y: float = 10.0       # m — 2m real × 5 scale
    spawn_zone_margin: float = 2.0        # m — robot clearance from zone edges (keep away from walls)
    spawn_random_yaw: bool = True         # random heading each episode

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=75.0,
        replicate_physics=True,
    )

    # ── robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = LUNABOTICS_DIRECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ── physics constants ─────────────────────────────────────────────────────
    wheel_radius: float = 0.34544  # 13.6 inches — verified from CAD

    # USD PhysicsMassAPI stores absolute kg values that do not scale with geometry.
    # Divide all body masses (and inertias) by this factor at runtime to match
    # the real robot. Set to (usd_total_mass / real_total_mass).
    # USD total = 9038 kg; tune denominator once real robot is weighed.
    mass_correction_factor: float = 100.0

    # ── curriculum (reward-threshold based) ──────────────────────────────────
    # Phase 0 — straight only:   advance when EMA(track_lin) > phase1_threshold
    # Phase 1 — straight+pivot:  advance when EMA(track_lin) > phase2_lin_threshold
    #                                      AND EMA(track_ang) > phase2_ang_threshold
    # Phase 2 — arcs unlocked:   full reward suite
    #
    # EMA alpha of 0.002 gives ~500-step (~10 iter) smoothing window so a single
    # good/bad batch doesn't prematurely advance or stall the curriculum.
    # min_steps guards against advancing on lucky early rollouts before the EMA
    # has had time to reflect true policy performance.
    curriculum_ema_alpha: float = 0.002
    curriculum_phase1_threshold: float = 0.5     # EMA(track_lin) to unlock phase 1
    curriculum_phase2_lin_threshold: float = 0.65 # EMA(track_lin) to unlock phase 2
    curriculum_phase2_ang_threshold: float = 0.50 # EMA(track_ang) to unlock phase 2
    curriculum_phase0_min_steps: int = 2_400      # ~50 iters minimum in phase 0
    curriculum_phase1_min_steps: int = 2_400      # ~50 iters minimum in phase 1

    # ── reward scales ────────────────────────────────────────────────────────
    # Follows legged_gym / WheeledLab conventions with three additions for
    # wheeled robots: alive_bonus, stall_penalty, and only_positive_rewards.
    #
    # Wheels can trivially output zero torque (unlike legs, where standing is
    # hard), so without explicit anti-stall mechanisms the policy converges to
    # doing nothing.
    #
    # alive_bonus:  constant +reward each step the robot is alive.  Gives the
    #   policy a baseline reason to stay in-bounds and not tip over.
    # stall_penalty:  discrete negative when |vx| < stall_threshold.  Catches
    #   the zero-output equilibrium that continuous rewards can't fully break.
    # only_positive_rewards:  clamp per-step total at 0 so penalty-dominated
    #   early training doesn't teach "do nothing to avoid penalties."
    #   (legged_gym default = True)
    alive_bonus_scale: float = 0.1
    stall_penalty_scale: float = -0.5       # light nudge — forward_progress is the real anti-stall
    stall_threshold: float = 0.05           # m/s
    only_positive_rewards: bool = True       # clamp dt total at 0 (legged_gym default)

    # ── reward scales ────────────────────────────────────────────────────────
    # Design rule: penalties must be 5–10× smaller than forward_progress so the
    # clamp doesn't mask the forward signal.  At 0.5 m/s the forward_progress
    # term yields +3.0/step — no single penalty should approach that magnitude.
    #
    #   forward_progress (6.0)  — THE primary reward, 60%+ of total
    #   track_lin/ang (1.0/0.5) — velocity quality shaping
    #   all penalties combined  — should sum to ≤ 1.0 at normal driving
    forward_progress_scale: float = 6.0     # body-frame forward vel — dominant
    lin_vel_reward_scale: float = 1.0       # exp(-||vx_error||² / 0.25)
    ang_vel_reward_scale: float = 0.5       # exp(-||wz_error||² / 0.25)
    lin_vel_z_scale: float = -0.5           # vz² — was -2.0, too harsh on rough terrain
    ang_vel_xy_l2_scale: float = -0.05      # roll/pitch rate
    flat_orientation_l2_scale: float = -0.5  # tilt — was -1.0
    action_rate_l2_scale: float = -0.01     # smoothness
    overspeed_scale: float = -1.0           # was -2.0
    preferred_speed_threshold: float = 0.71 # m/s

    # ── arena collision ───────────────────────────────────────────────────────
    wall_contact_threshold: float = 1.0     # N — sensor is on arena, only robot can touch it, any force = collision
    wall_collision_scale: float = -100.0    # painful but recoverable — -500 caused training divergence

    # ── differential drive quality (always active) ───────────────────────────
    # Penalties here must stay small relative to forward_progress (6.0).
    # At -1.0 scale, wheel_consistency generates ~-0.6 typical penalty vs
    # +3.0 forward reward — a meaningful but not dominant constraint.
    wheel_consistency_scale: float = -1.0   # was -5.0 — overwhelmed forward signal
    wheel_slip_scale: float = -0.3
    wheel_slip_deadzone: float = 0.35


@configclass
class LunaboticsDirectEnvCfg_PLAY(LunaboticsDirectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 70.0
