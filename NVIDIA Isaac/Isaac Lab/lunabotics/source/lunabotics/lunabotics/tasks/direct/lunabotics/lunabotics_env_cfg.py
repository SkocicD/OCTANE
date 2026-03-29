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
    decimation: int = 4
    action_scale: float = 210.0     # 35 RPM = 210 deg/s — full range available for evasive action
    action_space: int = 2           # [forward, turn_rate]
    observation_space: int = 13
    state_space: int = 0

    # ── simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physx=PhysxCfg(solver_type=0),  # PGS — TGS has confirmed velocity-reporting bug
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
    regolith_horizontal_scale: float = 0.0625      # 6.25 cm/cell (1/16 — exact in binary, no FP truncation in height_field_to_mesh)
    regolith_vertical_scale: float = 0.001         # 1 mm/count
    regolith_particle_density_range: tuple = (0.5, 2.5)   # mounds/m²
    regolith_particle_radius_range: tuple = (0.40, 2.0)   # m
    regolith_crest_height_range: tuple = (0.02, 0.15)     # m

    # ── robot spawn ───────────────────────────────────────────────────────────
    # XY offset from each env's origin where the robot spawns.
    # Negative x = back of arena (toward excavation zone start), negative y = left side.
    robot_spawn_x_offset: float = -5.0    # m — place robot at bottom of arena
    robot_spawn_y_offset: float = -3.0    # m — place robot at left side of arena

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=75.0,
        replicate_physics=True,
    )

    # ── robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = LUNABOTICS_DIRECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ── physics constants ─────────────────────────────────────────────────────
    wheel_radius: float = 0.91

    # ── curriculum thresholds (in _pre_physics_step calls) ───────────────────
    # Phase 0 (0 → 150 iters):  straight only
    # Phase 1 (150 → 500 iters): straight OR pivot turn, exclusive
    # Phase 2 (500+):            arcs unlocked
    curriculum_phase1_steps: int = 7_200    # 150 iters
    curriculum_phase2_steps: int = 24_000   # 500 iters

    # ── reward scales (legged_gym / WheeledLab research standard) ────────────
    lin_vel_reward_scale: float = 1.0       # exp(-||vx_error||² / 0.25)
    ang_vel_reward_scale: float = 0.5       # exp(-||wz_error||² / 0.25)
    lin_vel_z_scale: float = -2.0           # vz² — vertical bouncing
    ang_vel_xy_l2_scale: float = -0.05      # roll/pitch rate
    flat_orientation_l2_scale: float = -1.0 # tilt
    action_rate_l2_scale: float = -0.01     # smoothness
    overspeed_scale: float = -2.0           # linear ramp above preferred_speed_threshold
    preferred_speed_threshold: float = 0.71 # m/s ≈ 45 deg/s wheel speed — preferred cruise

    # ── arena collision ───────────────────────────────────────────────────────
    # ContactSensor (horizontal XY forces only) detects robot↔wall contacts.
    # Increase threshold if normal driving triggers false positives;
    # decrease if soft wall grazes aren't being caught.
    wall_contact_threshold: float = 1.0    # Newtons — horizontal force to classify as wall hit
    wall_collision_scale: float = -50.0    # one-time penalty per wall-hit event (not dt-scaled)


@configclass
class LunaboticsDirectEnvCfg_PLAY(LunaboticsDirectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 45.0   # must be > regolith_size[1]=38.0 to prevent terrain overlap
