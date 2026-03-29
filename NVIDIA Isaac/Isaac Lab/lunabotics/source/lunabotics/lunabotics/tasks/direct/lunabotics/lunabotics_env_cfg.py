"""Direct RL environment config for the CSU Lunabotics 6-wheel skid-steer rover.

Clean research-grounded implementation based on legged_gym / WheeledLab / ANYmal standards.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from lunabotics.assets.lunabotics import LUNABOTICS_DIRECT_CFG  # isort: skip
from lunabotics.terrains import RegolithTerrainCfg  # isort: skip


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

    # ── terrain ───────────────────────────────────────────────────────────────
    # Procedural regolith terrain — 7.220 m × 5.350 m Artemis Arena footprint.
    # Each sub-terrain tile is arena-sized; num_rows × num_cols tiles are
    # generated with difficulty increasing left→right across columns.
    # Switch terrain_type to "plane" for a flat baseline run.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        collision_group=-1,
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(14.14, 9.14),         # arena footprint: 14.141 m long axis; short axis TBD — update once confirmed
            num_rows=2,
            num_cols=4,
            horizontal_scale=0.05,      # 5 cm/cell — good balance of detail vs speed
            vertical_scale=0.001,       # 1 mm/count
            slope_threshold=None,
            sub_terrains={
                "regolith": RegolithTerrainCfg(
                    proportion=1.0,
                    size=(14.14, 9.14),
                    horizontal_scale=0.05,
                    vertical_scale=0.001,
                    border_width=0.0,
                    # ── tune these to change surface roughness ───────────────
                    particle_density_range=(0.5, 2.5),     # mounds/m²  (sparse — large features)
                    particle_radius_range=(0.40, 2.0),     # metres     (8× wider mounds)
                    crest_height_range=(0.02, 0.15),       # metres     (more depth)
                ),
            },
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.67, 0.67, 0.65),  # lunar grey
            roughness=0.95,
            metallic=0.0,
        ),
        debug_vis=False,
    )

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=20.0,
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
        self.scene.num_envs = 16
        self.scene.env_spacing = 20.0
        # smaller tile grid for visual runs — 2×2 = 4 tiles
        self.terrain.terrain_generator.num_rows = 2
        self.terrain.terrain_generator.num_cols = 2
