"""Direct RL environment config — EXPERIMENTAL particle-bed variant.

Flat ground plane + Gaussian-mound-distributed rigid-sphere regolith bed.
Rigid spheres physically react to the robot (wheels scatter particles).
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
class LunaboticsParticleEnvCfg(DirectRLEnvCfg):
    # ── env ───────────────────────────────────────────────────────────────────
    episode_length_s: float = 20.0
    decimation: int = 4
    action_scale: float = 210.0
    action_space: int = 2
    observation_space: int = 13
    state_space: int = 0

    # ── simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physx=PhysxCfg(
            solver_type=0,
            gpu_max_rigid_contact_count=2**23,   # raised for particle contacts
            gpu_max_rigid_patch_count=2**22,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── terrain ───────────────────────────────────────────────────────────────
    # Flat plane — the rigid-sphere particle bed is the actual surface.
    terrain = TerrainImporterCfg(
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
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.38, 0.38, 0.36),   # dark moon-dust base under particles
            roughness=0.95,
            metallic=0.0,
        ),
        debug_vis=False,
    )

    # ── scene ─────────────────────────────────────────────────────────────────
    # Particle simulation is GPU-heavy — 16 envs is a practical training limit.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=250.0,
        replicate_physics=True,
    )

    # ── robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = LUNABOTICS_DIRECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ── physics constants ─────────────────────────────────────────────────────
    wheel_radius: float = 0.91

    # ── regolith particle bed ─────────────────────────────────────────────────
    # Rigid spheres stacked in Gaussian-mound columns — same mound distribution
    # as RegolithTerrainCfg but as physical objects that react to the robot.
    # ~4 000–6 000 particles × 16 envs at current radius/spacing settings.
    particle_radius:  float = 0.07    # m — grain radius
    particle_spacing: float = 0.18    # m — horizontal centre-to-centre spacing
    particle_area_x:  float = 13.0    # m — full x coverage (≤ arena width)
    particle_area_y:  float = 8.5     # m — full y coverage (≤ arena depth)

    # Gaussian mound distribution — controls layering depth across the bed.
    # Mound peaks get 2–4 stacked layers; flat areas get a single base layer.
    mound_density:       float = 0.4             # mounds / m²  (~44 mounds total)
    mound_radius_range:  tuple = (1.0, 3.0)      # m — Gaussian sigma×2 per mound
    mound_height_range:  tuple = (0.10, 0.30)    # m — peak extra height (2–6 extra layers)

    # ── robot spawn ───────────────────────────────────────────────────────────
    # South end of arena — robot drives north into the particle bed.
    robot_spawn_y_offset: float = -5.0   # m

    # ── curriculum thresholds ─────────────────────────────────────────────────
    curriculum_phase1_steps: int = 7_200
    curriculum_phase2_steps: int = 24_000

    # ── reward scales ─────────────────────────────────────────────────────────
    lin_vel_reward_scale: float = 1.0
    ang_vel_reward_scale: float = 0.5
    lin_vel_z_scale: float = -2.0
    ang_vel_xy_l2_scale: float = -0.05
    flat_orientation_l2_scale: float = -1.0
    action_rate_l2_scale: float = -0.01
    overspeed_scale: float = -2.0
    preferred_speed_threshold: float = 0.71

    # ── arena collision ───────────────────────────────────────────────────────
    wall_contact_threshold: float = 1.0
    wall_collision_scale: float = -50.0


@configclass
class LunaboticsParticleEnvCfg_PLAY(LunaboticsParticleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2
        self.scene.env_spacing = 20.0
