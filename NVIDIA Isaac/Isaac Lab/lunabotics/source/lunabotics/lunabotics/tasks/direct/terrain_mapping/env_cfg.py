"""Config for the terrain data collection environment."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from lunabotics.assets.lunabotics import LUNABOTICS_DIRECT_CFG  # isort: skip


@configclass
class TerrainCollectionEnvCfg(DirectRLEnvCfg):
    # ── dummy RL params — data collection doesn't use rewards ────────────────
    episode_length_s: float = 1.0
    decimation: int = 4
    action_space: int = 1
    observation_space: int = 7   # robot pose: x y z qw qx qy qz
    state_space: int = 0

    # ── simulation ────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=4,
        physx=PhysxCfg(solver_type=0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── flat ground fallback ─────────────────────────────────────────────────
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

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=50.0,
        replicate_physics=True,
    )

    # ── robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = LUNABOTICS_DIRECT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ── regolith terrain ─────────────────────────────────────────────────────
    regolith_size: tuple = (15.0, 15.0)
    regolith_horizontal_scale: float = 0.05   # 5 cm/cell
    regolith_vertical_scale: float = 0.001
    regolith_particle_density_range: tuple = (0.5, 2.5)
    regolith_particle_radius_range: tuple = (0.15, 0.40)
    regolith_crest_height_range: tuple = (0.01, 0.08)

    # ── obstacle generation ───────────────────────────────────────────────────
    arena_size: tuple = (10.0, 10.0)   # data collection area within terrain
    spawn_margin: float = 1.0

    crater_count_range: tuple = (3, 15)
    crater_diameter_range: tuple = (0.40, 1.50)
    crater_depth_ratio: float = 0.25

    rock_count_range: tuple = (5, 20)
    rock_diameter_range: tuple = (0.20, 0.50)

    wall_count_range: tuple = (0, 4)
    wall_length_range: tuple = (0.5, 3.0)
    wall_height: float = 0.3

    # ── BEV parameters ────────────────────────────────────────────────────────
    bev_grid_size: int = 200
    bev_cell_size: float = 0.05
