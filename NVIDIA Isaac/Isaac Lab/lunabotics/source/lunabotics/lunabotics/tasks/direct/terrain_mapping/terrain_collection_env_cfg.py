"""Direct RL environment config for terrain data collection."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import os

from lunabotics.assets.lunabotics import LUNABOTICS_DIRECT_CFG  # isort: skip

# CSU_Lunabotics_Isaac_Lab_Model.usd does not exist yet — fall back to the
# complex Isaac Sim USD.  Both are authored in mm, so scale=0.1 is correct.
_ASSETS_DIR       = os.path.dirname(os.path.abspath(__file__))
_NVIDIA_ISAAC_ROOT = os.path.normpath(os.path.join(_ASSETS_DIR, "..", "..", "..", "..", "..", "..", "..", ".."))
_COMPLEX_USD_PATH  = os.path.join(_NVIDIA_ISAAC_ROOT, "Isaac Sim", "USD", "Lunabotics Isaac Sim (Complex) - Isaac.usd")

TERRAIN_COLLECTION_ROBOT_CFG = LUNABOTICS_DIRECT_CFG.replace(
    spawn=LUNABOTICS_DIRECT_CFG.spawn.replace(
        usd_path=_COMPLEX_USD_PATH,
        scale=(0.1, 0.1, 0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={j: 0.0 for j in [
            "Left_Front_Wheel", "Left_Center_Wheel", "Left_Rear_Wheel",
            "Right_Front_Wheel", "Right_Center_Wheel", "Right_Rear_Wheel",
        ]},
        joint_vel={".*": 0.0},
    ),
)


@configclass
class TerrainCollectionEnvCfg(DirectRLEnvCfg):
    # ── env ───────────────────────────────────────────────────────────────────
    episode_length_s: float = 20.0
    decimation: int = 4
    action_scale: float = 3.665
    action_space: int = 2
    observation_space: int = 7
    state_space: int = 0

    # ── simulation ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physx=PhysxCfg(
            solver_type=0,
            gpu_collision_stack_size=2**28,
            gpu_max_rigid_patch_count=2**18,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # ── infinite flat ground plane ─────────────────────────────────────────
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
    regolith_size: tuple = (20.0, 20.0)
    regolith_horizontal_scale: float = 0.05
    regolith_vertical_scale: float = 0.001
    regolith_particle_density_range: tuple = (0.5, 2.5)
    regolith_particle_radius_range: tuple = (0.40, 2.0)
    regolith_crest_height_range: tuple = (0.03, 0.12)

    # ── crater config ──────────────────────────────────────────────────────
    crater_count_range: tuple = (3, 10)
    crater_diameter_range: tuple = (0.40, 1.50)  # meters
    crater_depth_ratio: float = 0.50             # depth = diameter × ratio (0.5 → semicircle)

    # ── spawn zone ─────────────────────────────────────────────────────────
    spawn_zone_center_x: float = 0.0
    spawn_zone_center_y: float = 0.0
    spawn_zone_size_x: float = 20.0
    spawn_zone_size_y: float = 20.0
    spawn_zone_margin: float = 0.0
    spawn_random_yaw: bool = True
    spawn_position_scale: float = 0.0   # 0.0 = always spawn at center

    # ── visual randomization ──────────────────────────────────────────────
    randomize_materials: bool = True

    # ── obstacle config ────────────────────────────────────────────────────────
    obstacle_spawn_margin: float = 1.5
    rock_count_range: tuple = (3, 12)
    rock_diameter_range: tuple = (0.15, 0.60)
    wall_count_range: tuple = (2, 5)
    wall_length_range: tuple = (5.0, 14.0)
    wall_height: float = 3.0

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=45.0,
        replicate_physics=True,
    )

    # ── robot ─────────────────────────────────────────────────────────────────
    robot: ArticulationCfg = TERRAIN_COLLECTION_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # ── physics constants ─────────────────────────────────────────────────────
    mass_correction_factor: float = 100.0
