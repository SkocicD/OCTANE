"""Direct RL environment config for the CSU Lunabotics 6-wheel skid-steer rover."""

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
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 3.67     # 35 RPM = 3.67 rad/s; actions in [-1,1] → vel targets in [-3.67, 3.67] rad/s
    action_space = 2        # (left_side_vel, right_side_vel)
    observation_space = 19  # see lunabotics_env.py _get_observations
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physx=PhysxCfg(solver_type=0),  # PGS — TGS has confirmed velocity-reporting bug for skid-steer
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # terrain
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
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=12.0,  # scale=0.1: robot is ~10× larger, needs wider spacing to avoid overlap
        replicate_physics=True,
    )

    # robot
    robot: ArticulationCfg = LUNABOTICS_DIRECT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # physics constants
    # Wheel: 13.6 in diameter = 0.3454 m in real life.
    # USD in mm (mislabeled metersPerUnit=0.01). BBoxCache shows 18.2 USD-units diameter.
    # With scale=0.1: wheel_radius = 18.2/2 × 0.1 = 0.91 m in simulation.
    # NOTE: if physics material uses density (not fixed mass), re-run find_resting_height.py
    # after scale change and update init_state.pos in assets/lunabotics.py accordingly.
    wheel_radius: float = 0.91

    # curriculum — phase thresholds in _pre_physics_step calls (all envs, one call per RL step)
    # ~333 iters × 48 steps_per_env = 16 000 to learn straight; another 333 for turning.
    curriculum_phase1_steps: int = 16_000   # straight only  → add turning after this
    curriculum_phase2_steps: int = 36_000   # turning added  → full mix after this

    # reward scales
    # Wheel: 13.6 in diameter = 0.1727 m radius. 35 RPM = 3.67 rad/s → max ~0.63 m/s linear.
    lin_vel_reward_scale = 1.5       # exp(-||vel_xy_error||^2 / 0.25)
    yaw_rate_reward_scale = 1.5      # exp(-yaw_rate_error^2 / 0.25) — equal weight with forward
    z_vel_reward_scale = -2.0        # penalize vertical bouncing
    ang_vel_reward_scale = -0.05     # penalize roll/pitch rates
    flat_orientation_reward_scale = -1.0   # penalize tilt
    action_rate_reward_scale = -0.2        # penalize jerky/oscillatory commands — smooth accel/decel
    excessive_yaw_reward_scale = -1.0      # penalize yaw rate > 1.5 rad/s to prevent spin-bounce
    lateral_vel_reward_scale = -3.0        # penalize body-Y (sideways) drift — skid-steer can't strafe
    overspeed_reward_scale = -3.0          # penalize XY speed above 0.65 m/s (35 RPM physical limit)


@configclass
class LunaboticsDirectEnvCfg_PLAY(LunaboticsDirectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 12.0
