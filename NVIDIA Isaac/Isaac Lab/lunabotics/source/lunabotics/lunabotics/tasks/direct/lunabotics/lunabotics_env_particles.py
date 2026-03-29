"""DirectRLEnv — EXPERIMENTAL particle-bed variant.

Gaussian-mound-distributed rigid-sphere regolith bed.
Robot spawns at the south end and drives north into the particle bed.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from lunabotics.assets.lunabotics import ARENA_USD_PATH  # isort: skip
from .lunabotics_env_cfg_particles import LunaboticsParticleEnvCfg


class LunaboticsParticleEnv(DirectRLEnv):
    cfg: LunaboticsParticleEnvCfg

    def __init__(self, cfg: LunaboticsParticleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions      = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._commands     = torch.zeros(self.num_envs, 3, device=self.device)

        self._wheel_ids = self._robot.actuators["wheels"].joint_indices
        wheel_names     = self._robot.actuators["wheels"].joint_names
        self._left_mask = torch.tensor(
            ["Left" in name for name in wheel_names],
            device=self.device, dtype=torch.bool,
        )

        self._wheel_vel_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self._rl_step = 0

        sensor_body_names = self._arena_contact.body_names
        self._chassis_body_ids = torch.tensor(
            [i for i, n in enumerate(sensor_body_names) if "Wheel" not in n],
            device=self.device, dtype=torch.long,
        )

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel",
                "track_ang_vel",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "flat_orientation_l2",
                "action_rate_l2",
                "overspeed",
                "wall_collision",
            ]
        }

    def _spawn_particles(self):
        """Spawn a Gaussian-mound-distributed rigid-sphere regolith bed in env_0.

        Particle column heights at each (x, y) grid position are determined by
        superimposing random Gaussian mounds — the same statistical approach as
        RegolithTerrainCfg, but producing stacked rigid spheres instead of a
        solid mesh.  Mound peaks get several stacked layers; flat areas get one.

        Created BEFORE clone_environments() → replicated to every env with
        independent rigid-body physics.  Particles are NOT reset between
        episodes — the bed deforms progressively, adding realistic variability.
        """
        import numpy as np

        r   = self.cfg.particle_radius
        sp  = self.cfg.particle_spacing
        vsp = r * 1.8
        hx  = self.cfg.particle_area_x / 2.0
        hy  = self.cfg.particle_area_y / 2.0

        # ── Gaussian mound height field ─────────────────────────────────────
        rng  = np.random.default_rng(seed=42)
        area = self.cfg.particle_area_x * self.cfg.particle_area_y
        n_mounds = max(1, int(self.cfg.mound_density * area))

        cx = rng.uniform(-hx, hx, n_mounds)
        cy = rng.uniform(-hy, hy, n_mounds)
        cr = rng.uniform(self.cfg.mound_radius_range[0], self.cfg.mound_radius_range[1], n_mounds)
        ch = rng.uniform(self.cfg.mound_height_range[0], self.cfg.mound_height_range[1], n_mounds)

        xs = np.arange(-hx + sp / 2.0, hx, sp)
        ys = np.arange(-hy + sp / 2.0, hy, sp)

        height_field = np.zeros((len(xs), len(ys)))
        for i in range(n_mounds):
            sigma2 = (cr[i] / 2.0) ** 2
            dx = xs - cx[i]
            dy = ys - cy[i]
            height_field += ch[i] * np.exp(
                -(dx[:, np.newaxis] ** 2 + dy[np.newaxis, :] ** 2) / (2.0 * sigma2)
            )

        # ── Sphere template ─────────────────────────────────────────────────
        particle_cfg = sim_utils.SphereCfg(
            radius=r,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_linear_velocity=10.0,
                max_angular_velocity=200.0,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=1500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.80,
                dynamic_friction=0.70,
                restitution=0.05,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.67, 0.67, 0.65),
                roughness=0.95,
                metallic=0.0,
            ),
        )

        # ── Spawn stacked columns ────────────────────────────────────────────
        count = 0
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                max_z = r + height_field[ix, iy]
                z = r
                while z <= max_z + 1e-6:
                    particle_cfg.func(
                        f"/World/envs/env_0/Particles/p_{count:05d}",
                        particle_cfg,
                        translation=(float(x), float(y), float(z)),
                    )
                    count += 1
                    z += vsp

    def _setup_scene(self):
        # 1. Static geometry FIRST
        arena_cfg = sim_utils.UsdFileCfg(
            usd_path=ARENA_USD_PATH,
            scale=(5.0, 5.0, 5.0),
        )
        arena_cfg.func(
            "/World/envs/env_.*/Arena",
            arena_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.7071, 0.0, 0.0, 0.7071),
        )

        # 2. Regolith particle bed (env_0 only — cloned to all envs below)
        self._spawn_particles()

        # 3. Robot articulation
        self._robot = Articulation(self.cfg.robot)

        # 4. Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 5. Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 6. Scene registration AFTER clone
        self.scene.articulations["robot"] = self._robot

        # 7. Contact sensor
        self._arena_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                history_length=2,
                update_period=0.0,
                track_air_time=False,
            )
        )
        self.scene.sensors["arena_contact"] = self._arena_contact

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._rl_step += 1
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        fwd = self._actions[:, 0:1].clamp(0.0, 1.0)
        trn = self._actions[:, 1:2]
        left_vel  =  (fwd - trn) * self.cfg.action_scale
        right_vel = -(fwd + trn) * self.cfg.action_scale

        raw_targets = torch.where(
            self._left_mask,
            left_vel.expand(-1, 6),
            right_vel.expand(-1, 6),
        )

        self._wheel_vel_targets = 0.5 * self._wheel_vel_targets + 0.5 * raw_targets

    def _apply_action(self):
        self._robot.set_joint_velocity_target(self._wheel_vel_targets, joint_ids=self._wheel_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._commands[:, [0, 2]],
            self._actions,
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        vx  = self._robot.data.root_lin_vel_b[:, 0]
        vz  = self._robot.data.root_lin_vel_b[:, 2]
        wz  = self._robot.data.root_ang_vel_b[:, 2]
        cmd_vx = self._commands[:, 0]
        cmd_wz = self._commands[:, 2]

        track_lin = torch.exp(-torch.square(cmd_vx - vx) / 0.25)
        track_ang = torch.exp(-torch.square(cmd_wz - wz) / 0.25)

        lin_vel_z  = torch.square(vz)
        ang_vel_xy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        flat_ori   = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        act_rate   = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

        speed_xy  = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        overspeed = torch.clamp(speed_xy - self.cfg.preferred_speed_threshold, min=0.0)

        chassis_forces = self._arena_contact.data.net_forces_w[:, self._chassis_body_ids, :]
        wall_hit = (torch.norm(chassis_forces, dim=-1).amax(dim=-1)
                    > self.cfg.wall_contact_threshold).float()

        dt_rewards = {
            "track_lin_vel":       track_lin  * self.cfg.lin_vel_reward_scale,
            "track_ang_vel":       track_ang  * self.cfg.ang_vel_reward_scale,
            "lin_vel_z_l2":        lin_vel_z  * self.cfg.lin_vel_z_scale,
            "ang_vel_xy_l2":       ang_vel_xy * self.cfg.ang_vel_xy_l2_scale,
            "flat_orientation_l2": flat_ori   * self.cfg.flat_orientation_l2_scale,
            "action_rate_l2":      act_rate   * self.cfg.action_rate_l2_scale,
            "overspeed":           overspeed  * self.cfg.overspeed_scale,
        }

        total = torch.zeros(self.num_envs, device=self.device)
        for key, val in dt_rewards.items():
            total += val * self.step_dt
            self._episode_sums[key] += val * self.step_dt

        wall_penalty = wall_hit * self.cfg.wall_collision_scale
        total += wall_penalty
        self._episode_sums["wall_collision"] += wall_penalty

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        tipped   = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > 0.9
        chassis_forces = self._arena_contact.data.net_forces_w[:, self._chassis_body_ids, :]
        wall_hit = (torch.norm(chassis_forces, dim=-1).amax(dim=-1)
                    > self.cfg.wall_contact_threshold)
        return tipped | wall_hit, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids]           = 0.0
        self._prev_actions[env_ids]      = 0.0
        self._wheel_vel_targets[env_ids] = 0.0

        n     = len(env_ids)
        zeros = torch.zeros(n, device=self.device)

        if self._rl_step < self.cfg.curriculum_phase1_steps:
            vx = torch.empty(n, device=self.device).uniform_(0.3, 0.65)
            self._commands[env_ids, 0] = vx
            self._commands[env_ids, 2] = zeros

        elif self._rl_step < self.cfg.curriculum_phase2_steps:
            mode = torch.rand(n, device=self.device)
            vx   = torch.empty(n, device=self.device).uniform_(0.3, 0.65)
            wz   = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
            straight = mode < 0.6
            self._commands[env_ids, 0] = torch.where(straight, vx,    zeros)
            self._commands[env_ids, 2] = torch.where(straight, zeros, wz)

        else:
            mode = torch.rand(n, device=self.device)
            vx   = torch.empty(n, device=self.device).uniform_(0.2, 0.65)
            wz   = torch.empty(n, device=self.device).uniform_(-0.8, 0.8)
            straight = mode < 0.5
            self._commands[env_ids, 0] = torch.where(straight, vx, vx * 0.5)
            self._commands[env_ids, 2] = torch.where(straight, zeros, wz)

        self._commands[env_ids, 1] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 1]  += self.cfg.robot_spawn_y_offset
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras
