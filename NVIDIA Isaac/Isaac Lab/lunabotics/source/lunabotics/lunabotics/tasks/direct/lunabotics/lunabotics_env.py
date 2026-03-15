"""DirectRLEnv for the CSU Lunabotics 6-wheel skid-steer rover.

Action space (2):  [left_side_vel, right_side_vel] in [-1, 1]
  → scaled by action_scale (3.67 rad/s) and broadcast to all 3 left / 3 right wheels.

Observation space (19):
  root_lin_vel_b    (3)  — body-frame linear velocity
  root_ang_vel_b    (3)  — body-frame angular velocity
  projected_gravity (3)  — tilt indicator
  commands          (2)  — [cmd_vx, cmd_wz]
  joint_vel_wheels  (6)  — actual wheel velocities
  last_actions      (2)  — previous action for action-rate penalty
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .lunabotics_env_cfg import LunaboticsDirectEnvCfg


class LunaboticsDirectEnv(DirectRLEnv):
    cfg: LunaboticsDirectEnvCfg

    def __init__(self, cfg: LunaboticsDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # buffers
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        # [cmd_vx, cmd_vy=0, cmd_wz]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Wheel joint indices pulled directly from the DCMotorCfg actuator — these are the
        # exact same indices _apply_actuator_model() reads from _data.joint_vel_target.
        self._wheel_ids = self._robot.actuators["wheels"].joint_indices

        # Build left/right mask from actual joint names — USD traversal order is NOT
        # guaranteed to be [L,L,L,R,R,R], so we must not assume a fixed split index.
        wheel_names = self._robot.actuators["wheels"].joint_names
        self._left_wheel_mask = torch.tensor(
            ["Left" in name for name in wheel_names],
            device=self.device, dtype=torch.bool,
        )  # [6] — True for left wheels, False for right wheels

        # velocity target buffer: [N, 6]
        self._wheel_vel_targets = torch.zeros(self.num_envs, 6, device=self.device)

        # curriculum step counter — incremented once per _pre_physics_step call
        self._rl_step = 0

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy",
                "track_ang_vel_z",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "flat_orientation_l2",
                "action_rate_l2",
                "excessive_yaw_rate",
                "lateral_vel_l2",
                "overspeed",
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._rl_step += 1
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Compute per-side velocity targets.
        # Left wheels are negated: action=(+1,+1) → left=-scale, right=+scale → forward.
        left_vel  = -self._actions[:, 0:1] * self.cfg.action_scale  # [N, 1]
        right_vel =  self._actions[:, 1:2] * self.cfg.action_scale  # [N, 1]

        # Map to all 6 wheels using the name-based left/right mask.
        # This is robust to any joint traversal order in the USD — never assume L,L,L,R,R,R.
        raw_targets = torch.where(
            self._left_wheel_mask,           # [6] broadcast → [N, 6]
            left_vel.expand(-1, 6),
            right_vel.expand(-1, 6),
        )

        # Exponential smoothing — prevents policy from exploiting physics ratcheting
        # by oscillating wheel commands. alpha=0.4: ~55 ms time constant at dt=0.02 s.
        self._wheel_vel_targets = 0.6 * self._wheel_vel_targets + 0.4 * raw_targets

    def _apply_action(self):
        self._robot.set_joint_velocity_target(self._wheel_vel_targets, joint_ids=self._wheel_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self._robot.data.root_lin_vel_b,                            # 3
            self._robot.data.root_ang_vel_b,                            # 3
            self._robot.data.projected_gravity_b,                       # 3
            self._commands[:, [0, 2]],                                  # 2  (vx, wz)
            self._robot.data.joint_vel[:, self._wheel_ids],             # 6
            self._actions,                                               # 2
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking (xy in body frame)
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        track_lin = torch.exp(-lin_vel_error / 0.25)

        # yaw rate tracking
        yaw_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        track_yaw = torch.exp(-yaw_error / 0.25)

        # penalties
        z_vel      = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        flat_ori   = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        act_rate   = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)
        # Penalize yaw rate above 1.5 rad/s — discourages spin-bounce and runaway rotation
        excessive_yaw = torch.clamp(torch.abs(self._robot.data.root_ang_vel_b[:, 2]) - 1.5, min=0.0)
        # Penalize body-frame lateral (Y) velocity — skid-steer cannot strafe
        lat_vel = torch.square(self._robot.data.root_lin_vel_b[:, 1])
        # Penalize XY speed above 35 RPM physical limit (~0.63 m/s); penalty ramps linearly above 0.65
        speed_xy = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        overspeed = torch.clamp(speed_xy - 0.65, min=0.0)

        rewards = {
            "track_lin_vel_xy":   track_lin      * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z":    track_yaw      * self.cfg.yaw_rate_reward_scale,
            "lin_vel_z_l2":       z_vel          * self.cfg.z_vel_reward_scale,
            "ang_vel_xy_l2":      ang_vel_xy     * self.cfg.ang_vel_reward_scale,
            "flat_orientation_l2":flat_ori       * self.cfg.flat_orientation_reward_scale,
            "action_rate_l2":     act_rate       * self.cfg.action_rate_reward_scale,
            "excessive_yaw_rate": excessive_yaw  * self.cfg.excessive_yaw_reward_scale,
            "lateral_vel_l2":     lat_vel        * self.cfg.lateral_vel_reward_scale,
            "overspeed":          overspeed      * self.cfg.overspeed_reward_scale,
        }
        total = torch.zeros(self.num_envs, device=self.device)
        for key, val in rewards.items():
            total += val * self.step_dt
            self._episode_sums[key] += val * self.step_dt

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Terminate if robot tips over significantly
        tipped = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > 0.9
        # Physics-grounded slip reset: body XY speed cannot exceed 2× the fastest wheel's
        # tangential speed. If it does, the robot was launched by non-wheel forces (arm swing,
        # collision explosion, etc.) and the episode is invalid.
        wheel_vels = self._robot.data.joint_vel[:, self._wheel_ids]  # [N, 6]
        max_wheel_tangential = torch.max(torch.abs(wheel_vels), dim=1).values * self.cfg.wheel_radius
        body_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        # 2× factor allows for natural skid-steer slip; +0.05 prevents false positives at rest
        physics_violation = body_speed > (max_wheel_tangential * 2.0 + 0.05)
        return tipped | physics_violation, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        # Curriculum command sampling — complexity grows as training progresses.
        #
        # Phase 0 (<phase1_steps): straight only, low speed.
        #   Robot must first learn to drive without lateral drift.
        # Phase 1 (phase1→phase2): pure turns added, speed increases.
        #   80% straight, 20% pure in-place pivot — no mixed curves yet.
        # Phase 2 (phase2+): full mix — straight, turns, arcs.
        #   50% straight, 30% turn, 20% arc.
        n = len(env_ids)
        zeros = torch.zeros(n, device=self.device)

        if self._rl_step < self.cfg.curriculum_phase1_steps:
            # Phase 0: straight movement only, conservative speed
            vx = torch.empty(n, device=self.device).uniform_(-0.3, 0.3)
            self._commands[env_ids, 0] = vx
            self._commands[env_ids, 2] = zeros

        elif self._rl_step < self.cfg.curriculum_phase2_steps:
            # Phase 1: mostly straight, add pure turning
            mode = torch.rand(n, device=self.device)
            vx = torch.empty(n, device=self.device).uniform_(-0.4, 0.4)
            wz = torch.empty(n, device=self.device).uniform_(-0.8, 0.8)
            self._commands[env_ids, 0] = torch.where(mode < 0.8, vx, zeros)   # 80% straight
            self._commands[env_ids, 2] = torch.where(mode >= 0.8, wz, zeros)  # 20% turn

        else:
            # Phase 2: full mix — straight, pivot, arc
            mode = torch.rand(n, device=self.device)
            vx = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
            wz = torch.empty(n, device=self.device).uniform_(-1.0, 1.0)
            has_vx = (mode < 0.5) | (mode >= 0.8)   # 70%: straight + arc
            has_wz = mode >= 0.5                      # 50%: turn + arc
            self._commands[env_ids, 0] = torch.where(has_vx, vx, zeros)
            self._commands[env_ids, 2] = torch.where(has_wz, wz, zeros)

        self._commands[env_ids, 1] = 0.0  # vy always zero — skid-steer can't strafe

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = {}
        for key in self._episode_sums:
            episodic_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras
