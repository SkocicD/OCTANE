"""DirectRLEnv for the CSU Lunabotics 6-wheel skid-steer rover.

Clean research-grounded implementation based on legged_gym / WheeledLab / ANYmal standards.

Action space (2):  [forward, turn_rate]
  forward    ∈ [-1, 1] clamped [0, 1] — forward only
  turn_rate  ∈ [-1, 1] — positive = CCW (left turn), negative = CW (right turn)

  left_joint_vel  =  (forward - turn_rate) * action_scale
  right_joint_vel = -(forward + turn_rate) * action_scale

  Verification:
    (fwd=1, trn=0)  → left=+scale, right=-scale → straight forward ✓
    (fwd=0, trn=1)  → left=-scale, right=-scale → left backward, right forward → pivot left (CCW) ✓
    (fwd=0, trn=-1) → left=+scale, right=+scale → left forward, right backward → pivot right (CW) ✓

Observation space (13):
  root_lin_vel_b    (3) — body-frame linear velocity
  root_ang_vel_b    (3) — body-frame angular velocity
  projected_gravity (3) — tilt indicator
  commands          (2) — [cmd_vx, cmd_wz]
  last_actions      (2) — previous action
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .lunabotics_env_cfg import LunaboticsDirectEnvCfg


class LunaboticsDirectEnv(DirectRLEnv):
    cfg: LunaboticsDirectEnvCfg

    def __init__(self, cfg: LunaboticsDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions      = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._commands     = torch.zeros(self.num_envs, 3, device=self.device)  # [vx, vy=0, wz]

        # wheel joint indices and left/right mask from actual joint names
        self._wheel_ids = self._robot.actuators["wheels"].joint_indices
        wheel_names     = self._robot.actuators["wheels"].joint_names
        self._left_mask = torch.tensor(
            ["Left" in name for name in wheel_names],
            device=self.device, dtype=torch.bool,
        )

        self._wheel_vel_targets = torch.zeros(self.num_envs, 6, device=self.device)
        self._rl_step = 0

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

        # differential drive: fwd clamped [0,1] (forward only), trn in [-1,1]
        fwd = self._actions[:, 0:1].clamp(0.0, 1.0)
        trn = self._actions[:, 1:2]
        left_vel  =  (fwd - trn) * self.cfg.action_scale
        right_vel = -(fwd + trn) * self.cfg.action_scale

        raw_targets = torch.where(
            self._left_mask,
            left_vel.expand(-1, 6),
            right_vel.expand(-1, 6),
        )

        # light smoothing — prevents instantaneous torque spikes
        self._wheel_vel_targets = 0.5 * self._wheel_vel_targets + 0.5 * raw_targets

    def _apply_action(self):
        self._robot.set_joint_velocity_target(self._wheel_vel_targets, joint_ids=self._wheel_ids)

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self._robot.data.root_lin_vel_b,       # 3
            self._robot.data.root_ang_vel_b,       # 3
            self._robot.data.projected_gravity_b,  # 3
            self._commands[:, [0, 2]],             # 2: cmd_vx, cmd_wz
            self._actions,                         # 2
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        vx  = self._robot.data.root_lin_vel_b[:, 0]
        vz  = self._robot.data.root_lin_vel_b[:, 2]
        wz  = self._robot.data.root_ang_vel_b[:, 2]
        cmd_vx = self._commands[:, 0]
        cmd_wz = self._commands[:, 2]

        # ── velocity tracking (research standard: sigma=0.25) ─────────────────
        track_lin = torch.exp(-torch.square(cmd_vx - vx) / 0.25)
        track_ang = torch.exp(-torch.square(cmd_wz - wz) / 0.25)

        # ── stability penalties ────────────────────────────────────────────────
        lin_vel_z  = torch.square(vz)
        ang_vel_xy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        flat_ori   = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        act_rate   = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

        # soft overspeed — linear ramp above preferred cruise speed (~45 deg/s wheel)
        # allows bursts to 210 deg/s for evasive action but discourages sustaining them
        speed_xy  = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        overspeed = torch.clamp(speed_xy - self.cfg.preferred_speed_threshold, min=0.0)

        rewards = {
            "track_lin_vel":       track_lin  * self.cfg.lin_vel_reward_scale,
            "track_ang_vel":       track_ang  * self.cfg.ang_vel_reward_scale,
            "lin_vel_z_l2":        lin_vel_z  * self.cfg.lin_vel_z_scale,
            "ang_vel_xy_l2":       ang_vel_xy * self.cfg.ang_vel_xy_l2_scale,
            "flat_orientation_l2": flat_ori   * self.cfg.flat_orientation_l2_scale,
            "action_rate_l2":      act_rate   * self.cfg.action_rate_l2_scale,
            "overspeed":           overspeed  * self.cfg.overspeed_scale,
        }

        total = torch.zeros(self.num_envs, device=self.device)
        for key, val in rewards.items():
            total += val * self.step_dt
            self._episode_sums[key] += val * self.step_dt

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        tipped   = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > 0.9
        return tipped, time_out

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
            # Phase 0: straight forward only — learn basic locomotion
            vx = torch.empty(n, device=self.device).uniform_(0.3, 0.65)
            self._commands[env_ids, 0] = vx
            self._commands[env_ids, 2] = zeros

        elif self._rl_step < self.cfg.curriculum_phase2_steps:
            # Phase 1: 60% straight, 40% pivot turn — exclusive, no arcs
            mode = torch.rand(n, device=self.device)
            vx   = torch.empty(n, device=self.device).uniform_(0.3, 0.65)
            wz   = torch.empty(n, device=self.device).uniform_(-0.5, 0.5)
            straight = mode < 0.6
            self._commands[env_ids, 0] = torch.where(straight, vx,    zeros)
            self._commands[env_ids, 2] = torch.where(straight, zeros, wz)

        else:
            # Phase 2: arcs unlocked — forward + turn simultaneously
            mode = torch.rand(n, device=self.device)
            vx   = torch.empty(n, device=self.device).uniform_(0.2, 0.65)
            wz   = torch.empty(n, device=self.device).uniform_(-0.8, 0.8)
            straight = mode < 0.5
            self._commands[env_ids, 0] = torch.where(straight, vx, vx * 0.5)
            self._commands[env_ids, 2] = torch.where(straight, zeros, wz)

        self._commands[env_ids, 1] = 0.0

        # reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # episode reward logging
        extras = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = (
                torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = extras
