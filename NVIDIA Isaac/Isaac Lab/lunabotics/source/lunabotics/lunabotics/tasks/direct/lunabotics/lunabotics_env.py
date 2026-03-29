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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.terrains.utils import create_prim_from_mesh

from lunabotics.assets.lunabotics import ARENA_USD_PATH  # isort: skip
from lunabotics.terrains import RegolithTerrainCfg, regolith_terrain  # isort: skip
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

        # Chassis body indices for wall-contact detection.
        # Wheels are always in contact with terrain; chassis only contacts walls.
        # Use the sensor's own body_names to get correct indices.
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

    def _generate_regolith_textures(self) -> str:
        """Generate and cache procedural PBR textures for lunar grey-sand regolith.

        Uses FFT-based Gaussian noise at multiple scales so each octave is
        smooth (no blocky nearest-neighbour artefacts).  Runs once at startup;
        delete assets/regolith_textures/ to regenerate.
        """
        import os
        import numpy as np
        from PIL import Image

        tex_dir = os.path.join(os.path.dirname(__file__), "assets", "regolith_textures")
        os.makedirs(tex_dir, exist_ok=True)
        albedo_path = os.path.join(tex_dir, "albedo.png")
        normal_path = os.path.join(tex_dir, "normal.png")
        rough_path  = os.path.join(tex_dir, "rough.png")

        if all(os.path.exists(p) for p in (albedo_path, normal_path, rough_path)):
            return tex_dir

        S   = 2048
        rng = np.random.default_rng(42)

        def smooth_noise(sigma_px: float) -> np.ndarray:
            """Band-limited Gaussian noise at the given pixel-space sigma (FFT filter)."""
            raw = rng.standard_normal((S, S)).astype(np.float32)
            fx  = np.fft.fftfreq(S).reshape(1, S).astype(np.float32)
            fy  = np.fft.fftfreq(S).reshape(S, 1).astype(np.float32)
            kernel = np.exp(-2.0 * np.pi ** 2 * sigma_px ** 2 * (fx ** 2 + fy ** 2))
            return np.real(np.fft.ifft2(np.fft.fft2(raw) * kernel)).astype(np.float32)

        # Octaves: (sigma_pixels, weight) — coarse regional → fine grain
        h = sum(w * smooth_noise(s) for s, w in [(300, 0.40), (80, 0.30), (25, 0.20), (8, 0.10)])
        h -= h.min(); h /= h.max()

        # Albedo — lighter lunar grey (55–70 % reflectance), subtle cool tint
        v = h * 0.15 + 0.55
        albedo = np.stack([
            (v * 0.94 * 255).clip(0, 255).astype(np.uint8),
            (v * 0.96 * 255).clip(0, 255).astype(np.uint8),
            (v * 1.00 * 255).clip(0, 255).astype(np.uint8),
        ], axis=-1)
        Image.fromarray(albedo).save(albedo_path)

        # Normal map from height gradient
        dx = np.gradient(h, axis=1) * 5.0
        dy = np.gradient(h, axis=0) * 5.0
        nz = np.ones_like(dx)
        L  = np.sqrt(dx ** 2 + dy ** 2 + nz ** 2)
        normal = np.stack([
            ((-dx / L) * 0.5 + 0.5) * 255,
            ((-dy / L) * 0.5 + 0.5) * 255,
            (( nz / L) * 0.5 + 0.5) * 255,
        ], axis=-1).clip(0, 255).astype(np.uint8)
        Image.fromarray(normal).save(normal_path)

        # Roughness — high base (sand ~0.87), small variation
        rough = ((h * 0.07 + 0.85) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rough).save(rough_path)

        print(f"[LunaboticsEnv] Regolith textures generated → {tex_dir}")
        return tex_dir

    def _apply_regolith_textures(self, mesh_prim_path: str, tex_dir: str):
        """Bind regolith PBR textures to the OmniPBR shader on a mesh prim.

        The shader lives at {mesh_prim_path}/visualMaterial/Shader;
        inputs are set as inputs:* attributes directly on the shader prim.
        texture_scale=0.04 → each tile covers ~25 m, so the 27×38 m terrain
        shows ~1.1×1.5 tile repetitions — barely noticeable at any viewing angle.
        """
        import os
        import omni.usd
        from pxr import UsdShade, Sdf

        stage = omni.usd.get_context().get_stage()
        shader_prim = stage.GetPrimAtPath(f"{mesh_prim_path}/visualMaterial/Shader")
        if not shader_prim.IsValid():
            return

        shader = UsdShade.Shader(shader_prim)

        def asset(fname: str) -> Sdf.AssetPath:
            return Sdf.AssetPath(os.path.join(tex_dir, fname).replace("\\", "/"))

        shader.CreateInput("diffuse_texture",             Sdf.ValueTypeNames.Asset).Set(asset("albedo.png"))
        shader.CreateInput("normalmap_texture",           Sdf.ValueTypeNames.Asset).Set(asset("normal.png"))
        shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(asset("rough.png"))
        shader.CreateInput("texture_scale",               Sdf.ValueTypeNames.Float2).Set((0.04, 0.04))

    def _spawn_per_env_terrain(self):
        """Generate a regolith height-field mesh and spawn it in env_0.

        Called in _setup_scene() BEFORE clone_environments() so the mesh is
        replicated to every env as a fully independent physics prim.  Each env
        gets a separate /World/envs/env_N/Ground prim that is never reset — only
        the robot state is reset between episodes, leaving each env's terrain
        intact and unaffected by resets in neighbouring envs.
        """
        terrain_cfg = RegolithTerrainCfg(
            size=self.cfg.regolith_size,
            horizontal_scale=self.cfg.regolith_horizontal_scale,
            vertical_scale=self.cfg.regolith_vertical_scale,
            border_width=0.0,
            slope_threshold=None,
            particle_density_range=self.cfg.regolith_particle_density_range,
            particle_radius_range=self.cfg.regolith_particle_radius_range,
            crest_height_range=self.cfg.regolith_crest_height_range,
        )

        meshes, _ = regolith_terrain(difficulty=0.5, cfg=terrain_cfg)
        mesh = meshes[0]

        # Centre the vertices at (0,0) in-place so the prim needs no translation
        # transform.  Using a USD translation on the mesh sub-prim can behave
        # unexpectedly with USD instancing (copy_from_source=False); baking the
        # offset directly into the vertex positions is more robust.
        cx = float((mesh.bounds[0][0] + mesh.bounds[1][0]) / 2)
        cy = float((mesh.bounds[0][1] + mesh.bounds[1][1]) / 2)
        mesh.vertices[:, 0] -= cx
        mesh.vertices[:, 1] -= cy

        create_prim_from_mesh(
            "/World/envs/env_0/Ground",
            mesh,
            translation=None,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="E:/IsaacLab/_isaac_sim/kit/mdl/core/Base/OmniPBR.mdl",
                project_uvw=True,
            ),
        )

        # Generate procedural grey-sand textures and bind to the OmniPBR shader.
        # Textures are cached to assets/regolith_textures/ after first run.
        tex_dir = self._generate_regolith_textures()
        self._apply_regolith_textures("/World/envs/env_0/Ground", tex_dir)

    def _randomize_per_env_terrain(self):
        """Overwrite each env's Ground/mesh with a uniquely-random terrain.

        Called AFTER clone_environments() so all Ground/mesh prims already exist
        as independent USD prims.  Each call to regolith_terrain() advances
        numpy's RNG, producing a different mound layout per env.  All envs share
        the same cfg parameters (density, radius, height ranges) — only the
        random seed varies.
        """
        import numpy as np
        import omni.usd
        from pxr import UsdGeom

        print(f"[LunaboticsEnv] Randomizing terrain for {self.num_envs} envs …")
        stage = omni.usd.get_context().get_stage()
        terrain_cfg = RegolithTerrainCfg(
            size=self.cfg.regolith_size,
            horizontal_scale=self.cfg.regolith_horizontal_scale,
            vertical_scale=self.cfg.regolith_vertical_scale,
            border_width=0.0,
            slope_threshold=None,
            particle_density_range=self.cfg.regolith_particle_density_range,
            particle_radius_range=self.cfg.regolith_particle_radius_range,
            crest_height_range=self.cfg.regolith_crest_height_range,
        )

        for env_id in range(self.num_envs):
            meshes, _ = regolith_terrain(difficulty=0.5, cfg=terrain_cfg)
            mesh = meshes[0]

            cx = float((mesh.bounds[0][0] + mesh.bounds[1][0]) / 2)
            cy = float((mesh.bounds[0][1] + mesh.bounds[1][1]) / 2)
            mesh.vertices[:, 0] -= cx
            mesh.vertices[:, 1] -= cy

            mesh_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Ground/mesh")
            if not mesh_prim.IsValid():
                continue

            geom = UsdGeom.Mesh(mesh_prim)
            geom.GetPointsAttr().Set(mesh.vertices.astype(np.float32))
            geom.GetFaceVertexIndicesAttr().Set(mesh.faces.flatten().astype(np.int32))
            geom.GetFaceVertexCountsAttr().Set(np.full(len(mesh.faces), 3, dtype=np.int32))

        print(f"[LunaboticsEnv] Terrain randomization complete.")

    def _setup_scene(self):
        # 1. Static geometry FIRST — must be spawned before Articulation and clone_environments
        arena_cfg = sim_utils.UsdFileCfg(
            usd_path=ARENA_USD_PATH,
            scale=(5.0, 5.0, 5.0),
        )
        arena_cfg.func(
            "/World/envs/env_.*/Arena",
            arena_cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.7071, 0.0, 0.0, 0.7071),   # +90° around Z (counter-clockwise from above)
        )

        # 1b. Per-env regolith terrain — spawned in env_0 so clone_environments()
        #     replicates it to every env as a physically independent mesh.
        self._spawn_per_env_terrain()

        # 2. Robot articulation
        self._robot = Articulation(self.cfg.robot)

        # 3. Infinite flat ground plane — visual backdrop + physics fallback.
        #    Not used for robot env_origins (we use GridCloner origins instead).
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # Sink the plane below the arena floor.  The prim is loaded from a USD
        # file so XformCommonAPI is incompatible; use AddXformOp directly instead.
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath(self.cfg.terrain.prim_path + "/terrain")
        if ground_prim.IsValid():
            for op in UsdGeom.Xformable(ground_prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(0.0, 0.0, -0.8))
                    break

        # 4. Clone environments — replicates arena + terrain + robot from env_0 to all envs.
        #    Each env ends up at a grid position determined by env_spacing (via GridCloner).
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 4b. Overwrite each env's terrain with a unique random mesh.
        #     Must be called after clone_environments() so the Ground/mesh prims exist.
        self._randomize_per_env_terrain()

        # Env origins (world-frame XYZ centre of each env) are pre-computed by the
        # GridCloner in InteractiveScene.__init__ and stored here.
        self._env_origins = self.scene._default_env_origins

        # 5. Scene registration AFTER clone (matches assembly_env.py working pattern)
        self.scene.articulations["robot"] = self._robot

        # 6. Contact sensor — detects chassis↔wall contacts.
        # No filter needed: chassis is elevated above terrain so only walls can touch it.
        # Wheels are excluded at runtime via self._chassis_body_ids (set in __init__).
        self._arena_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                history_length=2,
                update_period=0.0,
                track_air_time=False,
            )
        )
        self.scene.sensors["arena_contact"] = self._arena_contact

        # Single global sun — distant light applies to every env identically.
        # Elevation ~35° above horizon, azimuth from the +X/+Y diagonal.
        # q = (cos27.5°, sin27.5°·(-0.707, 0.707, 0)) = (0.887, -0.326, 0.326, 0.0)
        sun_cfg = sim_utils.DistantLightCfg(
            intensity=50_000.0,
            exposure=0.5,
            color=(1.00, 0.95, 0.85),   # warm sunlight
            angle=0.5,                  # angular diameter ~0.5° (realistic sun disc)
        )
        sun_cfg.func("/World/Sun", sun_cfg, orientation=(0.887, -0.326, 0.326, 0.0))

        # Enable shadow casting on the sun prim via UsdLux ShadowAPI
        import omni.usd
        from pxr import UsdLux
        _stage = omni.usd.get_context().get_stage()
        _sun_prim = _stage.GetPrimAtPath("/World/Sun")
        if _sun_prim.IsValid():
            shadow_api = UsdLux.ShadowAPI.Apply(_sun_prim)
            shadow_api.CreateShadowEnableAttr(True)
            shadow_api.CreateShadowColorAttr((0.0, 0.0, 0.0))
            shadow_api.CreateShadowFalloffAttr(0.0)
            shadow_api.CreateShadowDistanceAttr(500.0)

        # Dim global fill/sky light
        dome_cfg = sim_utils.DomeLightCfg(intensity=375.0, color=(0.75, 0.80, 0.90))
        dome_cfg.func("/World/Light", dome_cfg)

        # Two off-centre overhead disk lights per arena
        disk_cfg = sim_utils.DiskLightCfg(
            intensity=45_000.0,
            radius=1.5,
            color=(1.00, 0.95, 0.88),  # warm-white
        )
        disk_cfg.func(
            "/World/envs/env_.*/SpotA",
            disk_cfg,
            translation=(-7.0, 5.0, 12.0),
        )
        disk_cfg.func(
            "/World/envs/env_.*/SpotB",
            disk_cfg,
            translation=(7.0, -5.0, 12.0),
        )

        # Two side lights on the Y axis, outside the long walls (±19 m),
        # angled 45° down toward the arena centre — pure X-axis rotation.
        # SpotC at (0, +25, 25): rotate -Z by 45° around -X → q=(0.924, -0.383, 0, 0)
        # SpotD at (0, -25, 25): rotate -Z by 45° around +X → q=(0.924,  0.383, 0, 0)
        angled_cfg = sim_utils.DiskLightCfg(
            intensity=67_500.0,
            radius=2.5,
            color=(0.88, 0.93, 1.00),  # cooler tint for contrast
        )
        angled_cfg.func(
            "/World/envs/env_.*/SpotC",
            angled_cfg,
            translation=(0.0, 25.0, 25.0),
            orientation=(0.924, -0.383, 0.0, 0.0),
        )
        angled_cfg.func(
            "/World/envs/env_.*/SpotD",
            angled_cfg,
            translation=(0.0, -25.0, 25.0),
            orientation=(0.924, 0.383, 0.0, 0.0),
        )

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

        # ── arena wall collision ───────────────────────────────────────────────
        # net_forces_w: (num_envs, num_bodies, 3)
        # Only check chassis bodies — wheels are always touching terrain (false positive).
        # Chassis is elevated above terrain, so any force on it = wall contact.
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

        # wall collision is a one-time event penalty — not scaled by dt
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
        default_root_state[:, :3] += self._env_origins[env_ids]
        default_root_state[:, 0]  += self.cfg.robot_spawn_x_offset
        default_root_state[:, 1]  += self.cfg.robot_spawn_y_offset
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
