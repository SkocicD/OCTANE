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

Observation space (15):
  root_lin_vel_b    (3) — body-frame linear velocity
  root_ang_vel_b    (3) — body-frame angular velocity
  projected_gravity (3) — tilt indicator
  commands          (2) — [cmd_vx, cmd_wz]
  last_actions      (2) — previous action
  heading           (2) — [cos(yaw), sin(yaw)] — 0° = +X world
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

        # ── curriculum state ──────────────────────────────────────────────────
        self._phase            = 0      # 0 / 1 / 2
        self._phase_entry_step = 0      # _rl_step when current phase was entered
        self._ema_lin          = 0.0    # EMA of mean track_lin_vel across envs
        self._ema_ang          = 0.0    # EMA of mean track_ang_vel across envs
        self._base_episode_length_s = self.cfg.episode_length_s  # original value before increases
        self._ema_fwd              = 0.0    # EMA of mean forward speed (m/s)

        # ── diagnostics ───────────────────────────────────────────────────────
        body_names = self._robot.data.body_names
        live_masses = self._robot.root_physx_view.get_masses()[0]  # env_0, shape (num_bodies,)
        total_mass = live_masses.sum().item()
        print(f"[LunaboticsEnv] Wheel joints ({len(wheel_names)}): {wheel_names}")
        print(f"[LunaboticsEnv] Left mask: {self._left_mask.tolist()}")
        print(f"[LunaboticsEnv] Robot total mass: {total_mass:.2f} kg  (factor={self.cfg.mass_correction_factor:.0f}×)")
        print(f"[LunaboticsEnv] Per-link masses:")
        for name, mass in zip(body_names, live_masses.tolist()):
            print(f"  {name:40s} {mass:10.3f} kg")

        # Body positions relative to root (env_0, default pose).
        # wheel center world z ≈ wheel radius when robot rests on z=0 ground.
        root_pos   = self._robot.data.default_root_state[0, :3]
        body_pos_w = self._robot.data.body_pos_w[0]
        body_pos_rel = body_pos_w - root_pos
        print(f"[LunaboticsEnv] Body positions relative to root (spawn pose, env 0):")
        print(f"  {'body':40s}  {'x':>8}  {'y':>8}  {'z':>8}")
        for name, pos in zip(body_names, body_pos_rel.tolist()):
            print(f"  {name:40s}  {pos[0]:8.4f}  {pos[1]:8.4f}  {pos[2]:8.4f}")
        wheel_world_zs = [
            body_pos_w[i, 2].item()
            for i, name in enumerate(body_names) if "Wheel" in name
        ]
        if wheel_world_zs:
            avg_wheel_z = sum(wheel_world_zs) / len(wheel_world_zs)
            print(f"[LunaboticsEnv] Root spawn z:          {root_pos[2].item():.4f} m")
            print(f"[LunaboticsEnv] Avg wheel center z:    {avg_wheel_z:.4f} m  (≈ wheel radius if resting on z=0)")
            print(f"[LunaboticsEnv] CAD wheel radius:      0.34544 m  (13.6 in)")


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
                "wheel_consistency",
                "wheel_slip",
                "alive_bonus",
                "forward_progress",
                "stall_penalty",
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
        """Overwrite each env's Ground/mesh points with a uniquely-random terrain.

        Optimisations vs the naive loop:
        - Calls regolith_terrain.__wrapped__ (raw height-field fn) to skip
          trimesh object creation entirely.
        - Precomputes x,y vertex positions once — they are grid-fixed and
          identical across all envs; only z changes per env.
        - Skips writing faceVertexIndices/faceVertexCounts — topology is
          already correct on every cloned prim from clone_environments().
        - Parallel generation via ThreadPoolExecutor (numpy releases the GIL
          for array operations).
        - Batches all USD writes inside a single Sdf.ChangeBlock.
        """
        import os
        import numpy as np
        import omni.usd
        from pxr import UsdGeom, Sdf
        from concurrent.futures import ThreadPoolExecutor
        from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

        print(f"[LunaboticsEnv] Randomizing terrain for {self.num_envs} envs …")
        stage = omni.usd.get_context().get_stage()

        h_scale   = self.cfg.regolith_horizontal_scale
        v_scale   = self.cfg.regolith_vertical_scale
        # The height_field_to_mesh wrapper always adds 1 border pixel even when
        # border_width=0 (border_pixels = int(0/h_scale) + 1 = 1).
        border_px = 1
        width_px  = int(self.cfg.regolith_size[0] / h_scale) + 1
        length_px = int(self.cfg.regolith_size[1] / h_scale) + 1
        inner_size = (
            (width_px  - 2 * border_px) * h_scale,
            (length_px - 2 * border_px) * h_scale,
        )

        # cfg sized to match what __wrapped__ receives inside the decorator
        terrain_cfg = RegolithTerrainCfg(
            size=inner_size,
            horizontal_scale=h_scale,
            vertical_scale=v_scale,
            border_width=0.0,
            slope_threshold=None,
            particle_density_range=self.cfg.regolith_particle_density_range,
            particle_radius_range=self.cfg.regolith_particle_radius_range,
            crest_height_range=self.cfg.regolith_crest_height_range,
        )

        # Precompute centred x,y positions from a flat (zero) height field —
        # they never change, so we do this once and reuse across all envs.
        zero_hf = np.zeros((width_px, length_px), dtype=np.int16)
        verts_template, _ = convert_height_field_to_mesh(zero_hf, h_scale, v_scale, None)
        half_x = float(verts_template[:, 0].max() / 2.0)
        half_y = float(verts_template[:, 1].max() / 2.0)
        xy_base = verts_template[:, :2].copy().astype(np.float32)
        xy_base[:, 0] -= half_x
        xy_base[:, 1] -= half_y

        raw_fn = regolith_terrain.__wrapped__

        def _gen_verts(_):
            hf_inner = raw_fn(0.5, terrain_cfg)   # (width_px-2, length_px-2) int16
            heights = np.zeros((width_px, length_px), dtype=np.int16)
            heights[border_px:-border_px, border_px:-border_px] = hf_inner
            z = heights.flatten().astype(np.float32) * v_scale
            verts = np.empty((len(xy_base), 3), dtype=np.float32)
            verts[:, :2] = xy_base
            verts[:, 2]  = z
            return verts

        workers = min(self.num_envs, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_verts = list(pool.map(_gen_verts, range(self.num_envs)))

        # Single ChangeBlock — batches all USD attribute writes as one transaction
        with Sdf.ChangeBlock():
            for env_id, verts in enumerate(all_verts):
                mesh_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Ground/mesh")
                if mesh_prim.IsValid():
                    UsdGeom.Mesh(mesh_prim).GetPointsAttr().Set(verts)

        print("[LunaboticsEnv] Terrain randomization complete.")

    def _correct_robot_masses_in_usd(self):
        """Divide all PhysicsMassAPI mass and inertia values under env_0/Robot by
        mass_correction_factor.

        Must be called AFTER the robot prim is spawned in USD but BEFORE
        clone_environments() and before PhysX initializes the articulation.
        That window is inside _setup_scene(), which is exactly where this is called.
        PhysX reads USD values once at sim start; edits made here are permanent for
        the session.  clone_environments() replicates the corrected prim to all envs.
        """
        import omni.usd
        from pxr import Gf, Usd

        factor = self.cfg.mass_correction_factor
        stage  = omni.usd.get_context().get_stage()
        robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_root.IsValid():
            print("[LunaboticsEnv] WARNING: /World/envs/env_0/Robot not found — mass correction skipped")
            return

        # Masses are density-computed from physics material prims — scale those.
        corrected = 0
        for prim in stage.Traverse():
            density_attr = prim.GetAttribute("physics:density")
            if not density_attr.IsValid():
                continue
            val = density_attr.Get()
            if val is None or float(val) == 0.0:
                continue
            density_attr.Set(float(val) / factor)
            print(f"[LunaboticsEnv]   {prim.GetPath().pathString}  density {val:.1f} → {float(val)/factor:.2f}")
            corrected += 1

        print(f"[LunaboticsEnv] Mass correction: scaled {corrected} density attrs by 1/{factor:.0f}")

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

        # Add RigidBodyAPI (kinematic) + contact reporting to the arena so we can
        # detect when the robot touches it.  Done on env_0 before clone so every
        # env gets the APIs.  The arena is static geometry — this just enables
        # PhysX to report contacts on it without changing its physics behaviour.
        import omni.usd
        from pxr import UsdPhysics as _UsdPhysics
        _stage = omni.usd.get_context().get_stage()
        _arena_prim = _stage.GetPrimAtPath("/World/envs/env_0/Arena")
        if _arena_prim.IsValid():
            rb_api = _UsdPhysics.RigidBodyAPI.Apply(_arena_prim)
            rb_api.CreateKinematicEnabledAttr(True)
            from pxr import PhysxSchema
            cr_api = PhysxSchema.PhysxContactReportAPI.Apply(_arena_prim)
            cr_api.CreateThresholdAttr(0)

        # 1b. Per-env regolith terrain — spawned in env_0 so clone_environments()
        #     replicates it to every env as a physically independent mesh.
        self._spawn_per_env_terrain()

        # 2. Robot articulation
        self._robot = Articulation(self.cfg.robot)

        # 2b. Correct USD mass values before PhysX initializes — must happen here,
        #     after prim is spawned but before clone_environments() and sim start.
        self._correct_robot_masses_in_usd()

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

        # 6. Contact sensor on the ARENA (not the robot).
        #    The arena is kinematic (fixed) — nothing touches it except the robot.
        #    Terrain is a separate static prim and static-on-static produces zero
        #    contact forces in PhysX.  So ANY force on the arena = robot collision.
        #    No threshold tuning needed, no false positives from terrain or digging.
        self._arena_contact = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Arena",
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
        # Heading as (cos, sin) — 0° = +X world, avoids discontinuity at 0/2π.
        # Extracted from root quaternion (w, x, y, z).
        quat = self._robot.data.root_quat_w  # (N, 4)
        yaw  = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        heading = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (N, 2)

        obs = torch.cat([
            self._robot.data.root_lin_vel_b,       # 3
            self._robot.data.root_ang_vel_b,       # 3
            self._robot.data.projected_gravity_b,  # 3
            self._commands[:, [0, 2]],             # 2: cmd_vx, cmd_wz
            self._actions,                         # 2
            heading,                               # 2: cos(yaw), sin(yaw)
        ], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        vx  = self._robot.data.root_lin_vel_b[:, 1]  # body +Y = robot forward (bucket side)
        vz  = self._robot.data.root_lin_vel_b[:, 2]
        wz  = self._robot.data.root_ang_vel_b[:, 2]
        cmd_vx = self._commands[:, 0]
        cmd_wz = self._commands[:, 2]

        # ── one-time direction diagnostic (first 3 steps) ─────────────────────
        if self._rl_step <= 3:
            vy_w = self._robot.data.root_lin_vel_w[0, 1].item()
            print(f"[DEBUG step={self._rl_step}] act={self._actions[0].tolist()}  "
                  f"vy_body={vx[0]:.4f}  vy_world={vy_w:.4f}  cmd_vx={cmd_vx[0]:.3f}")

        # ── velocity tracking (sigma=0.25 — wide basin of attraction) ─────────
        # At sigma=0.25 the gradient at vx=0 toward cmd_vx=0.5 is 1.47/step,
        # nearly 2× stronger than sigma=0.10 (0.82/step).  Wider basin makes it
        # much easier for a random initial policy to discover forward motion.
        track_lin = torch.exp(-torch.square(cmd_vx - vx) / 0.25)
        track_ang = torch.exp(-torch.square(cmd_wz - wz) / 0.25)

        # ── alive bonus — constant per-step survival reward ─────────────────
        # Gives the policy a baseline reason to stay in-bounds, avoid walls,
        # and not tip over.  Without this, early penalty-dominated training can
        # teach "do nothing."
        alive_bonus = torch.ones_like(vx)

        # ── forward progress — body-frame forward velocity ─────────────────
        # Body +Y = robot forward (bucket side).  Rewards path-length along
        # current heading regardless of world orientation — works for any spawn
        # heading and after turns.  Unclamped so backward body motion is penalised.
        forward_progress = vx  # same as root_lin_vel_b[:, 1]

        # ── stall penalty — discrete negative for near-zero velocity ──────────
        # Catches the zero-output local optimum that continuous rewards can't
        # fully break.  Only fires when the robot is truly stationary.
        stall = (torch.abs(vx) < self.cfg.stall_threshold).float()

        # ── stability penalties ────────────────────────────────────────────────
        lin_vel_z  = torch.square(vz)
        ang_vel_xy = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        flat_ori   = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        act_rate   = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

        # soft overspeed
        speed_xy  = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        overspeed = torch.clamp(speed_xy - self.cfg.preferred_speed_threshold, min=0.0)

        # ── wheel consistency: variance within each side ───────────────────────
        wheel_vels = self._robot.data.joint_vel[:, self._wheel_ids]
        left_vels  = wheel_vels[:,  self._left_mask]
        right_vels = wheel_vels[:, ~self._left_mask]
        wheel_consistency = torch.var(left_vels, dim=1) + torch.var(right_vels, dim=1)

        # ── wheel slip ─────────────────────────────────────────────────────────
        avg_wheel_cmd   = torch.mean(torch.abs(self._wheel_vel_targets), dim=1)
        predicted_speed = avg_wheel_cmd * self.cfg.wheel_radius
        wheel_slip      = torch.clamp(predicted_speed - speed_xy - self.cfg.wheel_slip_deadzone, min=0.0)

        # ── arena wall collision (force on the arena itself) ──────────────────
        # The arena is kinematic — nothing touches it except the robot.
        # ANY force on any arena body = robot collision.  Low threshold (1 N)
        # to catch even gentle bumps; no false positives possible.
        arena_forces = self._arena_contact.data.net_forces_w  # (N, arena_bodies, 3)
        max_arena_force = arena_forces.norm(dim=-1).amax(dim=-1)  # (N,)
        arena_hit = (max_arena_force > self.cfg.wall_contact_threshold).float()

        # ── curriculum EMA + phase advancement ────────────────────────────────
        alpha = self.cfg.curriculum_ema_alpha
        self._ema_lin = (1.0 - alpha) * self._ema_lin + alpha * track_lin.mean().item()
        self._ema_ang = (1.0 - alpha) * self._ema_ang + alpha * track_ang.mean().item()

        steps_in_phase = self._rl_step - self._phase_entry_step
        if self._phase == 0 and steps_in_phase >= self.cfg.curriculum_phase0_min_steps:
            if self._ema_lin >= self.cfg.curriculum_phase1_threshold:
                self._phase = 1
                self._phase_entry_step = self._rl_step
                print(f"[Curriculum] → Phase 1  (step={self._rl_step}  ema_lin={self._ema_lin:.3f})")
        elif self._phase == 1 and steps_in_phase >= self.cfg.curriculum_phase1_min_steps:
            if (self._ema_lin >= self.cfg.curriculum_phase2_lin_threshold
                    and self._ema_ang >= self.cfg.curriculum_phase2_ang_threshold):
                self._phase = 2
                self._phase_entry_step = self._rl_step
                print(f"[Curriculum] → Phase 2  (step={self._rl_step}  ema_lin={self._ema_lin:.3f}  ema_ang={self._ema_ang:.3f})")

        # ── episode length curriculum (speed-based) ─────────────────────────
        # Ramps linearly from base (20s) to max (300s) as the EMA of forward
        # speed approaches target_speed.  Short episodes early = less wasted
        # compute; longer episodes as the robot improves = practice sustained drive.
        self._ema_fwd = (1.0 - alpha) * self._ema_fwd + alpha * forward_progress.mean().item()
        frac = min(1.0, max(0.0, self._ema_fwd / self.cfg.episode_length_target_speed))
        new_s = self._base_episode_length_s + (self.cfg.episode_length_max_s - self._base_episode_length_s) * frac
        new_s = round(new_s)
        if new_s != self.cfg.episode_length_s:
            self.cfg.episode_length_s = new_s

        # ── reward curriculum weights ─────────────────────────────────────────
        p2 = float(self._phase >= 2)

        dt_rewards = {
            # ── always active ─────────────────────────────────────────────────
            "alive_bonus":         alive_bonus       * self.cfg.alive_bonus_scale,
            "track_lin_vel":       track_lin         * self.cfg.lin_vel_reward_scale,
            "track_ang_vel":       track_ang         * self.cfg.ang_vel_reward_scale,
            "forward_progress":    forward_progress  * self.cfg.forward_progress_scale,
            "stall_penalty":       stall             * self.cfg.stall_penalty_scale,
            "lin_vel_z_l2":        lin_vel_z         * self.cfg.lin_vel_z_scale,
            "ang_vel_xy_l2":       ang_vel_xy        * self.cfg.ang_vel_xy_l2_scale,
            "flat_orientation_l2": flat_ori          * self.cfg.flat_orientation_l2_scale,
            "action_rate_l2":      act_rate          * self.cfg.action_rate_l2_scale,
            # ── differential drive enforcement (always active) ────────────────
            "wheel_consistency":   wheel_consistency * self.cfg.wheel_consistency_scale,
            "wheel_slip":          wheel_slip        * self.cfg.wheel_slip_scale,
            # ── phase 2+ ─────────────────────────────────────────────────────
            "overspeed":           overspeed  * self.cfg.overspeed_scale  * p2,
        }

        total = torch.zeros(self.num_envs, device=self.device)
        for key, val in dt_rewards.items():
            total += val * self.step_dt
            self._episode_sums[key] += val * self.step_dt

        # only_positive_rewards (legged_gym default): clamp dt-reward total at 0
        # so penalty-dominated early steps don't teach "do nothing to avoid loss."
        # Applied BEFORE wall collision so termination penalties still bite.
        if self.cfg.only_positive_rewards:
            total = torch.clamp(total, min=0.0)

        # wall collision — one-time event penalty, not dt-scaled
        wall_penalty = arena_hit * self.cfg.wall_collision_scale
        total += wall_penalty
        self._episode_sums["wall_collision"] += wall_penalty

        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out  = self.episode_length_buf >= self.max_episode_length - 1
        tipped    = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1) > 0.9
        arena_forces = self._arena_contact.data.net_forces_w
        arena_hit = arena_forces.norm(dim=-1).amax(dim=-1) > self.cfg.wall_contact_threshold
        return tipped | arena_hit, time_out

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

        if self._phase == 0:
            # Phase 0: straight forward only — learn basic locomotion
            vx = torch.empty(n, device=self.device).uniform_(0.3, 0.65)
            self._commands[env_ids, 0] = vx
            self._commands[env_ids, 2] = zeros

        elif self._phase == 1:
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

        # reset robot state — random position + yaw within start zone
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._env_origins[env_ids]

        # Random XY within the start zone (inset by margin for clearance)
        cfg = self.cfg
        half_x = cfg.spawn_zone_size_x / 2.0 - cfg.spawn_zone_margin
        half_y = cfg.spawn_zone_size_y / 2.0 - cfg.spawn_zone_margin
        default_root_state[:, 0] += cfg.spawn_zone_center_x + torch.empty(n, device=self.device).uniform_(-half_x, half_x)
        default_root_state[:, 1] += cfg.spawn_zone_center_y + torch.empty(n, device=self.device).uniform_(-half_y, half_y)

        # Random yaw (rotation about Z) — quaternion (w, x, y, z) = (cos θ/2, 0, 0, sin θ/2)
        if cfg.spawn_random_yaw:
            yaw = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * 3.14159265)
            default_root_state[:, 3] = torch.cos(yaw * 0.5)  # w
            default_root_state[:, 4] = 0.0                     # x
            default_root_state[:, 5] = 0.0                     # y
            default_root_state[:, 6] = torch.sin(yaw * 0.5)  # z

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
        extras["Curriculum/phase"]          = float(self._phase)
        extras["Curriculum/ema_lin"]        = self._ema_lin
        extras["Curriculum/ema_ang"]        = self._ema_ang
        extras["Curriculum/ema_fwd_speed"]  = self._ema_fwd
        extras["Curriculum/episode_len_s"]  = self.cfg.episode_length_s
        self.extras["log"] = extras
