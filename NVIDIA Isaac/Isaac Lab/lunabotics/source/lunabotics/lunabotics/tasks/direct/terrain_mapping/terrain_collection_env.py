"""DirectRLEnv for terrain data collection.

No arena, no RL curriculum, no commands.  The robot spawns on open regolith
terrain with randomly placed rocks and walls each episode.  The collect script
drives it with zero actions while cameras capture frames.

Action space  (2): [forward, turn_rate]
Observation   (7): root_state_w pos + quat  (not used by collector)
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.terrains.utils import create_prim_from_mesh

from lunabotics.terrains import RegolithTerrainCfg, regolith_terrain  # isort: skip
from .terrain_collection_env_cfg import TerrainCollectionEnvCfg


def _jagged_sphere_mesh(
    rng: np.random.Generator,
    radius: float,
    n_lat: int = 9,
    n_lon: int = 13,
    roughness: float = 0.35,
) -> tuple:
    """Return (pts list-of-tuples, face_counts list, face_indices list) for a jagged sphere.

    Each vertex radius is perturbed independently, giving a rough, non-spherical rock shape.
    Topology (face_counts / face_indices) is identical every call for the same n_lat/n_lon —
    only the vertex positions change per call, so callers can skip re-setting topology when
    the per-episode shape update only needs new points.
    """
    pts = []
    for i in range(n_lat + 1):
        theta = float(np.pi * i / n_lat)
        st = float(np.sin(theta))
        ct = float(np.cos(theta))
        for j in range(n_lon):
            phi = float(2.0 * np.pi * j / n_lon)
            r = float(radius * rng.uniform(1.0 - roughness, 1.0 + roughness))
            pts.append((r * st * float(np.cos(phi)), r * st * float(np.sin(phi)), r * ct))

    fc, fi = [], []
    for i in range(n_lat):
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            fc.append(4)
            fi.extend([i * n_lon + j, i * n_lon + jn, (i + 1) * n_lon + jn, (i + 1) * n_lon + j])

    return pts, fc, fi


def _sample_rock_mat(rng: np.random.Generator) -> tuple:
    """Return (diffuse_rgb, roughness, metallic) sampled from a continuous rock distribution."""
    roll = float(rng.random())
    v    = float(rng.uniform(0.12, 0.88))
    if roll < 0.55:                             # pure grey
        t = float(rng.uniform(-0.04, 0.06))
        d = (v + t, v, max(0.0, v - t * 0.5))
    elif roll < 0.80:                           # warm brownish-grey
        b = float(rng.uniform(0.04, 0.22))
        d = (min(1.0, v + b), v, max(0.0, v - b * 0.4))
    elif roll < 0.92:                           # cool grey (slight blue)
        c = float(rng.uniform(0.02, 0.08))
        d = (max(0.0, v - c), v, min(1.0, v + c))
    elif roll < 0.97:                           # reddish-brown / iron oxide
        d = (min(1.0, v * 1.30), v * 0.76, v * 0.55)
    else:                                       # dark basalt
        v = float(rng.uniform(0.06, 0.22))
        d = (v, v, min(1.0, v + 0.01))
    diffuse   = tuple(float(np.clip(x, 0.0, 1.0)) for x in d)
    roughness = float(rng.uniform(0.70, 0.96))
    metallic  = float(rng.uniform(0.0, 0.08)) if float(rng.random()) < 0.12 else 0.0
    return diffuse, roughness, metallic


def _sample_wall_mat(rng: np.random.Generator, allow_glass: bool = True) -> tuple:
    """Return (diffuse_rgb, roughness, metallic, opacity) for a random wall type."""
    # Glass / acrylic 50% of the time (when allowed); remaining types split evenly.
    t = 1 if (allow_glass and float(rng.random()) < 0.5) else [0, 2, 3, 4][int(rng.integers(4))]
    if t == 0:                                  # concrete
        v = float(rng.uniform(0.28, 0.78))
        w = float(rng.uniform(-0.03, 0.04))
        d = (v + w, v, max(0.0, v - w * 0.8))
        r, m, o = float(rng.uniform(0.82, 0.98)), 0.0, 1.0
    elif t == 1:                                # hazed glass / acrylic
        h   = float(rng.uniform(0.60, 0.96))
        tnt = int(rng.integers(3))
        if tnt == 0:   d = (h * 0.82, h * 0.90, h)
        elif tnt == 1: d = (h * 0.86, h, h * 0.86)
        else:          d = (h, h, h)
        # Low roughness → strong specular highlight makes glass clearly visible
        r, m, o = float(rng.uniform(0.02, 0.12)), 0.0, float(rng.uniform(0.40, 0.70))
    elif t == 2:                                # metal / aluminium framing
        v = float(rng.uniform(0.22, 0.68))
        d = (v, v, min(1.0, v * 1.04))
        r, m, o = float(rng.uniform(0.08, 0.50)), float(rng.uniform(0.72, 0.98)), 1.0
    elif t == 3:                                # safety / hi-vis colour
        palette = [(0.92, 0.46, 0.05), (1.0, 0.82, 0.0),
                   (0.08, 0.42, 0.90), (0.85, 0.08, 0.08)]
        d = palette[int(rng.integers(len(palette)))]
        r, m, o = float(rng.uniform(0.55, 0.82)), 0.0, 1.0
    else:                                       # painted board / plywood
        v = float(rng.uniform(0.68, 0.96))
        w = float(rng.uniform(-0.06, 0.06))
        d = (min(1.0, v + w), v, max(0.0, v - w))
        r, m, o = float(rng.uniform(0.60, 0.88)), 0.0, 1.0
    diffuse = tuple(float(np.clip(x, 0.0, 1.0)) for x in d)
    return diffuse, r, m, o


def _sample_ground_mat(rng: np.random.Generator) -> tuple:
    """Return (diffuse_rgb, roughness, metallic) for a random ground type."""
    t = int(rng.integers(6))
    if t == 0:                                  # lunar regolith (tan-grey)
        v = float(rng.uniform(0.50, 0.82))
        w = float(rng.uniform(-0.03, 0.08))
        d = (v + w, v, max(0.0, v - w * 0.5))
        r = float(rng.uniform(0.84, 0.96))
    elif t == 1:                                # sand (warm yellow-tan)
        v = float(rng.uniform(0.60, 0.86))
        d = (min(1.0, v * 1.10), v, v * 0.68)
        r = float(rng.uniform(0.78, 0.92))
    elif t == 2:                                # gravel (grey, rough)
        v = float(rng.uniform(0.28, 0.62))
        d = (v, v, min(1.0, v * 1.03))
        r = float(rng.uniform(0.84, 0.96))
    elif t == 3:                                # dirt / red soil
        v = float(rng.uniform(0.32, 0.62))
        d = (min(1.0, v * 1.18), v * 0.84, v * 0.62)
        r = float(rng.uniform(0.82, 0.94))
    elif t == 4:                                # light rock slab
        v = float(rng.uniform(0.42, 0.80))
        d = (min(1.0, v * 1.02), v, v * 0.97)
        r = float(rng.uniform(0.70, 0.88))
    else:                                       # basalt (very dark)
        v = float(rng.uniform(0.08, 0.26))
        d = (v, v, min(1.0, v * 1.06))
        r = float(rng.uniform(0.68, 0.82))
    diffuse  = tuple(float(np.clip(x, 0.0, 1.0)) for x in d)
    metallic = float(rng.uniform(0.0, 0.04)) if float(rng.random()) < 0.10 else 0.0
    return diffuse, r, metallic


_POLYHAVEN_SLUGS = [
    "sand_01",
    "coast_sand_01",
    "sandy_gravel_02",
    "rocky_dirt_02",
    "dried_mud_03",
    "red_muddy_ground",
    "ground_grey_var1",
    "rock_ground_02",
    "gravel_dirt_path",
    "aerial_rocks_01",
]


def _try_download_polyhaven(slug: str, dest_dir: str) -> bool:
    """Download one Poly Haven 1k PBR texture set into dest_dir. Returns True on success."""
    import os
    import urllib.request

    os.makedirs(dest_dir, exist_ok=True)
    base = f"https://dl.polyhaven.org/file/ph-assets/Textures/{slug}/1k"

    def _get(url: str, path: str) -> bool:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(path, "wb") as f:
                    f.write(resp.read())
            return True
        except Exception:
            return False

    albedo_path = os.path.join(dest_dir, "albedo.jpg")
    downloaded = False
    for suffix in ("_diff_1k.jpg", "_col_1k.jpg", "_diff_1k.png"):
        if _get(f"{base}/{slug}{suffix}", albedo_path):
            downloaded = True
            break
    if not downloaded:
        return False

    if not _get(f"{base}/{slug}_nor_gl_1k.jpg", os.path.join(dest_dir, "normal.jpg")):
        return False
    if not _get(f"{base}/{slug}_rough_1k.jpg", os.path.join(dest_dir, "rough.jpg")):
        return False

    return True


def _make_texture_variants(
    base_dir: str,
    variants_dir: str,
    n: int,
    rng: np.random.Generator,
) -> list[str]:
    """Produce n brightness / contrast / hue-shifted variants of a PBR texture set."""
    import os
    import shutil
    from PIL import Image, ImageEnhance

    def _first(names):
        return next(
            (os.path.join(base_dir, fn)
             for fn in names if os.path.exists(os.path.join(base_dir, fn))),
            None,
        )

    albedo_src = _first(("albedo.jpg", "albedo.png"))
    normal_src = _first(("normal.jpg", "normal.png"))
    rough_src  = _first(("rough.jpg",  "rough.png"))
    if albedo_src is None:
        return []

    dirs = []
    for v in range(n):
        vdir = os.path.join(variants_dir, f"v{v:02d}")
        os.makedirs(vdir, exist_ok=True)

        img = Image.open(albedo_src).convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.80, 1.20)))
        img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.88, 1.12)))
        img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.78, 1.22)))

        hue_shift = float(rng.uniform(-10.0, 10.0))
        if abs(hue_shift) > 0.5:
            try:
                from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
                arr = np.array(img).astype(np.float32) / 255.0
                hsv = rgb_to_hsv(arr)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 360.0) % 1.0
                img = Image.fromarray(
                    (hsv_to_rgb(hsv) * 255.0).clip(0, 255).astype(np.uint8)
                )
            except ImportError:
                pass

        img.save(os.path.join(vdir, "albedo.png"))

        if normal_src:
            shutil.copy(normal_src, os.path.join(vdir, "normal.png"))
        else:
            Image.fromarray(np.full((64, 64, 3), (128, 128, 255), dtype=np.uint8)).save(
                os.path.join(vdir, "normal.png")
            )

        if rough_src:
            rimg = ImageEnhance.Brightness(Image.open(rough_src).convert("L")).enhance(
                float(rng.uniform(0.88, 1.12))
            )
            rimg.save(os.path.join(vdir, "rough.png"))
        else:
            Image.fromarray(np.full((64, 64), 200, dtype=np.uint8)).save(
                os.path.join(vdir, "rough.png")
            )

        dirs.append(vdir)

    return dirs


def _gaussian_smooth_hf(heights: np.ndarray, sigma: float) -> np.ndarray:
    """FFT-based Gaussian smooth on a float32 height field — no scipy needed."""
    h, w = heights.shape
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.fftfreq(w).reshape(1, -1)
    kernel = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (fy ** 2 + fx ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(heights.astype(np.float64)) * kernel)).astype(np.float32)


class TerrainCollectionEnv(DirectRLEnv):
    cfg: TerrainCollectionEnvCfg

    def __init__(self, cfg: TerrainCollectionEnvCfg, render_mode: str | None = None, **kwargs):
        self._current_gt = None
        self._rng = np.random.default_rng()
        super().__init__(cfg, render_mode, **kwargs)

        self._actions           = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_actions      = torch.zeros(self.num_envs, 2, device=self.device)
        self._wheel_vel_targets = torch.zeros(self.num_envs, 6, device=self.device)

        self._wheel_ids = self._robot.actuators["wheels"].joint_indices
        wheel_names     = self._robot.actuators["wheels"].joint_names
        self._left_mask = torch.tensor(
            ["Left" in name for name in wheel_names],
            device=self.device, dtype=torch.bool,
        )

        live_masses = self._robot.root_physx_view.get_masses()[0]
        print(f"[TerrainCollectionEnv] Robot total mass: {live_masses.sum().item():.2f} kg")
        print(f"[TerrainCollectionEnv] Wheel joints: {wheel_names}")

    # ── terrain texture helpers ───────────────────────────────────────────────

    def _generate_regolith_textures(self) -> str:
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
            raw = rng.standard_normal((S, S)).astype(np.float32)
            fx  = np.fft.fftfreq(S).reshape(1, S).astype(np.float32)
            fy  = np.fft.fftfreq(S).reshape(S, 1).astype(np.float32)
            kernel = np.exp(-2.0 * np.pi ** 2 * sigma_px ** 2 * (fx ** 2 + fy ** 2))
            return np.real(np.fft.ifft2(np.fft.fft2(raw) * kernel)).astype(np.float32)

        h = sum(w * smooth_noise(s) for s, w in [(300, 0.40), (80, 0.30), (25, 0.20), (8, 0.10)])
        h -= h.min(); h /= h.max()

        v = h * 0.15 + 0.55
        albedo = np.stack([
            (v * 0.94 * 255).clip(0, 255).astype(np.uint8),
            (v * 0.96 * 255).clip(0, 255).astype(np.uint8),
            (v * 1.00 * 255).clip(0, 255).astype(np.uint8),
        ], axis=-1)
        Image.fromarray(albedo).save(albedo_path)

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

        rough = ((h * 0.07 + 0.85) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rough).save(rough_path)

        print(f"[TerrainCollectionEnv] Regolith textures generated -> {tex_dir}")
        return tex_dir

    def _apply_regolith_textures(self, mesh_prim_path: str, tex_dir: str):
        import os
        import omni.usd
        from pxr import UsdShade, Sdf, Usd

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(mesh_prim_path)
        if not root.IsValid():
            print(f"[TerrainCollectionEnv] WARNING: prim not found at {mesh_prim_path}")
            return

        shader_prim = None
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() == "Shader":
                shader_prim = prim
                break

        if shader_prim is None:
            print(f"[TerrainCollectionEnv] WARNING: no Shader found under {mesh_prim_path}")
            return

        print(f"[TerrainCollectionEnv] Applying regolith textures via {shader_prim.GetPath()}")
        shader = UsdShade.Shader(shader_prim)

        def asset(fname: str) -> Sdf.AssetPath:
            return Sdf.AssetPath(os.path.join(tex_dir, fname).replace("\\", "/"))

        shader.CreateInput("diffuse_texture",             Sdf.ValueTypeNames.Asset).Set(asset("albedo.png"))
        shader.CreateInput("normalmap_texture",           Sdf.ValueTypeNames.Asset).Set(asset("normal.png"))
        shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(asset("rough.png"))
        shader.CreateInput("texture_scale",               Sdf.ValueTypeNames.Float2).Set((0.04, 0.04))

    def _spawn_per_env_terrain(self):
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

        tex_dir = self._generate_regolith_textures()
        self._apply_regolith_textures("/World/envs/env_0/Ground", tex_dir)

    def _randomize_per_env_terrain(self):
        import os
        import numpy as np
        import omni.usd
        from pxr import UsdGeom, Sdf
        from concurrent.futures import ThreadPoolExecutor
        from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

        print(f"[TerrainCollectionEnv] Randomizing terrain for {self.num_envs} envs ...")
        stage = omni.usd.get_context().get_stage()

        h_scale   = self.cfg.regolith_horizontal_scale
        v_scale   = self.cfg.regolith_vertical_scale
        border_px = 1
        width_px  = int(self.cfg.regolith_size[0] / h_scale) + 1
        length_px = int(self.cfg.regolith_size[1] / h_scale) + 1
        inner_size = (
            (width_px  - 2 * border_px) * h_scale,
            (length_px - 2 * border_px) * h_scale,
        )

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

        zero_hf = np.zeros((width_px, length_px), dtype=np.int16)
        verts_template, _ = convert_height_field_to_mesh(zero_hf, h_scale, v_scale, None)
        half_x = float(verts_template[:, 0].max() / 2.0)
        half_y = float(verts_template[:, 1].max() / 2.0)
        xy_base = verts_template[:, :2].copy().astype(np.float32)
        xy_base[:, 0] -= half_x
        xy_base[:, 1] -= half_y

        # Generate crater parameters for every env before spawning threads.
        # Each crater: (local_x, local_y, diameter, depth) in metres, origin = terrain centre.
        cfg = self.cfg
        all_crater_params: list[list[tuple]] = []
        for _ in range(self.num_envs):
            n = int(self._rng.integers(cfg.crater_count_range[0], cfg.crater_count_range[1] + 1))
            params: list[tuple] = []
            for _ in range(n):
                d    = float(self._rng.uniform(cfg.crater_diameter_range[0], cfg.crater_diameter_range[1]))
                depth = d * cfg.crater_depth_ratio
                margin = d / 2.0 + 0.5
                hx = cfg.regolith_size[0] / 2.0 - margin
                hy = cfg.regolith_size[1] / 2.0 - margin
                if hx <= 0 or hy <= 0:
                    continue
                lx = float(self._rng.uniform(-hx, hx))
                ly = float(self._rng.uniform(-hy, hy))
                params.append((lx, ly, d, depth))
            all_crater_params.append(params)
        self._episode_craters_per_env = all_crater_params

        raw_fn = regolith_terrain.__wrapped__

        def _gen_verts(env_idx: int) -> np.ndarray:
            hf_inner = raw_fn(0.5, terrain_cfg)
            # Use float for accumulation so crater subtraction is exact.
            heights = np.zeros((width_px, length_px), dtype=np.float32)
            heights[border_px:-border_px, border_px:-border_px] = hf_inner.astype(np.float32)

            # Carve craters as spherical-cap (semicircle cross-section) depressions.
            for lx, ly, diameter, depth in all_crater_params[env_idx]:
                cx_m   = lx + half_x          # terrain-local X (0 = left edge of HF)
                cy_m   = ly + half_y
                radius = diameter / 2.0
                R2     = radius ** 2
                depth_counts = depth / v_scale

                # Spherical cap is exactly zero outside the rim, so bound by radius.
                ix_lo = max(0, int((cx_m - radius) / h_scale))
                ix_hi = min(width_px,  int((cx_m + radius) / h_scale) + 1)
                iy_lo = max(0, int((cy_m - radius) / h_scale))
                iy_hi = min(length_px, int((cy_m + radius) / h_scale) + 1)

                dx  = np.arange(ix_lo, ix_hi, dtype=np.float32) * h_scale - cx_m
                dy  = np.arange(iy_lo, iy_hi, dtype=np.float32) * h_scale - cy_m
                r2  = dx[:, np.newaxis] ** 2 + dy[np.newaxis, :] ** 2
                # sqrt(1 - r²/R²) gives a perfect semicircle in cross-section; zero outside R.
                bowl = depth_counts * np.sqrt(np.maximum(0.0, 1.0 - r2 / R2))
                heights[ix_lo:ix_hi, iy_lo:iy_hi] -= bowl

            # Gaussian smooth to remove grid staircase aliasing on crater rims.
            heights = _gaussian_smooth_hf(heights, sigma=1.5)

            heights_m = (heights * v_scale).astype(np.float32)  # store in metres for GT

            z = heights.flatten() * v_scale
            verts = np.empty((len(xy_base), 3), dtype=np.float32)
            verts[:, :2] = xy_base
            verts[:, 2]  = z

            # Per-vertex normals from world-space height gradient → smooth shading.
            scale = v_scale / h_scale
            dzdx  = np.gradient(heights * scale, axis=0).flatten().astype(np.float32)
            dzdy  = np.gradient(heights * scale, axis=1).flatten().astype(np.float32)
            nz    = np.ones(len(dzdx), dtype=np.float32)
            inv_L = (1.0 / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)).astype(np.float32)
            normals = np.stack([-dzdx * inv_L, -dzdy * inv_L, inv_L], axis=-1)

            return verts, normals, heights_m

        workers = min(self.num_envs, (os.cpu_count() or 4) * 2)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            all_verts = list(pool.map(_gen_verts, range(self.num_envs)))

        # Store height fields in metres for BEV GT computation at episode time.
        self._episode_hf_m = [hm for _, _, hm in all_verts]
        self._terrain_half_x = half_x
        self._terrain_half_y = half_y

        with Sdf.ChangeBlock():
            for env_id, (verts, normals, _) in enumerate(all_verts):
                mesh_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Ground/mesh")
                if mesh_prim.IsValid():
                    geom = UsdGeom.Mesh(mesh_prim)
                    geom.GetPointsAttr().Set(verts)
                    geom.GetNormalsAttr().Set(normals)
                    geom.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        n_craters = sum(len(p) for p in all_crater_params)
        print(f"[TerrainCollectionEnv] Terrain randomization complete — {n_craters} craters carved.")

    def _correct_robot_masses_in_usd(self):
        import omni.usd

        factor = self.cfg.mass_correction_factor
        stage  = omni.usd.get_context().get_stage()
        robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if not robot_root.IsValid():
            print("[TerrainCollectionEnv] WARNING: /World/envs/env_0/Robot not found — mass correction skipped")
            return

        corrected = 0
        for prim in stage.Traverse():
            density_attr = prim.GetAttribute("physics:density")
            if not density_attr.IsValid():
                continue
            val = density_attr.Get()
            if val is None or float(val) == 0.0:
                continue
            density_attr.Set(float(val) / factor)
            corrected += 1

        print(f"[TerrainCollectionEnv] Mass correction: scaled {corrected} density attrs by 1/{factor:.0f}")

    # ── obstacle pool (created once at scene setup, teleported each episode) ───

    def _setup_obstacle_pool(self):
        """Pre-allocate all obstacle prims and per-type material pools at scene-setup time."""
        import omni.usd
        from pxr import UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Vt

        stage  = omni.usd.get_context().get_stage()
        stage.DefinePrim("/World/collect_obstacles", "Xform")
        stage.DefinePrim("/World/collect_obstacles/Looks", "Scope")

        PARK = Gf.Vec3d(1000.0, 0.0, 0.0)
        cfg  = self.cfg

        def _make_mat(path, diffuse, roughness=0.85, metallic=0.0, opacity=1.0):
            mat    = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(diffuse)
            shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(metallic)
            if opacity < 1.0:
                shader.CreateInput("opacity",  Sdf.ValueTypeNames.Float).Set(opacity)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat

        # ── one mutable episode material per type — shader inputs updated each episode ─
        def _make_ep_mat(path):
            mat    = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + "/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.5, 0.5, 0.5))
            shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.85)
            shader.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("opacity",      Sdf.ValueTypeNames.Float).Set(1.0)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat, shader

        base = "/World/collect_obstacles/Looks"
        self._ep_rock_mat,   self._ep_rock_shader   = _make_ep_mat(f"{base}/EpRockMat")
        self._ep_wall_mat,   self._ep_wall_shader   = _make_ep_mat(f"{base}/EpWallMat")
        self._ep_footer_mat, self._ep_footer_shader = _make_ep_mat(f"{base}/EpFooterMat")

        # ── rocks (jagged meshes) ──────────────────────────────────────────
        n_max_rocks = cfg.rock_count_range[1]
        self._rock_slots: list[tuple] = []          # (UsdGeom.Mesh, translate_op)
        for i in range(n_max_rocks):
            path      = f"/World/collect_obstacles/rock_{i:03d}"
            rock_geom = UsdGeom.Mesh.Define(stage, path)
            pts, fc, fi = _jagged_sphere_mesh(self._rng, cfg.rock_diameter_range[1] / 2.0)
            rock_geom.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
            rock_geom.CreateFaceVertexCountsAttr().Set(fc)
            rock_geom.CreateFaceVertexIndicesAttr().Set(fi)
            rock_geom.CreateSubdivisionSchemeAttr().Set("none")
            rock_geom.CreateDoubleSidedAttr().Set(True)
            xf   = UsdGeom.Xformable(rock_geom)
            xf.ClearXformOpOrder()
            t_op = xf.AddTranslateOp()
            t_op.Set(PARK)
            UsdPhysics.CollisionAPI.Apply(rock_geom.GetPrim())
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(rock_geom.GetPrim())
            mesh_col.CreateApproximationAttr().Set("convexHull")
            UsdShade.MaterialBindingAPI.Apply(rock_geom.GetPrim()).Bind(self._ep_rock_mat)
            self._rock_slots.append((rock_geom, t_op))

        # ── walls (thin panels) ────────────────────────────────────────────
        n_max_walls  = cfg.wall_count_range[1]
        self._wall_width = 0.06
        self._wall_slots: list[tuple] = []          # (cube_prim, translate_op, rotate_op, scale_op)
        for i in range(n_max_walls):
            path = f"/World/collect_obstacles/wall_{i:03d}"
            cube = UsdGeom.Cube.Define(stage, path)
            xf    = UsdGeom.Xformable(cube)
            xf.ClearXformOpOrder()
            t_op  = xf.AddTranslateOp()
            r_op  = xf.AddRotateZOp()
            s_op  = xf.AddScaleOp()
            t_op.Set(PARK)
            r_op.Set(0.0)
            s_op.Set(Gf.Vec3d(cfg.wall_length_range[1] / 2.0,
                               self._wall_width / 2.0,
                               cfg.wall_height / 2.0))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(self._ep_wall_mat)
            self._wall_slots.append((cube, t_op, r_op, s_op))

        # ── wall footers (thicker base trim) ───────────────────────────────
        self._footer_slots: list[tuple] = []
        for i in range(n_max_walls):
            path  = f"/World/collect_obstacles/footer_{i:03d}"
            cube  = UsdGeom.Cube.Define(stage, path)
            xf    = UsdGeom.Xformable(cube)
            xf.ClearXformOpOrder()
            t_op  = xf.AddTranslateOp()
            r_op  = xf.AddRotateZOp()
            s_op  = xf.AddScaleOp()
            t_op.Set(PARK)
            r_op.Set(0.0)
            s_op.Set(Gf.Vec3d(cfg.wall_length_range[1] / 2.0, 0.15, 0.20))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(self._ep_footer_mat)
            self._footer_slots.append((cube, t_op, r_op, s_op))

        print(f"[TerrainCollectionEnv] Obstacle pool: {n_max_rocks} rock slots, {n_max_walls} wall slots")

    def _randomize_ground_material(self):
        """Switch the OmniPBR ground shader to a random cached texture set with random UV scale."""
        if not self.cfg.randomize_materials:
            return
        if not getattr(self, "_ground_tex_sets", None) or self._ground_omni_shader is None:
            return
        import os
        from PIL import Image
        from pxr import Sdf

        tex_dir  = self._ground_tex_sets[int(self._rng.integers(len(self._ground_tex_sets)))]
        uv_scale = float(self._rng.uniform(0.025, 0.065))

        def _asset(fname: str) -> Sdf.AssetPath:
            return Sdf.AssetPath(os.path.join(tex_dir, fname).replace("\\", "/"))

        sh = self._ground_omni_shader
        sh.CreateInput("diffuse_texture",             Sdf.ValueTypeNames.Asset).Set(_asset("albedo.png"))
        sh.CreateInput("normalmap_texture",           Sdf.ValueTypeNames.Asset).Set(_asset("normal.png"))
        sh.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(_asset("rough.png"))
        sh.CreateInput("texture_scale",               Sdf.ValueTypeNames.Float2).Set((uv_scale, uv_scale))

        # Sample a 32×32 centre crop of the albedo for rock colour-matching
        try:
            img  = Image.open(os.path.join(tex_dir, "albedo.png")).convert("RGB")
            w, h = img.size
            crop = img.crop((w // 2 - 16, h // 2 - 16, w // 2 + 16, h // 2 + 16))
            avg  = np.asarray(crop, dtype=np.float32).mean(axis=(0, 1)) / 255.0
            self._current_ground_avg_rgb = tuple(float(c) for c in avg)
        except Exception:
            self._current_ground_avg_rgb = None

    def _generate_biome_textures(self) -> list[str]:
        """Generate procedurally distinct ground biome textures with Gaussian splotches."""
        import os
        import numpy as np
        from PIL import Image

        # (folder, primary_rgb, secondary_rgb, bright_lo, bright_hi, detail_mul, n_splotches, splotch_px)
        # primary/secondary: two colors that blend across the texture via Gaussian splotches
        # detail_mul: scales grain noise frequency (higher = rougher grain)
        # n_splotches: number of large colour-blend blobs
        # splotch_px: (min, max) Gaussian sigma in pixels for colour blobs
        BIOMES = [
            ("lunar_regolith",    (0.72, 0.68, 0.60), (0.52, 0.50, 0.44), 0.38, 0.80, 1.2, 28, (35, 170)),
            ("fine_sand",         (0.90, 0.80, 0.56), (0.76, 0.66, 0.42), 0.52, 0.90, 0.6, 14, (70, 240)),
            ("coarse_sand",       (0.80, 0.63, 0.42), (0.60, 0.46, 0.28), 0.36, 0.74, 1.0, 22, (30, 140)),
            ("rocky_gravel",      (0.40, 0.41, 0.42), (0.24, 0.25, 0.26), 0.18, 0.72, 2.0, 45, (18, 90)),
            ("red_soil",          (0.66, 0.37, 0.24), (0.46, 0.22, 0.12), 0.28, 0.72, 1.3, 32, (35, 160)),
            ("dark_basalt",       (0.14, 0.14, 0.15), (0.06, 0.06, 0.07), 0.05, 0.32, 2.2, 55, (12, 70)),
            ("tan_mud",           (0.60, 0.52, 0.38), (0.42, 0.35, 0.24), 0.28, 0.74, 1.0, 26, (50, 210)),
            ("chalk",             (0.90, 0.88, 0.84), (0.76, 0.74, 0.68), 0.52, 0.92, 0.7, 18, (65, 260)),
            ("white_salt_flat",   (0.93, 0.91, 0.87), (0.72, 0.70, 0.68), 0.62, 0.97, 0.4, 16, (90, 320)),
            ("orange_dust",       (0.84, 0.54, 0.28), (0.65, 0.36, 0.14), 0.35, 0.78, 1.1, 30, (40, 180)),
            ("gray_clay",         (0.52, 0.50, 0.47), (0.36, 0.34, 0.32), 0.30, 0.72, 0.9, 24, (55, 210)),
            ("iron_oxide",        (0.57, 0.28, 0.14), (0.40, 0.16, 0.06), 0.24, 0.68, 1.4, 36, (28, 140)),
            ("volcanic_ash",      (0.20, 0.20, 0.21), (0.10, 0.10, 0.11), 0.05, 0.40, 2.4, 55, (12, 65)),
            ("pale_limestone",    (0.84, 0.80, 0.68), (0.68, 0.64, 0.52), 0.44, 0.86, 0.8, 22, (60, 250)),
            ("dark_red_volcanic", (0.46, 0.18, 0.10), (0.28, 0.08, 0.03), 0.14, 0.60, 1.6, 42, (22, 110)),
            ("mixed_gravel",      (0.50, 0.46, 0.40), (0.30, 0.28, 0.24), 0.20, 0.75, 1.8, 48, (18, 85)),
            ("rust_sand",         (0.74, 0.48, 0.26), (0.52, 0.30, 0.12), 0.32, 0.76, 1.1, 28, (40, 160)),
            ("pale_clay",         (0.76, 0.70, 0.60), (0.58, 0.52, 0.44), 0.40, 0.80, 0.8, 20, (60, 230)),
            ("dark_gravel",       (0.28, 0.27, 0.26), (0.18, 0.17, 0.16), 0.14, 0.55, 1.9, 42, (16, 80)),
            ("red_clay",          (0.62, 0.34, 0.22), (0.44, 0.20, 0.10), 0.26, 0.66, 1.0, 30, (45, 180)),
        ]

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        biomes_dir = os.path.join(assets_dir, "biome_textures")

        if os.path.isdir(biomes_dir):
            dirs = sorted(
                os.path.join(biomes_dir, d)
                for d in os.listdir(biomes_dir)
                if os.path.isdir(os.path.join(biomes_dir, d))
                and not d.endswith("_variants")
            )
            if len(dirs) == len(BIOMES):
                return dirs

        S = 512
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)

        def _fft_noise(rng: np.random.Generator, sigma_px: float) -> np.ndarray:
            raw = rng.standard_normal((S, S)).astype(np.float32)
            fx  = np.fft.fftfreq(S).reshape(1, S).astype(np.float32)
            fy  = np.fft.fftfreq(S).reshape(S, 1).astype(np.float32)
            k   = np.exp(-2.0 * np.pi ** 2 * sigma_px ** 2 * (fx ** 2 + fy ** 2))
            out = np.real(np.fft.ifft2(np.fft.fft2(raw) * k)).astype(np.float32)
            return out

        def _splotch_blend(rng: np.random.Generator, n: int, size_range: tuple) -> np.ndarray:
            """Build a [0,1] blend map from overlapping Gaussian blobs."""
            blend = np.zeros((S, S), dtype=np.float32)
            for _ in range(n):
                cx = rng.uniform(0, S)
                cy = rng.uniform(0, S)
                sig = rng.uniform(*size_range)
                strength = rng.uniform(0.3, 1.0)
                blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig ** 2))
                blend += strength * blob
            # Normalise to [0, 1]
            lo, hi = blend.min(), blend.max()
            if hi > lo:
                blend = (blend - lo) / (hi - lo)
            return blend

        dirs = []
        for (folder, pri, sec, bright_lo, bright_hi, detail_mul, n_spl, spl_px) in BIOMES:
            bdir = os.path.join(biomes_dir, folder)
            os.makedirs(bdir, exist_ok=True)

            rng = np.random.default_rng(abs(hash(folder)) % (2 ** 32))

            # ── brightness noise: mid + fine grain, no dominant low-freq wash ──
            h = (
                0.30 * _fft_noise(rng, 80 * detail_mul)
              + 0.35 * _fft_noise(rng, 30 * detail_mul)
              + 0.25 * _fft_noise(rng, 10 * detail_mul)
              + 0.10 * _fft_noise(rng,  4 * detail_mul)
            )
            h -= h.min(); h /= h.max()

            # ── colour blend map: Gaussian splotches drive primary→secondary ──
            blend = _splotch_blend(rng, n_spl, spl_px)

            # ── brightness splotches: local darkening / lightening patches ───
            bright_mod = _splotch_blend(rng, n_spl // 2, (spl_px[0] * 0.6, spl_px[1] * 0.8))
            bright_mod = (bright_mod - 0.5) * 0.30   # ±0.15 additive

            v = np.clip(h * (bright_hi - bright_lo) + bright_lo + bright_mod, 0.0, 1.0)

            # ── per-pixel colour: lerp primary ↔ secondary via blend map ─────
            def _ch(p_val: float, s_val: float) -> np.ndarray:
                return np.clip(v * (p_val * (1.0 - blend) + s_val * blend), 0.0, 1.0)

            albedo = np.stack([
                (_ch(pri[0], sec[0]) * 255).astype(np.uint8),
                (_ch(pri[1], sec[1]) * 255).astype(np.uint8),
                (_ch(pri[2], sec[2]) * 255).astype(np.uint8),
            ], axis=-1)
            Image.fromarray(albedo).save(os.path.join(bdir, "albedo.png"))

            # ── normal map from combined height ──────────────────────────────
            bump = h + blend * 0.25   # blend seams add slight bump
            dx = np.gradient(bump, axis=1) * 12.0
            dy = np.gradient(bump, axis=0) * 12.0
            nz = np.ones_like(dx)
            L  = np.sqrt(dx ** 2 + dy ** 2 + nz ** 2)
            normal = np.stack([
                ((-dx / L) * 0.5 + 0.5) * 255,
                ((-dy / L) * 0.5 + 0.5) * 255,
                (( nz / L) * 0.5 + 0.5) * 255,
            ], axis=-1).clip(0, 255).astype(np.uint8)
            Image.fromarray(normal).save(os.path.join(bdir, "normal.png"))

            # ── roughness: fine grain drives roughness variation ──────────────
            rough_base = 0.55 + (1.0 - bright_hi) * 0.30
            rough = np.clip(h * 0.20 + rough_base, 0.0, 1.0)
            Image.fromarray((rough * 255).astype(np.uint8), mode="L").save(
                os.path.join(bdir, "rough.png")
            )

            dirs.append(bdir)
            print(f"[TerrainCollectionEnv]   biome: {folder}")

        print(f"[TerrainCollectionEnv] Generated {len(dirs)} biome textures → {biomes_dir}")
        return dirs

    def _setup_ground_textures(self):
        """Build ground texture pool: Poly Haven downloads + 8 procedural biomes w/ variants."""
        import os
        import omni.usd
        from pxr import Usd, UsdShade

        stage = omni.usd.get_context().get_stage()
        self._ground_tex_sets: list[str] = []
        self._ground_omni_shader = None

        root = stage.GetPrimAtPath("/World/envs/env_0/Ground")
        if root.IsValid():
            for prim in Usd.PrimRange(root):
                if prim.GetTypeName() == "Shader":
                    self._ground_omni_shader = UsdShade.Shader(prim)
                    break
        if self._ground_omni_shader is None:
            print("[TerrainCollectionEnv] WARNING: ground OmniPBR shader not found — textures disabled")

        assets_dir   = os.path.join(os.path.dirname(__file__), "assets")
        ph_cache_dir = os.path.join(assets_dir, "polyhaven_textures")

        # ── Poly Haven (internet, first run only) ────────────────────────────
        for slug in _POLYHAVEN_SLUGS:
            slug_dir     = os.path.join(ph_cache_dir, slug)
            variants_dir = os.path.join(ph_cache_dir, f"{slug}_variants")

            if os.path.isdir(variants_dir):
                vdirs = sorted(
                    os.path.join(variants_dir, d)
                    for d in os.listdir(variants_dir)
                    if os.path.isdir(os.path.join(variants_dir, d))
                )
                if vdirs:
                    self._ground_tex_sets.extend(vdirs)
                    continue

            if not os.path.isfile(os.path.join(slug_dir, "albedo.jpg")):
                if not _try_download_polyhaven(slug, slug_dir):
                    print(f"[TerrainCollectionEnv] Skipped Poly Haven: {slug}")
                    continue
                print(f"[TerrainCollectionEnv] Downloaded: {slug}")

            vdirs = _make_texture_variants(slug_dir, variants_dir, n=3, rng=self._rng)
            self._ground_tex_sets.extend(vdirs)

        # ── Procedural biomes (always available) ─────────────────────────────
        for bdir in self._generate_biome_textures():
            bname        = os.path.basename(bdir)
            variants_dir = os.path.join(os.path.dirname(bdir), f"{bname}_variants")
            if os.path.isdir(variants_dir):
                vdirs = sorted(
                    os.path.join(variants_dir, d)
                    for d in os.listdir(variants_dir)
                    if os.path.isdir(os.path.join(variants_dir, d))
                )
                if vdirs:
                    self._ground_tex_sets.extend(vdirs)
                    continue
            vdirs = _make_texture_variants(bdir, variants_dir, n=2, rng=self._rng)
            self._ground_tex_sets.extend(vdirs if vdirs else [bdir])

        print(f"[TerrainCollectionEnv] Ground texture pool: {len(self._ground_tex_sets)} variants")

    def _compute_bev_gt(
        self,
        hf_m: np.ndarray,
        robot_lx: float, robot_ly: float, robot_yaw: float,
        rocks: list,    # [[lx, ly, wz, radius], ...]  terrain-centered
        craters: list,  # [(lx, ly, diameter, depth), ...]  terrain-centered
        walls: list,    # [[x1, y1, x2, y2, height], ...]  terrain-centered
    ) -> dict:
        """Compute robot-centric BEV ground truth arrays from terrain data."""
        from scipy.ndimage import map_coordinates

        BEV_N    = 200
        CELL     = 0.05        # m per BEV cell
        BEV_HALF = BEV_N * CELL / 2.0   # 5.0 m
        h_scale  = self.cfg.regolith_horizontal_scale

        # ── height_gt: bilinear-sampled, yaw-rotated BEV window ──────────
        rx_c = (robot_lx + self._terrain_half_x) / h_scale  # terrain cell coords
        ry_c = (robot_ly + self._terrain_half_y) / h_scale

        offsets = np.arange(-BEV_N // 2, BEV_N // 2, dtype=np.float32)
        ii, jj = np.meshgrid(offsets, offsets, indexing="ij")  # (200, 200)

        cos_y, sin_y = float(np.cos(robot_yaw)), float(np.sin(robot_yaw))
        tx = cos_y * ii - sin_y * jj + rx_c
        ty = sin_y * ii + cos_y * jj + ry_c

        height_gt = map_coordinates(hf_m, [tx, ty], order=1, mode="nearest").astype(np.float32)

        # ── coordinate helpers ────────────────────────────────────────────
        cos_ny, sin_ny = np.cos(-robot_yaw), np.sin(-robot_yaw)

        def to_robot(wx, wy):
            dx, dy = wx - robot_lx, wy - robot_ly
            return float(cos_ny * dx - sin_ny * dy), float(sin_ny * dx + cos_ny * dy)

        def in_bev(rx, ry):
            return abs(rx) < BEV_HALF and abs(ry) < BEV_HALF

        def to_cell(v):
            return int((v + BEV_HALF) / CELL)

        def draw_disk(grid, cx, cy, r_cells, label):
            x0, x1 = max(0, cx - r_cells), min(BEV_N, cx + r_cells + 1)
            y0, y1 = max(0, cy - r_cells), min(BEV_N, cy + r_cells + 1)
            if x0 >= x1 or y0 >= y1:
                return
            xs = np.arange(x0, x1) - cx
            ys = np.arange(y0, y1) - cy
            grid[x0:x1, y0:y1][xs[:, None] ** 2 + ys[None, :] ** 2 <= r_cells ** 2] = label

        # ── semantic_gt ───────────────────────────────────────────────────
        semantic = np.zeros((BEV_N, BEV_N), dtype=np.uint8)

        for x1w, y1w, x2w, y2w, _ in walls:
            seg = float(np.hypot(x2w - x1w, y2w - y1w))
            for t in np.linspace(0.0, 1.0, max(2, int(seg / CELL * 2))):
                rx, ry = to_robot(x1w + t * (x2w - x1w), y1w + t * (y2w - y1w))
                if in_bev(rx, ry):
                    cx, cy = to_cell(rx), to_cell(ry)
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            xi, yj = cx + di, cy + dj
                            if 0 <= xi < BEV_N and 0 <= yj < BEV_N:
                                semantic[xi, yj] = 3

        for lx, ly, diameter, *_ in craters:
            rx, ry = to_robot(lx, ly)
            if in_bev(rx, ry):
                draw_disk(semantic, to_cell(rx), to_cell(ry),
                          max(1, int(diameter / 2.0 / CELL)), 2)

        for lx, ly, _wz, radius in rocks:
            rx, ry = to_robot(lx, ly)
            if in_bev(rx, ry):
                draw_disk(semantic, to_cell(rx), to_cell(ry),
                          max(1, int(radius / CELL)), 1)

        # ── objects_gt ────────────────────────────────────────────────────
        obj_list = []
        for lx, ly, _wz, radius in rocks:
            rx, ry = to_robot(lx, ly)
            if in_bev(rx, ry):
                obj_list.append((rx * rx + ry * ry, rx, ry, radius * 2.0, 0.0))
        for lx, ly, diameter, *_ in craters:
            rx, ry = to_robot(lx, ly)
            if in_bev(rx, ry):
                obj_list.append((rx * rx + ry * ry, rx, ry, diameter, 1.0))
        obj_list.sort(key=lambda o: o[0])
        objects_gt = (np.array([[o[1], o[2], o[3], o[4]] for o in obj_list], dtype=np.float32)
                      if obj_list else np.zeros((0, 4), dtype=np.float32))

        # ── walls_gt ──────────────────────────────────────────────────────
        wall_list = []
        for x1w, y1w, x2w, y2w, _ in walls:
            rx1, ry1 = to_robot(x1w, y1w)
            rx2, ry2 = to_robot(x2w, y2w)
            wall_list.append([rx1, ry1, rx2, ry2])
        walls_gt = (np.array(wall_list, dtype=np.float32)
                    if wall_list else np.zeros((0, 4), dtype=np.float32))

        return {
            "height_gt":   height_gt,
            "semantic_gt": semantic,
            "objects_gt":  objects_gt,
            "walls_gt":    walls_gt,
            "robot_yaw":   np.float32(robot_yaw),
        }

    def _randomize_obstacles(self, env_ox: float, env_oy: float,
                              robot_wx: float, robot_wy: float, robot_yaw: float):
        """Teleport pooled obstacle prims to new random positions."""
        from pxr import Gf

        rng  = self._rng
        cfg  = self.cfg
        PARK = Gf.Vec3d(1000.0, 0.0, 0.0)

        half_ax = cfg.regolith_size[0] / 2.0 - cfg.obstacle_spawn_margin
        half_ay = cfg.regolith_size[1] / 2.0 - cfg.obstacle_spawn_margin
        spawn_rx = (cfg.spawn_zone_size_x / 2.0 - cfg.spawn_zone_margin) * cfg.spawn_position_scale
        spawn_ry = (cfg.spawn_zone_size_y / 2.0 - cfg.spawn_zone_margin) * cfg.spawn_position_scale
        excl_x   = spawn_rx + cfg.obstacle_spawn_margin
        excl_y   = spawn_ry + cfg.obstacle_spawn_margin

        def _far_from_spawn(lx: float, ly: float) -> bool:
            return (abs(lx - cfg.spawn_zone_center_x) > excl_x
                    or abs(ly - cfg.spawn_zone_center_y) > excl_y)

        def _sample_pos():
            for _ in range(30):
                lx = float(rng.uniform(-half_ax, half_ax))
                ly = float(rng.uniform(-half_ay, half_ay))
                if _far_from_spawn(lx, ly):
                    return lx, ly
            return float(rng.uniform(-half_ax, -excl_x)), float(rng.uniform(-half_ay, half_ay))

        def _wall_perimeter() -> tuple[float, float, float]:
            """Return (lx, ly, ang_deg) with wall near a random arena edge, parallel to it."""
            side   = int(rng.integers(4))              # 0=N +Y, 1=S -Y, 2=E +X, 3=W -X
            inset  = float(rng.uniform(0.3, 2.0))      # metres inside from edge
            wobble = float(rng.uniform(-20.0, 20.0))   # slight angle variation
            if side == 0:   # North
                return (float(rng.uniform(-half_ax * 0.85, half_ax * 0.85)),
                        half_ay - inset, 0.0 + wobble)
            elif side == 1: # South
                return (float(rng.uniform(-half_ax * 0.85, half_ax * 0.85)),
                        -(half_ay - inset), 0.0 + wobble)
            elif side == 2: # East
                return (half_ax - inset,
                        float(rng.uniform(-half_ay * 0.85, half_ay * 0.85)),
                        90.0 + wobble)
            else:           # West
                return (-(half_ax - inset),
                        float(rng.uniform(-half_ay * 0.85, half_ay * 0.85)),
                        90.0 + wobble)

        # ── update episode-shared materials once before the obstacle loops ─────
        # Ground material must be set first so _current_ground_avg_rgb is current
        if cfg.randomize_materials:
            self._randomize_ground_material()

        from pxr import Vt, UsdShade
        if cfg.randomize_materials and hasattr(self, "_ep_rock_shader"):
            ground_rgb = getattr(self, "_current_ground_avg_rgb", None)
            if ground_rgb is not None and float(rng.random()) < 0.85:
                # Match the ground colour, darkened slightly so rocks read as distinct objects
                darken = float(rng.uniform(0.62, 0.78))
                d = tuple(float(np.clip(c * darken, 0.0, 1.0)) for c in ground_rgb)
                r = float(rng.uniform(0.72, 0.96))
                m = 0.0
            else:
                d, r, m = _sample_rock_mat(rng)
            self._ep_rock_shader.GetInput("diffuseColor").Set(d)
            self._ep_rock_shader.GetInput("roughness").Set(r)
            self._ep_rock_shader.GetInput("metallic").Set(m)
            d, r, m, o = _sample_wall_mat(rng, allow_glass=True)
            self._ep_wall_shader.GetInput("diffuseColor").Set(d)
            self._ep_wall_shader.GetInput("roughness").Set(r)
            self._ep_wall_shader.GetInput("metallic").Set(m)
            self._ep_wall_shader.GetInput("opacity").Set(o)
            d, r, m, _ = _sample_wall_mat(rng, allow_glass=False)
            self._ep_footer_shader.GetInput("diffuseColor").Set(d)
            self._ep_footer_shader.GetInput("roughness").Set(r)
            self._ep_footer_shader.GetInput("metallic").Set(m)
            self._ep_footer_shader.GetInput("opacity").Set(1.0)

        # ── rocks (jagged meshes — shape regenerated each episode) ─────────────
        n_rocks = int(rng.integers(cfg.rock_count_range[0], cfg.rock_count_range[1] + 1))
        rocks: list[list[float]] = []
        for i, (rock_geom, t_op) in enumerate(self._rock_slots):
            if i < n_rocks:
                lx, ly = _sample_pos()
                radius = float(rng.uniform(cfg.rock_diameter_range[0] / 2.0,
                                            cfg.rock_diameter_range[1] / 2.0))
                pts, fc, fi = _jagged_sphere_mesh(rng, radius)
                rock_geom.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
                rock_geom.GetFaceVertexCountsAttr().Set(fc)
                rock_geom.GetFaceVertexIndicesAttr().Set(fi)
                wz = radius          # center at 1r above z=0 → bottom vertex above surface
                t_op.Set(Gf.Vec3d(env_ox + lx, env_oy + ly, wz))
                rocks.append([lx, ly, float(wz), radius])
            else:
                t_op.Set(PARK)

        # ── walls (perimeter-placed) + footers ────────────────────────────
        n_walls = int(rng.integers(cfg.wall_count_range[0], cfg.wall_count_range[1] + 1))
        walls: list[list[float]] = []
        for i, (cube, t_op, r_op, s_op) in enumerate(self._wall_slots):
            _, ft_op, fr_op, fs_op = self._footer_slots[i]
            if i < n_walls:
                lx, ly, ang_deg = _wall_perimeter()
                ang_rad  = float(np.deg2rad(ang_deg))
                length   = float(rng.uniform(cfg.wall_length_range[0], cfg.wall_length_range[1]))
                wz       = cfg.wall_height / 2.0
                t_op.Set(Gf.Vec3d(env_ox + lx, env_oy + ly, wz))
                r_op.Set(ang_deg)
                s_op.Set(Gf.Vec3d(length / 2.0, self._wall_width / 2.0, cfg.wall_height / 2.0))
                # Footer: thicker base trim, randomised height
                footer_h = float(rng.uniform(0.25, 0.55))
                footer_w = self._wall_width + float(rng.uniform(0.06, 0.18))
                ft_op.Set(Gf.Vec3d(env_ox + lx, env_oy + ly, footer_h / 2.0))
                fr_op.Set(ang_deg)
                fs_op.Set(Gf.Vec3d(length / 2.0, footer_w / 2.0, footer_h / 2.0))
                dx = float(np.cos(ang_rad)) * length / 2.0
                dy = float(np.sin(ang_rad)) * length / 2.0
                walls.append([lx - dx, ly - dy, lx + dx, ly + dy, cfg.wall_height])
            else:
                t_op.Set(PARK)
                ft_op.Set(PARK)

        crater_params = getattr(self, "_episode_craters_per_env", [[]])[0]
        robot_lx = robot_wx - env_ox
        robot_ly = robot_wy - env_oy
        self._current_gt = self._compute_bev_gt(
            hf_m=self._episode_hf_m[0],
            robot_lx=robot_lx,
            robot_ly=robot_ly,
            robot_yaw=robot_yaw,
            rocks=rocks,
            craters=crater_params,
            walls=walls,
        )
        print(f"[TerrainCollectionEnv] Episode obstacles: {n_rocks} rocks, {n_walls} walls")

    # ── scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._spawn_per_env_terrain()

        self._robot = Articulation(self.cfg.robot)
        self._correct_robot_masses_in_usd()

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        ground_prim = stage.GetPrimAtPath(self.cfg.terrain.prim_path + "/terrain")
        if ground_prim.IsValid():
            for op in UsdGeom.Xformable(ground_prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(0.0, 0.0, -0.8))
                    break

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self._randomize_per_env_terrain()
        self._env_origins = self.scene._default_env_origins
        self.scene.articulations["robot"] = self._robot

        self._setup_obstacle_pool()
        self._setup_ground_textures()

        dome_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.85, 0.88, 1.00))
        dome_cfg.func("/World/Light", dome_cfg)

        from .camera_capture import CameraCapture
        self._cam_capture = CameraCapture()
        # setup() is called lazily on first capture() so the render pipeline
        # is guaranteed active — calling it here caused random black cameras.

        disk_cfg = sim_utils.DiskLightCfg(intensity=45_000.0, radius=1.5, color=(1.00, 0.95, 0.88))
        disk_cfg.func("/World/envs/env_.*/SpotA", disk_cfg, translation=(-7.0,  5.0, 12.0))
        disk_cfg.func("/World/envs/env_.*/SpotB", disk_cfg, translation=( 7.0, -5.0, 12.0))

        angled_cfg = sim_utils.DiskLightCfg(intensity=67_500.0, radius=2.5, color=(0.88, 0.93, 1.00))
        angled_cfg.func("/World/envs/env_.*/SpotC", angled_cfg,
                        translation=(0.0,  25.0, 25.0), orientation=(0.924, -0.383, 0.0, 0.0))
        angled_cfg.func("/World/envs/env_.*/SpotD", angled_cfg,
                        translation=(0.0, -25.0, 25.0), orientation=(0.924,  0.383, 0.0, 0.0))

    # ── step callbacks ────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
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
        self._last_frames = self._cam_capture.capture()
        return {"policy": self._robot.data.root_state_w[:, :7]}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids]           = 0.0
        self._prev_actions[env_ids]      = 0.0
        self._wheel_vel_targets[env_ids] = 0.0

        n = len(env_ids)
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._env_origins[env_ids]

        cfg  = self.cfg
        hx   = (cfg.spawn_zone_size_x / 2.0 - cfg.spawn_zone_margin) * cfg.spawn_position_scale
        hy   = (cfg.spawn_zone_size_y / 2.0 - cfg.spawn_zone_margin) * cfg.spawn_position_scale
        default_root_state[:, 0] += cfg.spawn_zone_center_x + torch.empty(n, device=self.device).uniform_(-hx, hx)
        default_root_state[:, 1] += cfg.spawn_zone_center_y + torch.empty(n, device=self.device).uniform_(-hy, hy)

        if cfg.spawn_random_yaw:
            yaw = torch.empty(n, device=self.device).uniform_(0.0, 2.0 * 3.14159265)
            default_root_state[:, 3] = torch.cos(yaw * 0.5)
            default_root_state[:, 4] = 0.0
            default_root_state[:, 5] = 0.0
            default_root_state[:, 6] = torch.sin(yaw * 0.5)

        if len(env_ids) == self.num_envs and hasattr(self, "_rock_slots"):
            self._randomize_per_env_terrain()
            env_o = self._env_origins[env_ids[0]]
            quat  = default_root_state[0, 3:7]
            w, x, y, z = (float(quat[i]) for i in range(4))
            _ryaw = float(torch.atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2)
            ).item())
            self._randomize_obstacles(
                float(env_o[0]), float(env_o[1]),
                float(default_root_state[0, 0]), float(default_root_state[0, 1]),
                _ryaw,
            )
            # Append roll/pitch to GT after _randomize_obstacles populates _current_gt
            if self._current_gt is not None:
                _roll  = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
                _pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
                self._current_gt["robot_roll"]  = np.float32(_roll)
                self._current_gt["robot_pitch"] = np.float32(_pitch)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            None, env_ids,
        )
