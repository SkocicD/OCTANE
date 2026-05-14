"""Arena generator for Lunabotics navigation training.

Generates randomised arena configurations and terrain maps based on the
KSC Artemis arena layout, with scale/proportion variation to cover both
the full KSC arena and the half-size UCF Exolith Lab lanes.

Arena coordinate system (metres, origin bottom-left):
  X = right (width)
  Y = up    (length / depth into arena)

Zones (from image):
  Start Zone   — bottom-left corner, no obstacles
  Excavation   — left column, full depth (minus start/deposit)
  Obstacle/Nav — right column, full depth (minus start/deposit)
  Deposit      — bottom-right, construction zone
  Berm Target  — high-value sub-region inside deposit zone
"""

import math
import random
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class Rect:
    x: float   # left edge (m)
    y: float   # bottom edge (m)
    w: float   # width (m)
    h: float   # height (m)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def centre(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def random_point(self, rng: random.Random, margin: float = 0.0) -> Tuple[float, float]:
        x = rng.uniform(self.x + margin, self.x + self.w - margin)
        y = rng.uniform(self.y + margin, self.y + self.h - margin)
        return x, y


@dataclass
class Obstacle:
    x: float         # centre (m)
    y: float         # centre (m)
    diameter: float
    kind: str        # 'rock' | 'crater'


@dataclass
class ArenaConfig:
    width: float
    length: float
    scale: float
    start_zone: Rect
    excavation_zone: Rect
    nav_zone: Rect
    deposit_zone: Rect
    berm_target: Rect
    obstacles: list = field(default_factory=list)


def generate_arena(cfg: dict, rng: random.Random | None = None) -> ArenaConfig:
    """Build one randomised arena from config."""
    if rng is None:
        rng = random.Random()

    ac = cfg['arena']
    tc = cfg['training']

    scale  = rng.uniform(tc['scale_min'], tc['scale_max'])
    width  = ac['base_width']  * scale
    length = ac['base_length'] * scale

    exc_w    = width  * ac['excavation_width_frac']
    nav_w    = width  - exc_w
    start_h  = length * ac['start_length_frac']
    dep_h    = length * ac['deposit_length_frac']
    mid_h    = length - start_h - dep_h

    # Zone rects
    start_zone      = Rect(0,          0,          exc_w, start_h)
    excavation_zone = Rect(0,          start_h,    exc_w, mid_h)
    nav_zone        = Rect(exc_w,      0,          nav_w, length - dep_h)
    deposit_zone    = Rect(exc_w,      length - dep_h, nav_w, dep_h)

    # Berm target (centred in deposit zone, fixed physical size clamped to zone)
    bw = min(ac['berm_w'] * scale, deposit_zone.w * 0.7)
    bh = min(ac['berm_h'] * scale, deposit_zone.h * 0.7)
    bx = deposit_zone.x + (deposit_zone.w - bw) / 2
    by = deposit_zone.y + (deposit_zone.h - bh) / 2
    berm_target = Rect(bx, by, bw, bh)

    # Obstacles — placed inside excavation + nav zones, not in start/deposit
    obstacle_zones = [excavation_zone, nav_zone]
    n_rocks   = max(1, round(ac['rocks_mean']   * scale * rng.uniform(0.7, 1.3)))
    n_craters = max(1, round(ac['craters_mean'] * scale * rng.uniform(0.7, 1.3)))
    obstacles = []

    for kind, n, diam in [('rock',   n_rocks,   ac['rock_diameter']),
                           ('crater', n_craters, ac['crater_diameter'])]:
        placed = 0
        attempts = 0
        while placed < n and attempts < n * 30:
            attempts += 1
            zone = rng.choice(obstacle_zones)
            margin = diam
            if zone.w <= 2 * margin or zone.h <= 2 * margin:
                continue
            ox = rng.uniform(zone.x + margin, zone.x + zone.w - margin)
            oy = rng.uniform(zone.y + margin, zone.y + zone.h - margin)
            # Keep away from start zone
            if start_zone.contains(ox, oy):
                continue
            # Avoid overlap with existing obstacles
            too_close = any(
                math.hypot(ox - o.x, oy - o.y) < (diam + o.diameter) * 0.6
                for o in obstacles
            )
            if not too_close:
                obstacles.append(Obstacle(ox, oy, diam, kind))
                placed += 1

    return ArenaConfig(
        width=width, length=length, scale=scale,
        start_zone=start_zone,
        excavation_zone=excavation_zone,
        nav_zone=nav_zone,
        deposit_zone=deposit_zone,
        berm_target=berm_target,
        obstacles=obstacles,
    )


# ── Terrain map generation ────────────────────────────────────────────────────

def build_terrain_maps(arena: ArenaConfig, cfg: dict,
                       rng: np.random.Generator | None = None) -> dict:
    """
    Returns terrain maps for the FULL arena (not robot-centred).
    Each map is (arena_rows, arena_cols) float32.

    Keys: 'height', 'rocks', 'craters', 'walls'
    """
    if rng is None:
        rng = np.random.default_rng()

    tc = cfg['terrain']
    cs = tc.get('cell_size', 0.1)

    cols = max(1, round(arena.width  / cs))
    rows = max(1, round(arena.length / cs))

    def to_cell(x, y):
        return int(x / cs), int(y / cs)   # col, row

    # ── Height map — smooth random + slight slope toward deposit zone ─────────
    height = rng.standard_normal((rows, cols)).astype(np.float32) * tc['height_noise_std']
    height = gaussian_filter(height, sigma=3.0)
    # Gentle slope: terrain rises slightly away from robot start (bottom)
    slope = np.linspace(0, 0.08 * arena.scale, rows)[:, None]
    height += slope
    # Flatten start zone and deposit zone
    for zone in [arena.start_zone, arena.deposit_zone]:
        c0, r0 = to_cell(zone.x, zone.y)
        c1, r1 = to_cell(zone.x + zone.w, zone.y + zone.h)
        height[r0:r1, c0:c1] *= 0.1

    # ── Rock heatmap ──────────────────────────────────────────────────────────
    rocks = np.zeros((rows, cols), dtype=np.float32)
    for obs in arena.obstacles:
        if obs.kind != 'rock':
            continue
        cx, cy = int(obs.x / cs), int(obs.y / cs)
        sigma = max(1.0, (obs.diameter / 2) / cs)
        r = int(sigma * 3)
        x0, x1 = max(0, cx - r), min(cols, cx + r + 1)
        y0, y1 = max(0, cy - r), min(rows, cy + r + 1)
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xx, yy = np.meshgrid(xs, ys)
        rocks[y0:y1, x0:x1] = np.maximum(
            rocks[y0:y1, x0:x1],
            np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        )
        # Raise height at rock location
        height[y0:y1, x0:x1] += rocks[y0:y1, x0:x1] * 0.25 * arena.scale

    # ── Crater heatmap ────────────────────────────────────────────────────────
    craters = np.zeros((rows, cols), dtype=np.float32)
    for obs in arena.obstacles:
        if obs.kind != 'crater':
            continue
        cx, cy = int(obs.x / cs), int(obs.y / cs)
        sigma = max(1.0, (obs.diameter / 2) / cs)
        r = int(sigma * 3)
        x0, x1 = max(0, cx - r), min(cols, cx + r + 1)
        y0, y1 = max(0, cy - r), min(rows, cy + r + 1)
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xx, yy = np.meshgrid(xs, ys)
        blob = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        craters[y0:y1, x0:x1] = np.maximum(craters[y0:y1, x0:x1], blob)
        height[y0:y1, x0:x1] -= blob * 0.3 * arena.scale

    # ── Wall mask — arena boundary ────────────────────────────────────────────
    walls = np.zeros((rows, cols), dtype=np.float32)
    wall_thick = max(1, int(0.1 / cs))
    walls[:wall_thick,  :] = 1.0
    walls[-wall_thick:, :] = 1.0
    walls[:,  :wall_thick] = 1.0
    walls[:, -wall_thick:] = 1.0

    # Normalise height
    height = (height / tc['height_scale']).astype(np.float32)

    return {'height': height, 'rocks': rocks, 'craters': craters, 'walls': walls,
            'rows': rows, 'cols': cols}


def _view_cell_size(cfg: dict) -> tuple[float, float]:
    """Returns (cell_size_x, cell_size_y) in metres/cell based on view extents."""
    tc = cfg['terrain']
    gs = tc['grid_size']
    return tc['view_width'] / gs, tc['view_height'] / gs


def crop_robot_view(terrain: dict, robot_x: float, robot_y: float,
                    cfg: dict) -> np.ndarray:
    """
    Resample the full arena terrain into a grid_size × grid_size patch
    centred on the robot. view_width × view_height defines the real-world
    extents — change those in config.yaml to calibrate to actual sensor range.
    Out-of-bounds areas are filled with wall values.
    Returns (4, grid_size, grid_size) float32.
    """
    tc       = cfg['terrain']
    gs       = tc['grid_size']
    csx, csy = _view_cell_size(cfg)
    arena_cs = tc.get('cell_size', 0.1)
    rows     = terrain['rows']
    cols     = terrain['cols']
    half     = gs // 2

    # Vectorised: build world coordinates for every output cell
    pj = np.arange(gs)
    pi = np.arange(gs)
    wx = robot_x + (pj - half) * csx   # (gs,)  world-x per output column
    wy = robot_y + (pi - half) * csy   # (gs,)  world-y per output row

    ci = (wx / arena_cs).astype(np.intp)   # column indices in full terrain
    ri = (wy / arena_cs).astype(np.intp)   # row    indices in full terrain

    col_valid = (ci >= 0) & (ci < cols)
    row_valid = (ri >= 0) & (ri < rows)
    valid     = row_valid[:, None] & col_valid[None, :]   # (gs, gs) bool

    ci_safe = np.clip(ci, 0, cols - 1)
    ri_safe = np.clip(ri, 0, rows - 1)

    channels = []
    for key in ('height', 'rocks', 'craters', 'walls'):
        patch = terrain[key][np.ix_(ri_safe, ci_safe)]          # (gs, gs)
        fill  = 1.0 if key == 'walls' else 0.0
        channels.append(np.where(valid, patch, fill).astype(np.float32))

    return np.stack(channels, axis=0)   # (4, gs, gs)


def build_goal_heatmap(arena: ArenaConfig, goal_zone: Rect,
                       robot_x: float, robot_y: float, cfg: dict) -> np.ndarray:
    """
    Build a (grid_size, grid_size) goal heatmap centred on the robot.
    Uses the same view_width/view_height extents as crop_robot_view so
    the goal channel is spatially consistent with the terrain channels.
    """
    tc       = cfg['terrain']
    gs       = tc['grid_size']
    csx, csy = _view_cell_size(cfg)
    half     = gs // 2

    pj = np.arange(gs)
    pi = np.arange(gs)
    wx = robot_x + (pj - half) * csx   # (gs,)
    wy = robot_y + (pi - half) * csy   # (gs,)

    in_x    = (wx >= goal_zone.x) & (wx < goal_zone.x + goal_zone.w)
    in_y    = (wy >= goal_zone.y) & (wy < goal_zone.y + goal_zone.h)
    heatmap = (in_y[:, None] & in_x[None, :]).astype(np.float32)   # (gs, gs)

    heatmap = gaussian_filter(heatmap, sigma=3.0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def current_zone(arena: ArenaConfig, x: float, y: float) -> str:
    if arena.berm_target.contains(x, y):
        return 'berm_target'
    if arena.deposit_zone.contains(x, y):
        return 'deposit'
    if arena.start_zone.contains(x, y):
        return 'start'
    if arena.excavation_zone.contains(x, y):
        return 'excavation'
    if arena.nav_zone.contains(x, y):
        return 'nav'
    return 'outside'
