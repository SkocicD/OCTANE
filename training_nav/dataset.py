"""Navigation dataset — generates samples on the fly from random arenas.

Each sample:
  terrain:    (5, gs, gs) float32 — height, rocks, craters, walls, goal_heatmap
  heading:    (2,)        float32 — [sin(yaw), cos(yaw)]
  arena_type: ()          float32 — 0.0=UCF, 1.0=KSC
  action:     (3,)        float32 — [left_motor, right_motor, bucket_class(0/1/2)]

Mission phases (randomly sampled per episode):
  to_excavation  — navigate from start/nav zone to excavation zone   (bucket UP)
  digging        — slow forward movement in excavation zone           (bucket COLLECT)
  to_deposit     — navigate from excavation zone to berm target       (bucket UP)
  dumping        — stopped at berm zone                               (bucket DUMP)
"""

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from training_nav.arena import (
    ArenaConfig, Rect, build_goal_heatmap, build_terrain_maps,
    crop_robot_view, generate_arena,
)
from training_nav.planner import plan_action


_PHASES = ['to_excavation', 'digging', 'to_deposit', 'dumping']


def _goal_zone_for_phase(arena: ArenaConfig, phase: str) -> Rect:
    if phase in ('to_excavation', 'digging'):
        return arena.excavation_zone
    return arena.berm_target


def _start_zones_for_phase(arena: ArenaConfig, phase: str) -> list[Rect]:
    """Return valid starting zones for robot placement given the phase."""
    if phase == 'to_excavation':
        return [arena.start_zone, arena.nav_zone]
    elif phase == 'digging':
        # Place robot in the excavation zone, but avoid any overlap with the start
        # zone (KSC start sits at the bottom of the excavation column).
        ez = arena.excavation_zone
        sz = arena.start_zone
        overlaps = (sz.x < ez.x + ez.w and sz.x + sz.w > ez.x and
                    sz.y < ez.y + ez.h and sz.y + sz.h > ez.y)
        y_min = max(ez.y, sz.y + sz.h) if overlaps else ez.y + ez.h * 0.1
        inner = Rect(ez.x, y_min, ez.w, ez.y + ez.h - y_min)
        return [inner if inner.h >= 0.5 else ez]
    elif phase == 'to_deposit':
        return [arena.excavation_zone, arena.nav_zone]
    elif phase == 'dumping':
        return [arena.deposit_zone]
    return [arena.start_zone]


def _sample_robot_pose(arena: ArenaConfig, phase: str,
                       rng: random.Random) -> tuple[float, float, float]:
    """Sample a valid robot start position (clear of obstacles) for the given phase."""
    margin     = 0.4
    clearance  = 0.55   # min distance from robot centre to any obstacle edge
    candidates = _start_zones_for_phase(arena, phase)

    for _ in range(100):
        zone = rng.choice(candidates)
        if zone.w <= 2 * margin or zone.h <= 2 * margin:
            continue
        x = rng.uniform(zone.x + margin, zone.x + zone.w - margin)
        y = rng.uniform(zone.y + margin, zone.y + zone.h - margin)
        if all(math.hypot(x - o.x, y - o.y) > o.diameter / 2 + clearance
               for o in arena.obstacles):
            return x, y, rng.uniform(-math.pi, math.pi)

    z = candidates[0]
    return z.x + z.w / 2, z.y + z.h / 2, 0.0


class NavDataset(Dataset):
    """Generates navigation samples on the fly.

    Each call to __getitem__ generates a fresh random arena (KSC or UCF,
    sampled according to arena_mix_ucf), places the robot at a valid position
    for the sampled phase, and returns the expert-supervised action.
    """

    def __init__(self, cfg: dict, n_samples: int, seed: int = 0):
        self.cfg      = cfg
        self.n        = n_samples
        self.base_rng = random.Random(seed)
        self.np_rng   = np.random.default_rng(seed)
        self._seeds   = [self.base_rng.randint(0, 2**31) for _ in range(n_samples)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        seed   = self._seeds[idx]
        rng    = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        mix        = self.cfg['training'].get('arena_mix_ucf', 0.4)
        atype      = 'ucf' if rng.random() < mix else 'ksc'
        atype_val  = np.float32(0.0 if atype == 'ucf' else 1.0)

        arena   = generate_arena(self.cfg, rng, arena_type=atype)
        terrain = build_terrain_maps(arena, self.cfg, np_rng)

        phase     = rng.choice(_PHASES)
        goal_zone = _goal_zone_for_phase(arena, phase)
        rx, ry, heading = _sample_robot_pose(arena, phase, rng)

        terrain_crop = crop_robot_view(terrain, rx, ry, self.cfg)
        goal_map     = build_goal_heatmap(arena, goal_zone, rx, ry, self.cfg)
        terrain_5ch  = np.concatenate(
            [terrain_crop, goal_map[None]], axis=0
        ).astype(np.float32)

        heading_vec = np.array([math.sin(heading), math.cos(heading)], dtype=np.float32)

        action = plan_action(terrain_crop, goal_map, heading, self.cfg, phase=phase)
        if action is None:
            action = (0.0, 0.0, 0)

        left, right, bucket = action
        action_vec = np.array([left, right, float(bucket)], dtype=np.float32)

        return (
            torch.from_numpy(terrain_5ch),
            torch.from_numpy(heading_vec),
            torch.tensor(atype_val),
            torch.from_numpy(action_vec),
        )

    def reshuffle(self, epoch: int):
        base = random.Random(epoch)
        self._seeds = [base.randint(0, 2**31) for _ in range(self.n)]
