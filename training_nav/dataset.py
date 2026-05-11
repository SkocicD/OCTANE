"""Navigation dataset — generates samples on the fly from random arenas.

Each sample:
  terrain:  (5, gs, gs) float32 — height, rocks, craters, walls, goal_heatmap
  heading:  (2,)        float32 — [sin(yaw), cos(yaw)]
  action:   (2,)        float32 — [linear_vel, angular_vel]

Mission phases (goal zones the robot is trained to navigate toward):
  0 — start → excavation zone entry
  1 — anywhere → berm target (deposit run)
The phase is sampled randomly per episode so the model learns both legs.
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


_PHASES = ['to_excavation', 'to_deposit']


def _goal_zone_for_phase(arena: ArenaConfig, phase: str) -> Rect:
    if phase == 'to_excavation':
        return arena.excavation_zone
    return arena.berm_target


def _sample_robot_pose(arena: ArenaConfig, phase: str,
                       rng: random.Random) -> tuple[float, float, float]:
    """Sample a valid robot start position and random heading for the given phase."""
    margin = 0.4  # keep robot away from walls

    if phase == 'to_excavation':
        # Robot starts anywhere in the start or nav zone
        candidates = [arena.start_zone, arena.nav_zone]
    else:
        # Robot starts anywhere except the deposit zone (doing a deposit run)
        candidates = [arena.start_zone, arena.nav_zone, arena.excavation_zone]

    for _ in range(50):
        zone = rng.choice(candidates)
        if zone.w <= 2 * margin or zone.h <= 2 * margin:
            continue
        x = rng.uniform(zone.x + margin, zone.x + zone.w - margin)
        y = rng.uniform(zone.y + margin, zone.y + zone.h - margin)
        heading = rng.uniform(-math.pi, math.pi)
        return x, y, heading

    # Fallback: centre of arena
    return arena.width / 2, arena.length / 2, 0.0


class NavDataset(Dataset):
    """Generates navigation samples on-the-fly.

    Each call to __getitem__ generates a fresh random arena, placing the
    robot at a random valid position, and returns the A*-supervised action.
    The dataset length controls how many samples are drawn per epoch.
    """

    def __init__(self, cfg: dict, n_samples: int, seed: int = 0):
        self.cfg      = cfg
        self.n        = n_samples
        self.base_rng = random.Random(seed)
        self.np_rng   = np.random.default_rng(seed)
        # Pre-generate seeds for reproducibility across epochs
        self._seeds   = [self.base_rng.randint(0, 2**31) for _ in range(n_samples)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        seed = self._seeds[idx]
        rng  = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        # Generate arena
        arena   = generate_arena(self.cfg, rng)
        terrain = build_terrain_maps(arena, self.cfg, np_rng)

        # Pick mission phase
        phase     = rng.choice(_PHASES)
        goal_zone = _goal_zone_for_phase(arena, phase)

        # Sample robot pose
        rx, ry, heading = _sample_robot_pose(arena, phase, rng)

        # Build model inputs
        terrain_crop = crop_robot_view(terrain, rx, ry, self.cfg)  # (4, gs, gs)
        goal_map     = build_goal_heatmap(arena, goal_zone, rx, ry, self.cfg)  # (gs, gs)
        terrain_5ch  = np.concatenate(
            [terrain_crop, goal_map[None]], axis=0
        ).astype(np.float32)  # (5, gs, gs)

        heading_vec = np.array([math.sin(heading), math.cos(heading)], dtype=np.float32)

        # Get expert action from A*
        action = plan_action(terrain_crop, goal_map, heading, self.cfg)
        if action is None:
            # No path found — teach the model to stop
            action = (0.0, 0.0)

        action_vec = np.array(action, dtype=np.float32)

        return (
            torch.from_numpy(terrain_5ch),
            torch.from_numpy(heading_vec),
            torch.from_numpy(action_vec),
        )

    def reshuffle(self, epoch: int):
        """Call between epochs to get fresh arenas while keeping reproducibility."""
        base = random.Random(epoch)
        self._seeds = [base.randint(0, 2**31) for _ in range(self.n)]
