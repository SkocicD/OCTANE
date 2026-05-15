"""Navigation dataset — generates samples on the fly from random arenas.

Each sample is a sequence of T timesteps (T = cfg['model']['seq_len'], default 32):
  terrain:    (T, 5, gs, gs) float32 — height, rocks, craters, walls, goal_heatmap
  heading:    (T, 2)         float32 — [sin(yaw), cos(yaw)]
  arena_type: ()             float32 — 0.0=UCF, 1.0=KSC
  action:     (T, 3)         float32 — [left_motor, right_motor, bucket_class(0/1/2)]
                             motors are normalised to [-1, 1] (divide by nav_speed_limit).

Mission phases (randomly sampled per episode):
  to_excavation  — navigate from start/nav zone to excavation zone   (bucket UP)
  digging        — slow forward movement in excavation zone           (bucket COLLECT)
  to_deposit     — navigate from excavation zone to berm target       (bucket UP)
  dumping        — stopped at berm zone                               (bucket DUMP)

Staged curriculum (controlled by config.curriculum):
  Stage 0  — no obstacles; robot learns pure goal-seeking.
  Stage 1  — 1-2 obstacles; basic avoidance.
  Stage 2  — full obstacles + DART positional noise; recovery training.
  Stage 3  — full obstacles + heavier DART; hardening.

DART (Dataset Aggregation via Random Trajectory) perturbs the robot's starting
position with Gaussian noise so the model sees off-path recovery states without
requiring interactive expert rollouts.
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
    margin    = 0.4
    clearance = 0.55
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
    """Generates navigation samples on the fly with a staged curriculum.

    reshuffle(epoch) must be called each epoch before iterating so that
    workers receive fresh seeds AND the correct curriculum stage.
    """

    def __init__(self, cfg: dict, n_samples: int, seed: int = 0):
        self.cfg       = cfg
        self.n         = n_samples
        self.epoch     = 0
        self.max_epochs = cfg['training'].get('epochs', 200)
        self._cc       = cfg.get('curriculum', {})
        self.base_rng  = random.Random(seed)
        self.np_rng    = np.random.default_rng(seed)
        self._seeds    = [self.base_rng.randint(0, 2**31) for _ in range(n_samples)]

    def __len__(self):
        return self.n

    # ── Curriculum helpers ────────────────────────────────────────────────────

    def _stage(self) -> int:
        prog = self.epoch / max(self.max_epochs, 1)
        cc   = self._cc
        if prog < cc.get('stage0_end', 0.15): return 0
        if prog < cc.get('stage1_end', 0.40): return 1
        if prog < cc.get('stage2_end', 0.75): return 2
        return 3

    def _dart_params(self, stage: int) -> tuple[float, float]:
        cc = self._cc
        if stage == 2:
            return cc.get('dart_prob_s2', 0.30), cc.get('dart_std_s2', 0.18)
        if stage == 3:
            return cc.get('dart_prob_s3', 0.50), cc.get('dart_std_s3', 0.32)
        return 0.0, 0.0

    def _seq_len(self) -> int:
        """Sequence length grows with curriculum stage.

        Short sequences in early stages keep training fast and match task
        complexity — goal-seeking doesn't need 3 s of history, but DART
        recovery does.  Lengths are configurable via curriculum.seq_len_s*.
        """
        base = self.cfg['model'].get('seq_len', 32)
        cc   = self._cc
        stage = self._stage()
        lengths = {
            0: cc.get('seq_len_s0', max(4,  base // 4)),
            1: cc.get('seq_len_s1', max(8,  base // 2)),
            2: cc.get('seq_len_s2', base),
            3: cc.get('seq_len_s3', base),
        }
        return lengths[stage]

    # ── Sample generation ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        seed   = self._seeds[idx]
        rng    = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        stage  = self._stage()

        mix       = self.cfg['training'].get('arena_mix_ucf', 0.4)
        atype     = 'ucf' if rng.random() < mix else 'ksc'
        atype_val = np.float32(0.0 if atype == 'ucf' else 1.0)

        arena = generate_arena(self.cfg, rng, arena_type=atype)

        # ── Stage 0 / 1: prune obstacles to teach basic goal-seeking first ────
        if stage == 0:
            # No obstacles at all — pure goal-seeking (structural column kept)
            arena.obstacles = [o for o in arena.obstacles if o.kind == 'column']
        elif stage == 1:
            # Keep column + at most 2 non-column obstacles
            non_col = [o for o in arena.obstacles if o.kind != 'column']
            rng.shuffle(non_col)
            arena.obstacles = ([o for o in arena.obstacles if o.kind == 'column']
                               + non_col[:2])

        terrain = build_terrain_maps(arena, self.cfg, np_rng)

        phase     = rng.choice(_PHASES)
        goal_zone = _goal_zone_for_phase(arena, phase)
        rx, ry, heading = _sample_robot_pose(arena, phase, rng)

        # ── DART: perturb position to create recovery training states ─────────
        dart_prob, dart_std = self._dart_params(stage)
        if dart_prob > 0 and rng.random() < dart_prob:
            margin = 0.4
            for _ in range(50):
                rx_p = rx + rng.gauss(0, dart_std)
                ry_p = ry + rng.gauss(0, dart_std)
                rx_p = float(np.clip(rx_p, margin, arena.width  - margin))
                ry_p = float(np.clip(ry_p, margin, arena.length - margin))
                clearance = 0.35
                if all(math.hypot(rx_p - o.x, ry_p - o.y) > o.diameter / 2 + clearance
                       for o in arena.obstacles):
                    rx, ry = rx_p, ry_p
                    break

        seq_len   = self._seq_len()
        nav_limit = self.cfg['robot'].get('nav_speed_limit', 1.0)
        sim_dt    = self.cfg['training'].get('sim_dt', 0.10)
        wb        = self.cfg['robot']['wheel_base']

        terrain_list = []
        heading_list = []
        action_list  = []

        for _ in range(seq_len):
            # Record heading BEFORE update
            heading_vec = np.array([math.sin(heading), math.cos(heading)],
                                   dtype=np.float32)
            heading_list.append(heading_vec)

            terrain_crop = crop_robot_view(terrain, rx, ry, self.cfg)
            goal_map     = build_goal_heatmap(arena, goal_zone, rx, ry, self.cfg)
            terrain_5ch  = np.concatenate(
                [terrain_crop, goal_map[None]], axis=0
            ).astype(np.float32)
            terrain_list.append(terrain_5ch)

            action = plan_action(terrain_crop, goal_map, heading, self.cfg, phase=phase)
            if action is None:
                action = (0.0, 0.0, 0)
            left, right, bucket = action
            action_vec = np.array([left / nav_limit, right / nav_limit, float(bucket)],
                                   dtype=np.float32)
            action_list.append(action_vec)

            # Differential-drive kinematics: advance robot pose
            v     = (left + right) / 2.0
            omega = (right - left) / wb
            rx    += v * math.cos(heading) * sim_dt
            ry    += v * math.sin(heading) * sim_dt
            heading += omega * sim_dt
            heading  = ((heading + math.pi) % (2 * math.pi)) - math.pi
            rx = float(np.clip(rx, 0.2, arena.width  - 0.2))
            ry = float(np.clip(ry, 0.2, arena.length - 0.2))

        return (
            torch.from_numpy(np.stack(terrain_list).astype(np.float32)),   # (T, 5, gs, gs)
            torch.from_numpy(np.stack(heading_list).astype(np.float32)),   # (T, 2)
            torch.tensor(atype_val),                                        # scalar
            torch.from_numpy(np.stack(action_list).astype(np.float32)),    # (T, 3)
        )

    def reshuffle(self, epoch: int):
        self.epoch = epoch
        base = random.Random(epoch)
        self._seeds = [base.randint(0, 2**31) for _ in range(self.n)]
