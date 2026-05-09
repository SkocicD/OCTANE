from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class CraterCfg:
    horizontal_scale: float = 0.05
    crater_count_range: Tuple[int, int] = (5, 20)
    crater_diameter_range: Tuple[float, float] = (0.40, 0.50)
    crater_depth_ratio: float = 0.25

def carve_craters(
    hf: np.ndarray,
    cfg: CraterCfg,
    arena_size: Tuple[float, float],
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Carve inverted-paraboloid crater depressions into a float32 height field.
    Returns (updated_hf, records) where records is list of {cx, cy, diameter, depth}."""
    if rng is None:
        rng = np.random.default_rng()
    hf = hf.copy()
    num_x, num_y = hf.shape
    records = []
    n = int(rng.integers(cfg.crater_count_range[0], cfg.crater_count_range[1] + 1))
    for _ in range(n):
        diameter = float(rng.uniform(cfg.crater_diameter_range[0], cfg.crater_diameter_range[1]))
        radius = diameter / 2.0
        depth = diameter * cfg.crater_depth_ratio
        cx = float(rng.uniform(radius, arena_size[0] - radius))
        cy = float(rng.uniform(radius, arena_size[1] - radius))
        records.append({"cx": cx, "cy": cy, "diameter": diameter, "depth": depth})
        ix_lo = max(0, int((cx - radius) / cfg.horizontal_scale))
        ix_hi = min(num_x, int((cx + radius) / cfg.horizontal_scale) + 1)
        iy_lo = max(0, int((cy - radius) / cfg.horizontal_scale))
        iy_hi = min(num_y, int((cy + radius) / cfg.horizontal_scale) + 1)
        xs = np.arange(ix_lo, ix_hi) * cfg.horizontal_scale
        ys = np.arange(iy_lo, iy_hi) * cfg.horizontal_scale
        dist2 = (xs[:, None] - cx) ** 2 + (ys[None, :] - cy) ** 2
        r2 = radius ** 2
        within = dist2 <= r2
        depression = depth * (1.0 - dist2 / r2)
        hf[ix_lo:ix_hi, iy_lo:iy_hi] -= np.where(within, depression, 0.0)
    return hf, records
