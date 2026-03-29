"""Procedural regolith terrain for the CSU Lunabotics Artemis Arena.

Simulates lunar regolith surface as a height field composed of overlapping
Gaussian mounds — one mound per "particle". Difficulty scales particle density
and crest heights from the low end to the high end of each configured range.

Arena footprint: 7.220 m × 5.350 m (centered at origin)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def regolith_terrain(difficulty: float, cfg: "RegolithTerrainCfg") -> np.ndarray:
    """Generate a height field simulating loose lunar regolith.

    Each "particle" is a Gaussian mound placed at a random position. The
    resulting height field is the superposition of all mounds, clipped to the
    configured crest height range.

    Args:
        difficulty: Float in [0, 1]. Scales density and heights from the low
                    end (0) to the high end (1) of each range.
        cfg: ``RegolithTerrainCfg`` instance with terrain parameters.

    Returns:
        Integer height field array of shape ``(num_x, num_y)`` in units of
        ``cfg.vertical_scale`` meters per count, suitable for Isaac Lab's
        ``HfTerrainBaseCfg`` pipeline.
    """
    num_x = int(cfg.size[0] / cfg.horizontal_scale)
    num_y = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((num_x, num_y), dtype=np.float32)

    rng = np.random.default_rng()

    # interpolate density and crest height from low → high based on difficulty
    density = cfg.particle_density_range[0] + (
        cfg.particle_density_range[1] - cfg.particle_density_range[0]
    ) * difficulty
    max_crest = cfg.crest_height_range[0] + (
        cfg.crest_height_range[1] - cfg.crest_height_range[0]
    ) * difficulty

    area = cfg.size[0] * cfg.size[1]
    num_particles = max(1, int(density * area))

    # particle centre positions in metres (within terrain bounds)
    cx = rng.uniform(0.0, cfg.size[0], size=num_particles)
    cy = rng.uniform(0.0, cfg.size[1], size=num_particles)

    # per-particle radius and height sampled uniformly within configured ranges
    radii = rng.uniform(
        cfg.particle_radius_range[0], cfg.particle_radius_range[1], size=num_particles
    )
    heights = rng.uniform(cfg.crest_height_range[0], max_crest, size=num_particles)

    # grid coordinates in metres
    xs = np.arange(num_x) * cfg.horizontal_scale  # (num_x,)
    ys = np.arange(num_y) * cfg.horizontal_scale  # (num_y,)

    for i in range(num_particles):
        r = radii[i]
        h = heights[i]

        # bounding box in grid indices to avoid iterating over the whole field
        ix_lo = max(0, int((cx[i] - 3.0 * r) / cfg.horizontal_scale))
        ix_hi = min(num_x, int((cx[i] + 3.0 * r) / cfg.horizontal_scale) + 1)
        iy_lo = max(0, int((cy[i] - 3.0 * r) / cfg.horizontal_scale))
        iy_hi = min(num_y, int((cy[i] + 3.0 * r) / cfg.horizontal_scale) + 1)

        dx = xs[ix_lo:ix_hi] - cx[i]   # (nx_local,)
        dy = ys[iy_lo:iy_hi] - cy[i]   # (ny_local,)

        # Gaussian: h * exp(-(dx² + dy²) / (2 * sigma²)), sigma = r / 2
        sigma2 = (r / 2.0) ** 2
        gauss = h * np.exp(
            -(dx[:, np.newaxis] ** 2 + dy[np.newaxis, :] ** 2) / (2.0 * sigma2)
        )
        hf[ix_lo:ix_hi, iy_lo:iy_hi] += gauss

    # convert metres → integer height counts
    return (hf / cfg.vertical_scale).astype(np.int16)


@configclass
class RegolithTerrainCfg(HfTerrainBaseCfg):
    """Configuration for the procedural regolith height-field terrain.

    Typical usage in an env config::

        from lunabotics.terrains import RegolithTerrainCfg
        from isaaclab.terrains import TerrainGeneratorCfg

        terrain = TerrainImporterCfg(
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                size=(7.22, 5.35),
                num_rows=8,
                num_cols=8,
                horizontal_scale=0.05,
                vertical_scale=0.001,
                sub_terrains={"regolith": RegolithTerrainCfg(proportion=1.0)},
            ),
            ...
        )
    """

    function: Callable = regolith_terrain

    # ── surface composition ───────────────────────────────────────────────────
    particle_density_range: tuple[float, float] = (20.0, 80.0)
    """Number of Gaussian mounds (particles) per m². Low end = difficulty 0, high end = difficulty 1."""

    particle_radius_range: tuple[float, float] = (0.02, 0.12)
    """Radius of each Gaussian mound in metres (sampled uniformly per particle)."""

    crest_height_range: tuple[float, float] = (0.005, 0.05)
    """Peak height of each mound in metres. High end is also scaled by difficulty."""
