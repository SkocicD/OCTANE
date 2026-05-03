from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class TerrainMappingEnvCfg:
    arena_size: Tuple[float, float] = (10.0, 10.0)
    regolith_noise_amplitude: float = 0.02
    rock_count_range: Tuple[int, int] = (5, 20)
    rock_diameter_range: Tuple[float, float] = (0.30, 0.40)
    crater_count_range: Tuple[int, int] = (5, 20)
    crater_diameter_range: Tuple[float, float] = (0.40, 0.50)
    crater_depth_ratio: float = 0.25
    wall_count_range: Tuple[int, int] = (0, 4)
    wall_length_range: Tuple[float, float] = (0.5, 3.0)
    wall_height: float = 0.6
    wall_materials: Tuple[str, ...] = ("concrete", "glass", "metal", "wood")
    ground_texture_variants: int = 8
    spawn_margin: float = 1.0
    height_field_resolution: float = 0.05
    bev_grid_size: int = 200
    bev_cell_size: float = 0.05
    episodes_per_run: int = 50_000
    output_dir: str = "data/terrain_mapping"
