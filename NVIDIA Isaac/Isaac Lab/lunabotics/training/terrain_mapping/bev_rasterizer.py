from __future__ import annotations
import numpy as np

def rasterize_bev(
    points: np.ndarray,
    grid_size: int = 200,
    cell_size: float = 0.05,
) -> np.ndarray:
    """Project XYZ+RGB point cloud (base_link frame) into BEV grid.
    points: (N,4) float32 — x,y,z,rgb_packed (uint32 as float, 0x00RRGGBB)
    Returns (6,grid_size,grid_size) float32: [height_max,height_mean,r,g,b,occupancy]
    Empty cells are all zeros."""
    half = grid_size * cell_size / 2.0
    grid = np.zeros((6, grid_size, grid_size), dtype=np.float32)
    if len(points) == 0:
        return grid
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rgb_packed = points[:, 3].view(np.uint32)
    r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_packed >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (rgb_packed & 0xFF).astype(np.float32) / 255.0
    ix = ((x + half) / cell_size).astype(np.int32)
    iy = ((y + half) / cell_size).astype(np.int32)
    mask = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size)
    ix, iy, z, r, g, b = ix[mask], iy[mask], z[mask], r[mask], g[mask], b[mask]
    height_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.int32)
    np.maximum.at(grid[0], (ix, iy), z)
    np.add.at(height_sum, (ix, iy), z)
    np.add.at(count, (ix, iy), 1)
    np.add.at(grid[2], (ix, iy), r)
    np.add.at(grid[3], (ix, iy), g)
    np.add.at(grid[4], (ix, iy), b)
    occ = count > 0
    grid[1][occ] = height_sum[occ] / count[occ]
    grid[2][occ] /= count[occ]
    grid[3][occ] /= count[occ]
    grid[4][occ] /= count[occ]
    grid[5][occ] = 1.0
    return grid
