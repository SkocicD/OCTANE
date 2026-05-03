"""BEV rasterizer — point cloud to 6-channel bird's-eye-view grid."""
from __future__ import annotations
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


def rasterize_from_ros_points(
    points: np.ndarray,
    grid_size: int = 200,
    cell_size: float = 0.05,
) -> np.ndarray:
    """Project XYZ+RGB point cloud into BEV grid.
    points: (N,4) float32 — x,y,z,rgb_packed
    Returns (6,grid_size,grid_size) float32: [height_max,height_mean,r,g,b,occupancy]
    """
    half = grid_size * cell_size / 2.0
    grid = np.zeros((6, grid_size, grid_size), dtype=np.float32)
    if len(points) == 0:
        return grid
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rgb_packed = points[:, 3].view(np.uint32)
    r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((rgb_packed >> 8)  & 0xFF).astype(np.float32) / 255.0
    b = ( rgb_packed        & 0xFF).astype(np.float32) / 255.0
    ix = ((x + half) / cell_size).astype(np.int32)
    iy = ((y + half) / cell_size).astype(np.int32)
    mask = (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size)
    ix, iy, z, r, g, b = ix[mask], iy[mask], z[mask], r[mask], g[mask], b[mask]
    height_sum = np.zeros((grid_size, grid_size), dtype=np.float32)
    count      = np.zeros((grid_size, grid_size), dtype=np.int32)
    np.maximum.at(grid[0], (ix, iy), z)
    np.add.at(height_sum,  (ix, iy), z)
    np.add.at(count,       (ix, iy), 1)
    np.add.at(grid[2],     (ix, iy), r)
    np.add.at(grid[3],     (ix, iy), g)
    np.add.at(grid[4],     (ix, iy), b)
    occ = count > 0
    grid[1][occ]  = height_sum[occ] / count[occ]
    grid[2][occ] /= count[occ]
    grid[3][occ] /= count[occ]
    grid[4][occ] /= count[occ]
    grid[5][occ]  = 1.0
    return grid


def unpack_point_cloud2(msg: PointCloud2) -> np.ndarray:
    """Read PointCloud2 into (N,4) float32 array [x,y,z,rgb_packed]."""
    pts = list(pc2.read_points(msg, field_names=(x, y, z, rgb), skip_nans=True))
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(pts, dtype=np.float32)
