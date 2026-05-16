"""BEV (Bird's-Eye-View) grid utilities.

Matches the coordinate system and constants used during Nexus model training.
The BEV grid is centred on the robot (base_link):
  axis-0 (row)    = forward/back  (+X in ROS, row 0 = +5 m ahead)
  axis-1 (col)    = left/right    (+Y in ROS, col 0 = +5 m left)
  cell size       = 0.05 m  (5 cm)
  grid            = 200 x 200  =>  10 m x 10 m coverage

Ported from training/dataset.py on the ml/terrain_model branch.
"""

import numpy as np
import cv2

GRID_SIZE = 200        # cells
CELL_SIZE = 0.05       # metres per cell
BEV_HALF  = GRID_SIZE * CELL_SIZE / 2   # 5.0 m — half-extent in each axis


def world_to_cell(v: float) -> int:
    """Convert a signed world coordinate (metres, centred on robot) to a grid cell index."""
    return int((v + BEV_HALF) / CELL_SIZE)


def cell_to_world(idx: int) -> float:
    """Convert a grid cell index to the world coordinate at its centre."""
    return (idx + 0.5) * CELL_SIZE - BEV_HALF


def build_object_heatmap(objects_gt: np.ndarray, class_id: int,
                          grid_size: int = GRID_SIZE,
                          cell_size: float = CELL_SIZE) -> np.ndarray:
    """Gaussian heatmap (0-1) for one object class.

    objects_gt: (N, 4) float — [rx, ry, diameter, class_id]
      rx  = forward offset from robot (metres, +X)
      ry  = lateral offset from robot (metres, +Y / left)
      diameter = object diameter in metres
      class_id = integer class label
    """
    half = grid_size * cell_size / 2
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    for obj in objects_gt:
        rx, ry, diameter, cls = float(obj[0]), float(obj[1]), float(obj[2]), int(obj[3])
        if cls != class_id:
            continue
        cx = int((rx + half) / cell_size)
        cy = int((ry + half) / cell_size)
        if not (0 <= cx < grid_size and 0 <= cy < grid_size):
            continue
        sigma  = max(1.0, (diameter / 2.0) / cell_size)
        radius = int(sigma * 3)
        x0, x1 = max(0, cx - radius), min(grid_size, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(grid_size, cy + radius + 1)
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        blob = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        heatmap[x0:x1, y0:y1] = np.maximum(heatmap[x0:x1, y0:y1], blob)
    return heatmap


def build_wall_mask(walls_gt: np.ndarray,
                    grid_size: int = GRID_SIZE,
                    cell_size: float = CELL_SIZE) -> np.ndarray:
    """Binary mask with wall segments rasterised as 3-pixel-thick lines.

    walls_gt: (M, 4) float — [rx1, ry1, rx2, ry2] endpoint pairs in metres.
    """
    half = grid_size * cell_size / 2
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    for wall in walls_gt:
        rx1, ry1, rx2, ry2 = float(wall[0]), float(wall[1]), float(wall[2]), float(wall[3])
        px1 = int((rx1 + half) / cell_size)
        py1 = int((ry1 + half) / cell_size)
        px2 = int((rx2 + half) / cell_size)
        py2 = int((ry2 + half) / cell_size)
        cv2.line(mask, (py1, px1), (py2, px2), 1.0, thickness=3)
    return mask


def terrain_to_colormap(height: np.ndarray,
                         vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """Convert a (H, W) float32 height map to an (H, W, 3) uint8 BGR image for display."""
    normed = np.clip((height - vmin) / (vmax - vmin), 0.0, 1.0)
    grey   = (normed * 255).astype(np.uint8)
    return cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)
