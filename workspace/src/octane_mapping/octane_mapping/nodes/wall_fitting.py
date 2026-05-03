"""Fit wall line segments from semantic grid using RANSAC."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import RANSACRegressor


def fit_walls(
    semantic: np.ndarray,
    cell_size: float = 0.05,
    min_wall_cells: int = 10,
    max_walls: int = 8,
) -> list[dict]:
    """semantic: (H,W) int — Returns list of {x1,y1,x2,y2} in meters."""
    H, W = semantic.shape
    half_x = H * cell_size / 2.0
    half_y = W * cell_size / 2.0
    wall_cells = np.argwhere(semantic == 3)
    if len(wall_cells) < min_wall_cells:
        return []
    pts_x = wall_cells[:, 0] * cell_size - half_x
    pts_y = wall_cells[:, 1] * cell_size - half_y
    walls     = []
    remaining = np.ones(len(pts_x), dtype=bool)
    for _ in range(max_walls):
        if remaining.sum() < min_wall_cells:
            break
        rx, ry = pts_x[remaining], pts_y[remaining]
        ransac = RANSACRegressor(min_samples=0.3, residual_threshold=0.1, max_trials=100)
        x_spread  = rx.max() - rx.min() if len(rx) > 1 else 0.0
        y_spread  = ry.max() - ry.min() if len(ry) > 1 else 0.0
        fit_y_on_x = x_spread >= y_spread
        try:
            if fit_y_on_x:
                ransac.fit(rx.reshape(-1, 1), ry)
            else:
                ransac.fit(ry.reshape(-1, 1), rx)
        except ValueError:
            break
        inlier_mask = ransac.inlier_mask_
        if inlier_mask.sum() < min_wall_cells:
            break
        if fit_y_on_x:
            inlier_x = rx[inlier_mask]
            x1, x2  = float(inlier_x.min()), float(inlier_x.max())
            y1 = float(ransac.predict([[x1]])[0])
            y2 = float(ransac.predict([[x2]])[0])
        else:
            inlier_y = ry[inlier_mask]
            y1, y2  = float(inlier_y.min()), float(inlier_y.max())
            x1 = float(ransac.predict([[y1]])[0])
            x2 = float(ransac.predict([[y2]])[0])
        walls.append({x1: x1, y1: y1, x2: x2, y2: y2})
        orig = np.where(remaining)[0]
        remaining[orig[inlier_mask]] = False
    return walls
