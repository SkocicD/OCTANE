from __future__ import annotations
import numpy as np
from scipy import ndimage


def extract_instances(
    semantic: np.ndarray,
    cell_size: float = 0.05,
) -> list[dict]:
    """Extract rock and crater instances from semantic grid via connected components.

    semantic: (H,W) int array — 0=free,1=rock,2=crater,3=wall
    cell_size: meters per cell
    Returns list of {type (0=rock,1=crater), x, y, diameter} in robot-frame meters.
    """
    H, W = semantic.shape
    half_x = H * cell_size / 2.0
    half_y = W * cell_size / 2.0
    instances = []

    for label_val, obj_type in [(1, 0), (2, 1)]:
        mask = (semantic == label_val).astype(np.uint8)
        labeled, n = ndimage.label(mask)
        for comp_id in range(1, n + 1):
            comp = labeled == comp_id
            area = comp.sum()
            if area < 4:
                continue
            rows, cols = np.where(comp)
            x = rows.mean() * cell_size - half_x
            y = cols.mean() * cell_size - half_y
            diameter = 2.0 * np.sqrt(area / np.pi) * cell_size
            instances.append({"type": obj_type, "x": float(x), "y": float(y), "diameter": float(diameter)})

    return instances
