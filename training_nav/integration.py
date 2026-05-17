"""Terrain model → Navigation model integration contract.

This module documents and implements the exact bridge between the two ML models
at deployment time.  The nav model is trained to accept input in a format that
matches the terrain model's output directly — NO lossy resizing is needed.

═══════════════════════════════════════════════════════════════════════════════
TERRAIN MODEL  (ml/terrain_model branch)
  Output: dict with keys 'height', 'rocks', 'craters', 'walls'
          Each tensor: (B, 200, 200)  float32
          Resolution:  CELL_SIZE = 0.05 m/cell
          Coverage:    BEV_HALF  = 5.0 m  →  10 m × 10 m centred on robot
          Robot pixel: (100, 100)  ← centre of the 200×200 grid

NAV MODEL  (ml/navigation_model branch)
  Input ch0: height   (B, T, 80, 80)
  Input ch1: rocks    (B, T, 80, 80)
  Input ch2: craters  (B, T, 80, 80)
  Input ch3: walls    (B, T, 80, 80)
  Input ch4: goal     (B, T, 80, 80)  ← from mission planner, NOT terrain model
  Full input: (B, T, 5, 80, 80)
  Resolution: 0.05 m/cell  (== terrain model)
  Coverage:   4.0 m × 4.0 m centred on robot  (80 cells × 0.05 m)

THE CROP
  The terrain model's 200×200 BEV covers ±5 m from the robot.
  The nav model needs ±2 m (80 cells × 0.05 m / 2 = 2 m).
  Centre crop: rows [60:140], cols [60:140]  (exactly 80×80, no resize needed)

      terrain 200×200
      ┌──────────────────────────┐
      │        5m margin         │
      │   ┌──────────────┐   60  │
      │   │  nav crop    │       │
      │ 2m│  80×80       │       │
      │   │  (±2m)       │       │
      │   └──────────────┘  140  │
      │                          │
      └──────────────────────────┘
      0    60       140      200

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import torch


# Crop indices into the terrain model's 200×200 BEV grid
_TERRAIN_GRID   = 200
_NAV_GRID       = 80
_CROP_START     = (_TERRAIN_GRID - _NAV_GRID) // 2   # 60
_CROP_END       = _CROP_START + _NAV_GRID             # 140
_CROP_SLICE     = slice(_CROP_START, _CROP_END)


def terrain_to_nav_input(terrain_output: dict,
                         goal_heatmap: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert terrain model output dict → nav model input tensor.

    Args:
        terrain_output: dict with keys 'height', 'rocks', 'craters', 'walls'.
                        Each value is (B, 200, 200) float32 tensor or ndarray.
        goal_heatmap:   (B, 80, 80) float32 — goal zone heatmap from mission
                        planner.  Must already be at nav grid resolution (80×80
                        at 0.05 m/cell).

    Returns:
        (B, 5, 80, 80) float32 tensor ready for NavPolicy.forward().
    """
    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    height  = _to_tensor(terrain_output['height']) [..., _CROP_SLICE, _CROP_SLICE]
    rocks   = _to_tensor(terrain_output['rocks'])  [..., _CROP_SLICE, _CROP_SLICE]
    craters = _to_tensor(terrain_output['craters'])[..., _CROP_SLICE, _CROP_SLICE]
    walls   = _to_tensor(terrain_output['walls'])  [..., _CROP_SLICE, _CROP_SLICE]
    goal    = _to_tensor(goal_heatmap)

    return torch.stack([height, rocks, craters, walls, goal], dim=-3)  # (B, 5, 80, 80)


def make_goal_heatmap(goal_zone_centre_m: tuple[float, float],
                      robot_x_m: float, robot_y_m: float,
                      sigma_m: float = 0.5,
                      cell_size: float = 0.05,
                      grid_size: int = 80) -> np.ndarray:
    """Generate a goal heatmap in nav-model coordinates.

    Args:
        goal_zone_centre_m: (gx, gy) world position of goal centre in metres.
        robot_x_m, robot_y_m: robot world position in metres.
        sigma_m: Gaussian spread in metres.
        cell_size: metres per cell (must equal terrain model cell size = 0.05).
        grid_size: nav grid size (must equal 80).

    Returns:
        (grid_size, grid_size) float32 ndarray, peak value 1.0 at goal direction,
        0.0 if goal is outside the view.
    """
    half   = grid_size // 2
    sigma  = sigma_m / cell_size                # convert to cells

    gx_rel = goal_zone_centre_m[0] - robot_x_m  # goal relative to robot (metres)
    gy_rel = goal_zone_centre_m[1] - robot_y_m

    # Goal in grid coordinates (robot at centre = half, half)
    gc = half + gx_rel / cell_size
    gr = half + gy_rel / cell_size

    # Build Gaussian centred on goal position
    cols = np.arange(grid_size, dtype=np.float32)
    rows = np.arange(grid_size, dtype=np.float32)
    cc, rr = np.meshgrid(cols, rows)
    heatmap = np.exp(-((cc - gc) ** 2 + (rr - gr) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap.astype(np.float32)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


# ── Sanity check ───────────────────────────────────────────────────────────────

def verify_crop_alignment():
    """Quick smoke-test that crop indices are correct."""
    assert _CROP_START == 60,  f"Expected crop start 60, got {_CROP_START}"
    assert _CROP_END   == 140, f"Expected crop end 140, got {_CROP_END}"
    assert _CROP_END - _CROP_START == _NAV_GRID

    coverage_m = _NAV_GRID * 0.05   # 80 × 0.05 = 4.0 m
    assert abs(coverage_m - 4.0) < 1e-6, f"Nav coverage should be 4.0 m, got {coverage_m}"
    print(f"Crop alignment OK: [{_CROP_START}:{_CROP_END}] → {_NAV_GRID}×{_NAV_GRID} cells "
          f"= {coverage_m:.1f}m × {coverage_m:.1f}m at 0.05 m/cell")


if __name__ == '__main__':
    verify_crop_alignment()
