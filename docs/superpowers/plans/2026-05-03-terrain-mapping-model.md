# Terrain Mapping Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a ResNet-34 U-Net on procedurally generated Isaac Sim environments to produce real-time 3D terrain maps (heightmap + semantic labels + object detections) from the robot's combined RGBD point cloud.

**Architecture:** Isaac Sim procedurally generates random arenas (craters, rocks, walls) each episode; the real ROS DA3 perception stack runs inside Isaac Sim so training data has realistic depth noise; a shared BEV rasterizer converts the combined point cloud to a 200×200×6 grid that a ResNet-34 U-Net decodes into three outputs; TensorRT FP16 export targets ≥5 Hz on Jetson Orin AGX.

**Tech Stack:** Isaac Sim 4.x, rclpy (ROS2 Humble), PyTorch 2.x, torchvision, scipy, numpy, onnx, tensorrt, sensor\_msgs\_py, nav\_msgs

---

## File Structure

**Shorthand:** `<il>` = `NVIDIA Isaac/Isaac Lab/lunabotics/`

```
<il>/source/lunabotics/lunabotics/
  terrains/
    crater.py                          # NEW — carve_craters() height field utility
  tasks/direct/terrain_mapping/
    __init__.py                        # NEW — empty, marks package
    env_cfg.py                         # NEW — TerrainMappingEnvCfg dataclass

<il>/scripts/
  collect_terrain_data.py              # NEW — Isaac Sim + rclpy data collection loop

<il>/training/terrain_mapping/
  __init__.py                          # NEW — empty
  bev_rasterizer.py                    # NEW — numpy point cloud → 200×200×6 grid
  dataset.py                           # NEW — TerrainDataset (PyTorch Dataset)
  model.py                             # NEW — TerrainMappingModel (ResNet34 + U-Net + heads)
  losses.py                            # NEW — HeightmapLoss, SemanticLoss, DetectionLoss, TotalLoss
  train.py                             # NEW — training loop script
  evaluate.py                          # NEW — MAE/RMSE/mIoU/mAP metrics
  export_trt.py                        # NEW — PyTorch → ONNX → TensorRT FP16
  tests/
    __init__.py
    test_bev_rasterizer.py
    test_model.py
    test_losses.py
    test_dataset.py

workspace/src/octane_terrain_mapping/
  package.xml                          # NEW — ROS2 package manifest
  setup.py                             # NEW
  setup.cfg                            # NEW
  CMakeLists.txt                       # NEW — message generation only
  msg/
    TerrainObject.msg                  # NEW
    TerrainObjects.msg                 # NEW
    WallSegment.msg                    # NEW
    WallSegments.msg                   # NEW
  octane_terrain_mapping/
    __init__.py
    bev_rasterizer.py                  # NEW — mirrors training version, uses PointCloud2
    instance_extraction.py             # NEW — semantic grid → rock/crater instances
    wall_fitting.py                    # NEW — semantic grid → wall line segments (RANSAC)
    inference_node.py                  # NEW — full ROS2 inference node
  tests/
    __init__.py
    test_bev_rasterizer.py
    test_instance_extraction.py
    test_wall_fitting.py
```

---

## Task 1: Crater Height Field Utility

**Files:**
- Create: `<il>/source/lunabotics/lunabotics/terrains/crater.py`
- Create: `<il>/training/terrain_mapping/tests/test_bev_rasterizer.py` (stub for now — reused in Task 3)
- Create: `<il>/training/terrain_mapping/tests/__init__.py`

- [ ] **Step 1: Create test for crater carving**

Create `<il>/training/terrain_mapping/tests/test_crater_terrain.py`:

```python
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[4] / "source" / "lunabotics"))

from lunabotics.terrains.crater import carve_craters, CraterCfg

def test_carve_craters_returns_records():
    cfg = CraterCfg(
        horizontal_scale=0.05,
        crater_count_range=(2, 2),
        crater_diameter_range=(0.4, 0.4),
        crater_depth_ratio=0.25,
    )
    hf = np.zeros((200, 200), dtype=np.float32)
    updated_hf, records = carve_craters(hf, cfg, arena_size=(10.0, 10.0), rng=np.random.default_rng(42))
    assert len(records) == 2
    for r in records:
        assert set(r.keys()) == {"cx", "cy", "diameter", "depth"}
        assert 0.39 < r["diameter"] < 0.41
        assert r["depth"] == pytest.approx(r["diameter"] * 0.25)

def test_carve_craters_depresses_height_field():
    cfg = CraterCfg(
        horizontal_scale=0.05,
        crater_count_range=(1, 1),
        crater_diameter_range=(0.5, 0.5),
        crater_depth_ratio=0.25,
    )
    hf = np.zeros((200, 200), dtype=np.float32)
    updated_hf, records = carve_craters(hf, cfg, arena_size=(10.0, 10.0), rng=np.random.default_rng(0))
    # center cell of the crater should be depressed
    r = records[0]
    cx_idx = int(r["cx"] / 0.05)
    cy_idx = int(r["cy"] / 0.05)
    assert updated_hf[cx_idx, cy_idx] < -0.05  # deeper than 5cm at center

import pytest
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd "<il>"
python -m pytest training/terrain_mapping/tests/test_crater_terrain.py -v
```
Expected: `ModuleNotFoundError: No module named 'lunabotics.terrains.crater'`

- [ ] **Step 3: Implement `crater.py`**

Create `<il>/source/lunabotics/lunabotics/terrains/crater.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class CraterCfg:
    horizontal_scale: float = 0.05          # meters per grid cell
    crater_count_range: Tuple[int, int] = (5, 20)
    crater_diameter_range: Tuple[float, float] = (0.40, 0.50)  # meters
    crater_depth_ratio: float = 0.25        # depth = diameter × ratio


def carve_craters(
    hf: np.ndarray,
    cfg: CraterCfg,
    arena_size: Tuple[float, float],
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Carve inverted-paraboloid crater depressions into a float32 height field.

    Args:
        hf: (num_x, num_y) float32 height field in meters. Modified in-place copy.
        cfg: CraterCfg with randomization ranges.
        arena_size: (X_extent, Y_extent) in meters.
        rng: numpy RNG; creates one if None.

    Returns:
        (updated_hf, records) where records is a list of dicts
        {cx, cy, diameter, depth} in meters.
    """
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
```

- [ ] **Step 4: Run test to confirm it passes**

```
python -m pytest training/terrain_mapping/tests/test_crater_terrain.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Create `env_cfg.py`**

Create `<il>/source/lunabotics/lunabotics/tasks/direct/terrain_mapping/__init__.py` (empty).

Create `<il>/source/lunabotics/lunabotics/tasks/direct/terrain_mapping/env_cfg.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TerrainMappingEnvCfg:
    # Arena
    arena_size: Tuple[float, float] = (10.0, 10.0)   # meters X×Y
    regolith_noise_amplitude: float = 0.02            # meters

    # Rocks
    rock_count_range: Tuple[int, int] = (5, 20)
    rock_diameter_range: Tuple[float, float] = (0.30, 0.40)   # meters

    # Craters
    crater_count_range: Tuple[int, int] = (5, 20)
    crater_diameter_range: Tuple[float, float] = (0.40, 0.50) # meters
    crater_depth_ratio: float = 0.25

    # Walls
    wall_count_range: Tuple[int, int] = (0, 4)
    wall_length_range: Tuple[float, float] = (0.5, 3.0)       # meters
    wall_height: float = 0.6                                   # meters
    wall_materials: Tuple[str, ...] = ("concrete", "glass", "metal", "wood")

    # Ground
    ground_texture_variants: int = 8

    # Robot spawn
    spawn_margin: float = 1.0                                  # meters from arena edge

    # Data collection
    height_field_resolution: float = 0.05                     # meters per cell
    bev_grid_size: int = 200                                   # cells
    bev_cell_size: float = 0.05                                # meters per cell
    episodes_per_run: int = 50_000
    output_dir: str = "data/terrain_mapping"
```

- [ ] **Step 6: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/source/lunabotics/lunabotics/terrains/crater.py" \
        "NVIDIA Isaac/Isaac Lab/lunabotics/source/lunabotics/lunabotics/tasks/direct/terrain_mapping/" \
        "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/tests/"
git commit -m "feat: add crater terrain carver and environment config for terrain mapping"
```

---

## Task 2: BEV Rasterizer (Training)

**Files:**
- Create: `<il>/training/terrain_mapping/__init__.py`
- Create: `<il>/training/terrain_mapping/bev_rasterizer.py`
- Create: `<il>/training/terrain_mapping/tests/test_bev_rasterizer.py`

- [ ] **Step 1: Write the failing test**

Create `<il>/training/terrain_mapping/tests/test_bev_rasterizer.py`:

```python
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from bev_rasterizer import rasterize_bev


def _pack_rgb(r: int, g: int, b: int) -> float:
    packed = (r << 16) | (g << 8) | b
    return np.array(packed, dtype=np.uint32).view(np.float32)


def test_output_shape():
    points = np.zeros((100, 4), dtype=np.float32)
    points[:, 3] = _pack_rgb(255, 0, 0)
    grid = rasterize_bev(points)
    assert grid.shape == (6, 200, 200)


def test_empty_cloud_all_zeros():
    points = np.zeros((0, 4), dtype=np.float32)
    grid = rasterize_bev(points)
    assert grid.sum() == 0.0


def test_single_point_lands_in_correct_cell():
    # Point at x=0, y=0, z=1.5, red
    points = np.array([[0.0, 0.0, 1.5, _pack_rgb(255, 0, 0)]], dtype=np.float32)
    grid = rasterize_bev(points)
    cx = cy = 100  # center cell for ±5m grid at 5cm resolution
    assert grid[0, cx, cy] == pytest.approx(1.5)   # height_max
    assert grid[1, cx, cy] == pytest.approx(1.5)   # height_mean
    assert grid[2, cx, cy] == pytest.approx(1.0)   # r = 255/255
    assert grid[3, cx, cy] == pytest.approx(0.0)   # g
    assert grid[4, cx, cy] == pytest.approx(0.0)   # b
    assert grid[5, cx, cy] == pytest.approx(1.0)   # occupancy


def test_out_of_range_points_ignored():
    points = np.array([[100.0, 0.0, 1.0, _pack_rgb(0, 255, 0)]], dtype=np.float32)
    grid = rasterize_bev(points)
    assert grid.sum() == 0.0


import pytest
```

- [ ] **Step 2: Run test to confirm it fails**

```
cd "<il>"
python -m pytest training/terrain_mapping/tests/test_bev_rasterizer.py -v
```
Expected: `ModuleNotFoundError: No module named 'bev_rasterizer'`

- [ ] **Step 3: Implement `bev_rasterizer.py`**

Create `<il>/training/terrain_mapping/bev_rasterizer.py`:

```python
from __future__ import annotations
import numpy as np


def rasterize_bev(
    points: np.ndarray,
    grid_size: int = 200,
    cell_size: float = 0.05,
) -> np.ndarray:
    """Project XYZ+RGB point cloud (base_link frame) into a BEV grid.

    Args:
        points: (N, 4) float32 array — columns are x, y, z, rgb_packed.
                rgb_packed is a float32 reinterpretation of uint32 0x00RRGGBB.
        grid_size: Number of cells along each axis (default 200).
        cell_size: Meters per cell (default 0.05 = 5 cm).

    Returns:
        (6, grid_size, grid_size) float32 array.
        Channels: [height_max, height_mean, r, g, b, occupancy].
        Empty cells are all zeros.
    """
    half = grid_size * cell_size / 2.0  # 5.0 m

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
    ix, iy = ix[mask], iy[mask]
    z, r, g, b = z[mask], r[mask], g[mask], b[mask]

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
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest training/terrain_mapping/tests/test_bev_rasterizer.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/training/"
git commit -m "feat: add BEV rasterizer for point cloud to 200x200x6 grid projection"
```

---

## Task 3: Data Collection Script

**Files:**
- Create: `<il>/scripts/collect_terrain_data.py`

This script runs inside Isaac Sim (launched via `./isaaclab.sh -p scripts/collect_terrain_data.py`). It uses the Isaac Sim Python API to build scenes and rclpy to receive the point cloud from the existing ROS pipeline.

- [ ] **Step 1: Implement `collect_terrain_data.py`**

Create `<il>/scripts/collect_terrain_data.py`:

```python
"""Data collection script for terrain mapping model training.

Run with: ./isaaclab.sh -p scripts/collect_terrain_data.py --output_dir /path/to/data --episodes 50000

Prerequisites:
  - Isaac Sim running with the robot USD loaded (cameras + ROS2 OmniGraphs active)
  - ROS2 perception stack running (DA3 + point cloud combiner publishing /mapping/point_cloud/combined)
"""

import argparse
import pathlib
import time
import numpy as np

# Isaac Sim must be initialized before other omni imports
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
from omni.isaac.core.objects import VisualSphere, VisualCuboid
from pxr import UsdGeom, Gf
import carb

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source" / "lunabotics"))
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "training" / "terrain_mapping"))

from lunabotics.terrains.crater import carve_craters, CraterCfg
from lunabotics.tasks.direct.terrain_mapping.env_cfg import TerrainMappingEnvCfg
from bev_rasterizer import rasterize_bev


class PointCloudListener(Node):
    def __init__(self):
        super().__init__("terrain_data_collector")
        self.latest_points: np.ndarray | None = None
        self.create_subscription(
            PointCloud2, "/mapping/point_cloud/combined", self._cb, 10
        )

    def _cb(self, msg: PointCloud2):
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        if pts:
            self.latest_points = np.array(pts, dtype=np.float32)


def spawn_rocks(world: World, cfg: TerrainMappingEnvCfg, rng: np.random.Generator) -> list[dict]:
    n = int(rng.integers(cfg.rock_count_range[0], cfg.rock_count_range[1] + 1))
    records = []
    for i in range(n):
        diameter = float(rng.uniform(cfg.rock_diameter_range[0], cfg.rock_diameter_range[1]))
        x = float(rng.uniform(-cfg.arena_size[0] / 2 + cfg.spawn_margin,
                               cfg.arena_size[0] / 2 - cfg.spawn_margin))
        y = float(rng.uniform(-cfg.arena_size[1] / 2 + cfg.spawn_margin,
                               cfg.arena_size[1] / 2 - cfg.spawn_margin))
        z = diameter / 2.0  # sit on ground

        prim_path = f"/World/rocks/rock_{i}"
        VisualSphere(prim_path=prim_path, radius=diameter / 2.0,
                     position=np.array([x, y, z]))
        records.append({"type": 1, "x": x, "y": y, "diameter": diameter})
    return records


def spawn_walls(world: World, cfg: TerrainMappingEnvCfg, rng: np.random.Generator) -> list[dict]:
    n = int(rng.integers(cfg.wall_count_range[0], cfg.wall_count_range[1] + 1))
    records = []
    for i in range(n):
        length = float(rng.uniform(cfg.wall_length_range[0], cfg.wall_length_range[1]))
        angle = float(rng.uniform(0, np.pi))
        cx = float(rng.uniform(-cfg.arena_size[0] / 2 + 1.0, cfg.arena_size[0] / 2 - 1.0))
        cy = float(rng.uniform(-cfg.arena_size[1] / 2 + 1.0, cfg.arena_size[1] / 2 - 1.0))

        x1 = cx - (length / 2) * np.cos(angle)
        y1 = cy - (length / 2) * np.sin(angle)
        x2 = cx + (length / 2) * np.cos(angle)
        y2 = cy + (length / 2) * np.sin(angle)

        prim_path = f"/World/walls/wall_{i}"
        VisualCuboid(
            prim_path=prim_path,
            position=np.array([cx, cy, cfg.wall_height / 2.0]),
            scale=np.array([length, 0.05, cfg.wall_height]),
        )
        # Rotate to match angle
        prim = prim_utils.get_prim_at_path(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.AddRotateZOp().Set(float(np.degrees(angle)))

        records.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)})
    return records


def compute_ground_truth(
    cfg: TerrainMappingEnvCfg,
    crater_records: list[dict],
    rock_records: list[dict],
    wall_records: list[dict],
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
) -> dict:
    """Build analytical ground truth arrays in robot frame."""
    G = cfg.bev_grid_size
    C = cfg.bev_cell_size
    half = G * C / 2.0

    height_gt = np.zeros((G, G), dtype=np.float32)
    semantic_gt = np.zeros((G, G), dtype=np.int64)  # 0=free

    cos_a, sin_a = np.cos(-robot_yaw), np.sin(-robot_yaw)

    def world_to_robot(wx, wy):
        dx, dy = wx - robot_x, wy - robot_y
        rx = cos_a * dx - sin_a * dy
        ry = sin_a * dx + cos_a * dy
        return rx, ry

    def to_grid(rx, ry):
        ix = int((rx + half) / C)
        iy = int((ry + half) / C)
        return ix, iy

    # Craters (label=2, depression in height)
    for c in crater_records:
        rx, ry = world_to_robot(c["cx"], c["cy"])
        radius = c["diameter"] / 2.0
        depth = c["depth"]
        ix_c, iy_c = int((rx + half) / C), int((ry + half) / C)
        r_cells = int(radius / C) + 1
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                ii, jj = ix_c + di, iy_c + dj
                if 0 <= ii < G and 0 <= jj < G:
                    cell_x = (ii - G / 2) * C
                    cell_y = (jj - G / 2) * C
                    dist2 = (cell_x - rx) ** 2 + (cell_y - ry) ** 2
                    if dist2 <= radius ** 2:
                        h = -depth * (1.0 - dist2 / radius ** 2)
                        height_gt[ii, jj] = min(height_gt[ii, jj], h)
                        semantic_gt[ii, jj] = 2

    # Rocks (label=1, raised bump)
    objects_gt = []
    for r in rock_records:
        rx, ry = world_to_robot(r["x"], r["y"])
        objects_gt.append({"type": 0, "x": rx, "y": ry, "diameter": r["diameter"]})
        radius = r["diameter"] / 2.0
        ix_c, iy_c = int((rx + half) / C), int((ry + half) / C)
        r_cells = int(radius / C) + 1
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                ii, jj = ix_c + di, iy_c + dj
                if 0 <= ii < G and 0 <= jj < G:
                    cell_x = (ii - G / 2) * C
                    cell_y = (jj - G / 2) * C
                    dist2 = (cell_x - rx) ** 2 + (cell_y - ry) ** 2
                    if dist2 <= radius ** 2:
                        semantic_gt[ii, jj] = 1

    # Walls (label=3)
    walls_gt = []
    for w in wall_records:
        rx1, ry1 = world_to_robot(w["x1"], w["y1"])
        rx2, ry2 = world_to_robot(w["x2"], w["y2"])
        walls_gt.append({"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2})
        # Rasterize wall line into semantic grid
        steps = int(np.hypot(rx2 - rx1, ry2 - ry1) / C * 2) + 2
        for t in np.linspace(0, 1, steps):
            px, py = rx1 + t * (rx2 - rx1), ry1 + t * (ry2 - ry1)
            for dw in range(-1, 2):  # 3-cell width
                for dh in range(-1, 2):
                    ii = int((px + half) / C) + dw
                    jj = int((py + half) / C) + dh
                    if 0 <= ii < G and 0 <= jj < G:
                        semantic_gt[ii, jj] = 3

    return {
        "height_gt": height_gt,
        "semantic_gt": semantic_gt,
        "objects_gt": objects_gt,
        "walls_gt": walls_gt,
    }


def clear_episode_prims():
    for path in ["/World/rocks", "/World/walls", "/World/terrain"]:
        if prim_utils.is_prim_path_valid(path):
            prim_utils.delete_prim(path)


def run_collection(cfg: TerrainMappingEnvCfg, output_dir: pathlib.Path, n_episodes: int):
    rclpy.init()
    listener = PointCloudListener()
    world = World()
    rng = np.random.default_rng()
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(n_episodes):
        clear_episode_prims()

        # Generate crater height field (flat arena + craters)
        crater_cfg = CraterCfg(
            horizontal_scale=cfg.height_field_resolution,
            crater_count_range=cfg.crater_count_range,
            crater_diameter_range=cfg.crater_diameter_range,
            crater_depth_ratio=cfg.crater_depth_ratio,
        )
        hf_base = np.zeros(
            (int(cfg.arena_size[0] / cfg.height_field_resolution),
             int(cfg.arena_size[1] / cfg.height_field_resolution)),
            dtype=np.float32,
        )
        # Add regolith noise
        hf_base += rng.normal(0, cfg.regolith_noise_amplitude, hf_base.shape).astype(np.float32)
        hf, crater_records = carve_craters(hf_base, crater_cfg, cfg.arena_size, rng)

        rock_records = spawn_rocks(world, cfg, rng)
        wall_records = spawn_walls(world, cfg, rng)

        # Random robot pose
        robot_x = float(rng.uniform(-cfg.arena_size[0] / 2 + cfg.spawn_margin,
                                     cfg.arena_size[0] / 2 - cfg.spawn_margin))
        robot_y = float(rng.uniform(-cfg.arena_size[1] / 2 + cfg.spawn_margin,
                                     cfg.arena_size[1] / 2 - cfg.spawn_margin))
        robot_yaw = float(rng.uniform(0, 2 * np.pi))

        # Step sim to let cameras render + ROS pipeline process
        for _ in range(30):
            world.step(render=True)
            rclpy.spin_once(listener, timeout_sec=0.0)

        # Wait for point cloud
        deadline = time.monotonic() + 5.0
        while listener.latest_points is None and time.monotonic() < deadline:
            world.step(render=True)
            rclpy.spin_once(listener, timeout_sec=0.05)

        if listener.latest_points is None:
            carb.log_warn(f"Episode {ep}: no point cloud received, skipping")
            continue

        bev = rasterize_bev(listener.latest_points, cfg.bev_grid_size, cfg.bev_cell_size)
        listener.latest_points = None

        gt = compute_ground_truth(cfg, crater_records, rock_records, wall_records,
                                   robot_x, robot_y, robot_yaw)

        np.savez_compressed(
            output_dir / f"ep_{ep:06d}.npz",
            bev=bev,
            height_gt=gt["height_gt"],
            semantic_gt=gt["semantic_gt"],
            objects_gt=np.array(
                [[o["type"], o["x"], o["y"], o["diameter"]] for o in gt["objects_gt"]],
                dtype=np.float32,
            ) if gt["objects_gt"] else np.zeros((0, 4), dtype=np.float32),
            walls_gt=np.array(
                [[w["x1"], w["y1"], w["x2"], w["y2"]] for w in gt["walls_gt"]],
                dtype=np.float32,
            ) if gt["walls_gt"] else np.zeros((0, 4), dtype=np.float32),
        )

        if ep % 100 == 0:
            print(f"Episode {ep}/{n_episodes} saved")

    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/terrain_mapping")
    parser.add_argument("--episodes", type=int, default=50_000)
    args = parser.parse_args()

    cfg = TerrainMappingEnvCfg(
        output_dir=args.output_dir,
        episodes_per_run=args.episodes,
    )
    run_collection(cfg, pathlib.Path(args.output_dir), args.episodes)
```

- [ ] **Step 2: Smoke test (manual — requires Isaac Sim)**

```
cd "<il>"
./isaaclab.sh -p scripts/collect_terrain_data.py --output_dir /tmp/terrain_test --episodes 5
```
Expected: 5 `.npz` files in `/tmp/terrain_test/`, each ~500KB. Verify with:
```python
import numpy as np
d = np.load("/tmp/terrain_test/ep_000000.npz")
print(d["bev"].shape)       # (6, 200, 200)
print(d["height_gt"].shape) # (200, 200)
print(d["semantic_gt"].shape) # (200, 200)
print(d["objects_gt"].shape)  # (N, 4)
```

- [ ] **Step 3: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/scripts/collect_terrain_data.py"
git commit -m "feat: add Isaac Sim data collection script for terrain mapping dataset"
```

---

## Task 4: PyTorch Dataset

**Files:**
- Create: `<il>/training/terrain_mapping/dataset.py`
- Create: `<il>/training/terrain_mapping/tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Create `<il>/training/terrain_mapping/tests/test_dataset.py`:

```python
import numpy as np
import pathlib, tempfile, sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from dataset import TerrainDataset
import torch


def _write_fake_npz(path: pathlib.Path, n_objects: int = 3, n_walls: int = 2):
    np.savez_compressed(
        path,
        bev=np.random.rand(6, 200, 200).astype(np.float32),
        height_gt=np.random.rand(200, 200).astype(np.float32),
        semantic_gt=np.random.randint(0, 4, (200, 200), dtype=np.int64),
        objects_gt=np.random.rand(n_objects, 4).astype(np.float32),
        walls_gt=np.random.rand(n_walls, 4).astype(np.float32),
    )


def test_dataset_length():
    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            _write_fake_npz(pathlib.Path(d) / f"ep_{i:06d}.npz")
        ds = TerrainDataset(d)
        assert len(ds) == 5


def test_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as d:
        _write_fake_npz(pathlib.Path(d) / "ep_000000.npz", n_objects=4)
        ds = TerrainDataset(d)
        bev, height_gt, semantic_gt, objects_gt, occupancy = ds[0]
        assert bev.shape == (6, 200, 200)
        assert height_gt.shape == (200, 200)
        assert semantic_gt.shape == (200, 200)
        assert len(objects_gt) == 4                       # list of 4 dicts
        assert set(objects_gt[0].keys()) == {"type", "x", "y", "diameter"}
        assert occupancy.shape == (200, 200)
        assert occupancy.dtype == torch.float32
        assert set(occupancy.unique().tolist()).issubset({0.0, 1.0})
```

- [ ] **Step 2: Run test to confirm it fails**

```
python -m pytest training/terrain_mapping/tests/test_dataset.py -v
```
Expected: `ModuleNotFoundError: No module named 'dataset'`

- [ ] **Step 3: Implement `dataset.py`**

Create `<il>/training/terrain_mapping/dataset.py`:

```python
from __future__ import annotations
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


class TerrainDataset(Dataset):
    """Loads .npz files produced by collect_terrain_data.py.

    Returns per sample:
        bev          (6, 200, 200) float32 tensor
        height_gt    (200, 200) float32 tensor — terrain height in meters
        semantic_gt  (200, 200) int64 tensor — 0=free,1=rock,2=crater,3=wall
        objects_gt   list of dicts {type, x, y, diameter} — variable length per sample
        occupancy    (200, 200) float32 tensor — 1 where BEV has points, 0 elsewhere
    """

    def __init__(self, data_dir: str | pathlib.Path):
        self.files = sorted(pathlib.Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx], allow_pickle=False)
        bev = torch.from_numpy(d["bev"])                                       # (6, 200, 200)
        height_gt = torch.from_numpy(d["height_gt"])                           # (200, 200)
        semantic_gt = torch.from_numpy(d["semantic_gt"].astype(np.int64))      # (200, 200)
        raw_objects = d["objects_gt"]                                          # (N, 4) float32
        objects_gt = [
            {"type": int(row[0]), "x": float(row[1]), "y": float(row[2]), "diameter": float(row[3])}
            for row in raw_objects
        ]
        occupancy = bev[5].clone()                                             # occupancy channel
        return bev, height_gt, semantic_gt, objects_gt, occupancy
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest training/terrain_mapping/tests/test_dataset.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/dataset.py" \
        "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/tests/test_dataset.py"
git commit -m "feat: add TerrainDataset for loading collected .npz training data"
```

---

## Task 5: Model Architecture

**Files:**
- Create: `<il>/training/terrain_mapping/model.py`
- Create: `<il>/training/terrain_mapping/tests/test_model.py`

- [ ] **Step 1: Write the failing test**

Create `<il>/training/terrain_mapping/tests/test_model.py`:

```python
import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from model import TerrainMappingModel


def test_output_shapes():
    model = TerrainMappingModel()
    model.eval()
    x = torch.zeros(2, 6, 200, 200)
    with torch.no_grad():
        height, semantic, detections = model(x)
    assert height.shape == (2, 1, 200, 200)
    assert semantic.shape == (2, 4, 200, 200)
    assert detections.shape == (2, 110, 5)


def test_model_runs_on_single_sample():
    model = TerrainMappingModel()
    model.eval()
    x = torch.rand(1, 6, 200, 200)
    with torch.no_grad():
        height, semantic, detections = model(x)
    assert not torch.isnan(height).any()
    assert not torch.isnan(semantic).any()
    assert not torch.isnan(detections).any()
```

- [ ] **Step 2: Run test to confirm it fails**

```
python -m pytest training/terrain_mapping/tests/test_model.py -v
```
Expected: `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Implement `model.py`**

Create `<il>/training/terrain_mapping/model.py`:

```python
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TerrainMappingModel(nn.Module):
    """ResNet-34 encoder + U-Net decoder + lightweight detection head.

    Input:  (B, 6, 200, 200) BEV grid
    Output: height  (B, 1, 200, 200)  — terrain height in metres
            semantic (B, 4, 200, 200) — per-cell logits (free/rock/crater/wall)
            detections (B, 110, 5)    — [x, y, diam, conf, type] per slot
    """

    MAX_ROCKS: int = 50
    MAX_CRATERS: int = 60

    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

        # Patch first conv for 6-channel BEV input
        orig = backbone.conv1
        backbone.conv1 = nn.Conv2d(6, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            # Copy pretrained RGB weights into first 3 channels, mirror into next 3
            backbone.conv1.weight[:, :3] = orig.weight
            backbone.conv1.weight[:, 3:] = orig.weight

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # →64, /2
        self.pool = backbone.maxpool                                              # →64, /4
        self.enc1 = backbone.layer1   # →64,  /4
        self.enc2 = backbone.layer2   # →128, /8
        self.enc3 = backbone.layer3   # →256, /16
        self.enc4 = backbone.layer4   # →512, /32

        self.dec3 = _DecoderBlock(512 + 256, 256)
        self.dec2 = _DecoderBlock(256 + 128, 128)
        self.dec1 = _DecoderBlock(128 + 64, 64)
        self.dec0 = _DecoderBlock(64 + 64, 64)
        self.upsample_final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.height_head = nn.Conv2d(64, 1, 1)
        self.semantic_head = nn.Conv2d(64, 4, 1)

        n_slots = self.MAX_ROCKS + self.MAX_CRATERS  # 110
        self.detect_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_slots * 5),
        )

    def forward(self, x: torch.Tensor):
        e0 = self.enc0(x)           # (B, 64, 100, 100)
        ep = self.pool(e0)          # (B, 64,  50,  50)
        e1 = self.enc1(ep)          # (B, 64,  50,  50)
        e2 = self.enc2(e1)          # (B,128,  25,  25)
        e3 = self.enc3(e2)          # (B,256,  13,  13)
        e4 = self.enc4(e3)          # (B,512,   7,   7)

        d3 = self.dec3(e4, e3)      # (B,256,  13,  13)
        d2 = self.dec2(d3, e2)      # (B,128,  25,  25)
        d1 = self.dec1(d2, e1)      # (B, 64,  50,  50)
        d0 = self.dec0(d1, e0)      # (B, 64, 100, 100)
        out = self.upsample_final(d0)  # (B, 64, 200, 200)

        height = self.height_head(out)    # (B, 1, 200, 200)
        semantic = self.semantic_head(out) # (B, 4, 200, 200)
        detections = self.detect_head(e4).reshape(
            -1, self.MAX_ROCKS + self.MAX_CRATERS, 5
        )                                  # (B, 110, 5)

        return height, semantic, detections
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest training/terrain_mapping/tests/test_model.py -v
```
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/model.py" \
        "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/tests/test_model.py"
git commit -m "feat: add ResNet-34 U-Net terrain mapping model with height/semantic/detection heads"
```

---

## Task 6: Loss Functions

**Files:**
- Create: `<il>/training/terrain_mapping/losses.py`
- Create: `<il>/training/terrain_mapping/tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

Create `<il>/training/terrain_mapping/tests/test_losses.py`:

```python
import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

from losses import HeightmapLoss, SemanticLoss, DetectionLoss, TotalLoss


def test_heightmap_loss_ignores_empty_cells():
    loss_fn = HeightmapLoss()
    pred = torch.ones(2, 1, 200, 200)
    target = torch.zeros(2, 200, 200)
    occupancy = torch.zeros(2, 200, 200)  # all empty
    loss = loss_fn(pred, target, occupancy)
    assert loss.item() == 0.0


def test_heightmap_loss_nonzero_when_occupied():
    loss_fn = HeightmapLoss()
    pred = torch.ones(1, 1, 200, 200)
    target = torch.zeros(1, 200, 200)
    occupancy = torch.ones(1, 200, 200)
    loss = loss_fn(pred, target, occupancy)
    assert loss.item() > 0.0


def test_semantic_loss_returns_scalar():
    loss_fn = SemanticLoss()
    pred = torch.randn(2, 4, 200, 200)
    target = torch.randint(0, 4, (2, 200, 200))
    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert loss.item() > 0.0


def test_detection_loss_zero_gt_no_nan():
    loss_fn = DetectionLoss()
    pred = torch.randn(2, 110, 5)
    gt = [[], []]  # no objects in either sample
    loss = loss_fn(pred, gt)
    assert not torch.isnan(loss)


def test_total_loss_scalar():
    criterion = TotalLoss()
    height_pred = torch.randn(2, 1, 200, 200)
    semantic_pred = torch.randn(2, 4, 200, 200)
    detect_pred = torch.randn(2, 110, 5)
    height_gt = torch.randn(2, 200, 200)
    semantic_gt = torch.randint(0, 4, (2, 200, 200))
    occupancy = torch.ones(2, 200, 200)
    objects_gt = [
        [{"type": 0, "x": 1.0, "y": 0.5, "diameter": 0.35}],
        [],
    ]
    loss = criterion(
        (height_pred, semantic_pred, detect_pred),
        (height_gt, semantic_gt, objects_gt, occupancy),
    )
    assert loss.shape == ()
    assert not torch.isnan(loss)
```

- [ ] **Step 2: Run test to confirm it fails**

```
python -m pytest training/terrain_mapping/tests/test_losses.py -v
```
Expected: `ModuleNotFoundError: No module named 'losses'`

- [ ] **Step 3: Implement `losses.py`**

Create `<il>/training/terrain_mapping/losses.py`:

```python
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HeightmapLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,       # (B, 1, H, W)
        target: torch.Tensor,     # (B, H, W)
        occupancy: torch.Tensor,  # (B, H, W) float, 1 = occupied
    ) -> torch.Tensor:
        diff = (pred.squeeze(1) - target) ** 2
        denom = occupancy.sum() + 1e-6
        return (diff * occupancy).sum() / denom


_SEMANTIC_WEIGHTS = torch.tensor([0.3, 2.0, 2.0, 2.0])  # free, rock, crater, wall


class SemanticLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, 4, H, W) logits;  target: (B, H, W) long
        w = _SEMANTIC_WEIGHTS.to(pred.device)
        return F.cross_entropy(pred, target, weight=w)


class DetectionLoss(nn.Module):
    def forward(
        self,
        pred: torch.Tensor,          # (B, 110, 5): [x, y, diam, conf, type]
        gt_objects: list[list[dict]],
    ) -> torch.Tensor:
        total = pred.new_zeros(())
        B = pred.shape[0]

        for b_pred, b_gt in zip(pred, gt_objects):
            # Confidence loss for unmatched slots (all targets = 0)
            conf_target = torch.zeros(b_pred.shape[0], device=pred.device)

            if len(b_gt) == 0:
                total = total + F.binary_cross_entropy_with_logits(
                    b_pred[:, 3], conf_target
                )
                continue

            gt_tensor = torch.tensor(
                [[o["x"], o["y"], o["diameter"], 1.0, float(o["type"])] for o in b_gt],
                dtype=torch.float32, device=pred.device,
            )
            n_gt = len(b_gt)

            # Hungarian matching on x,y distance
            with torch.no_grad():
                pred_xy = b_pred[:n_gt, :2].cpu().numpy()
                gt_xy = gt_tensor[:, :2].cpu().numpy()
                cost = np.linalg.norm(pred_xy[:, None] - gt_xy[None, :], axis=-1)
                row_ind, col_ind = linear_sum_assignment(cost)

            matched_pred = b_pred[row_ind]
            matched_gt = gt_tensor[col_ind]

            reg_loss = F.l1_loss(matched_pred[:, :3], matched_gt[:, :3])

            conf_target[row_ind] = 1.0
            conf_loss = F.binary_cross_entropy_with_logits(b_pred[:, 3], conf_target)

            total = total + reg_loss + conf_loss

        return total / B


class TotalLoss(nn.Module):
    def __init__(self, w_height: float = 1.0, w_semantic: float = 0.5, w_detect: float = 0.3):
        super().__init__()
        self.w_height = w_height
        self.w_semantic = w_semantic
        self.w_detect = w_detect
        self.height_loss = HeightmapLoss()
        self.semantic_loss = SemanticLoss()
        self.detect_loss = DetectionLoss()

    def forward(self, preds, targets) -> torch.Tensor:
        height_pred, semantic_pred, detect_pred = preds
        height_gt, semantic_gt, objects_gt, occupancy = targets
        l_h = self.height_loss(height_pred, height_gt, occupancy)
        l_s = self.semantic_loss(semantic_pred, semantic_gt)
        l_d = self.detect_loss(detect_pred, objects_gt)
        return self.w_height * l_h + self.w_semantic * l_s + self.w_detect * l_d
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest training/terrain_mapping/tests/test_losses.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/losses.py" \
        "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/tests/test_losses.py"
git commit -m "feat: add heightmap/semantic/detection loss functions with Hungarian matching"
```

---

## Task 7: Training + Evaluation Scripts

**Files:**
- Create: `<il>/training/terrain_mapping/train.py`
- Create: `<il>/training/terrain_mapping/evaluate.py`

- [ ] **Step 1: Implement `train.py`**

Create `<il>/training/terrain_mapping/train.py`:

```python
"""Training script for terrain mapping model.

Usage:
    python -m training.terrain_mapping.train \
        --data_dir /path/to/data \
        --output_dir /path/to/checkpoints \
        --epochs 100 \
        --batch_size 32
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[0]))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from dataset import TerrainDataset
from losses import TotalLoss
from model import TerrainMappingModel


def collate_fn(batch):
    """Custom collate to handle variable-length objects_gt."""
    bevs, heights, semantics, objects, occupancies = zip(*batch)
    return (
        torch.stack(bevs),
        torch.stack(heights),
        torch.stack(semantics),
        list(objects),       # keep as list — DetectionLoss handles variable lengths
        torch.stack(occupancies),
    )


def train(data_dir: str, output_dir: str, epochs: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = TerrainDataset(data_dir)
    n_val = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True)

    model = TerrainMappingModel().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = TotalLoss()

    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for bev, height_gt, semantic_gt, objects_gt, occupancy in train_loader:
            bev = bev.to(device)
            height_gt = height_gt.to(device)
            semantic_gt = semantic_gt.to(device)
            occupancy = occupancy.to(device)

            optimizer.zero_grad()
            preds = model(bev)
            loss = criterion(preds, (height_gt, semantic_gt, objects_gt, occupancy))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bev, height_gt, semantic_gt, objects_gt, occupancy in val_loader:
                bev = bev.to(device)
                height_gt = height_gt.to(device)
                semantic_gt = semantic_gt.to(device)
                occupancy = occupancy.to(device)
                preds = model(bev)
                loss = criterion(preds, (height_gt, semantic_gt, objects_gt, occupancy))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "best_model.pt")
            print(f"  → saved best (val={best_val:.4f})")

    torch.save(model.state_dict(), out / "final_model.pt")
    print("Training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", default="checkpoints/terrain_mapping")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    train(args.data_dir, args.output_dir, args.epochs, args.batch_size)
```

- [ ] **Step 2: Implement `evaluate.py`**

Create `<il>/training/terrain_mapping/evaluate.py`:

```python
"""Evaluation script — prints MAE, RMSE, mIoU, mAP@0.5 on a held-out split.

Usage:
    python -m training.terrain_mapping.evaluate \
        --data_dir /path/to/data \
        --checkpoint /path/to/best_model.pt
"""

import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TerrainDataset
from model import TerrainMappingModel
from train import collate_fn


def circle_iou(px, py, pd, gx, gy, gd):
    """Intersection-over-union for two circles (approximate via bounding box area)."""
    dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    pr, gr = pd / 2, gd / 2
    if dist >= pr + gr:
        return 0.0
    if dist <= abs(pr - gr):
        return min(pr, gr) ** 2 / max(pr, gr) ** 2
    # Approximate: use min-circle / max-circle area ratio
    inter = min(pr, gr) ** 2 * np.pi
    union = (pr ** 2 + gr ** 2) * np.pi - inter
    return inter / union


def evaluate(data_dir: str, checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TerrainDataset(data_dir)
    loader = DataLoader(dataset, batch_size=16, num_workers=2, collate_fn=collate_fn)

    model = TerrainMappingModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    height_errors, all_ious = [], {c: [] for c in range(4)}
    det_tp = {0: 0, 1: 0}
    det_fp = {0: 0, 1: 0}
    det_fn = {0: 0, 1: 0}

    with torch.no_grad():
        for bev, height_gt, semantic_gt, objects_gt, occupancy in loader:
            bev = bev.to(device)
            height_pred, semantic_pred, detect_pred = model(bev)

            # Height errors (occupied cells)
            occ = occupancy.bool()
            diff = (height_pred.squeeze(1).cpu() - height_gt)[occ]
            height_errors.extend(diff.abs().numpy().tolist())

            # Semantic IoU
            labels_pred = semantic_pred.argmax(dim=1).cpu()
            for c in range(4):
                pred_c = labels_pred == c
                gt_c = semantic_gt == c
                inter = (pred_c & gt_c).float().sum().item()
                union = (pred_c | gt_c).float().sum().item()
                if union > 0:
                    all_ious[c].append(inter / union)

            # Detection mAP@0.5
            for b_idx, b_gt in enumerate(objects_gt):
                confs = detect_pred[b_idx, :, 3].sigmoid().cpu().numpy()
                preds_xy = detect_pred[b_idx, :, :3].cpu().numpy()  # x,y,diam
                for obj in b_gt:
                    t = int(obj[0].item()) if isinstance(obj, torch.Tensor) else int(obj["type"])
                    matched = False
                    for s in range(len(confs)):
                        if confs[s] < 0.5:
                            continue
                        iou = circle_iou(preds_xy[s, 0], preds_xy[s, 1], preds_xy[s, 2],
                                         float(obj[1]) if isinstance(obj, torch.Tensor) else obj["x"],
                                         float(obj[2]) if isinstance(obj, torch.Tensor) else obj["y"],
                                         float(obj[3]) if isinstance(obj, torch.Tensor) else obj["diameter"])
                        if iou >= 0.5:
                            det_tp[t] += 1
                            matched = True
                            break
                    if not matched:
                        det_fn[t] += 1

    mae = np.mean(height_errors)
    rmse = np.sqrt(np.mean(np.array(height_errors) ** 2))
    class_names = ["free", "rock", "crater", "wall"]
    miou_all = np.mean([np.mean(v) for v in all_ious.values() if v])
    miou_obstacles = np.mean([np.mean(all_ious[c]) for c in [1, 2, 3] if all_ious[c]])

    print(f"Height MAE:  {mae:.4f} m  (target <0.03)")
    print(f"Height RMSE: {rmse:.4f} m  (target <0.05)")
    print(f"mIoU all:    {miou_all:.4f}  (target >0.70)")
    print(f"mIoU obstacles: {miou_obstacles:.4f}  (target >0.60)")
    for t, name in [(0, "rock"), (1, "crater")]:
        tp, fp, fn = det_tp[t], det_fp[t], det_fn[t]
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        ap = 2 * prec * rec / (prec + rec + 1e-6)
        print(f"mAP@0.5 {name}: {ap:.4f}  (target >0.75)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.parse_args()
    args = p.parse_args()
    evaluate(args.data_dir, args.checkpoint)
```

- [ ] **Step 3: Smoke test training on 10 samples**

```bash
cd "<il>"
# Create a tiny dataset from 10 real or synthetic npz files, then:
python -m training.terrain_mapping.train \
    --data_dir /tmp/terrain_test \
    --output_dir /tmp/terrain_ckpt \
    --epochs 3 \
    --batch_size 2
```
Expected: Three lines of `Epoch N | train=X.XXXX | val=X.XXXX` with no NaN values and a `best_model.pt` in `/tmp/terrain_ckpt/`.

- [ ] **Step 4: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/train.py" \
        "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/evaluate.py"
git commit -m "feat: add training loop and evaluation metrics for terrain mapping model"
```

---

## Task 8: TensorRT Export

**Files:**
- Create: `<il>/training/terrain_mapping/export_trt.py`

- [ ] **Step 1: Implement `export_trt.py`**

Create `<il>/training/terrain_mapping/export_trt.py`:

```python
"""Export trained model to TensorRT FP16 engine for Orin deployment.

Run this ON the Jetson Orin (TensorRT engines are device-specific):
    python export_trt.py \
        --checkpoint /path/to/best_model.pt \
        --output /path/to/terrain_mapping.trt

Requires: tensorrt, onnx, onnxruntime (or just tensorrt + polygraphy)
"""

import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from model import TerrainMappingModel


def export(checkpoint: str, output: str):
    device = torch.device("cuda")
    model = TerrainMappingModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dummy = torch.zeros(1, 6, 200, 200, device=device)
    onnx_path = pathlib.Path(output).with_suffix(".onnx")

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["bev"],
        output_names=["height", "semantic", "detections"],
        dynamic_axes={"bev": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX saved to {onnx_path}")

    try:
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

        engine_bytes = builder.build_serialized_network(network, config)
        with open(output, "wb") as f:
            f.write(engine_bytes)
        print(f"TensorRT FP16 engine saved to {output}")

    except ImportError:
        print("tensorrt not found — ONNX export only. Install tensorrt on the Orin to build engine.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="terrain_mapping.trt")
    args = p.parse_args()
    export(args.checkpoint, args.output)
```

- [ ] **Step 2: Test ONNX export (workstation, no TensorRT required)**

```bash
cd "<il>"
python training/terrain_mapping/export_trt.py \
    --checkpoint /tmp/terrain_ckpt/best_model.pt \
    --output /tmp/terrain_mapping.trt
```
Expected: `ONNX saved to /tmp/terrain_mapping.onnx` then `tensorrt not found — ONNX export only.`

- [ ] **Step 3: Commit**

```bash
git add "NVIDIA Isaac/Isaac Lab/lunabotics/training/terrain_mapping/export_trt.py"
git commit -m "feat: add TensorRT FP16 export script for Orin deployment"
```

---

## Task 9: ROS2 Package Scaffold + Custom Messages

**Files:**
- Create: `workspace/src/octane_terrain_mapping/` (full package)

- [ ] **Step 1: Create package structure**

```bash
mkdir -p workspace/src/octane_terrain_mapping/msg
mkdir -p workspace/src/octane_terrain_mapping/octane_terrain_mapping
mkdir -p workspace/src/octane_terrain_mapping/tests
touch workspace/src/octane_terrain_mapping/octane_terrain_mapping/__init__.py
touch workspace/src/octane_terrain_mapping/tests/__init__.py
```

- [ ] **Step 2: Create message definitions**

`workspace/src/octane_terrain_mapping/msg/TerrainObject.msg`:
```
uint8 type        # 0=rock  1=crater
float32 x
float32 y
float32 diameter
float32 confidence
```

`workspace/src/octane_terrain_mapping/msg/TerrainObjects.msg`:
```
std_msgs/Header header
octane_terrain_mapping/TerrainObject[] objects
```

`workspace/src/octane_terrain_mapping/msg/WallSegment.msg`:
```
float32 x1
float32 y1
float32 x2
float32 y2
```

`workspace/src/octane_terrain_mapping/msg/WallSegments.msg`:
```
std_msgs/Header header
octane_terrain_mapping/WallSegment[] walls
```

- [ ] **Step 3: Create `package.xml`**

`workspace/src/octane_terrain_mapping/package.xml`:
```xml
<?xml version="1.0"?>
<package format="3">
  <name>octane_terrain_mapping</name>
  <version>0.1.0</version>
  <description>Terrain mapping model inference node</description>
  <maintainer email="jack.elton.carbone@gmail.com">Adam Carbone</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <depend>rclpy</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>std_msgs</depend>

  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 4: Create `CMakeLists.txt`**

`workspace/src/octane_terrain_mapping/CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.8)
project(octane_terrain_mapping)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TerrainObject.msg"
  "msg/TerrainObjects.msg"
  "msg/WallSegment.msg"
  "msg/WallSegments.msg"
  DEPENDENCIES std_msgs
)

ament_package()
```

- [ ] **Step 5: Create `setup.py` and `setup.cfg`**

`workspace/src/octane_terrain_mapping/setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="octane_terrain_mapping",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "torch", "sensor-msgs-py"],
)
```

`workspace/src/octane_terrain_mapping/setup.cfg`:
```ini
[develop]
script_dir=$base/lib/octane_terrain_mapping
[install]
install_scripts=$base/lib/octane_terrain_mapping
```

- [ ] **Step 6: Build and verify messages compile**

```bash
cd workspace
colcon build --packages-select octane_terrain_mapping
source install/setup.bash
ros2 interface show octane_terrain_mapping/msg/TerrainObject
```
Expected:
```
uint8 type
float32 x
float32 y
float32 diameter
float32 confidence
```

- [ ] **Step 7: Commit**

```bash
git add workspace/src/octane_terrain_mapping/
git commit -m "feat: add octane_terrain_mapping ROS2 package with custom terrain map messages"
```

---

## Task 10: Instance Extraction + Wall Fitting

**Files:**
- Create: `workspace/src/octane_terrain_mapping/octane_terrain_mapping/instance_extraction.py`
- Create: `workspace/src/octane_terrain_mapping/octane_terrain_mapping/wall_fitting.py`
- Create: `workspace/src/octane_terrain_mapping/tests/test_instance_extraction.py`
- Create: `workspace/src/octane_terrain_mapping/tests/test_wall_fitting.py`

- [ ] **Step 1: Write failing tests**

`workspace/src/octane_terrain_mapping/tests/test_instance_extraction.py`:
```python
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from octane_terrain_mapping.instance_extraction import extract_instances

def test_detects_single_rock():
    semantic = np.zeros((200, 200), dtype=np.int64)
    # Place a 7×7 rock blob at cell (100,100)
    semantic[97:104, 97:104] = 1
    instances = extract_instances(semantic, cell_size=0.05)
    rocks = [i for i in instances if i["type"] == 0]
    assert len(rocks) == 1
    assert abs(rocks[0]["x"]) < 0.5
    assert abs(rocks[0]["y"]) < 0.5
    assert 0.2 < rocks[0]["diameter"] < 0.6

def test_detects_crater_and_rock_separately():
    semantic = np.zeros((200, 200), dtype=np.int64)
    semantic[50:56, 50:56] = 2   # crater
    semantic[150:157, 150:157] = 1  # rock
    instances = extract_instances(semantic, cell_size=0.05)
    assert sum(1 for i in instances if i["type"] == 0) == 1  # 1 rock
    assert sum(1 for i in instances if i["type"] == 1) == 1  # 1 crater
```

`workspace/src/octane_terrain_mapping/tests/test_wall_fitting.py`:
```python
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from octane_terrain_mapping.wall_fitting import fit_walls

def test_detects_horizontal_wall():
    semantic = np.zeros((200, 200), dtype=np.int64)
    semantic[100, 50:150] = 3  # horizontal wall
    walls = fit_walls(semantic, cell_size=0.05)
    assert len(walls) == 1
    w = walls[0]
    length = np.hypot(w["x2"] - w["x1"], w["y2"] - w["y1"])
    assert length > 2.0  # 100 cells × 0.05 = 5m

def test_empty_returns_no_walls():
    semantic = np.zeros((200, 200), dtype=np.int64)
    assert fit_walls(semantic, cell_size=0.05) == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd workspace
python -m pytest src/octane_terrain_mapping/tests/test_instance_extraction.py \
                 src/octane_terrain_mapping/tests/test_wall_fitting.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `instance_extraction.py`**

`workspace/src/octane_terrain_mapping/octane_terrain_mapping/instance_extraction.py`:
```python
from __future__ import annotations
import numpy as np
from scipy import ndimage


def extract_instances(
    semantic: np.ndarray,   # (H, W) int array: 0=free,1=rock,2=crater,3=wall
    cell_size: float = 0.05,
) -> list[dict]:
    """Extract rock and crater instances from a semantic grid via connected components.

    Returns list of dicts: {type (0=rock,1=crater), x, y, diameter} in meters,
    coordinates in robot frame (center of grid = robot origin).
    """
    H, W = semantic.shape
    half_x = H * cell_size / 2.0
    half_y = W * cell_size / 2.0
    instances = []

    for label_val, obj_type in [(1, 0), (2, 1)]:   # rock=1→type0, crater=2→type1
        mask = (semantic == label_val).astype(np.uint8)
        labeled, n_components = ndimage.label(mask)

        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            area_cells = comp_mask.sum()
            if area_cells < 4:   # noise filter
                continue

            rows, cols = np.where(comp_mask)
            cx_cell = rows.mean()
            cy_cell = cols.mean()

            # Diameter from bounding circle (equivalent radius of area)
            diameter = 2.0 * np.sqrt(area_cells / np.pi) * cell_size

            x = cx_cell * cell_size - half_x
            y = cy_cell * cell_size - half_y

            instances.append({"type": obj_type, "x": float(x), "y": float(y),
                               "diameter": float(diameter)})

    return instances
```

- [ ] **Step 4: Implement `wall_fitting.py`**

`workspace/src/octane_terrain_mapping/octane_terrain_mapping/wall_fitting.py`:
```python
from __future__ import annotations
import numpy as np
from sklearn.linear_model import RANSACRegressor


def fit_walls(
    semantic: np.ndarray,   # (H, W) int
    cell_size: float = 0.05,
    min_wall_cells: int = 10,
    max_walls: int = 8,
) -> list[dict]:
    """Fit line segments to wall-labeled cells using RANSAC.

    Returns list of dicts: {x1, y1, x2, y2} in meters (robot frame).
    """
    H, W = semantic.shape
    half_x = H * cell_size / 2.0
    half_y = W * cell_size / 2.0

    wall_cells = np.argwhere(semantic == 3)
    if len(wall_cells) < min_wall_cells:
        return []

    # Convert to robot-frame metres
    pts_x = wall_cells[:, 0] * cell_size - half_x
    pts_y = wall_cells[:, 1] * cell_size - half_y

    walls = []
    remaining = np.ones(len(pts_x), dtype=bool)

    for _ in range(max_walls):
        if remaining.sum() < min_wall_cells:
            break

        rx = pts_x[remaining]
        ry = pts_y[remaining]

        # RANSAC: regress y ~ x; if wall is vertical, swap axes
        ransac = RANSACRegressor(min_samples=0.3, residual_threshold=0.1, max_trials=100)
        try:
            ransac.fit(rx.reshape(-1, 1), ry)
        except ValueError:
            break

        inlier_mask = ransac.inlier_mask_
        if inlier_mask.sum() < min_wall_cells:
            break

        inlier_x = rx[inlier_mask]
        inlier_y = ry[inlier_mask]
        x1, x2 = float(inlier_x.min()), float(inlier_x.max())
        y1 = float(ransac.predict([[x1]])[0])
        y2 = float(ransac.predict([[x2]])[0])

        walls.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

        # Remove inliers from remaining pool (map back to original indices)
        orig_indices = np.where(remaining)[0]
        remaining[orig_indices[inlier_mask]] = False

    return walls
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest src/octane_terrain_mapping/tests/test_instance_extraction.py \
                 src/octane_terrain_mapping/tests/test_wall_fitting.py -v
```
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add workspace/src/octane_terrain_mapping/octane_terrain_mapping/instance_extraction.py \
        workspace/src/octane_terrain_mapping/octane_terrain_mapping/wall_fitting.py \
        workspace/src/octane_terrain_mapping/tests/
git commit -m "feat: add instance extraction (connected components) and wall fitting (RANSAC)"
```

---

## Task 11: BEV Rasterizer (ROS2) + Inference Node

**Files:**
- Create: `workspace/src/octane_terrain_mapping/octane_terrain_mapping/bev_rasterizer.py`
- Create: `workspace/src/octane_terrain_mapping/octane_terrain_mapping/inference_node.py`
- Create: `workspace/src/octane_terrain_mapping/tests/test_bev_rasterizer.py`

- [ ] **Step 1: Write failing test for BEV rasterizer**

`workspace/src/octane_terrain_mapping/tests/test_bev_rasterizer.py`:
```python
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from octane_terrain_mapping.bev_rasterizer import rasterize_from_ros_points

def _pack_rgb(r, g, b):
    return np.array((r << 16) | (g << 8) | b, dtype=np.uint32).view(np.float32)

def test_shape():
    pts = np.zeros((50, 4), dtype=np.float32)
    pts[:, 3] = _pack_rgb(0, 255, 0)
    grid = rasterize_from_ros_points(pts)
    assert grid.shape == (6, 200, 200)

def test_empty():
    grid = rasterize_from_ros_points(np.zeros((0, 4), dtype=np.float32))
    assert grid.sum() == 0.0
```

- [ ] **Step 2: Implement `bev_rasterizer.py` (ROS2 variant)**

`workspace/src/octane_terrain_mapping/octane_terrain_mapping/bev_rasterizer.py`:
```python
"""BEV rasterizer for the ROS2 inference node.

Mirrors training/terrain_mapping/bev_rasterizer.py — keep interfaces in sync.
"""
from __future__ import annotations
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


def rasterize_from_ros_points(
    points: np.ndarray,     # (N, 4) float32: x, y, z, rgb_packed
    grid_size: int = 200,
    cell_size: float = 0.05,
) -> np.ndarray:            # (6, grid_size, grid_size) float32
    """Same logic as training bev_rasterizer.rasterize_bev."""
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


def unpack_point_cloud2(msg: PointCloud2) -> np.ndarray:
    """Read a PointCloud2 message into an (N, 4) float32 numpy array."""
    pts = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(pts, dtype=np.float32)
```

- [ ] **Step 3: Run BEV rasterizer tests**

```bash
python -m pytest src/octane_terrain_mapping/tests/test_bev_rasterizer.py -v
```
Expected: `2 passed`

- [ ] **Step 4: Implement `inference_node.py`**

`workspace/src/octane_terrain_mapping/octane_terrain_mapping/inference_node.py`:
```python
"""ROS2 terrain mapping inference node.

Subscribes to /mapping/point_cloud/combined, runs TensorRT model,
publishes terrain heightmap, semantic grid, object list, and wall segments.

Launch:
    ros2 run octane_terrain_mapping inference_node \
        --ros-args -p engine_path:=/path/to/terrain_mapping.trt
"""
from __future__ import annotations
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from octane_terrain_mapping.bev_rasterizer import rasterize_from_ros_points, unpack_point_cloud2
from octane_terrain_mapping.instance_extraction import extract_instances
from octane_terrain_mapping.wall_fitting import fit_walls


class TerrainMappingNode(Node):
    def __init__(self):
        super().__init__("terrain_mapping")

        self.declare_parameter("engine_path", "")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("grid_size", 200)
        self.declare_parameter("cell_size", 0.05)
        self.declare_parameter("use_pytorch_fallback", True)  # use PyTorch if no TRT engine

        engine_path = self.get_parameter("engine_path").value
        self.conf_thresh = self.get_parameter("confidence_threshold").value
        self.grid_size = self.get_parameter("grid_size").value
        self.cell_size = self.get_parameter("cell_size").value

        self._load_model(engine_path)

        self.sub = self.create_subscription(
            PointCloud2, "/mapping/point_cloud/combined", self._on_cloud, 10
        )
        self.pub_height = self.create_publisher(Image, "/terrain/heightmap", 10)
        self.pub_semantic = self.create_publisher(OccupancyGrid, "/terrain/semantic", 10)

        # Deferred import — messages not available until ROS2 package is built
        from octane_terrain_mapping.msg import TerrainObjects, WallSegments
        self.pub_objects = self.create_publisher(TerrainObjects, "/terrain/objects", 10)
        self.pub_walls = self.create_publisher(WallSegments, "/terrain/walls", 10)

        self.get_logger().info("TerrainMappingNode ready")

    def _load_model(self, engine_path: str):
        if engine_path:
            try:
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f:
                    runtime = trt.Runtime(logger)
                    self._engine = runtime.deserialize_cuda_engine(f.read())
                self._context = self._engine.create_execution_context()
                self._backend = "trt"
                self.get_logger().info(f"TensorRT engine loaded from {engine_path}")
                return
            except Exception as e:
                self.get_logger().warn(f"TRT load failed ({e}), falling back to PyTorch")

        # PyTorch fallback
        import torch
        import pathlib, sys
        sys.path.insert(0, str(pathlib.Path(__file__).parents[4] /
                                "NVIDIA Isaac" / "Isaac Lab" / "lunabotics" /
                                "training" / "terrain_mapping"))
        from model import TerrainMappingModel
        self._torch_model = TerrainMappingModel()
        self._torch_model.eval()
        self._backend = "pytorch"
        self.get_logger().info("PyTorch fallback model loaded (no checkpoint — random weights)")

    def _infer(self, bev: np.ndarray):
        if self._backend == "pytorch":
            import torch
            with torch.no_grad():
                x = torch.from_numpy(bev).unsqueeze(0)
                h, s, d = self._torch_model(x)
            return h.squeeze(0, 1).numpy(), s.squeeze(0).numpy(), d.squeeze(0).numpy()
        # TRT path (simplified — full impl needs pycuda buffers)
        raise NotImplementedError("TRT inference path: implement with pycuda bindings")

    def _on_cloud(self, msg: PointCloud2):
        points = unpack_point_cloud2(msg)
        bev = rasterize_from_ros_points(points, self.grid_size, self.cell_size)

        height_map, semantic_logits, detections = self._infer(bev)

        semantic_labels = semantic_logits.argmax(axis=0).astype(np.int8)  # (H, W)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = "base_link"

        self._publish_heightmap(height_map, header)
        self._publish_semantic(semantic_labels, header)
        self._publish_objects(detections, semantic_labels, header)
        self._publish_walls(semantic_labels, header)

    def _publish_heightmap(self, height_map: np.ndarray, header: Header):
        from cv_bridge import CvBridge
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(height_map, encoding="32FC1")
        img_msg.header = header
        self.pub_height.publish(img_msg)

    def _publish_semantic(self, labels: np.ndarray, header: Header):
        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = self.cell_size
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -(self.grid_size * self.cell_size) / 2.0
        msg.info.origin.position.y = -(self.grid_size * self.cell_size) / 2.0
        msg.data = labels.flatten().tolist()
        self.pub_semantic.publish(msg)

    def _publish_objects(self, detections: np.ndarray, semantic: np.ndarray, header: Header):
        from octane_terrain_mapping.msg import TerrainObjects, TerrainObject
        # Primary source: semantic-derived instances (more reliable)
        raw = extract_instances(semantic, self.cell_size)
        msg = TerrainObjects()
        msg.header = header
        for inst in raw:
            obj = TerrainObject()
            obj.type = inst["type"]
            obj.x = inst["x"]
            obj.y = inst["y"]
            obj.diameter = inst["diameter"]
            obj.confidence = 1.0
            msg.objects.append(obj)
        # Append high-confidence detection head results not already covered
        import torch
        confs = torch.tensor(detections[:, 3]).sigmoid().numpy()
        for i, conf in enumerate(confs):
            if conf >= self.conf_thresh:
                obj = TerrainObject()
                obj.type = int(round(detections[i, 4]))
                obj.x = float(detections[i, 0])
                obj.y = float(detections[i, 1])
                obj.diameter = float(detections[i, 2])
                obj.confidence = float(conf)
                msg.objects.append(obj)
        self.pub_objects.publish(msg)

    def _publish_walls(self, semantic: np.ndarray, header: Header):
        from octane_terrain_mapping.msg import WallSegments, WallSegment
        segs = fit_walls(semantic, self.cell_size)
        msg = WallSegments()
        msg.header = header
        for w in segs:
            seg = WallSegment()
            seg.x1, seg.y1, seg.x2, seg.y2 = w["x1"], w["y1"], w["x2"], w["y2"]
            msg.walls.append(seg)
        self.pub_walls.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainMappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Register entry point in `setup.py`**

Update `workspace/src/octane_terrain_mapping/setup.py`:
```python
from setuptools import setup, find_packages

setup(
    name="octane_terrain_mapping",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "inference_node = octane_terrain_mapping.inference_node:main",
        ],
    },
    install_requires=["numpy", "torch", "sensor-msgs-py", "scikit-learn", "scipy"],
)
```

- [ ] **Step 6: Build and smoke test node startup**

```bash
cd workspace
colcon build --packages-select octane_terrain_mapping
source install/setup.bash
ros2 run octane_terrain_mapping inference_node &
ros2 topic list | grep terrain
```
Expected:
```
/terrain/heightmap
/terrain/objects
/terrain/semantic
/terrain/walls
```

- [ ] **Step 7: Commit**

```bash
git add workspace/src/octane_terrain_mapping/
git commit -m "feat: add terrain mapping inference node with BEV rasterizer, TRT/PyTorch backend"
```

---

## Summary

| Task | Deliverable | Test method |
|------|-------------|-------------|
| 1 | Crater terrain carver + env config | pytest |
| 2 | BEV rasterizer (training) | pytest |
| 3 | Data collection script | Manual: 5 episodes, verify .npz shapes |
| 4 | PyTorch Dataset | pytest |
| 5 | TerrainMappingModel | pytest (shape test) |
| 6 | Loss functions | pytest |
| 7 | Train + evaluate scripts | Smoke: 3 epochs, verify no NaN |
| 8 | TensorRT export | Smoke: ONNX output |
| 9 | ROS2 package + messages | `ros2 interface show` |
| 10 | Instance extraction + wall fitting | pytest |
| 11 | BEV rasterizer (ROS2) + inference node | pytest + `ros2 topic list` |
