"""Data collection script for terrain mapping model training.

Run with: ./isaaclab.sh -p scripts/collect_terrain_data.py --output_dir /path/to/data --episodes 50000

Prerequisites:
  - Isaac Sim running with robot USD loaded (cameras + ROS2 OmniGraphs active)
  - ROS2 perception stack running (DA3 + point cloud combiner publishing /mapping/point_cloud/combined)
"""

import argparse
import pathlib
import time
import numpy as np

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
        z = diameter / 2.0
        prim_path = f"/World/rocks/rock_{i}"
        VisualSphere(prim_path=prim_path, radius=diameter / 2.0, position=np.array([x, y, z]))
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
    G = cfg.bev_grid_size
    C = cfg.bev_cell_size
    half = G * C / 2.0

    height_gt = np.zeros((G, G), dtype=np.float32)
    semantic_gt = np.zeros((G, G), dtype=np.int64)

    cos_a, sin_a = np.cos(-robot_yaw), np.sin(-robot_yaw)

    def world_to_robot(wx, wy):
        dx, dy = wx - robot_x, wy - robot_y
        return cos_a * dx - sin_a * dy, sin_a * dx + cos_a * dy

    # Craters (label=2)
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

    # Rocks (label=1)
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
                    if (cell_x - rx) ** 2 + (cell_y - ry) ** 2 <= radius ** 2:
                        semantic_gt[ii, jj] = 1

    # Walls (label=3)
    walls_gt = []
    for w in wall_records:
        rx1, ry1 = world_to_robot(w["x1"], w["y1"])
        rx2, ry2 = world_to_robot(w["x2"], w["y2"])
        walls_gt.append({"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2})
        steps = int(np.hypot(rx2 - rx1, ry2 - ry1) / C * 2) + 2
        for t in np.linspace(0, 1, steps):
            px, py = rx1 + t * (rx2 - rx1), ry1 + t * (ry2 - ry1)
            for dw in range(-1, 2):
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
        hf_base += rng.normal(0, cfg.regolith_noise_amplitude, hf_base.shape).astype(np.float32)
        hf, crater_records = carve_craters(hf_base, crater_cfg, cfg.arena_size, rng)

        rock_records = spawn_rocks(world, cfg, rng)
        wall_records = spawn_walls(world, cfg, rng)

        robot_x = float(rng.uniform(-cfg.arena_size[0] / 2 + cfg.spawn_margin,
                                     cfg.arena_size[0] / 2 - cfg.spawn_margin))
        robot_y = float(rng.uniform(-cfg.arena_size[1] / 2 + cfg.spawn_margin,
                                     cfg.arena_size[1] / 2 - cfg.spawn_margin))
        robot_yaw = float(rng.uniform(0, 2 * np.pi))

        for _ in range(30):
            world.step(render=True)
            rclpy.spin_once(listener, timeout_sec=0.0)

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
    cfg = TerrainMappingEnvCfg(output_dir=args.output_dir, episodes_per_run=args.episodes)
    run_collection(cfg, pathlib.Path(args.output_dir), args.episodes)
