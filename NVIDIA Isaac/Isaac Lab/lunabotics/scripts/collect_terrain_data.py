"""Collect terrain ground-truth data using the TerrainCollectionEnv.

Saves per-episode NPZ files containing only ground-truth terrain maps.
Camera frames are NOT saved — point clouds are generated separately by the
ROS perception stack (DA3 + Orbbec) and paired with these NPZ files at
training time.

NPZ contents per episode:
    height_gt   (200, 200) float32   robot-centric BEV heightmap (metres)
    semantic_gt (200, 200) uint8     0=free 1=rock 2=crater 3=wall
    objects_gt  (N, 4)    float32   [rx, ry, diameter, type] per object
    walls_gt    (M, 4)    float32   [rx1, ry1, rx2, ry2] per wall segment
    robot_yaw   (1,)      float32   radians (baked into BEV rotation)
    robot_pitch (1,)      float32   radians
    robot_roll  (1,)      float32   radians
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Terrain GT data collection.")
parser.add_argument("--task",     type=str, default="Template-TerrainCollection-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--gt_dir",   type=str, default=r"E:\terrain_data\gt")
parser.add_argument("--episodes", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gc
import pathlib
import numpy as np
import gymnasium as gym

import lunabotics.tasks  # noqa: F401 — registers all envs, also sets up CUDA DLL paths
import torch             # import torch AFTER isaaclab/lunabotics so CUDA paths are ready

from lunabotics.tasks.direct.terrain_mapping.terrain_collection_env_cfg import TerrainCollectionEnvCfg

# Minimum physics steps per episode so the robot settles onto terrain before
# GT is read.  GT is computed at reset time so this only matters for physics
# accuracy; 10 steps is enough for the robot to land.
_WARMUP_STEPS = 10


def main():
    gt_dir = pathlib.Path(args_cli.gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = TerrainCollectionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)

    zero_actions = torch.zeros(args_cli.num_envs, 2)

    existing = sorted(gt_dir.glob("ep_*_gt.npz"))
    start_ep = int(existing[-1].name[3:9]) + 1 if existing else 0
    if start_ep > 0:
        print(f"[TerrainCollect] Resuming from episode {start_ep} ({len(existing)} existing)")

    obs, _ = env.reset()

    for ep in range(start_ep, start_ep + args_cli.episodes):
        if ep > start_ep:
            obs, _ = env.reset()

        for _ in range(_WARMUP_STEPS):
            obs, _, terminated, truncated, _ = env.step(zero_actions)

        ep_id = f"ep_{ep:06d}"

        gt = env.unwrapped._current_gt
        if gt is None:
            print(f"[TerrainCollect] WARNING: no GT for episode {ep}, skipping")
            continue

        save_kwargs = dict(
            height_gt   = gt["height_gt"],
            semantic_gt = gt["semantic_gt"],
            objects_gt  = gt["objects_gt"],
            walls_gt    = gt["walls_gt"],
            robot_yaw   = np.array([gt["robot_yaw"]]),
            robot_pitch = np.array([gt.get("robot_pitch", 0.0)]),
            robot_roll  = np.array([gt.get("robot_roll",  0.0)]),
        )

        np.savez_compressed(gt_dir / f"{ep_id}_gt.npz", **save_kwargs)
        (gt_dir / f"{ep_id}.ready").touch()

        del save_kwargs
        if ep % 50 == 0:
            gc.collect()
            print(f"[TerrainCollect] Episode {ep} / {start_ep + args_cli.episodes - 1}")

    print("[TerrainCollect] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
