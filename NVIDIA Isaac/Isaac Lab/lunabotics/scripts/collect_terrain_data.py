"""Collect terrain data using the TerrainCollectionEnv.

Saves per-episode NPZ files containing ground-truth terrain maps AND all 7
camera frames (5 RGB + Orbbec RGB + Orbbec depth).  No ROS or DDS required.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Terrain data collection.")
parser.add_argument("--task",     type=str, default="Template-TerrainCollection-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--gt_dir",   type=str, default=r"E:\terrain_data\gt")
parser.add_argument("--episodes", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pathlib
import numpy as np
import gymnasium as gym

import lunabotics.tasks  # noqa: F401 — registers all envs, also sets up CUDA DLL paths
import torch             # import torch AFTER isaaclab/lunabotics so CUDA paths are ready

from lunabotics.tasks.direct.terrain_mapping.terrain_collection_env_cfg import TerrainCollectionEnvCfg


def main():
    gt_dir = pathlib.Path(args_cli.gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = TerrainCollectionEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)

    zero_actions = torch.zeros(args_cli.num_envs, 2)

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()

        # Warm up — let physics and cameras settle
        for _ in range(30):
            obs, _, terminated, truncated, _ = env.step(zero_actions)

        ep_id = f"ep_{ep:06d}"

        gt = env.unwrapped._current_gt
        if gt is None:
            print(f"[TerrainCollect] WARNING: no GT for episode {ep}, skipping")
            continue

        frames = getattr(env.unwrapped, "_last_frames", {})

        save_kwargs = dict(
            height_gt   = gt["height_gt"],
            semantic_gt = gt["semantic_gt"],
            objects_gt  = gt["objects_gt"],
            walls_gt    = gt["walls_gt"],
            robot_pos   = gt["robot_pos"],
            robot_yaw   = np.array([gt["robot_yaw"]]),
        )
        # Camera frames — present only if cameras were successfully attached
        for serial, arr in frames.items():
            save_kwargs[f"cam_{serial}"] = arr

        np.savez_compressed(gt_dir / f"{ep_id}_gt.npz", **save_kwargs)
        (gt_dir / f"{ep_id}.ready").touch()

        if ep % 10 == 0:
            cam_count = len(frames)
            print(f"[TerrainCollect] Episode {ep}/{args_cli.episodes}  cameras={cam_count}")

    print("[TerrainCollect] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
