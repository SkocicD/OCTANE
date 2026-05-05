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

import gc
import pathlib
import numpy as np
import gymnasium as gym
from PIL import Image

import lunabotics.tasks  # noqa: F401 — registers all envs, also sets up CUDA DLL paths
import torch             # import torch AFTER isaaclab/lunabotics so CUDA paths are ready

from lunabotics.tasks.direct.terrain_mapping.terrain_collection_env_cfg import TerrainCollectionEnvCfg


def _save_images(frames: dict, img_root: pathlib.Path, ep_id: str) -> None:
    for serial, arr in frames.items():
        folder = img_root / serial
        folder.mkdir(parents=True, exist_ok=True)
        if arr.ndim == 2:
            # Depth — clip sky/infinity, invert so near=bright, save as PNG
            depth = np.where(np.isfinite(arr), arr, 0.0)
            depth = np.clip(depth, 0.0, 10.0)   # 10 m max range
            vis = (255 - (depth / 10.0 * 255)).astype(np.uint8)  # near=white, far=black
            Image.fromarray(vis, mode="L").save(folder / f"{ep_id}.png")
        else:
            Image.fromarray(arr, mode="RGB").save(folder / f"{ep_id}.jpg", quality=85)


def main():
    gt_dir  = pathlib.Path(args_cli.gt_dir)
    img_dir = gt_dir.parent / "images"
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

    # Setup cameras after first reset (renderer is active).
    # Retry up to 3 times if any camera is still black after warm-up —
    # some render products need a fresh attach to initialise reliably.
    obs, _ = env.reset()
    cam = env.unwrapped._cam_capture
    for attempt in range(3):
        cam.destroy()
        cam.setup()
        for _ in range(60):
            obs, _, terminated, truncated, _ = env.step(zero_actions)
        frames = cam.capture()
        black = [s for s, arr in frames.items()
                 if (arr.mean() < 3.0 if arr.ndim == 3 else not np.any(arr > 0.0))]
        if not black:
            print(f"[TerrainCollect] All cameras ready (attempt {attempt + 1})")
            break
        print(f"[TerrainCollect] Black cameras after attempt {attempt + 1}: {black} — retrying setup...")
    else:
        print(f"[TerrainCollect] WARNING: cameras still black after 3 attempts: {black}")

    for ep in range(start_ep, start_ep + args_cli.episodes):
        if ep > start_ep:
            obs, _ = env.reset()

        # Warm up — let physics and cameras settle.
        for _ in range(60):
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
            robot_yaw   = np.array([gt["robot_yaw"]]),
            robot_pitch = np.array([gt.get("robot_pitch", 0.0)]),
            robot_roll  = np.array([gt.get("robot_roll",  0.0)]),
        )
        # Camera frames — present only if cameras were successfully attached
        for serial, arr in frames.items():
            save_kwargs[f"cam_{serial}"] = arr

        np.savez_compressed(gt_dir / f"{ep_id}_gt.npz", **save_kwargs)
        _save_images(frames, img_dir, ep_id)
        (gt_dir / f"{ep_id}.ready").touch()

        del frames, save_kwargs
        if ep % 10 == 0:
            gc.collect()
            print(f"[TerrainCollect] Episode {ep}/{args_cli.episodes}  cameras={len(getattr(env.unwrapped, '_last_frames', {}))}")

    print("[TerrainCollect] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
