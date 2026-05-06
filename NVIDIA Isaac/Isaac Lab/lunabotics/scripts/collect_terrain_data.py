"""Collect terrain data using the TerrainCollectionEnv.

Saves per-episode NPZ files containing ground-truth terrain maps only.
Camera images are saved separately under <gt_dir>/../images/<serial>/ for
use by the ROS point-cloud generation pipeline.  No ROS or DDS required.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Terrain data collection.")
parser.add_argument("--task",     type=str, default="Template-TerrainCollection-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--gt_dir",   type=str, default=r"F:\terrain_data\gt")
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
            # Depth — encode as 16-bit PNG (millimetres, uint16, max 65.5 m).
            depth_m  = np.where(np.isfinite(arr), arr, 0.0)
            depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.int32)
            Image.fromarray(depth_mm, mode='I').save(folder / f"{ep_id}.png")
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

    # Setup cameras once after first reset — renderer is active at this point.
    obs, _ = env.reset()
    env.unwrapped._cam_capture.setup()

    _MAX_RETRIES = 3   # resets allowed per episode slot before giving up

    ep          = start_ep
    _need_reset = False   # first ep already reset above
    while ep < start_ep + args_cli.episodes:

        if _need_reset:
            obs, _ = env.reset()
        _need_reset = True

        # ── Adaptive warmup ──────────────────────────────────────────────────
        # Minimum 60 steps, then poll until all cameras return non-black frames
        # (up to 300 steps total).  The render pipeline varies per launch.
        for _ws in range(300):
            obs, _, terminated, truncated, _ = env.step(zero_actions)
            if _ws >= 59:
                _wf = getattr(env.unwrapped, "_last_frames", {})
                _wb = [s for s, a in _wf.items()
                       if (a.mean() < 3.0 if a.ndim == 3 else not np.any(a > 0.0))]
                if not _wb and _wf:
                    if _ws > 59:
                        print(f"[TerrainCollect] ep {ep}: cameras ready after {_ws + 1} warmup steps")
                    break

        gt = env.unwrapped._current_gt
        if gt is None:
            print(f"[TerrainCollect] WARNING: no GT for episode {ep}, skipping")
            ep += 1
            continue

        frames = getattr(env.unwrapped, "_last_frames", {})
        black  = [s for s, arr in frames.items()
                  if (arr.mean() < 3.0 if arr.ndim == 3 else not np.any(arr > 0.0))]

        # ── Black-camera guard ───────────────────────────────────────────────
        # If any camera is still black after full warmup, reset and try again.
        # After _MAX_RETRIES failures for this episode slot we give up and move on.
        if black:
            _need_reset = getattr(main, "_retry_count", 0) < _MAX_RETRIES - 1
            main._retry_count = getattr(main, "_retry_count", 0) + 1
            if main._retry_count < _MAX_RETRIES:
                print(f"[TerrainCollect] ep {ep}: black cameras {black} "
                      f"— retrying ({main._retry_count}/{_MAX_RETRIES})")
                continue
            print(f"[TerrainCollect] ep {ep}: black cameras {black} "
                  f"after {_MAX_RETRIES} retries — skipping episode")
            main._retry_count = 0
            ep += 1
            continue
        main._retry_count = 0

        # ── Save ─────────────────────────────────────────────────────────────
        ep_id = f"ep_{ep:06d}"
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
        _save_images(frames, img_dir, ep_id)
        (gt_dir / f"{ep_id}.ready").touch()

        del frames, save_kwargs
        if ep % 10 == 0:
            gc.collect()
            print(f"[TerrainCollect] Episode {ep}/{args_cli.episodes}"
                  f"  cameras={len(getattr(env.unwrapped, '_last_frames', {}))}")
        ep += 1

    print("[TerrainCollect] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
