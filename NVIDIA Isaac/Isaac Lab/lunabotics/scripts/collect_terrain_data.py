"""Collect terrain data using the TerrainCollectionEnv.

Saves per-episode NPZ files containing ground-truth terrain maps only.
Camera images are saved separately under <gt_dir>/../images/<serial>/ for
use by the ROS point-cloud generation pipeline.  No ROS or DDS required.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Terrain data collection.")
parser.add_argument("--task",        type=str, default="Template-TerrainCollection-v0")
parser.add_argument("--num_envs",    type=int, default=1)
parser.add_argument("--gt_dir",      type=str, default=r"F:\terrain_data\gt")
parser.add_argument("--episodes",    type=int, default=100)
parser.add_argument("--ray_tracing", action="store_true", default=False,
                    help="Use RTX RayTracedLighting for camera captures")
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

if args_cli.ray_tracing:
    import carb.settings
    carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
    carb.settings.get_settings().set("/rtx/post/aa/op", 0)
    print("[TerrainCollect] RTX RaytracedLighting enabled")

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
    n_envs  = args_cli.num_envs
    gt_dir  = pathlib.Path(args_cli.gt_dir)
    img_dir = gt_dir.parent / "images"
    gt_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = TerrainCollectionEnvCfg()
    env_cfg.scene.num_envs = n_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)

    zero_actions = torch.zeros(n_envs, 2)

    existing  = sorted(gt_dir.glob("ep_*_gt.npz"))
    start_ep  = int(existing[-1].name[3:9]) + 1 if existing else 0
    if start_ep > 0:
        print(f"[TerrainCollect] Resuming from episode {start_ep} ({len(existing)} existing)")

    # Setup cameras once after first reset — renderer is active at this point.
    obs, _ = env.reset()
    env.unwrapped._cam_capture.setup()

    _MAX_RETRIES = 3

    ep          = start_ep
    _need_reset = False   # first ep already reset above
    while ep < start_ep + args_cli.episodes:

        if _need_reset:
            obs, _ = env.reset()
        _need_reset = True

        # ── Adaptive warmup ──────────────────────────────────────────────────
        # Minimum 60 steps, then poll until all cameras return non-black frames.
        for _ws in range(120):
            obs, _, terminated, truncated, _ = env.step(zero_actions)
            if _ws >= 59:
                _wf = getattr(env.unwrapped, "_last_frames", {})
                _any_black = False
                for _env_frames in _wf.values():
                    if any(a.mean() < 3.0 if a.ndim == 3 else not np.any(a > 0.0)
                           for a in _env_frames.values()):
                        _any_black = True
                        break
                if not _any_black and _wf:
                    if _ws > 59:
                        print(f"[TerrainCollect] ep {ep}: cameras ready after {_ws + 1} warmup steps")
                    break

        # ── Per-env frames and GT ────────────────────────────────────────────
        all_frames = getattr(env.unwrapped, "_last_frames", {})  # {env_idx: {serial: arr}}
        gts        = getattr(env.unwrapped, "_current_gts", [])  # list[dict|None]

        black_per_env: dict[int, list[str]] = {}
        for env_idx, env_frames in all_frames.items():
            black = [s for s, a in env_frames.items()
                     if (a.mean() < 3.0 if a.ndim == 3 else not np.any(a > 0.0))]
            if black:
                black_per_env[env_idx] = black

        # ── Black-camera guard ───────────────────────────────────────────────
        if black_per_env:
            main._retry_count = getattr(main, "_retry_count", 0) + 1
            if main._retry_count < _MAX_RETRIES:
                print(f"[TerrainCollect] ep {ep}: black cameras {black_per_env} "
                      f"— reinitializing ({main._retry_count}/{_MAX_RETRIES})")
                for env_idx, black in black_per_env.items():
                    for _serial in black:
                        env.unwrapped._cam_capture.reinitialize_camera(env_idx, _serial)
                for _ in range(40):
                    env.step(zero_actions)
                _need_reset = False
                continue
            print(f"[TerrainCollect] ep {ep}: black cameras {black_per_env} "
                  f"after {_MAX_RETRIES} reinit attempts — saving without them")
        main._retry_count = 0

        # ── Save one episode per env ─────────────────────────────────────────
        saved = 0
        for env_idx in range(n_envs):
            gt = gts[env_idx] if env_idx < len(gts) else None
            if gt is None:
                print(f"[TerrainCollect] WARNING: no GT for ep {ep + env_idx} (env {env_idx}), skipping")
                continue

            ep_id       = f"ep_{ep + env_idx:06d}"
            env_frames  = all_frames.get(env_idx, {})
            black       = black_per_env.get(env_idx, [])
            good_frames = {s: a for s, a in env_frames.items() if s not in black}

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
            _save_images(good_frames, img_dir, ep_id)
            (gt_dir / f"{ep_id}.ready").touch()
            saved += 1

        del all_frames, gts
        if ep % max(10, 10 * n_envs) == 0:
            gc.collect()
            print(f"[TerrainCollect] Episode {ep}/{start_ep + args_cli.episodes}"
                  f"  envs={n_envs}  saved_this_batch={saved}")
        ep += n_envs

    print("[TerrainCollect] Done.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
