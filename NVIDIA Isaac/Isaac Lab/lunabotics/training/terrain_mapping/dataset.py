from __future__ import annotations
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


class TerrainDataset(Dataset):
    """Loads .npz files produced by collect_terrain_data.py.

    Returns per sample:
        bev          (6, 200, 200) float32 tensor
        height_gt    (200, 200) float32 tensor
        semantic_gt  (200, 200) int64 tensor  — 0=free,1=rock,2=crater,3=wall
        objects_gt   list of dicts {type, x, y, diameter}
        occupancy    (200, 200) float32 tensor — from bev channel 5
    """

    def __init__(self, data_dir: str | pathlib.Path):
        self.files = sorted(pathlib.Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx], allow_pickle=False)
        bev = torch.from_numpy(d["bev"])
        height_gt = torch.from_numpy(d["height_gt"])
        semantic_gt = torch.from_numpy(d["semantic_gt"].astype(np.int64))
        raw_objects = d["objects_gt"]
        objects_gt = [
            {"type": int(row[0]), "x": float(row[1]), "y": float(row[2]), "diameter": float(row[3])}
            for row in raw_objects
        ]
        occupancy = bev[5].clone()
        return bev, height_gt, semantic_gt, objects_gt, occupancy
