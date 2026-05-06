import numpy as np
import pathlib, tempfile, sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from dataset import TerrainDataset
import torch


def _write_fake_npz(path: pathlib.Path, n_objects: int = 3):
    np.savez_compressed(
        path,
        bev=np.random.rand(6, 200, 200).astype(np.float32),
        height_gt=np.random.rand(200, 200).astype(np.float32),
        semantic_gt=np.random.randint(0, 4, (200, 200)).astype(np.int64),
        objects_gt=np.random.rand(n_objects, 4).astype(np.float32),
        walls_gt=np.random.rand(2, 4).astype(np.float32),
    )


def test_dataset_length():
    with tempfile.TemporaryDirectory() as d:
        for i in range(5):
            _write_fake_npz(pathlib.Path(d) / f"ep_{i:06d}.npz")
        assert len(TerrainDataset(d)) == 5


def test_dataset_item_shapes():
    with tempfile.TemporaryDirectory() as d:
        _write_fake_npz(pathlib.Path(d) / "ep_000000.npz", n_objects=4)
        ds = TerrainDataset(d)
        bev, height_gt, semantic_gt, objects_gt, occupancy = ds[0]
        assert bev.shape == (6, 200, 200)
        assert height_gt.shape == (200, 200)
        assert semantic_gt.shape == (200, 200)
        assert len(objects_gt) == 4
        assert set(objects_gt[0].keys()) == {"type", "x", "y", "diameter"}
        assert occupancy.shape == (200, 200)
        assert occupancy.dtype == torch.float32

def test_raises_on_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            TerrainDataset(d)
