import os, sys, json, tempfile
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.train import compute_loss, create_splits
from training.model import TerrainModel
from training.dataset import TerrainDataset
from training.tests.test_dataset import _make_fake_episode


def _fake_preds(B=2):
    return {
        'height':  torch.randn(B, 200, 200),
        'rocks':   torch.sigmoid(torch.randn(B, 200, 200)),
        'craters': torch.sigmoid(torch.randn(B, 200, 200)),
        'walls':   torch.sigmoid(torch.randn(B, 200, 200)),
    }


def _fake_targets(B=2):
    return {
        'height':  torch.randn(B, 200, 200),
        'rocks':   torch.rand(B, 200, 200),
        'craters': torch.rand(B, 200, 200),
        'walls':   (torch.rand(B, 200, 200) > 0.8).float(),
    }


def test_compute_loss_returns_scalar():
    loss = compute_loss(_fake_preds(), _fake_targets())
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_compute_loss_decreases_on_perfect_preds():
    targets = _fake_targets()
    perfect_preds = {k: v.clone() for k, v in targets.items()}
    loss_random  = compute_loss(_fake_preds(), targets)
    loss_perfect = compute_loss(perfect_preds, targets)
    assert loss_perfect.item() < loss_random.item()


def test_create_splits():
    with tempfile.TemporaryDirectory() as root:
        gt_dir = os.path.join(root, 'gt')
        os.makedirs(gt_dir)
        for i in range(10):
            ep_id = f'ep_{i:06d}'
            np.savez(os.path.join(gt_dir, f'{ep_id}_gt.npz'),
                     height_gt=np.zeros((200, 200), dtype=np.float32),
                     objects_gt=np.zeros((0, 4), dtype=np.float32),
                     walls_gt=np.zeros((0, 4), dtype=np.float32),
                     robot_yaw=np.float32(0.0),
                     robot_pitch=np.float32(0.0),
                     robot_roll=np.float32(0.0))
        splits_file = os.path.join(root, 'splits.json')
        train_ids, val_ids = create_splits(root, splits_file, val_ratio=0.2, seed=42)
        assert len(train_ids) == 8
        assert len(val_ids)   == 2
        assert set(train_ids) | set(val_ids) == {f'ep_{i:06d}' for i in range(10)}
        # Idempotent
        train2, val2 = create_splits(root, splits_file, val_ratio=0.2, seed=42)
        assert train2 == train_ids
        assert val2   == val_ids


def test_single_training_step_updates_weights():
    with tempfile.TemporaryDirectory() as root:
        ep_id = 'ep_000000'
        _make_fake_episode(root, ep_id)
        depth_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}
        ds = TerrainDataset(root, [ep_id], depth_stats=depth_stats)
        loader = torch.utils.data.DataLoader(ds, batch_size=1)

        model = TerrainModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        before = [p.clone() for p in model.parameters()]

        images, rotation, gt = next(iter(loader))
        preds = model(images, rotation)
        loss  = compute_loss(preds, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = any(not torch.equal(a, b)
                      for a, b in zip(before, model.parameters()))
        assert changed, "No parameters were updated"
