import os
import json
import random
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.model import TerrainModel
from training.dataset import TerrainDataset, compute_depth_stats


def create_splits(data_root: str, splits_file: str,
                  val_ratio: float = 0.2, seed: int = 42):
    """Load splits from file, or create and save them if missing."""
    if os.path.exists(splits_file):
        with open(splits_file) as f:
            splits = json.load(f)
        return splits['train'], splits['val']

    gt_dir = os.path.join(data_root, 'gt')
    ep_ids = sorted(
        f.replace('_gt.npz', '')
        for f in os.listdir(gt_dir)
        if f.endswith('_gt.npz')
    )
    rng = random.Random(seed)
    rng.shuffle(ep_ids)
    n_val = max(1, int(len(ep_ids) * val_ratio))
    val_ids = ep_ids[:n_val]
    train_ids = ep_ids[n_val:]

    splits_dir = os.path.dirname(splits_file)
    if splits_dir:
        os.makedirs(splits_dir, exist_ok=True)
    with open(splits_file, 'w') as f:
        json.dump({'train': train_ids, 'val': val_ids}, f, indent=2)
    print(f"[train] Splits saved: {len(train_ids)} train / {len(val_ids)} val")
    return train_ids, val_ids


def compute_loss(preds: dict, targets: dict) -> torch.Tensor:
    height_loss  = F.l1_loss(preds['height'],  targets['height'])
    rocks_loss   = F.mse_loss(preds['rocks'],   targets['rocks'])
    craters_loss = F.mse_loss(preds['craters'], targets['craters'])
    walls_loss   = F.binary_cross_entropy(preds['walls'], targets['walls'])
    return height_loss + rocks_loss + craters_loss + walls_loss


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def train_epoch(model, loader, optimizer, device, grad_clip: float = 0.0):
    model.train()
    total = 0.0
    for images, rotation, gt in tqdm(loader, desc='train', leave=False):
        images   = _to_device(images, device)
        rotation = _to_device(rotation, device)
        gt       = _to_device(gt, device)
        preds = model(images, rotation)
        loss  = compute_loss(preds, gt)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, rotation, gt in tqdm(loader, desc='val', leave=False):
            images   = _to_device(images, device)
            rotation = _to_device(rotation, device)
            gt       = _to_device(gt, device)
            preds = model(images, rotation)
            total += compute_loss(preds, gt).item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    data_root   = cfg['data']['root']
    splits_file = cfg['data']['splits_file']
    stats_file  = cfg['data']['depth_stats_file']

    train_ids, val_ids = create_splits(
        data_root, splits_file,
        val_ratio=cfg['training']['val_ratio'],
        seed=cfg['training']['seed'],
    )

    if os.path.exists(stats_file):
        with open(stats_file) as f:
            depth_stats = json.load(f)
        print("[train] Loaded depth stats from cache")
    else:
        print("[train] Computing depth stats (first run)...")
        depth_stats = compute_depth_stats(data_root, train_ids)
        stats_dir = os.path.dirname(stats_file)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(depth_stats, f, indent=2)
        print(f"[train] Depth stats saved to {stats_file}")

    train_ds = TerrainDataset(data_root, train_ids, depth_stats, augment=True)
    val_ds   = TerrainDataset(data_root, val_ids,   depth_stats, augment=False)

    bs = cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = TerrainModel().to(device)
    warmup_epochs   = cfg['training'].get('warmup_epochs', 5)
    grad_clip       = cfg['training'].get('grad_clip', 1.0)
    patience        = cfg['training'].get('early_stop_patience', 15)
    max_epochs      = cfg['training']['epochs']

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )

    # Linear warmup for warmup_epochs, then cosine anneal for the remainder
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_epochs - warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    ckpt_dir   = cfg['checkpoints']['dir']
    save_every = cfg['checkpoints']['save_every']
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val        = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip)
        val_loss   = val_epoch(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"[epoch {epoch:03d}] train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': val_loss},
                       os.path.join(ckpt_dir, 'best.pt'))
            print(f"  -> best checkpoint saved (val={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[train] Early stopping at epoch {epoch} — no improvement for {patience} epochs")
                break

        if epoch % save_every == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict()},
                       os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'))


if __name__ == '__main__':
    main()
