import os
import json
import random
import argparse
import subprocess
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.model import TerrainModel
from training.dataset import TerrainDataset, compute_depth_stats


def create_splits(data_root: str, splits_file: str,
                  val_ratio: float = 0.2, seed: int = 42):
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
    n_val     = max(1, int(len(ep_ids) * val_ratio))
    val_ids   = ep_ids[:n_val]
    train_ids = ep_ids[n_val:]

    splits_dir = os.path.dirname(splits_file)
    if splits_dir:
        os.makedirs(splits_dir, exist_ok=True)
    with open(splits_file, 'w') as f:
        json.dump({'train': train_ids, 'val': val_ids}, f, indent=2)
    print(f"[train] Splits: {len(train_ids)} train / {len(val_ids)} val")
    return train_ids, val_ids


def compute_loss(preds: dict, targets: dict):
    height_loss  = F.l1_loss(preds['height'],  targets['height'])
    rocks_loss   = F.mse_loss(preds['rocks'],   targets['rocks'])
    craters_loss = F.mse_loss(preds['craters'], targets['craters'])
    walls_loss   = F.binary_cross_entropy(preds['walls'], targets['walls'])
    total = height_loss + rocks_loss + craters_loss + walls_loss
    return total, {
        'height':  height_loss.item(),
        'rocks':   rocks_loss.item(),
        'craters': craters_loss.item(),
        'walls':   walls_loss.item(),
    }


def find_batch_size(model, device, start_bs: int, gpu_margin: float = 0.20) -> int:
    if not torch.cuda.is_available():
        return start_bs

    total_mem  = torch.cuda.get_device_properties(device).total_memory
    target_max = total_mem * (1.0 - gpu_margin)
    bs         = start_bs

    print(f"[train] Auto batch size — GPU: {total_mem/1e9:.1f}GB, target ≤{(1-gpu_margin)*100:.0f}% usage")

    while bs >= 1:
        try:
            torch.cuda.empty_cache()
            model.zero_grad()
            dummy_img = torch.randn(bs, 12, 3, 224, 224, device=device)
            dummy_rot = torch.randn(bs, 6, device=device)
            dummy_gt  = {
                'height':  torch.randn(bs, 200, 200, device=device),
                'rocks':   torch.rand(bs, 200, 200, device=device),
                'craters': torch.rand(bs, 200, 200, device=device),
                'walls':   (torch.rand(bs, 200, 200, device=device) > 0.8).float(),
            }
            preds = model(dummy_img, dummy_rot)
            loss  = compute_loss(preds, dummy_gt)
            loss.backward()

            used = torch.cuda.memory_allocated(device)
            if used <= target_max:
                print(f"[train] Batch size {bs} — {used/1e9:.1f}GB / {total_mem/1e9:.1f}GB ({used/total_mem*100:.0f}%)")
                model.zero_grad()
                torch.cuda.empty_cache()
                return bs
            bs //= 2

        except torch.cuda.OutOfMemoryError:
            bs //= 2
            torch.cuda.empty_cache()

    return 1


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def train_epoch(model, loader, optimizer, device, writer: SummaryWriter,
                global_step: int, grad_clip: float = 0.0, log_every: int = 10):
    model.train()
    total      = 0.0
    head_sums  = {'height': 0.0, 'rocks': 0.0, 'craters': 0.0, 'walls': 0.0}
    running_sum, running_n = 0.0, 0

    bar = tqdm(loader, desc='train', leave=False,
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, loss={postfix}]')

    for images, rotation, gt in bar:
        images   = _to_device(images, device)
        rotation = _to_device(rotation, device)
        gt       = _to_device(gt, device)

        preds             = model(images, rotation)
        loss, head_losses = compute_loss(preds, gt)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_val     = loss.item()
        total       += loss_val
        running_sum += loss_val
        running_n   += 1
        global_step += 1

        for k, v in head_losses.items():
            head_sums[k] += v

        bar.set_postfix_str(f"{loss_val:.4f}")

        if global_step % log_every == 0:
            writer.add_scalar('Loss/train_running', running_sum / running_n, global_step)
            running_sum, running_n = 0.0, 0

    n = len(loader)
    return total / n, {k: v / n for k, v in head_sums.items()}, global_step


def val_epoch(model, loader, device):
    model.eval()
    total     = 0.0
    head_sums = {'height': 0.0, 'rocks': 0.0, 'craters': 0.0, 'walls': 0.0}
    with torch.no_grad():
        for images, rotation, gt in tqdm(loader, desc='val  ', leave=False):
            images   = _to_device(images, device)
            rotation = _to_device(rotation, device)
            gt       = _to_device(gt, device)
            preds             = model(images, rotation)
            loss, head_losses = compute_loss(preds, gt)
            total            += loss.item()
            for k, v in head_losses.items():
                head_sums[k] += v
    n = len(loader)
    return total / n, {k: v / n for k, v in head_sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/config.yaml')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs from config')
    parser.add_argument('--view',   action='store_true',
                        help='Auto-launch TensorBoard in browser')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    # Interactive epoch prompt if not passed as argument
    max_epochs = args.epochs
    if max_epochs is None:
        default_epochs = cfg['training']['epochs']
        raw = input(f"Epochs to train (press Enter for {default_epochs}): ").strip()
        max_epochs = int(raw) if raw else default_epochs

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

    warmup_epochs = cfg['training'].get('warmup_epochs', 5)
    grad_clip     = cfg['training'].get('grad_clip', 1.0)
    patience      = cfg['training'].get('early_stop_patience', 15)
    gpu_margin    = cfg['training'].get('gpu_memory_margin', 0.20)

    model = TerrainModel().to(device)

    bs = find_batch_size(model, device,
                         start_bs=cfg['training']['batch_size'],
                         gpu_margin=gpu_margin)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=pin)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )
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

    runs_dir = os.path.abspath('training/runs')
    writer   = SummaryWriter(log_dir=runs_dir)

    if args.view:
        subprocess.Popen(
            [sys.executable, '-m', 'tensorboard', '--logdir', runs_dir, '--bind_all'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[train] TensorBoard running — open http://localhost:6006")

    best_val          = float('inf')
    epochs_no_improve = 0
    global_step       = 0

    import time
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_heads, global_step = train_epoch(
            model, train_loader, optimizer, device,
            writer, global_step, grad_clip,
        )
        val_loss, val_heads = val_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]['lr']
        mem_gb = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0

        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val',         val_loss,   epoch)
        writer.add_scalar('LR',               lr_now,     epoch)
        for k in train_heads:
            writer.add_scalar(f'Heads/train_{k}', train_heads[k], epoch)
            writer.add_scalar(f'Heads/val_{k}',   val_heads[k],   epoch)
        writer.flush()

        flag = ' *' if val_loss < best_val else ''
        print(
            f"\n[epoch {epoch:03d}/{max_epochs}]  {elapsed/60:.1f}min  "
            f"gpu={mem_gb:.1f}GB  lr={lr_now:.2e}"
            f"\n  train  total={train_loss:.4f}  "
            f"height={train_heads['height']:.4f}  rocks={train_heads['rocks']:.4f}  "
            f"craters={train_heads['craters']:.4f}  walls={train_heads['walls']:.4f}"
            f"\n  val    total={val_loss:.4f}  "
            f"height={val_heads['height']:.4f}  rocks={val_heads['rocks']:.4f}  "
            f"craters={val_heads['craters']:.4f}  walls={val_heads['walls']:.4f}"
            f"{flag}"
        )

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': val_loss},
                       os.path.join(ckpt_dir, 'best.pt'))
            print(f"  -> saved best.pt (val={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[train] Early stopping — no improvement for {patience} epochs")
                break

        torch.save({'epoch': epoch, 'model': model.state_dict()},
                   os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'))

    writer.close()
    print("[train] Done.")


if __name__ == '__main__':
    main()
