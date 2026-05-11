"""Navigation policy training loop.

Usage:
  python training_nav/train.py
  python training_nav/train.py --config training_nav/config.yaml
  python training_nav/train.py --resume  # auto-resume from latest checkpoint
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training_nav.dataset import NavDataset
from training_nav.model import NavPolicy


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_latest_checkpoint(ckpt_dir: str) -> str | None:
    import glob
    paths = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))
    return paths[-1] if paths else None


def _save(ckpt_dir: str, epoch: int, model, optimizer, scheduler,
          val_loss: float, best_val: float):
    os.makedirs(ckpt_dir, exist_ok=True)
    data = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val': best_val,
    }
    torch.save(data, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt'))
    if val_loss <= best_val:
        torch.save(data, os.path.join(ckpt_dir, 'best.pt'))


def _run_epoch(loader: DataLoader, model: NavPolicy, criterion,
               optimizer, device: torch.device,
               train: bool, grad_clip: float) -> float:
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for terrain, heading, action_gt in loader:
            terrain   = terrain.to(device)
            heading   = heading.to(device)
            action_gt = action_gt.to(device)

            action_pred = model(terrain, heading)
            loss = criterion(action_pred, action_gt)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total += loss.item()

    return total / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training_nav/config.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    tc  = cfg['training']
    cc  = cfg['checkpoints']

    torch.manual_seed(tc['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[nav-train] device={device}')

    n_train = int(tc['arenas_per_epoch'] * tc['samples_per_arena'] * (1 - tc['val_ratio']))
    n_val   = int(tc['arenas_per_epoch'] * tc['samples_per_arena'] * tc['val_ratio'])

    train_ds = NavDataset(cfg, n_train, seed=tc['seed'])
    val_ds   = NavDataset(cfg, n_val,   seed=tc['seed'] + 1)

    train_loader = DataLoader(train_ds, batch_size=tc['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=tc['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True)

    model     = NavPolicy(cfg).to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tc['learning_rate'],
                                  weight_decay=tc['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc['epochs'], eta_min=tc['learning_rate'] * 0.01
    )

    start_epoch = 1
    best_val    = float('inf')
    no_improve  = 0

    if args.resume:
        ckpt_path = _find_latest_checkpoint(cc['dir'])
        if ckpt_path:
            print(f'[nav-train] Resuming from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch'] + 1
            best_val    = ckpt.get('best_val', float('inf'))
        else:
            print('[nav-train] No checkpoint found, starting fresh')

    print(f'[nav-train] {n_train} train / {n_val} val samples per epoch')

    for epoch in range(start_epoch, tc['epochs'] + 1):
        # Fresh arenas each epoch
        train_ds.reshuffle(epoch)
        val_ds.reshuffle(epoch + 10000)

        train_loss = _run_epoch(train_loader, model, criterion, optimizer,
                                device, train=True,  grad_clip=tc['grad_clip'])
        val_loss   = _run_epoch(val_loader,   model, criterion, optimizer,
                                device, train=False, grad_clip=0)
        scheduler.step()

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            no_improve = 0
            marker = ' *'
        else:
            no_improve += 1
            marker = ''

        lr = scheduler.get_last_lr()[0]
        print(f'[epoch {epoch:04d}/{tc["epochs"]}]  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={lr:.2e}{marker}')

        if epoch % cc['save_every'] == 0 or improved:
            _save(cc['dir'], epoch, model, optimizer, scheduler, val_loss, best_val)

        if no_improve >= tc['early_stop_patience']:
            print(f'[nav-train] Early stop — no val improvement for {no_improve} epochs')
            break

    print(f'[nav-train] Done. Best val loss: {best_val:.4f}')


if __name__ == '__main__':
    main()
