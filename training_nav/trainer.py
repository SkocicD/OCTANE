"""Core training loop for OCTANE navigation policy.

Public API:
    run_epoch()      — one forward/backward pass over a DataLoader
    run_training()   — full epoch loop with curriculum, logging, checkpointing
"""

import gc
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from training_nav import checkpoints, curriculum, dashboard
from training_nav.losses import compute_loss

_SEP = '  ' + '─' * 63

# Keys that can be overridden per curriculum stage
_WEIGHT_KEYS = [
    'speed_reg_weight',
    'proximity_reg_weight',
    'idle_reg_weight',
    'smooth_reg_weight',
    'diff_reg_weight',
    'forward_reg_weight',
    'stillness_reg_weight',
    'recovery_reg_weight',
]

_WEIGHT_DEFAULTS = {
    'speed_reg_weight':     0.05,
    'proximity_reg_weight': 0.03,
    'idle_reg_weight':      0.003,
    'smooth_reg_weight':    0.02,
    'diff_reg_weight':      0.01,
    'forward_reg_weight':   0.02,
    'stillness_reg_weight': 0.05,
    'recovery_reg_weight':  0.05,
}


def _stage_weights(stage: int, cc: dict, tc: dict) -> dict:
    """Return effective loss weights for the given curriculum stage.

    Starts from training-section defaults, then applies any per-stage
    overrides from curriculum.stage_weights.s{stage}.
    """
    base = {k: tc.get(k, _WEIGHT_DEFAULTS[k]) for k in _WEIGHT_KEYS}
    overrides = cc.get('stage_weights', {}).get(f's{stage}', {}) or {}
    base.update({k: v for k, v in overrides.items() if k in _WEIGHT_KEYS})
    return base


def run_epoch(loader, model, criterion, optimizer, device, *,
              train: bool, grad_clip: float,
              bucket_loss_weight: float,
              weights: dict,
              scaler: GradScaler | None = None):
    """One training or validation pass.

    Returns:
        (mean_loss, mean_pred_magnitude) averaged over all batches.
    """
    model.train(train)
    total_loss = 0.0
    total_pmag = 0.0
    desc = 'train' if train else 'val  '
    use_amp = scaler is not None and device.type == 'cuda'

    with torch.set_grad_enabled(train):
        bar = tqdm(loader, desc=desc, leave=False,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}  {postfix}]')

        for terrain, heading, zone_idx, arena_type, phase_idx, action_gt in bar:
            terrain    = terrain.to(device, non_blocking=True)
            heading    = heading.to(device, non_blocking=True)
            zone_idx   = zone_idx.to(device, non_blocking=True)
            arena_type = arena_type.to(device, non_blocking=True)
            phase_idx  = phase_idx.to(device, non_blocking=True)
            action_gt  = action_gt.to(device, non_blocking=True)

            with autocast('cuda', enabled=use_amp):
                action_pred_seq, _ = model(
                    terrain, heading, zone_idx, arena_type, phase_idx, hidden=None)
                motor_pred  = action_pred_seq[:, :, :2]   # (B, T, 2)
                bucket_pred = action_pred_seq[:, :, 2:]   # (B, T, 3)

                loss, pred_mag = compute_loss(
                    motor_pred, bucket_pred, action_gt, terrain, criterion,
                    bucket_loss_weight,
                    **weights)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            lv = loss.item()
            pm = pred_mag.mean().item()
            total_loss += lv
            total_pmag += pm
            bar.set_postfix_str(f'L={lv:.4f} mag={pm:.3f}')

    n = max(len(loader), 1)
    return total_loss / n, total_pmag / n


def run_training(cfg, model, optimizer, scheduler,
                 train_loader, val_loader, train_ds, val_ds,
                 device, max_epochs, start_epoch=1, best_val=float('inf'),
                 ckpt_dir='checkpoints', keep=10):
    """Run the full training loop.

    Handles:
      - Curriculum stage transitions with per-stage loss weight overrides
      - Per-epoch rich console logging
      - Live dashboard state updates
      - Checkpoint saving
      - Stale detection (non-stopping)
    """
    tc = cfg['training']
    cc = cfg.get('curriculum', {})

    criterion          = nn.HuberLoss(delta=0.1)
    bucket_loss_weight = tc.get('bucket_loss_weight', 0.5)
    display_scale      = tc.get('loss_display_scale', 100.0)

    scaler  = GradScaler('cuda') if device.type == 'cuda' else None
    floor   = curriculum.floor_epoch(cc)
    patience = max(tc['early_stop_patience'], max_epochs // 50)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    train_mags:   list[float] = []
    val_mags:     list[float] = []
    lrs:          list[float] = []

    no_improve = 0
    prev_stage = -1
    prev_val   = float('inf')
    t0         = time.time()
    weights    = _stage_weights(0, cc, tc)   # initialise for stage 0

    for epoch in range(start_epoch, max_epochs + 1):
        train_ds.reshuffle(epoch)
        val_ds.epoch = epoch

        cur_stage  = curriculum.get_stage(epoch, cc)
        stage_name = curriculum.STAGE_NAMES[cur_stage]
        next_ep    = curriculum.next_stage_epoch(cur_stage, cc)
        next_label = f'→ stage {cur_stage + 1} at epoch {next_ep}' if next_ep else 'final stage'

        if cur_stage != prev_stage:
            weights = _stage_weights(cur_stage, cc, tc)
            if prev_stage >= 0:
                print(f'\n  ┌─ Curriculum advance: stage {prev_stage} → {cur_stage} ──────────────────────')
                print(f'  │  Now training : {stage_name}')
                print(f'  │  Loss weights : '
                      f'spd={weights["speed_reg_weight"]:.3f}  '
                      f'prx={weights["proximity_reg_weight"]:.3f}  '
                      f'smt={weights["smooth_reg_weight"]:.3f}  '
                      f'dif={weights["diff_reg_weight"]:.3f}  '
                      f'fwd={weights["forward_reg_weight"]:.3f}  '
                      f'stn={weights["stillness_reg_weight"]:.3f}  '
                      f'rec={weights["recovery_reg_weight"]:.3f}')
                print(f'  └──────────────────────────────────────────────────────────────────')
                no_improve = 0
            prev_stage = cur_stage

        train_loss, train_pmag = run_epoch(
            train_loader, model, criterion, optimizer, device,
            train=True, grad_clip=tc['grad_clip'],
            bucket_loss_weight=bucket_loss_weight,
            weights=weights, scaler=scaler)

        val_loss, val_pmag = run_epoch(
            val_loader, model, criterion, optimizer, device,
            train=False, grad_clip=0,
            bucket_loss_weight=bucket_loss_weight,
            weights=weights)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mags.append(train_pmag)
        val_mags.append(val_pmag)
        lrs.append(lr)

        delta_val = val_loss - prev_val
        improved  = val_loss < best_val
        if improved:
            best_val   = val_loss
            no_improve = 0
        else:
            no_improve += 1
        prev_val = val_loss

        elapsed = time.time() - t0
        _print_epoch(epoch, max_epochs, cur_stage, stage_name, next_label,
                     train_loss, val_loss, delta_val, train_pmag, val_pmag,
                     best_val, no_improve, patience, lr, elapsed, improved,
                     display_scale=display_scale)

        dashboard.update(
            epoch=epoch, epochs=max_epochs,
            train_loss=train_losses, val_loss=val_losses,
            train_mag=train_mags, val_mag=val_mags,
            lr=lrs, best_val=best_val,
            no_improve=no_improve, patience=patience,
            elapsed_s=round(elapsed), status='running',
            stage=cur_stage, stage_name=stage_name,
        )

        checkpoints.save(ckpt_dir, epoch, model, optimizer, scheduler,
                         val_loss, best_val, keep=keep)

        if epoch >= floor and no_improve >= patience:
            print(f'  [ stale: no improvement for {no_improve} epochs — continuing ]')

        gc.collect()

    dashboard.update(status='done')
    print(f'\n  Done.  Best val loss: {best_val:.4f}\n')


def _print_epoch(epoch, max_epochs, stage, stage_name, next_label,
                 train_loss, val_loss, delta_val, train_pmag, val_pmag,
                 best_val, no_improve, patience, lr, elapsed, improved,
                 display_scale: float = 100.0):
    def _t(s):
        if s < 60:    return f'{int(s)}s'
        if s < 3600:  return f'{int(s)//60}m {int(s)%60:02d}s'
        return f'{int(s)//3600}h {(int(s)%3600)//60:02d}m'

    sc     = display_scale
    arrow  = '↓' if delta_val < 0 else ('↑' if delta_val > 0 else '─')
    marker = '  * new best' if improved else ''
    slabel = f'(×{int(sc)})' if sc != 1.0 else ''

    print(_SEP)
    print(f'  Epoch {epoch:05d} / {max_epochs}   │   Stage {stage} — {stage_name}  ({next_label})')
    print(_SEP)
    print(f'  train mean : {train_loss*sc:.2f} {slabel}   val mean   : {val_loss*sc:.2f} {arrow}   '
          f'Δval  : {delta_val*sc:+.2f}{marker}')
    print(f'  best val   : {best_val*sc:.2f}        no-improve : {no_improve} / {patience}    '
          f'lr    : {lr:.2e}')
    print(f'  train mag  : {train_pmag:.3f}          val mag    : {val_pmag:.3f}        '
          f'time  : {_t(elapsed)}')
