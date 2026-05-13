import os
import gc
import json
import math
import random
import argparse
import subprocess
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.model import TerrainModel
from training.dataset import TerrainDataset, compute_depth_stats


# ── Splits ────────────────────────────────────────────────────────────────────

def create_splits(data_root, splits_file, val_ratio=0.2, seed=42):
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


# ── Loss ──────────────────────────────────────────────────────────────────────

_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        dtype=torch.float32).view(1, 1, 3, 3) / 8.0
_SOBEL_Y = _SOBEL_X.transpose(-1, -2).contiguous()


def _height_grad_loss(pred, target):
    kx = _SOBEL_X.to(pred.device)
    ky = _SOBEL_Y.to(pred.device)
    p, t = pred.unsqueeze(1), target.unsqueeze(1)
    return (F.l1_loss(F.conv2d(p, kx, padding=1), F.conv2d(t, kx, padding=1)) +
            F.l1_loss(F.conv2d(p, ky, padding=1), F.conv2d(t, ky, padding=1)))


def _log_transform(h: torch.Tensor) -> torch.Tensor:
    """Map height to log-scale: log(1 + |h|) * sign(h). Compresses extreme values."""
    return torch.log1p(h.abs()) * h.sign()


def _berhu_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Reverse Huber: L1 for small errors, L2 for large ones.
    Threshold c = 20% of max error in the batch (adaptive per-batch)."""
    diff = (pred - target).abs()
    c = 0.2 * diff.detach().max()
    return torch.where(diff <= c, diff, (diff ** 2 + c ** 2) / (2 * c)).mean()


class UncertaintyWeightedLoss(nn.Module):
    """Per-task learned uncertainty weights (Kendall et al. 2018).

    L_total = sum_i [ exp(-s_i) * L_i + s_i ]
    where s_i = log(sigma_i^2) are learned parameters.
    High sigma → low weight on that task (model is uncertain about it).
    Avoids hand-tuning fixed loss weights across tasks.
    """

    TASKS = ('height', 'rocks', 'craters', 'walls')

    def __init__(self):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in self.TASKS
        })

    def forward(self, preds, targets):
        # Log-transform: compresses extreme heights so craters don't overwhelm gradients
        pred_h = _log_transform(preds['height'])
        tgt_h  = _log_transform(targets['height'])

        # Element-wise BerHu: L1 for small errors, L2 for large ones
        diff = (pred_h - tgt_h).abs()
        c    = (0.2 * diff.detach().max()).clamp(min=1e-6)
        berhu_map = torch.where(diff <= c, diff, (diff ** 2 + c ** 2) / (2 * c))

        # Slope weighting: crater-edge pixels get up to 4x the loss signal
        kx    = _SOBEL_X.to(tgt_h.device)
        ky    = _SOBEL_Y.to(tgt_h.device)
        t_h   = tgt_h.unsqueeze(1)
        slope = (F.conv2d(t_h, kx, padding=1).abs() +
                 F.conv2d(t_h, ky, padding=1).abs()).squeeze(1).detach()
        height_main = (berhu_map * (1.0 + 3.0 * slope)).mean()
        height_grad = _height_grad_loss(pred_h, tgt_h)
        raw = {
            'height':  height_main + 0.5 * height_grad,
            'rocks':   F.mse_loss(preds['rocks'],   targets['rocks']),
            'craters': F.mse_loss(preds['craters'], targets['craters']),
            'walls':   F.binary_cross_entropy(preds['walls'], targets['walls']),
        }
        total = torch.zeros(1, device=preds['height'].device)
        head_losses = {}
        for task, loss in raw.items():
            lv    = self.log_vars[task].clamp(-3.0, 3.0)  # weight range ~[0.05, 20]
            total = total + torch.exp(-lv) * loss + lv
            head_losses[task] = loss.item()
        return total.squeeze(), head_losses

    def weights(self):
        return {t: float(torch.exp(-lv).item()) for t, lv in self.log_vars.items()}


# ── Curriculum ────────────────────────────────────────────────────────────────

class CurriculumPool:
    """Progressive episode pool expansion sorted by height variance.

    Starts with high-variance episodes (strongest height gradient signal),
    expands when height val loss plateaus, until the full training set is active.
    Early stopping is suppressed while the pool is still expanding.
    """

    def __init__(self, train_ids, data_root, cfg):
        self.all_ids           = list(train_ids)
        self._expand_by        = cfg.get('expand_by', 20)
        self._plateau_patience = cfg.get('plateau_patience', 8)
        self._min_improvement  = cfg.get('min_improvement', 0.02)
        self._max_epochs_per_pool = cfg.get('max_epochs_per_pool', 40)

        print("[curriculum] Computing height variances...", end='', flush=True)
        variances = {}
        for ep_id in train_ids:
            try:
                h = np.load(os.path.join(data_root, 'gt', f'{ep_id}_gt.npz'))['height_gt']
                variances[ep_id] = float(h.std())
            except Exception:
                variances[ep_id] = 0.0
        self._sorted_ids = sorted(train_ids, key=lambda e: variances.get(e, 0), reverse=True)
        vmin, vmax = min(variances.values()), max(variances.values())
        print(f" done. Variance range: {vmin:.4f}–{vmax:.4f}")

        start = cfg.get('start_size', 20)
        self._pool_size   = min(start, len(self.all_ids))
        self._no_improve  = 0
        self._epochs_in_pool = 0
        self._best_height = float('inf')
        print(f"[curriculum] Starting with {self._pool_size} / {len(self.all_ids)} episodes")

    @property
    def current_ids(self):
        return self._sorted_ids[:self._pool_size]

    @property
    def is_full(self):
        return self._pool_size >= len(self.all_ids)

    @property
    def pool_size(self):
        return self._pool_size

    def step(self, height_val_loss):
        """Call after each epoch with train height loss. Returns True if pool expanded."""
        if self.is_full:
            return False

        self._epochs_in_pool += 1

        if height_val_loss < self._best_height * (1.0 - self._min_improvement):
            self._best_height = height_val_loss
            self._no_improve  = 0
        else:
            self._no_improve += 1

        plateau = self._no_improve >= self._plateau_patience
        timeout = self._epochs_in_pool >= self._max_epochs_per_pool

        if plateau or timeout:
            reason = "plateau" if plateau else "timeout"
            old                  = self._pool_size
            self._pool_size      = min(self._pool_size + self._expand_by, len(self.all_ids))
            self._no_improve     = 0
            self._epochs_in_pool = 0
            self._best_height    = float('inf')
            print(f"[curriculum] Pool expanded ({reason}): {old} → {self._pool_size} / {len(self.all_ids)}")
            return True
        return False

    def state_dict(self):
        return {
            'pool_size':      self._pool_size,
            'best_height':    self._best_height,
            'no_improve':     self._no_improve,
            'epochs_in_pool': self._epochs_in_pool,
        }

    def load_state_dict(self, d):
        self._pool_size      = d['pool_size']
        self._epochs_in_pool = d.get('epochs_in_pool', 0)
        # Reset best_height so stale values from a prior loss function don't block expansion
        self._best_height = float('inf')
        self._no_improve  = 0


# ── DataLoader helpers ────────────────────────────────────────────────────────

def _make_train_loader(ids, data_root, depth_stats, bs, target_steps, pin):
    """Build a training DataLoader, repeating episode IDs to reach target_steps."""
    n_needed = target_steps * bs
    ids = list(ids)
    if len(ids) < n_needed:
        repeats = math.ceil(n_needed / len(ids))
        ids = (ids * repeats)[:n_needed]
    ds = TerrainDataset(data_root, ids, depth_stats, augment=True)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=pin)


def _find_latest_checkpoint(ckpt_dir):
    import glob
    paths = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))
    return paths[-1] if paths else None


def _safe_empty_cache():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ── Batch size finder ─────────────────────────────────────────────────────────

def find_batch_size(model, loss_fn, device, start_bs, gpu_margin=0.20):
    if not torch.cuda.is_available():
        return start_bs

    total_mem  = torch.cuda.get_device_properties(device).total_memory
    target_max = total_mem * (1.0 - gpu_margin)
    bs         = min(start_bs, 8)
    print(f"[train] Auto batch size — GPU: {total_mem/1e9:.1f}GB, target ≤{(1-gpu_margin)*100:.0f}% usage")

    while bs >= 1:
        try:
            _safe_empty_cache()
            model.zero_grad()
            dummy_img = torch.randn(bs, 12, 3, 224, 224, device=device)
            dummy_rot = torch.randn(bs, 6, device=device)
            dummy_gt  = {
                'height':  torch.randn(bs, 200, 200, device=device),
                'rocks':   torch.rand(bs, 200, 200, device=device),
                'craters': torch.rand(bs, 200, 200, device=device),
                'walls':   (torch.rand(bs, 200, 200, device=device) > 0.8).float(),
            }
            preds     = model(dummy_img, dummy_rot)
            loss, _   = loss_fn(preds, dummy_gt)
            loss.backward()

            used = torch.cuda.memory_allocated(device)
            if used <= target_max:
                print(f"[train] Batch size {bs} — {used/1e9:.1f}GB / {total_mem/1e9:.1f}GB ({used/total_mem*100:.0f}%)")
                model.zero_grad()
                _safe_empty_cache()
                return bs
            bs //= 2

        except Exception as e:
            if not any(s in str(e) for s in ('out of memory', 'cudaErrorMemory', 'CUDA error')):
                raise
            bs //= 2
            _safe_empty_cache()

    return 1


# ── Training / val loops ──────────────────────────────────────────────────────

def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def train_epoch(model, loss_fn, loader, optimizer, device,
                writer, global_step, grad_clip=0.0, log_every=10):
    model.train()
    total     = 0.0
    head_sums = {t: 0.0 for t in UncertaintyWeightedLoss.TASKS}
    running_sum, running_n = 0.0, 0

    bar = tqdm(loader, desc='train', leave=False,
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, loss={postfix}]')

    for images, rotation, gt in bar:
        images   = _to_device(images, device)
        rotation = _to_device(rotation, device)
        gt       = _to_device(gt, device)

        preds             = model(images, rotation)
        loss, head_losses = loss_fn(preds, gt)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            all_params = list(model.parameters()) + list(loss_fn.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        lv           = loss.item()
        total       += lv
        running_sum += lv
        running_n   += 1
        global_step += 1

        for k, v in head_losses.items():
            head_sums[k] += v
        bar.set_postfix_str(f"{lv:.4f}")

        if global_step % log_every == 0:
            writer.add_scalar('Loss/train_running', running_sum / running_n, global_step)
            running_sum, running_n = 0.0, 0

    n = len(loader)
    return total / n, {k: v / n for k, v in head_sums.items()}, global_step


def val_epoch(model, loss_fn, loader, device):
    model.eval()
    total     = 0.0
    head_sums = {t: 0.0 for t in UncertaintyWeightedLoss.TASKS}
    with torch.no_grad():
        for images, rotation, gt in tqdm(loader, desc='val  ', leave=False):
            images   = _to_device(images, device)
            rotation = _to_device(rotation, device)
            gt       = _to_device(gt, device)
            preds             = model(images, rotation)
            loss, head_losses = loss_fn(preds, gt)
            total            += loss.item()
            for k, v in head_losses.items():
                head_sums[k] += v
    n = len(loader)
    return total / n, {k: v / n for k, v in head_sums.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/config.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--view',   action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

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

    # ── Curriculum ────────────────────────────────────────────────────────────
    cur_cfg      = cfg.get('curriculum', {})
    use_cur      = cur_cfg.get('enabled', False)
    target_steps = cur_cfg.get('target_steps_per_epoch', 200)
    curriculum   = CurriculumPool(train_ids, data_root, cur_cfg) if use_cur else None

    warmup_epochs = cfg['training'].get('warmup_epochs', 5)
    grad_clip     = cfg['training'].get('grad_clip', 1.0)
    patience      = cfg['training'].get('early_stop_patience', 15)
    gpu_margin    = cfg['training'].get('gpu_memory_margin', 0.20)

    model   = TerrainModel().to(device)
    loss_fn = UncertaintyWeightedLoss().to(device)

    bs = find_batch_size(model, loss_fn, device,
                         start_bs=cfg['training']['batch_size'],
                         gpu_margin=gpu_margin)

    pin = torch.cuda.is_available()
    val_ds     = TerrainDataset(data_root, val_ids, depth_stats, augment=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=pin)

    def _rebuild_train_loader():
        active = curriculum.current_ids if curriculum else train_ids
        if curriculum and not curriculum.is_full:
            # Cap repeats at 5x per epoch so small pools don't spin excessively
            pool = len(active)
            eff_steps = min(target_steps, pool * 5 // max(bs, 1))
            eff_steps = max(eff_steps, max(pool // max(bs, 1), 1))
            return _make_train_loader(active, data_root, depth_stats, bs, eff_steps, pin)
        ds = TerrainDataset(data_root, list(active), depth_stats, augment=True)
        return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=pin)

    train_loader = _rebuild_train_loader()

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
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
    os.makedirs(ckpt_dir, exist_ok=True)

    runs_dir = os.path.abspath('training/runs')
    writer   = SummaryWriter(log_dir=runs_dir)

    if args.view:
        import time, webbrowser, sysconfig
        tb_exe = None
        for scripts_dir in [
            sysconfig.get_path('scripts'),
            os.path.join(os.path.expandvars('%APPDATA%'), 'Python',
                         f'Python{sys.version_info.major}{sys.version_info.minor}', 'Scripts'),
        ]:
            candidate = os.path.join(scripts_dir, 'tensorboard.exe')
            if os.path.exists(candidate):
                tb_exe = candidate
                break
        if tb_exe is None:
            print("[train] TensorBoard not found — run: pip install tensorboard")
        else:
            subprocess.Popen([tb_exe, '--logdir', runs_dir, '--port', '6006'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[train] Starting TensorBoard...", end='', flush=True)
            time.sleep(4)
            print(" opening http://localhost:6006")
            webbrowser.open('http://localhost:6006')

    best_val          = float('inf')
    epochs_no_improve = 0
    global_step       = 0
    start_epoch       = 1

    resume_path = _find_latest_checkpoint(ckpt_dir)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        sd = ckpt['model']
        sd = {k: v for k, v in sd.items()
              if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        missing = [k for k in model.state_dict() if k not in sd]
        model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[train] New params (random init): {len(missing)} keys")
        if 'loss_fn' in ckpt:
            loss_fn.load_state_dict(ckpt['loss_fn'])
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except ValueError:
                print("[train] Optimizer state skipped (parameter count changed — new params added)")
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if curriculum and 'curriculum' in ckpt:
            curriculum.load_state_dict(ckpt['curriculum'])
            train_loader = _rebuild_train_loader()
        start_epoch       = ckpt['epoch'] + 1
        best_val          = float('inf')   # reset — stale across loss function changes
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        global_step       = ckpt.get('global_step', 0)
        print(f"[train] Resuming from {os.path.basename(resume_path)} "
              f"— epoch {ckpt['epoch']}, best_val={best_val:.4f}")
        if curriculum:
            print(f"[curriculum] Pool restored: {curriculum.pool_size}/{len(train_ids)} episodes")

    import time as _time
    for epoch in range(start_epoch, max_epochs + 1):
        t0 = _time.time()
        pool_info = (f"  pool={curriculum.pool_size}/{len(train_ids)}"
                     if curriculum and not curriculum.is_full else "")

        train_loss, train_heads, global_step = train_epoch(
            model, loss_fn, train_loader, optimizer, device,
            writer, global_step, grad_clip,
        )
        val_loss, val_heads = val_epoch(model, loss_fn, val_loader, device)
        scheduler.step()
        elapsed = _time.time() - t0

        lr_now  = optimizer.param_groups[0]['lr']
        mem_gb  = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0
        weights = loss_fn.weights()

        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val',         val_loss,   epoch)
        writer.add_scalar('LR',               lr_now,     epoch)
        for k in train_heads:
            writer.add_scalar(f'Heads/train_{k}', train_heads[k], epoch)
            writer.add_scalar(f'Heads/val_{k}',   val_heads[k],   epoch)
        for k, w in weights.items():
            writer.add_scalar(f'Weights/{k}', w, epoch)
        if curriculum:
            writer.add_scalar('Curriculum/pool_size', curriculum.pool_size, epoch)
        writer.flush()

        flag = ' *' if val_loss < best_val else ''
        print(
            f"\n[epoch {epoch:03d}/{max_epochs}]  {elapsed/60:.1f}min  "
            f"gpu={mem_gb:.1f}GB  lr={lr_now:.2e}{pool_info}"
            f"\n  train  total={train_loss:.4f}  "
            f"height={train_heads['height']:.4f}  rocks={train_heads['rocks']:.4f}  "
            f"craters={train_heads['craters']:.4f}  walls={train_heads['walls']:.4f}"
            f"\n  val    total={val_loss:.4f}  "
            f"height={val_heads['height']:.4f}  rocks={val_heads['rocks']:.4f}  "
            f"craters={val_heads['craters']:.4f}  walls={val_heads['walls']:.4f}"
            f"\n  weights  " + "  ".join(f"{k}={w:.2f}" for k, w in weights.items())
            + flag
        )

        # Curriculum expansion — must happen before checkpoint so state is saved
        # Use train height (not val) — val is always high because it tests unseen episodes
        if curriculum and not curriculum.is_full:
            if curriculum.step(train_heads['height']):
                train_loader = _rebuild_train_loader()

        ckpt_data = {
            'epoch': epoch, 'model': model.state_dict(),
            'loss_fn': loss_fn.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_val': best_val, 'epochs_no_improve': epochs_no_improve,
            'global_step': global_step,
        }
        if curriculum:
            ckpt_data['curriculum'] = curriculum.state_dict()

        if val_loss < best_val:
            best_val          = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # Suppress early stopping while curriculum is still expanding
            if epochs_no_improve >= patience and (curriculum is None or curriculum.is_full):
                print(f"[train] Early stopping — no improvement for {patience} epochs")
                break

        epoch_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt')
        torch.save(ckpt_data, epoch_path)

        # best.pt = always the latest checkpoint
        import shutil
        shutil.copy2(epoch_path, os.path.join(ckpt_dir, 'best.pt'))

        # Keep only the 10 most recent epoch checkpoints
        import glob as _glob
        old_ckpts = sorted(_glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))[:-10]
        for old in old_ckpts:
            os.remove(old)

        gc.collect()
        _safe_empty_cache()

    writer.close()
    print("[train] Done.")


if __name__ == '__main__':
    main()
