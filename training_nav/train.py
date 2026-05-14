"""Navigation policy training loop.

Usage:
  python training_nav/train.py
  python training_nav/train.py --config training_nav/config.yaml
  python training_nav/train.py --headless            # no prompts, use config defaults
  python training_nav/train.py --headless --view     # headless training + open dashboard
"""

import argparse
import gc
import glob
import http.server
import json
import os
import shutil
import sys
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training_nav.dataset import NavDataset
from training_nav.model import NavPolicy


# ── Live training dashboard ────────────────────────────────────────────────────

_state: dict = {
    'epoch': 0, 'epochs': 0, 'train_loss': [], 'val_loss': [],
    'lr': [], 'best_val': None, 'no_improve': 0, 'patience': 20,
    'status': 'starting', 'device': '', 'elapsed_s': 0,
}
_state_lock = threading.Lock()


def _update_state(**kw):
    with _state_lock:
        _state.update(kw)


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OCTANE Nav Training</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#080808;color:#e0e0e0;font-family:'Courier New',monospace;font-size:13px}
header{display:flex;align-items:center;gap:16px;padding:12px 20px;
       border-bottom:1px solid #1e1e1e;background:#0d0d0d}
.logo{color:#00e5ff;font-size:1.1rem;font-weight:bold;letter-spacing:.1em}
.device-badge{background:#1a2a1a;color:#4caf50;border:1px solid #2a4a2a;
              padding:2px 10px;border-radius:3px;font-size:.75rem}
.status-badge{padding:2px 10px;border-radius:3px;font-size:.75rem;border:1px solid #333}
.status-badge.running{background:#0a1a2a;color:#00e5ff;border-color:#1e3a5a}
.status-badge.done{background:#1a2a1a;color:#4caf50;border-color:#2a4a2a}
.status-badge.stopped{background:#2a1a1a;color:#ef5350;border-color:#4a2a2a}
main{display:grid;grid-template-columns:1fr 260px;gap:12px;padding:16px;
     height:calc(100vh - 53px)}
.chart-panel{background:#0d0d0d;border:1px solid #1e1e1e;border-radius:6px;
             padding:14px;display:flex;flex-direction:column;gap:8px}
.panel-label{font-size:.7rem;color:#555;letter-spacing:.08em;text-transform:uppercase}
canvas{display:block}
.side{display:flex;flex-direction:column;gap:12px}
.metric-panel{background:#0d0d0d;border:1px solid #1e1e1e;border-radius:6px;padding:14px}
.metric-row{display:flex;justify-content:space-between;align-items:baseline;
            padding:4px 0;border-bottom:1px solid #141414}
.metric-row:last-child{border-bottom:none}
.metric-key{color:#666;font-size:.75rem}
.metric-val{color:#00e5ff;font-size:.9rem}
.metric-val.best{color:#ffd700}
.metric-val.warn{color:#ff9800}
.metric-val.good{color:#4caf50}
.progress-bar{height:3px;background:#1a1a1a;border-radius:2px;margin-top:8px}
.progress-fill{height:100%;background:#00e5ff;border-radius:2px;transition:width .5s}
</style>
</head>
<body>
<header>
  <span class="logo">◈ OCTANE</span>
  <span style="color:#444">NAVIGATION TRAINING</span>
  <span class="device-badge" id="deviceBadge">—</span>
  <span class="status-badge running" id="statusBadge">STARTING</span>
  <span style="margin-left:auto;color:#444;font-size:.75rem" id="etaText">—</span>
</header>
<main>
  <div class="chart-panel">
    <div class="panel-label">Loss curves</div>
    <canvas id="chart"></canvas>
    <div style="display:flex;gap:20px;font-size:.7rem;color:#555;margin-top:4px">
      <span><span style="color:#00e5ff">■</span> Train</span>
      <span><span style="color:#ff9800">■</span> Val</span>
      <span style="margin-left:auto" id="epochLabel">epoch 0 / 0</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
  </div>
  <div class="side">
    <div class="metric-panel">
      <div class="panel-label" style="margin-bottom:8px">Current epoch</div>
      <div class="metric-row"><span class="metric-key">Train loss</span><span class="metric-val" id="mTrain">—</span></div>
      <div class="metric-row"><span class="metric-key">Val loss</span><span class="metric-val" id="mVal">—</span></div>
      <div class="metric-row"><span class="metric-key">Best val</span><span class="metric-val best" id="mBest">—</span></div>
      <div class="metric-row"><span class="metric-key">Learning rate</span><span class="metric-val" id="mLR">—</span></div>
      <div class="metric-row"><span class="metric-key">No-improve</span><span class="metric-val" id="mNoImprove">—</span></div>
      <div class="metric-row"><span class="metric-key">Elapsed</span><span class="metric-val" id="mElapsed">—</span></div>
    </div>
  </div>
</main>
<script>
const chart = document.getElementById('chart');
const ctx   = chart.getContext('2d');
let state = {};

function resize() {
  const p = chart.parentElement;
  chart.width  = p.clientWidth - 28;
  chart.height = Math.max(200, p.clientHeight - 100);
}
window.addEventListener('resize', () => { resize(); draw(); });
resize();

function fmt(v, digits=4) { return v == null ? '—' : v.toFixed(digits); }
function fmtTime(s) {
  if (s < 60)  return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${Math.round(s%60)}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
}

function draw() {
  const W = chart.width, H = chart.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#080808';
  ctx.fillRect(0,0,W,H);

  const tl = state.train_loss || [], vl = state.val_loss || [];
  if (tl.length < 2) {
    ctx.fillStyle = '#333'; ctx.font = '12px monospace';
    ctx.fillText('Waiting for data…', W/2-60, H/2);
    return;
  }

  const PAD = {l:50, r:20, t:16, b:30};
  const gW = W - PAD.l - PAD.r;
  const gH = H - PAD.t - PAD.b;
  const n  = tl.length;
  const maxE = state.epochs || n;

  const allVals = [...tl, ...vl].filter(v => v > 0);
  const minY = Math.min(...allVals) * 0.95;
  const maxY = Math.max(...allVals) * 1.05;

  function toX(i) { return PAD.l + (i / Math.max(maxE-1, 1)) * gW; }
  function toY(v) { return PAD.t + (1 - (v-minY)/(maxY-minY)) * gH; }

  // Grid
  ctx.strokeStyle = '#181818'; ctx.lineWidth = 1;
  for (let i=0; i<=4; i++) {
    const y = PAD.t + i * gH / 4;
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l+gW, y); ctx.stroke();
    const v = maxY - i*(maxY-minY)/4;
    ctx.fillStyle = '#444'; ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(4), PAD.l-4, y+3);
  }

  // Axes
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD.l,PAD.t); ctx.lineTo(PAD.l,PAD.t+gH);
  ctx.lineTo(PAD.l+gW,PAD.t+gH); ctx.stroke();

  // Best val line
  if (state.best_val != null) {
    const by = toY(state.best_val);
    ctx.strokeStyle = 'rgba(255,215,0,0.25)'; ctx.lineWidth = 1;
    ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(PAD.l,by); ctx.lineTo(PAD.l+gW,by); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ffd700'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
    ctx.fillText(`best ${state.best_val.toFixed(4)}`, PAD.l+4, by-3);
  }

  function drawLine(data, color) {
    if (data.length < 2) return;
    ctx.beginPath();
    data.forEach((v,i) => {
      const x = toX(i), y = toY(v);
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.stroke();
    // dot at latest point
    const lx = toX(data.length-1), ly = toY(data[data.length-1]);
    ctx.beginPath(); ctx.arc(lx,ly,3,0,2*Math.PI);
    ctx.fillStyle = color; ctx.fill();
  }

  drawLine(tl, '#00e5ff');
  drawLine(vl, '#ff9800');

  // X axis labels
  ctx.fillStyle = '#444'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(maxE/6));
  for (let i=0; i<=maxE; i+=step) {
    ctx.fillText(i, toX(i), PAD.t+gH+14);
  }
}

function update(s) {
  state = s;
  const epoch  = s.epoch || 0;
  const epochs = s.epochs || 0;
  const pct    = epochs > 0 ? (epoch / epochs * 100).toFixed(1) : 0;

  document.getElementById('deviceBadge').textContent   = s.device || '—';
  document.getElementById('epochLabel').textContent    = `epoch ${epoch} / ${epochs}`;
  document.getElementById('progressFill').style.width  = pct + '%';

  const tl = s.train_loss || [], vl = s.val_loss || [];
  const lr = s.lr || [];
  document.getElementById('mTrain').textContent     = fmt(tl[tl.length-1]);
  document.getElementById('mVal').textContent       = fmt(vl[vl.length-1]);
  document.getElementById('mBest').textContent      = fmt(s.best_val);
  document.getElementById('mLR').textContent        = s.lr?.length ? s.lr[s.lr.length-1].toExponential(2) : '—';
  const ni = s.no_improve ?? 0, pa = s.patience ?? 20;
  const noImpEl = document.getElementById('mNoImprove');
  noImpEl.textContent = `${ni} / ${pa}`;
  noImpEl.className   = 'metric-val' + (ni >= pa*0.7 ? ' warn' : ni === 0 ? ' good' : '');

  const el = s.elapsed_s || 0;
  document.getElementById('mElapsed').textContent = fmtTime(el);

  // ETA
  if (epoch > 0 && epochs > epoch) {
    const per_ep = el / epoch;
    const eta    = per_ep * (epochs - epoch);
    document.getElementById('etaText').textContent = `ETA ${fmtTime(eta)}`;
  }

  const badge = document.getElementById('statusBadge');
  badge.textContent  = (s.status || 'starting').toUpperCase();
  badge.className    = 'status-badge ' + (s.status === 'running' ? 'running'
                       : s.status === 'done' ? 'done' : 'stopped');
  draw();
}

async function poll() {
  try {
    const r = await fetch('/api/state');
    if (r.ok) update(await r.json());
  } catch(e) {}
  setTimeout(poll, 2000);
}

poll();
</script>
</body>
</html>"""


class _DashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path.startswith('/api/state'):
            with _state_lock:
                body = json.dumps(_state).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        else:
            body = _DASHBOARD_HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(body)


def _start_dashboard(port: int = 8767):
    server = http.server.HTTPServer(('localhost', port), _DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_latest_checkpoint(ckpt_dir: str):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))
    return paths[-1] if paths else None


def _save(ckpt_dir: str, epoch: int, model, optimizer, scheduler,
          val_loss: float, best_val: float, keep: int = 10):
    os.makedirs(ckpt_dir, exist_ok=True)
    data = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val': best_val,
    }
    epoch_path = os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt')
    torch.save(data, epoch_path)

    # best.pt = always the latest checkpoint (for easy inference)
    shutil.copy2(epoch_path, os.path.join(ckpt_dir, 'best.pt'))

    # Prune old checkpoints
    old = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt')))[:-keep]
    for p in old:
        os.remove(p)


def _run_epoch(loader: DataLoader, model: NavPolicy, criterion,
               optimizer, device: torch.device,
               train: bool, grad_clip: float,
               bucket_loss_weight: float = 0.5) -> float:
    model.train(train)
    total = 0.0
    desc  = 'train' if train else 'val  '
    with torch.set_grad_enabled(train):
        bar = tqdm(loader, desc=desc, leave=False,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}  {postfix}]')
        for terrain, heading, action_gt in bar:
            terrain   = terrain.to(device, non_blocking=True)
            heading   = heading.to(device, non_blocking=True)
            action_gt = action_gt.to(device, non_blocking=True)

            action_pred = model(terrain, heading)
            motor_loss  = criterion(action_pred[:, :2], action_gt[:, :2])
            bucket_loss = F.cross_entropy(action_pred[:, 2:], action_gt[:, 2].long())
            loss        = motor_loss + bucket_loss_weight * bucket_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            lv = loss.item()
            total += lv
            bar.set_postfix_str(f'{lv:.4f}')

    return total / max(len(loader), 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--config',   default=os.path.join(_script_dir, 'config.yaml'))
    parser.add_argument('--headless', action='store_true',
                        help='Skip interactive prompts — use config defaults')
    parser.add_argument('--view',     action='store_true',
                        help='Open live training dashboard in browser')
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    tc  = cfg['training']
    cc  = cfg['checkpoints']
    # Resolve checkpoints dir relative to the script, not CWD
    if not os.path.isabs(cc['dir']):
        cc['dir'] = os.path.join(_script_dir, cc['dir'])

    torch.manual_seed(tc['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device_label = f'cuda ({torch.cuda.get_device_name(0)})'
    else:
        device_label = 'cpu'

    print(f'\n  OCTANE  Navigation Policy Training')
    print(f'  ─────────────────────────────────')
    print(f'  Device : {device_label}')

    # ── Interactive prompts ────────────────────────────────────────────────────
    if args.headless:
        max_epochs = tc['epochs']
        open_gui   = args.view
    else:
        default_ep = tc['epochs']
        raw = input(f'\n  Epochs to train (Enter for {default_ep}): ').strip()
        max_epochs = int(raw) if raw else default_ep

        if not args.view:
            raw = input('  Open live training dashboard? [y/N]: ').strip().lower()
            open_gui = raw in ('y', 'yes')
        else:
            open_gui = True

    # ── Live dashboard ─────────────────────────────────────────────────────────
    if open_gui:
        _start_dashboard(port=8767)
        import webbrowser
        print('  Dashboard → http://localhost:8767')
        webbrowser.open('http://localhost:8767')

    _update_state(epochs=max_epochs, device=device_label, status='running')

    # ── Dataset ────────────────────────────────────────────────────────────────
    n_train = int(tc['arenas_per_epoch'] * tc['samples_per_arena'] * (1 - tc['val_ratio']))
    n_val   = int(tc['arenas_per_epoch'] * tc['samples_per_arena'] * tc['val_ratio'])
    nw      = tc.get('num_workers', 0)
    pin     = torch.cuda.is_available()

    train_ds = NavDataset(cfg, n_train, seed=tc['seed'])
    val_ds   = NavDataset(cfg, n_val,   seed=tc['seed'] + 1)

    train_loader = DataLoader(train_ds, batch_size=tc['batch_size'],
                              shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=tc['batch_size'],
                              shuffle=False, num_workers=nw, pin_memory=pin)

    print(f'  Train  : {n_train} samples/epoch')
    print(f'  Val    : {n_val} samples/epoch')

    # ── Model ──────────────────────────────────────────────────────────────────
    model              = NavPolicy(cfg).to(device)
    criterion          = nn.HuberLoss(delta=0.1)
    bucket_loss_weight = tc.get('bucket_loss_weight', 0.5)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tc['learning_rate'],
                                  weight_decay=tc['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=tc['learning_rate'] * 0.01
    )

    start_epoch = 1
    best_val    = float('inf')
    no_improve  = 0
    keep        = cc.get('keep', 10)

    # ── Resume ─────────────────────────────────────────────────────────────────
    ckpt_path = _find_latest_checkpoint(cc['dir'])
    if ckpt_path:
        print(f'\n  Resuming from {os.path.basename(ckpt_path)}')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
    else:
        print()

    # ── Training loop ──────────────────────────────────────────────────────────
    train_losses: list[float] = []
    val_losses:   list[float] = []
    lrs:          list[float] = []
    t0 = time.time()

    for epoch in range(start_epoch, max_epochs + 1):
        train_ds.reshuffle(epoch)
        val_ds.reshuffle(epoch + 10_000)

        print(f'\n  [{epoch:04d}/{max_epochs}]', end='  ', flush=True)

        train_loss = _run_epoch(train_loader, model, criterion, optimizer,
                                device, train=True,  grad_clip=tc['grad_clip'],
                                bucket_loss_weight=bucket_loss_weight)
        val_loss   = _run_epoch(val_loader,   model, criterion, optimizer,
                                device, train=False, grad_clip=0,
                                bucket_loss_weight=bucket_loss_weight)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lrs.append(lr)

        improved = val_loss < best_val
        if improved:
            best_val   = val_loss
            no_improve = 0
            marker     = ' *'
        else:
            no_improve += 1
            marker     = ''

        elapsed = time.time() - t0
        print(f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'lr={lr:.2e}  best={best_val:.4f}{marker}')

        _update_state(
            epoch=epoch, train_loss=train_losses, val_loss=val_losses,
            lr=lrs, best_val=best_val, no_improve=no_improve,
            patience=tc['early_stop_patience'], elapsed_s=round(elapsed),
        )

        _save(cc['dir'], epoch, model, optimizer, scheduler,
              val_loss, best_val, keep=keep)

        if no_improve >= tc['early_stop_patience']:
            print(f'\n  Early stop — no val improvement for {no_improve} epochs')
            break

        gc.collect()

    _update_state(status='done')
    print(f'\n  Done.  Best val loss: {best_val:.4f}\n')


if __name__ == '__main__':
    main()
