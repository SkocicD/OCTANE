"""Navigation policy visualizer — Tesla-style arena inspector.

Shows the full arena map with zone overlays, obstacles, the robot's sensor
view window, the A* planned path, a Tesla-style kinematic projection arc,
and a side panel with the robot's terrain crop + action comparison.

Usage:
  python training_nav/visualize.py          (from repo root)
  python visualize.py                        (from training_nav/)
  python training_nav/visualize.py --checkpoint path/to/best.pt
  python training_nav/visualize.py --seed 42 --phase to_excavation
  python training_nav/visualize.py --port 8766
"""

import argparse
import http.server
import json
import math
import os
import random
import sys
import threading
import webbrowser

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training_nav.arena import (
    build_goal_heatmap, build_terrain_maps, crop_robot_view, generate_arena,
)
from training_nav.dataset import _goal_zone_for_phase, _sample_robot_pose
from training_nav.planner import _build_cost_map, astar, plan_action


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Scene generation ──────────────────────────────────────────────────────────

def _generate_scene(cfg: dict, seed: int, phase: str) -> dict:
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    arena   = generate_arena(cfg, rng)
    terrain = build_terrain_maps(arena, cfg, np_rng)

    goal_zone       = _goal_zone_for_phase(arena, phase)
    rx, ry, heading = _sample_robot_pose(arena, phase, rng)

    terrain_crop = crop_robot_view(terrain, rx, ry, cfg)      # (4, gs, gs)
    goal_map     = build_goal_heatmap(arena, goal_zone, rx, ry, cfg)  # (gs, gs)

    cost = _build_cost_map(terrain_crop, cfg)
    gs   = cfg['terrain']['grid_size']
    half = gs // 2
    goal_idx  = np.unravel_index(np.argmax(goal_map), goal_map.shape)
    path      = astar(cost, (half, half), (int(goal_idx[0]), int(goal_idx[1])))

    expert_action = plan_action(terrain_crop, goal_map, heading, cfg)

    return {
        'arena':        arena,
        'terrain':      terrain,
        'terrain_crop': terrain_crop,
        'goal_map':     goal_map,
        'cost':         cost,
        'path':         path,
        'rx': rx, 'ry': ry, 'heading': heading,
        'expert_action': expert_action,
        'phase': phase,
        'goal_zone': goal_zone,
    }


def _run_model(cfg: dict, checkpoint_path: str,
               terrain_crop: np.ndarray, goal_map: np.ndarray,
               heading: float):
    try:
        import torch
        from training_nav.model import NavPolicy

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = NavPolicy(cfg).to(device)
        ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()

        import torch as _torch
        t5 = np.concatenate([terrain_crop, goal_map[None]], axis=0)
        t  = _torch.from_numpy(t5).unsqueeze(0).to(device)
        h  = _torch.tensor(
            [[math.sin(heading), math.cos(heading)]], dtype=_torch.float32
        ).to(device)
        with __import__('torch').inference_mode():
            out = model(t, h).squeeze(0).cpu().numpy()
        return float(out[0]), float(out[1])
    except Exception as e:
        print(f'[visualize] model inference failed: {e}')
        return None


# ── HTML generation ────────────────────────────────────────────────────────────

def _make_html(cfg: dict, scene: dict, model_action) -> str:  # noqa: C901
    arena   = scene['arena']
    terrain = scene['terrain']
    tc      = scene['terrain_crop']
    gm      = scene['goal_map']
    path    = scene['path']
    rx, ry  = scene['rx'], scene['ry']
    heading = scene['heading']
    ea      = scene['expert_action']
    phase   = scene['phase']
    gs      = cfg['terrain']['grid_size']
    vw      = cfg['terrain']['view_width']
    vh      = cfg['terrain']['view_height']
    cs      = cfg['terrain'].get('cell_size', 0.1)
    rc      = cfg.get('robot', {})
    v_max   = rc.get('v_max', 0.5)
    w_max   = rc.get('omega_max', 1.5)
    r_width = rc.get('width', 0.483)
    proj_t  = rc.get('projection_time', 3.0)

    # Full-arena terrain (downsample for display if large)
    h_full = terrain['height']
    r_full = terrain['rocks']
    c_full = terrain['craters']
    w_full = terrain['walls']
    t_rows = terrain['rows']
    t_cols = terrain['cols']

    # Obstacles as list of dicts
    obstacles = [
        {'x': o.x, 'y': o.y, 'd': o.diameter, 'k': o.kind}
        for o in arena.obstacles
    ]

    # A* path as [[r,c], ...]
    path_json = json.dumps([[int(r), int(c)] for r, c in path] if path else [])

    def _j(arr):
        """Convert 2-D numpy array to compact JSON list."""
        return json.dumps(arr.tolist())

    # Zone rects as {x,y,w,h}
    def _rect(r):
        return {'x': r.x, 'y': r.y, 'w': r.w, 'h': r.h}

    zones = {
        'start':      _rect(arena.start_zone),
        'excavation': _rect(arena.excavation_zone),
        'nav':        _rect(arena.nav_zone),
        'deposit':    _rect(arena.deposit_zone),
        'berm':       _rect(arena.berm_target),
    }

    ea_js = json.dumps(list(ea) if ea else None)
    ma_js = json.dumps(list(model_action) if model_action else None)

    data_block = f"""
const GS          = {gs};
const VIEW_W      = {vw};
const VIEW_H      = {vh};
const CELL_SIZE   = {cs};
const ARENA_W     = {arena.width:.4f};
const ARENA_L     = {arena.length:.4f};
const ARENA_SCALE = {arena.scale:.4f};
const T_ROWS      = {t_rows};
const T_COLS      = {t_cols};
const ROBOT_V_MAX = {v_max};
const ROBOT_W_MAX = {w_max};
const ROBOT_WIDTH = {r_width};
const PROJ_TIME   = {proj_t};
const PHASE       = {json.dumps(phase)};
const ZONES       = {json.dumps(zones)};
const OBSTACLES   = {json.dumps(obstacles)};
const H_FULL      = {_j(h_full)};
const R_FULL      = {_j(r_full)};
const C_FULL      = {_j(c_full)};
const W_FULL      = {_j(w_full)};
const CROP_H      = {_j(tc[0])};
const CROP_R      = {_j(tc[1])};
const CROP_C      = {_j(tc[2])};
const CROP_W      = {_j(tc[3])};
const GOAL_MAP    = {_j(gm)};
const ASTAR_PATH  = {path_json};
const ROBOT_X     = {rx:.4f};
const ROBOT_Y     = {ry:.4f};
const ROBOT_H     = {heading:.4f};
const EXPERT      = {ea_js};
const MODEL       = {ma_js};
"""

    return _HTML_TEMPLATE.replace('/*DATA_BLOCK*/', data_block)


# ── HTML / JS template ─────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCTANE Nav Policy</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080808;--panel:#0d0d0d;--border:#1e1e1e;--border2:#2a2a2a;
  --text:#d0d0d0;--dim:#555;--cyan:#00e5ff;--gold:#ffd700;
  --blue:#1e90ff;--orange:#ff6b35;--green:#4caf50;--red:#ef5350;
  --zone-start:rgba(30,90,50,.35);--zone-exc:rgba(110,65,0,.3);
  --zone-nav:rgba(10,35,80,.3);--zone-dep:rgba(80,10,100,.3);
  --zone-berm:rgba(200,160,0,.2);
}
body{background:var(--bg);color:var(--text);font-family:'Courier New',monospace;
     font-size:12px;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Top bar ── */
header{display:flex;align-items:center;gap:12px;padding:8px 16px;
       background:var(--panel);border-bottom:1px solid var(--border);
       flex-shrink:0}
.logo{color:var(--cyan);font-size:1rem;font-weight:bold;letter-spacing:.1em}
.phase-badge{padding:2px 10px;border-radius:3px;font-size:.7rem;font-weight:bold;
             letter-spacing:.08em;border:1px solid}
.phase-badge.exc{background:rgba(110,65,0,.3);color:#ffb74d;border-color:rgba(110,65,0,.6)}
.phase-badge.dep{background:rgba(80,10,100,.3);color:#ce93d8;border-color:rgba(80,10,100,.6)}
.ckpt-info{color:var(--dim);font-size:.7rem;margin-left:auto}

/* ── Main workspace ── */
.workspace{display:grid;grid-template-columns:1fr 296px;gap:8px;
           padding:8px;flex:1;min-height:0}

/* ── Arena panel ── */
.arena-wrap{position:relative;background:var(--panel);
            border:1px solid var(--border);border-radius:6px;
            overflow:hidden;display:flex;align-items:center;justify-content:center}
#arenaCanvas{display:block;cursor:crosshair}
.arena-label{position:absolute;top:8px;left:10px;font-size:.65rem;color:var(--dim);
             letter-spacing:.06em;text-transform:uppercase;pointer-events:none}
.legend{position:absolute;bottom:8px;left:10px;display:flex;flex-direction:column;
        gap:3px;pointer-events:none}
.legend-row{display:flex;align-items:center;gap:5px;font-size:.65rem;color:var(--dim)}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.legend-line{width:18px;height:2px;flex-shrink:0}

/* ── Side column ── */
.side-col{display:flex;flex-direction:column;gap:8px;min-height:0}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:10px}
.panel-title{font-size:.65rem;color:var(--dim);letter-spacing:.08em;
             text-transform:uppercase;margin-bottom:8px}

/* Robot view */
.robot-view-wrap{display:flex;justify-content:center}
#cropCanvas{display:block;border-radius:3px}

/* Action gauges */
.gauge-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.gauge-label{width:52px;font-size:.7rem;color:var(--dim);flex-shrink:0}
.gauge-track{flex:1;height:6px;background:#1a1a1a;border-radius:3px;position:relative;overflow:visible}
.gauge-zero{position:absolute;left:50%;top:-3px;width:1px;height:12px;background:#333}
.gauge-fill{position:absolute;top:0;height:100%;border-radius:3px;transition:width .15s,left .15s}
.gauge-val{width:48px;text-align:right;font-size:.8rem}
.gauge-val.pos{color:var(--cyan)}
.gauge-val.neg{color:var(--red)}
.divider{height:1px;background:var(--border);margin:8px 0}
.compare-row{display:flex;justify-content:space-between;font-size:.7rem;margin-bottom:3px}
.compare-row .key{color:var(--dim)}
.compare-row .a-val{color:var(--gold)}
.compare-row .m-val{color:var(--blue)}
.no-model{color:var(--dim);font-size:.7rem;text-align:center;padding:6px 0}

/* Info panel */
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px}
.info-item{display:flex;flex-direction:column}
.info-key{font-size:.65rem;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
.info-val{font-size:.8rem;color:var(--text)}

/* ── Controls ── */
footer{display:flex;align-items:center;gap:10px;padding:6px 16px;
       background:var(--panel);border-top:1px solid var(--border);flex-shrink:0}
select,input[type=number]{background:#111;color:var(--text);
  border:1px solid var(--border2);border-radius:3px;padding:3px 8px;
  font-family:inherit;font-size:.75rem}
button{background:#0a1a2a;color:var(--cyan);border:1px solid #1a3a5a;
       border-radius:3px;padding:4px 14px;font-family:inherit;font-size:.75rem;
       cursor:pointer;letter-spacing:.06em;transition:background .15s}
button:hover{background:#0f2a3a}
footer label{font-size:.75rem;color:var(--dim)}
</style>
</head>
<body>
<header>
  <span class="logo">◈ OCTANE</span>
  <span style="color:var(--border2)">NAVIGATION POLICY</span>
  <span class="phase-badge" id="phaseBadge">—</span>
  <span class="ckpt-info" id="ckptInfo">no model loaded</span>
</header>

<div class="workspace">
  <!-- Full arena -->
  <div class="arena-wrap" id="arenaWrap">
    <span class="arena-label" id="arenaLabel">—</span>
    <canvas id="arenaCanvas"></canvas>
    <div class="legend">
      <div class="legend-row"><div class="legend-dot" style="background:var(--cyan)"></div>Robot</div>
      <div class="legend-row">
        <div class="legend-line" style="background:var(--gold);box-shadow:0 0 4px var(--gold)"></div>
        A* path
      </div>
      <div class="legend-row">
        <div class="legend-line" style="background:var(--blue);box-shadow:0 0 6px var(--blue)"></div>
        Projected arc
      </div>
      <div class="legend-row">
        <div class="legend-line" style="background:transparent;border:1px dashed #666;width:18px"></div>
        Sensor view
      </div>
    </div>
  </div>

  <!-- Side column -->
  <div class="side-col">
    <!-- Terrain crop -->
    <div class="panel">
      <div class="panel-title">Sensor view  (what robot sees)</div>
      <div class="robot-view-wrap">
        <canvas id="cropCanvas" width="272" height="272"></canvas>
      </div>
    </div>

    <!-- Actions -->
    <div class="panel">
      <div class="panel-title">Actions</div>
      <div class="gauge-row">
        <span class="gauge-label">Linear</span>
        <div class="gauge-track">
          <div class="gauge-zero"></div>
          <div class="gauge-fill" id="aLinFill" style="height:6px"></div>
        </div>
        <span class="gauge-val" id="aLinVal">—</span>
      </div>
      <div class="gauge-row">
        <span class="gauge-label">Angular</span>
        <div class="gauge-track">
          <div class="gauge-zero"></div>
          <div class="gauge-fill" id="aAngFill" style="height:6px"></div>
        </div>
        <span class="gauge-val" id="aAngVal">—</span>
      </div>
      <div class="divider"></div>
      <div id="compareBlock">
        <div class="compare-row">
          <span class="key">Expert (A*)</span>
          <span class="a-val" id="cExpert">—</span>
        </div>
        <div class="compare-row">
          <span class="key">Model</span>
          <span class="m-val" id="cModel">—</span>
        </div>
      </div>
    </div>

    <!-- Info -->
    <div class="panel" style="flex:1">
      <div class="panel-title">Arena info</div>
      <div class="info-grid">
        <div class="info-item"><span class="info-key">Width</span><span class="info-val" id="iW">—</span></div>
        <div class="info-item"><span class="info-key">Length</span><span class="info-val" id="iL">—</span></div>
        <div class="info-item"><span class="info-key">Scale</span><span class="info-val" id="iSc">—</span></div>
        <div class="info-item"><span class="info-key">Obstacles</span><span class="info-val" id="iObs">—</span></div>
        <div class="info-item"><span class="info-key">Robot X</span><span class="info-val" id="iRx">—</span></div>
        <div class="info-item"><span class="info-key">Robot Y</span><span class="info-val" id="iRy">—</span></div>
        <div class="info-item"><span class="info-key">Heading</span><span class="info-val" id="iHd">—</span></div>
        <div class="info-item"><span class="info-key">Path len</span><span class="info-val" id="iPl">—</span></div>
      </div>
    </div>
  </div>
</div>

<footer>
  <label>Phase
    <select id="phaseSelect">
      <option value="to_excavation">to_excavation</option>
      <option value="to_deposit">to_deposit</option>
    </select>
  </label>
  <label>Seed
    <input type="number" id="seedInput" value="0" style="width:70px">
  </label>
  <button onclick="reload()">Regenerate</button>
  <span style="margin-left:auto;color:var(--dim);font-size:.7rem" id="footerInfo">—</span>
</footer>

<script>
/*DATA_BLOCK*/

// ── Helpers ────────────────────────────────────────────────────────────────

// World → canvas  (Y-flip: world bottom = canvas bottom)
function w2c(wx, wy, arenaW, arenaL, W, H) {
  return { x: wx / arenaW * W, y: (1 - wy / arenaL) * H };
}

// Lerp
function lerp(a, b, t) { return a + (b-a)*t; }

// Clamp
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// Heatmap colour: blue → cyan → yellow → red
function heatColor(v, lo, hi) {
  const t = clamp((v - lo) / (hi - lo + 1e-9), 0, 1);
  if (t < 0.5) {
    const s = t * 2;
    return `rgb(${Math.round(lerp(20,0,s))},${Math.round(lerp(30,160,s))},${Math.round(lerp(60,180,s))})`;
  }
  const s = (t - 0.5) * 2;
  return `rgb(${Math.round(lerp(0,220,s))},${Math.round(lerp(160,180,s))},${Math.round(lerp(180,20,s))})`;
}

function flatMinMax(arr2d) {
  let mn = Infinity, mx = -Infinity;
  for (const row of arr2d) for (const v of row) { if(v<mn)mn=v; if(v>mx)mx=v; }
  return [mn, mx];
}

// ── Kinematic arc simulation ────────────────────────────────────────────────

function simulateArc(rx, ry, heading, linVel, angVel) {
  const DT    = 0.08;
  const steps = Math.round(PROJ_TIME / DT);
  const pts   = [{x: rx, y: ry}];
  let x = rx, y = ry, h = heading;
  for (let i = 0; i < steps; i++) {
    x += ROBOT_V_MAX * linVel * Math.cos(h) * DT;
    y += ROBOT_V_MAX * linVel * Math.sin(h) * DT;
    h += ROBOT_W_MAX * angVel * DT;
    pts.push({x, y});
  }
  return pts;
}

// Same arc in robot-view grid coordinates (centre = half, half)
function simulateArcGrid(heading, linVel, angVel) {
  const CSX   = VIEW_W / GS;
  const CSY   = VIEW_H / GS;
  const DT    = 0.08;
  const steps = Math.round(PROJ_TIME / DT);
  const half  = GS / 2;
  const pts   = [{row: half, col: half}];
  let col = half, row = half, h = heading;
  for (let i = 0; i < steps; i++) {
    col +=  ROBOT_V_MAX * linVel * Math.cos(h) * DT / CSX;
    row += -ROBOT_V_MAX * linVel * Math.sin(h) * DT / CSY;  // +Y world = -row
    h   +=  ROBOT_W_MAX * angVel * DT;
    if (row < 0 || row >= GS || col < 0 || col >= GS) break;
    pts.push({row, col});
  }
  return pts;
}

// ── Draw helpers ────────────────────────────────────────────────────────────

// Tesla-style glowing arc (canvas-space array of {x,y})
function drawTeslaArc(ctx, pts) {
  if (pts.length < 2) return;
  // 3 passes: wide soft glow → medium glow → thin bright core
  const layers = [[14, 0.08], [5, 0.25], [2, 0.85]];
  for (const [lw, alpha] of layers) {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    // Gradient along path: bright at start, transparent at end
    const grad = ctx.createLinearGradient(
      pts[0].x, pts[0].y, pts[pts.length-1].x, pts[pts.length-1].y
    );
    grad.addColorStop(0,   `rgba(30,144,255,${alpha})`);
    grad.addColorStop(0.6, `rgba(100,200,255,${alpha * 0.5})`);
    grad.addColorStop(1,   `rgba(30,144,255,0)`);
    ctx.strokeStyle = grad;
    ctx.lineWidth   = lw;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.stroke();
  }
}

// Draw robot dot + heading arrow
function drawRobot(ctx, cx, cy, r, heading) {
  // Glow ring
  ctx.beginPath(); ctx.arc(cx, cy, r + 4, 0, 2*Math.PI);
  ctx.fillStyle = 'rgba(0,229,255,0.12)'; ctx.fill();
  // Dot
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI);
  ctx.fillStyle = '#00e5ff'; ctx.fill();
  // Heading arrow (canvas: +x right, +y down; heading: 0=right, π/2=up)
  const aLen = r * 3.5;
  const adx  =  Math.cos(-heading) * aLen;   // -heading to flip Y axis
  const ady  =  Math.sin(-heading) * aLen;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx+adx, cy+ady);
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
  ctx.lineCap = 'round'; ctx.stroke();
}

// ── Full Arena Canvas ────────────────────────────────────────────────────────

function drawArena(canvas) {
  const wrap = document.getElementById('arenaWrap');
  // Size canvas to fill wrapper, preserve arena aspect ratio
  const maxW  = wrap.clientWidth  - 2;
  const maxH  = wrap.clientHeight - 2;
  const ratio = ARENA_W / ARENA_L;
  let W, H;
  if (maxW / maxH > ratio) { H = maxH; W = H * ratio; }
  else                      { W = maxW; H = W / ratio; }
  canvas.width  = Math.floor(W);
  canvas.height = Math.floor(H);

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  // ── Height map as subtle texture ──
  const [hMin, hMax] = flatMinMax(H_FULL);
  const cellW = W / T_COLS;
  const cellH = H / T_ROWS;
  for (let r = 0; r < T_ROWS; r++) {
    for (let c = 0; c < T_COLS; c++) {
      const t = (H_FULL[r][c] - hMin) / (hMax - hMin + 1e-9);
      // Muted terrain grayscale so zone colours stay readable
      const v = Math.round(lerp(8, 35, t));
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(c * cellW, (T_ROWS-1-r) * cellH, cellW+1, cellH+1);
    }
  }

  // ── Zone overlays ──
  const zColors = {
    start:      'rgba(30,90,50,.35)',
    excavation: 'rgba(110,65,0,.3)',
    nav:        'rgba(10,35,80,.3)',
    deposit:    'rgba(80,10,100,.3)',
    berm:       'rgba(200,160,0,.2)',
  };
  const zLabels = {
    start:'START', excavation:'EXCAVATION', nav:'NAV', deposit:'DEPOSIT', berm:'BERM'
  };
  for (const [key, color] of Object.entries(zColors)) {
    const z = ZONES[key];
    const p = w2c(z.x, z.y + z.h, ARENA_W, ARENA_L, W, H);
    const q = w2c(z.x + z.w, z.y, ARENA_W, ARENA_L, W, H);
    ctx.fillStyle = color;
    ctx.fillRect(p.x, p.y, q.x - p.x, q.y - p.y);
    if (key === 'berm') {
      ctx.strokeStyle = 'rgba(255,215,0,0.6)';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4,3]);
      ctx.strokeRect(p.x, p.y, q.x - p.x, q.y - p.y);
      ctx.setLineDash([]);
    }
    // Zone label
    const lx = (p.x + q.x) / 2;
    const ly = (p.y + q.y) / 2;
    ctx.fillStyle = 'rgba(255,255,255,0.18)';
    ctx.font = `${Math.max(8, Math.min(11, W/35))}px Courier New`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(zLabels[key], lx, ly);
  }

  // ── Arena walls (border) ──
  ctx.strokeStyle = '#333'; ctx.lineWidth = 1.5; ctx.setLineDash([]);
  ctx.strokeRect(0.5, 0.5, W-1, H-1);

  // ── Obstacles ──
  for (const o of OBSTACLES) {
    const pt = w2c(o.x, o.y, ARENA_W, ARENA_L, W, H);
    const rr = (o.d / 2) / ARENA_W * W;
    ctx.beginPath(); ctx.arc(pt.x, pt.y, Math.max(2, rr), 0, 2*Math.PI);
    if (o.k === 'rock') {
      ctx.fillStyle = 'rgba(255,107,53,0.7)';
      ctx.strokeStyle = 'rgba(255,107,53,0.9)';
    } else {
      ctx.fillStyle = 'rgba(120,40,40,0.6)';
      ctx.strokeStyle = 'rgba(180,50,50,0.8)';
    }
    ctx.fill(); ctx.lineWidth = 1; ctx.stroke();
  }

  // ── Sensor view rectangle ──
  const vx0 = ROBOT_X - VIEW_W / 2;
  const vy0 = ROBOT_Y - VIEW_H / 2;
  const vTL = w2c(vx0, vy0 + VIEW_H, ARENA_W, ARENA_L, W, H);
  const vBR = w2c(vx0 + VIEW_W, vy0, ARENA_W, ARENA_L, W, H);
  const vRW = vBR.x - vTL.x, vRH = vBR.y - vTL.y;

  // Interior subtle highlight
  ctx.fillStyle = 'rgba(255,255,255,0.04)';
  ctx.fillRect(vTL.x, vTL.y, vRW, vRH);
  // Dashed border with glow
  ctx.shadowColor = 'rgba(255,255,255,0.3)'; ctx.shadowBlur = 4;
  ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(vTL.x + 0.5, vTL.y + 0.5, vRW - 1, vRH - 1);
  ctx.setLineDash([]); ctx.shadowBlur = 0;
  // Label
  ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '9px Courier New';
  ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText('SENSOR', vTL.x + 3, vTL.y + 2);

  // ── A* path in world coords ──
  if (ASTAR_PATH.length > 1) {
    const half  = GS / 2;
    const CSX   = VIEW_W / GS;
    const CSY   = VIEW_H / GS;
    const wPath = ASTAR_PATH.map(([r, c]) => ({
      x: ROBOT_X + (c - half) * CSX,
      y: ROBOT_Y + (r - half) * CSY,
    }));
    // Glow
    ctx.shadowColor = 'rgba(255,215,0,0.5)'; ctx.shadowBlur = 5;
    ctx.beginPath();
    wPath.forEach((p, i) => {
      const cp = w2c(p.x, p.y, ARENA_W, ARENA_L, W, H);
      i === 0 ? ctx.moveTo(cp.x, cp.y) : ctx.lineTo(cp.x, cp.y);
    });
    ctx.strokeStyle = '#ffd700'; ctx.lineWidth = 2;
    ctx.lineJoin = 'round'; ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // ── Tesla kinematic arc ──
  const action = MODEL || EXPERT;
  if (action) {
    const arcPts = simulateArc(ROBOT_X, ROBOT_Y, ROBOT_H, action[0], action[1]);
    const cPts   = arcPts.map(p => w2c(p.x, p.y, ARENA_W, ARENA_L, W, H));
    drawTeslaArc(ctx, cPts);
  }

  // ── Robot ──
  const rp = w2c(ROBOT_X, ROBOT_Y, ARENA_W, ARENA_L, W, H);
  drawRobot(ctx, rp.x, rp.y, Math.max(4, W/80), ROBOT_H);
}

// ── Robot view (sensor crop) canvas ─────────────────────────────────────────

function drawCrop(canvas) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cW = W / GS, cH = H / GS;

  ctx.clearRect(0, 0, W, H);

  // ── Height map ──
  const [hMin, hMax] = flatMinMax(CROP_H);
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    ctx.fillStyle = heatColor(CROP_H[r][c], hMin, hMax);
    ctx.fillRect(c*cW, r*cH, cW+.5, cH+.5);
  }

  // ── Rock overlay (orange, additive feel) ──
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    const v = CROP_R[r][c];
    if (v > 0.05) {
      ctx.fillStyle = `rgba(255,107,53,${v * 0.7})`;
      ctx.fillRect(c*cW, r*cH, cW+.5, cH+.5);
    }
  }

  // ── Crater overlay (dark red) ──
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    const v = CROP_C[r][c];
    if (v > 0.05) {
      ctx.fillStyle = `rgba(160,30,30,${v * 0.65})`;
      ctx.fillRect(c*cW, r*cH, cW+.5, cH+.5);
    }
  }

  // ── Wall overlay ──
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    if (CROP_W[r][c] > 0.5) {
      ctx.fillStyle = 'rgba(60,60,70,0.9)';
      ctx.fillRect(c*cW, r*cH, cW+.5, cH+.5);
    }
  }

  // ── Goal heatmap (green tint) ──
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    const v = GOAL_MAP[r][c];
    if (v > 0.02) {
      ctx.fillStyle = `rgba(76,175,80,${v * 0.5})`;
      ctx.fillRect(c*cW, r*cH, cW+.5, cH+.5);
    }
  }

  // ── A* path ──
  if (ASTAR_PATH.length > 1) {
    ctx.shadowColor = 'rgba(255,215,0,0.6)'; ctx.shadowBlur = 4;
    ctx.beginPath();
    ASTAR_PATH.forEach(([r, c], i) => {
      const x = (c + 0.5) * cW, y = (r + 0.5) * cH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#ffd700'; ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round'; ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // ── Tesla arc on crop ──
  const action = MODEL || EXPERT;
  if (action) {
    const gridPts = simulateArcGrid(ROBOT_H, action[0], action[1]);
    const cPts = gridPts.map(p => ({ x: (p.col + 0.5) * cW, y: (p.row + 0.5) * cH }));
    drawTeslaArc(ctx, cPts);
  }

  // ── Robot (centre) ──
  const cx = (GS/2 + 0.5) * cW, cy = (GS/2 + 0.5) * cH;
  drawRobot(ctx, cx, cy, Math.max(3, cW * 0.7), ROBOT_H);

  // ── Border ──
  ctx.strokeStyle = '#2a2a2a'; ctx.lineWidth = 1; ctx.setLineDash([]);
  ctx.strokeRect(0, 0, W, H);
}

// ── Action gauges ────────────────────────────────────────────────────────────

function setGauge(fillId, valId, v) {
  const el = document.getElementById(fillId);
  const tv = document.getElementById(valId);
  if (v == null) { el.style.width='0'; el.style.left='50%'; tv.textContent='—'; return; }
  const half = 50; // % for centre
  const pct  = Math.abs(v) * 50;
  if (v >= 0) { el.style.left = half + '%'; el.style.width = pct + '%'; el.style.background = '#00e5ff'; }
  else        { el.style.left = (half - pct) + '%'; el.style.width = pct + '%'; el.style.background = '#ef5350'; }
  tv.textContent = (v >= 0 ? '+' : '') + v.toFixed(2);
  tv.className   = 'gauge-val ' + (v >= 0 ? 'pos' : 'neg');
}

// ── Info panel ────────────────────────────────────────────────────────────────

function updateInfo() {
  const rocks   = OBSTACLES.filter(o => o.k === 'rock').length;
  const craters = OBSTACLES.filter(o => o.k === 'crater').length;
  document.getElementById('iW').textContent   = ARENA_W.toFixed(2) + 'm';
  document.getElementById('iL').textContent   = ARENA_L.toFixed(2) + 'm';
  document.getElementById('iSc').textContent  = (ARENA_SCALE * 100).toFixed(0) + '%';
  document.getElementById('iObs').textContent = `${rocks}r ${craters}c`;
  document.getElementById('iRx').textContent  = ROBOT_X.toFixed(2) + 'm';
  document.getElementById('iRy').textContent  = ROBOT_Y.toFixed(2) + 'm';
  document.getElementById('iHd').textContent  = (ROBOT_H * 180 / Math.PI).toFixed(1) + '°';
  document.getElementById('iPl').textContent  = ASTAR_PATH.length + ' cells';

  document.getElementById('arenaLabel').textContent =
    `Arena  ${ARENA_W.toFixed(2)}m × ${ARENA_L.toFixed(2)}m  ·  scale ${(ARENA_SCALE*100).toFixed(0)}%`;

  const pb = document.getElementById('phaseBadge');
  pb.textContent = PHASE === 'to_excavation' ? 'TO EXCAVATION' : 'TO DEPOSIT';
  pb.className   = 'phase-badge ' + (PHASE === 'to_excavation' ? 'exc' : 'dep');

  const action = MODEL || EXPERT;
  if (action) {
    setGauge('aLinFill', 'aLinVal', action[0]);
    setGauge('aAngFill', 'aAngVal', action[1]);
  }

  const fmt2 = (a) => a ? `lin=${a[0]>=0?'+':''}${a[0].toFixed(2)}  ang=${a[1]>=0?'+':''}${a[1].toFixed(2)}` : '—';
  document.getElementById('cExpert').textContent = fmt2(EXPERT);
  document.getElementById('cModel').textContent  = MODEL ? fmt2(MODEL) : 'no model loaded';

  document.getElementById('footerInfo').textContent =
    `v_max=${ROBOT_V_MAX}m/s  ω_max=${ROBOT_W_MAX}rad/s  proj=${PROJ_TIME}s`;
}

// ── Main render ───────────────────────────────────────────────────────────────

let _resizeTimer;
function render() {
  drawArena(document.getElementById('arenaCanvas'));
  drawCrop(document.getElementById('cropCanvas'));
  updateInfo();
}

function onResize() {
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(render, 60);
}

window.addEventListener('resize', onResize);
window.addEventListener('load', render);

// ── Controls ──────────────────────────────────────────────────────────────────

function reload() {
  const seed  = document.getElementById('seedInput').value;
  const phase = document.getElementById('phaseSelect').value;
  window.location.href = `/?seed=${seed}&phase=${phase}`;
}
document.getElementById('phaseSelect').value = PHASE;
document.getElementById('phaseSelect').addEventListener('change', reload);
document.addEventListener('keydown', e => {
  if (e.key === 'Enter') reload();
  if (e.key === 'ArrowRight') {
    const s = document.getElementById('seedInput');
    s.value = +s.value + 1; reload();
  }
  if (e.key === 'ArrowLeft') {
    const s = document.getElementById('seedInput');
    s.value = Math.max(0, +s.value - 1); reload();
  }
});
</script>
</body>
</html>"""


# ── HTTP server ────────────────────────────────────────────────────────────────

def _serve(cfg: dict, args: argparse.Namespace):
    checkpoint = args.checkpoint

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse
            qs    = parse_qs(urlparse(self.path).query)
            seed  = int(qs.get('seed',  ['0'])[0])
            phase = qs.get('phase', ['to_excavation'])[0]
            if phase not in ('to_excavation', 'to_deposit'):
                phase = 'to_excavation'

            scene        = _generate_scene(cfg, seed, phase)
            model_action = None
            if checkpoint and os.path.exists(checkpoint):
                model_action = _run_model(cfg, checkpoint,
                                          scene['terrain_crop'],
                                          scene['goal_map'],
                                          scene['heading'])

            html = _make_html(cfg, scene, model_action)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode())

    port   = args.port
    server = http.server.HTTPServer(('localhost', port), Handler)
    url    = f'http://localhost:{port}'
    print(f'[visualize]  {url}  (←→ arrow keys to step seeds,  Ctrl+C to stop)')
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--config',     default=os.path.join(_script_dir, 'config.yaml'))
    parser.add_argument('--checkpoint', default=os.path.join(_script_dir, 'checkpoints', 'best.pt'))
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--phase',      default='to_excavation',
                        choices=['to_excavation', 'to_deposit'])
    parser.add_argument('--port',       type=int, default=8766)
    args = parser.parse_args()
    cfg  = _load_cfg(args.config)
    _serve(cfg, args)


if __name__ == '__main__':
    main()
