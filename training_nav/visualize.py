"""Navigation policy live visualizer — Tesla-style arena inspector.

Runs a continuous differential-drive simulation (A* expert or trained model)
and streams state to a browser at ~10 Hz.  The path projection arc is drawn
as wide as the physical robot (0.75 m) matching Tesla's autopilot style.

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
import time
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


# ── Simulation state (shared between sim thread and HTTP handler) ─────────────

_sim_state: dict = {}
_sim_scene: dict = {}
_sim_lock  = threading.Lock()


# ── Full-arena A* (for display path in world coords) ─────────────────────────

def _build_full_cost_map(terrain: dict, cfg: dict) -> np.ndarray:
    """Build cost map on the full arena terrain (not a crop)."""
    pc = cfg['planner']
    from scipy.ndimage import sobel
    h = terrain['height']
    sx = sobel(h, axis=1); sy = sobel(h, axis=0)
    slope = np.sqrt(sx**2 + sy**2)
    cost = (1.0
            + pc['slope_weight']  * slope
            + pc['rock_weight']   * terrain['rocks']
            + pc['crater_weight'] * terrain['craters']
            + pc['wall_weight']   * terrain['walls'])
    return cost.astype(np.float32)


def _world_to_cell(x: float, y: float, cs: float):
    return int(y / cs), int(x / cs)   # (row, col)


# ── Model inference ───────────────────────────────────────────────────────────

def _load_model(cfg: dict, checkpoint_path: str):
    try:
        import torch
        from training_nav.model import NavPolicy
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = NavPolicy(cfg).to(device)
        ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        model._device = device
        print(f'[visualize] model loaded from {os.path.basename(checkpoint_path)}')
        return model
    except Exception as e:
        print(f'[visualize] no model ({e})')
        return None


def _infer(model, terrain_crop: np.ndarray, goal_map: np.ndarray, heading: float):
    try:
        import torch
        t5 = np.concatenate([terrain_crop, goal_map[None]], axis=0)
        t  = torch.from_numpy(t5).unsqueeze(0).to(model._device)
        h  = torch.tensor([[math.sin(heading), math.cos(heading)]],
                          dtype=torch.float32).to(model._device)
        with torch.inference_mode():
            out = model(t, h).squeeze(0).cpu().numpy()
        return float(out[0]), float(out[1])
    except Exception:
        return None


# ── Simulation loop ───────────────────────────────────────────────────────────

def _sim_loop(cfg: dict, checkpoint_path: str, init_seed: int, phase_req: list):
    """Continuous simulation loop — runs in a daemon thread."""
    rc         = cfg['robot']
    v_max      = rc['v_max']
    wheel_base = rc['wheel_base']
    cs         = cfg['terrain']['cell_size']
    DT         = 0.12     # sim step seconds (~8 Hz)
    REPLAN_N   = 15

    model    = (_load_model(cfg, checkpoint_path)
                if checkpoint_path and os.path.exists(checkpoint_path) else None)
    scene_id = 0
    seed     = init_seed

    while True:
        scene_id += 1
        rng    = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        seed  += 1

        phase   = phase_req[0]
        arena   = generate_arena(cfg, rng)
        terrain = build_terrain_maps(arena, cfg, np_rng)

        goal_zone       = _goal_zone_for_phase(arena, phase)
        rx, ry, heading = _sample_robot_pose(arena, phase, rng)

        cost_full = _build_full_cost_map(terrain, cfg)
        gx, gy    = goal_zone.centre()
        goal_rc   = _world_to_cell(gx, gy, cs)
        path_full = astar(cost_full, _world_to_cell(rx, ry, cs), goal_rc) or []

        def _rect(r):
            return {'x': r.x, 'y': r.y, 'w': r.w, 'h': r.h}

        with _sim_lock:
            _sim_scene.clear()
            _sim_scene.update({
                'scene_id':    scene_id,
                'phase':       phase,
                'arena_w':     arena.width,
                'arena_l':     arena.length,
                'arena_scale': arena.scale,
                't_rows':      terrain['rows'],
                't_cols':      terrain['cols'],
                'cell_size':   cs,
                'terrain_h':   terrain['height'].tolist(),
                'terrain_r':   terrain['rocks'].tolist(),
                'terrain_c':   terrain['craters'].tolist(),
                'terrain_w':   terrain['walls'].tolist(),
                'obstacles':   [{'x': o.x, 'y': o.y, 'd': o.diameter, 'k': o.kind}
                                for o in arena.obstacles],
                'zones': {
                    'start':      _rect(arena.start_zone),
                    'excavation': _rect(arena.excavation_zone),
                    'nav':        _rect(arena.nav_zone),
                    'deposit':    _rect(arena.deposit_zone),
                    'berm':       _rect(arena.berm_target),
                },
                'goal': {'x': gx, 'y': gy},
            })

        step = 0
        prev_rx, prev_ry = rx, ry
        stuck_steps = 0

        while True:
            t0 = time.monotonic()

            terrain_crop  = crop_robot_view(terrain, rx, ry, cfg)
            goal_map      = build_goal_heatmap(arena, goal_zone, rx, ry, cfg)
            expert_action = plan_action(terrain_crop, goal_map, heading, cfg)
            model_action  = _infer(model, terrain_crop, goal_map, heading) if model else None
            action        = model_action if model_action is not None else expert_action
            if action is None:
                action = (0.0, 0.0)

            left, right = action
            v_left  = left  * v_max
            v_right = right * v_max
            v       = (v_left + v_right) / 2.0
            omega   = (v_right - v_left) / wheel_base
            rx     += v * math.cos(heading) * DT
            ry     += v * math.sin(heading) * DT
            heading = (heading + omega * DT + math.pi) % (2 * math.pi) - math.pi

            if step % REPLAN_N == 0:
                path_full = astar(cost_full, _world_to_cell(rx, ry, cs), goal_rc) or []

            with _sim_lock:
                _sim_state.update({
                    'scene_id':    scene_id,
                    'rx':          rx,
                    'ry':          ry,
                    'heading':     heading,
                    'left':        left,
                    'right':       right,
                    'expert':      list(expert_action) if expert_action else None,
                    'model':       list(model_action)  if model_action  else None,
                    'using_model': model_action is not None,
                    'path_world':  [[r * cs, c * cs] for r, c in path_full],
                    'step':        step,
                    'phase':       phase,
                    'v':           v,
                    'omega':       omega,
                    'goal_x':      gx,
                    'goal_y':      gy,
                    'crop_h':      terrain_crop[0].tolist(),
                    'crop_r':      terrain_crop[1].tolist(),
                    'crop_c':      terrain_crop[2].tolist(),
                    'crop_w':      terrain_crop[3].tolist(),
                    'goal_map':    goal_map.tolist(),
                })

            step += 1

            if goal_zone.contains(rx, ry):
                time.sleep(1.5)
                break

            margin = 0.3
            if (rx < -margin or rx > arena.width  + margin or
                    ry < -margin or ry > arena.length + margin):
                break

            if step % 20 == 0:
                dist = math.hypot(rx - prev_rx, ry - prev_ry)
                if dist < 0.05:
                    stuck_steps += 1
                    if stuck_steps >= 4:
                        break
                else:
                    stuck_steps = 0
                prev_rx, prev_ry = rx, ry

            time.sleep(max(0.0, DT - (time.monotonic() - t0)))


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCTANE — Nav Policy</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: #0d1117;
  color: #c9d1d9;
  font-family: -apple-system, 'SF Pro Display', 'Segoe UI', sans-serif;
  font-size: 13px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── Top bar ── */
header {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 9px 20px;
  background: #161b22;
  border-bottom: 1px solid #21262d;
  flex-shrink: 0;
}
.logo {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: .08em;
  color: #f0f6fc;
}
.logo span { color: #148ce8; }
.badge {
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.badge-model { background: rgba(20,140,232,.15); color: #58a6ff; border: 1px solid rgba(20,140,232,.4); }
.badge-expert { background: rgba(255,215,0,.1); color: #d4a820; border: 1px solid rgba(255,215,0,.3); }
#statusDot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #3fb950; box-shadow: 0 0 6px #3fb950;
  flex-shrink: 0;
}
.header-right { margin-left: auto; display: flex; gap: 12px; align-items: center; }
.hval { font-size: 11px; color: #8b949e; }
.hval strong { color: #c9d1d9; }

/* ── Workspace ── */
.workspace {
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 0;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

/* ── Arena canvas ── */
.arena-wrap {
  position: relative;
  background: #090d14;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
#arenaCanvas { display: block; }
.arena-overlay {
  position: absolute;
  top: 12px; left: 14px;
  font-size: 10px;
  color: #484f58;
  letter-spacing: .08em;
  text-transform: uppercase;
  pointer-events: none;
}
.arena-overlay strong { color: #8b949e; }

/* ── Side panel ── */
.side {
  display: flex;
  flex-direction: column;
  background: #161b22;
  border-left: 1px solid #21262d;
  overflow-y: auto;
}
.panel {
  padding: 14px 16px;
  border-bottom: 1px solid #21262d;
}
.panel-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #484f58;
  margin-bottom: 12px;
}

/* Sensor view */
.crop-wrap { display: flex; justify-content: center; }
#cropCanvas { border-radius: 4px; display: block; }

/* Motor gauges */
.motor-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.motor-label {
  width: 14px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .06em;
  color: #8b949e;
  flex-shrink: 0;
}
.motor-track {
  flex: 1;
  height: 8px;
  background: #0d1117;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid #21262d;
}
.motor-fill {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, #0d6efd, #148ce8);
  transition: width .1s;
}
.motor-val {
  width: 36px;
  text-align: right;
  font-size: 12px;
  font-weight: 600;
  color: #58a6ff;
  font-variant-numeric: tabular-nums;
}
.motor-speed {
  font-size: 10px;
  color: #484f58;
  text-align: right;
  margin-top: -6px;
  margin-bottom: 8px;
}

.divider { height: 1px; background: #21262d; margin: 10px 0; }

/* Comparison block */
.cmp-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 5px;
  font-size: 11px;
}
.cmp-key { color: #484f58; }
.cmp-val-expert { color: #d4a820; font-variant-numeric: tabular-nums; }
.cmp-val-model  { color: #58a6ff; font-variant-numeric: tabular-nums; }

/* Info grid */
.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 12px;
}
.info-item { display: flex; flex-direction: column; gap: 1px; }
.info-key { font-size: 9px; text-transform: uppercase; letter-spacing: .08em; color: #484f58; }
.info-val { font-size: 12px; color: #c9d1d9; font-variant-numeric: tabular-nums; }

/* ── Footer ── */
footer {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 7px 20px;
  background: #161b22;
  border-top: 1px solid #21262d;
  flex-shrink: 0;
}
select, input[type=number] {
  background: #0d1117;
  color: #c9d1d9;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 12px;
}
button {
  background: #21262d;
  color: #c9d1d9;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 4px 14px;
  font-size: 12px;
  cursor: pointer;
}
button:hover { background: #30363d; }
.footer-right { margin-left: auto; font-size: 11px; color: #484f58; }
</style>
</head>
<body>

<header>
  <div class="logo">OCT<span>A</span>NE</div>
  <span style="color:#484f58;font-size:11px">NAVIGATION POLICY</span>
  <span class="badge badge-expert" id="modeBadge">A* EXPERT</span>
  <div id="statusDot"></div>
  <div class="header-right">
    <span class="hval">v <strong id="hVel">—</strong> m/s</span>
    <span class="hval">ω <strong id="hOmega">—</strong> rad/s</span>
    <span class="hval">hdg <strong id="hHdg">—</strong>°</span>
    <span class="hval" id="hPos">—</span>
  </div>
</header>

<div class="workspace">
  <div class="arena-wrap" id="arenaWrap">
    <canvas id="arenaCanvas"></canvas>
    <div class="arena-overlay" id="arenaOverlay">—</div>
  </div>

  <div class="side">

    <div class="panel">
      <div class="panel-title">Sensor view</div>
      <div class="crop-wrap">
        <canvas id="cropCanvas" width="268" height="268"></canvas>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Drive motors</div>

      <div class="motor-row">
        <span class="motor-label">L</span>
        <div class="motor-track"><div class="motor-fill" id="motorLFill" style="width:0%"></div></div>
        <span class="motor-val" id="motorLVal">—</span>
      </div>
      <div class="motor-speed" id="motorLSpd">—</div>

      <div class="motor-row">
        <span class="motor-label">R</span>
        <div class="motor-track"><div class="motor-fill" id="motorRFill" style="width:0%"></div></div>
        <span class="motor-val" id="motorRVal">—</span>
      </div>
      <div class="motor-speed" id="motorRSpd">—</div>

      <div class="divider"></div>

      <div class="cmp-row">
        <span class="cmp-key">Expert (A*)</span>
        <span class="cmp-val-expert" id="cmpExpert">—</span>
      </div>
      <div class="cmp-row">
        <span class="cmp-key">Model</span>
        <span class="cmp-val-model" id="cmpModel">—</span>
      </div>
    </div>

    <div class="panel" style="flex:1">
      <div class="panel-title">Arena info</div>
      <div class="info-grid">
        <div class="info-item"><span class="info-key">Width</span><span class="info-val" id="iW">—</span></div>
        <div class="info-item"><span class="info-key">Length</span><span class="info-val" id="iL">—</span></div>
        <div class="info-item"><span class="info-key">Scale</span><span class="info-val" id="iSc">—</span></div>
        <div class="info-item"><span class="info-key">Obstacles</span><span class="info-val" id="iObs">—</span></div>
        <div class="info-item"><span class="info-key">Robot X</span><span class="info-val" id="iRx">—</span></div>
        <div class="info-item"><span class="info-key">Robot Y</span><span class="info-val" id="iRy">—</span></div>
        <div class="info-item"><span class="info-key">Phase</span><span class="info-val" id="iPhase">—</span></div>
        <div class="info-item"><span class="info-key">Step</span><span class="info-val" id="iStep">—</span></div>
      </div>
    </div>

  </div>
</div>

<footer>
  <label style="font-size:11px;color:#8b949e">Phase
    <select id="phaseSelect">
      <option value="to_excavation">to_excavation</option>
      <option value="to_deposit">to_deposit</option>
    </select>
  </label>
  <button onclick="resetSim()">New Arena</button>
  <div class="footer-right" id="footerRight">—</div>
</footer>

<script>
// ── Constants injected server-side ────────────────────────────────────────────
const V_MAX       = /*V_MAX*/0.652;
const WHEEL_BASE  = /*WHEEL_BASE*/0.483;
const ROBOT_W     = /*ROBOT_W*/0.75;
const ROBOT_L     = /*ROBOT_L*/1.5;
const PROJ_TIME   = /*PROJ_TIME*/3.0;
const DIFF_LIMIT  = /*DIFF_LIMIT*/0.8;
const GS          = /*GS*/60;
const VIEW_W      = /*VIEW_W*/6.0;
const VIEW_H      = /*VIEW_H*/6.0;
const VIEW_L      = VIEW_H;   // alias — robot-view canvas height extent = VIEW_H

// ── State ─────────────────────────────────────────────────────────────────────
let scene    = null;   // full scene data (fetched once per arena)
let state    = null;   // latest live state
let lastSceneId = -1;

// ── Polling ───────────────────────────────────────────────────────────────────
async function pollState() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();
    state = s;

    if (s.scene_id !== lastSceneId) {
      lastSceneId = s.scene_id;
      const sr = await fetch('/api/scene');
      scene = await sr.json();
    }
  } catch(e) { /* server may be restarting */ }
}

setInterval(pollState, 100);   // 10 Hz

// ── Render loop ───────────────────────────────────────────────────────────────
function raf() {
  requestAnimationFrame(raf);
  if (!state || !scene) return;
  drawArena();
  drawCrop();
  updateUI();
}
requestAnimationFrame(raf);

// ── Coordinate helpers ────────────────────────────────────────────────────────
// World → canvas  (Y flipped: world y=0 at canvas bottom)
function w2c(wx, wy, W, H) {
  return { x: wx / scene.arena_w * W,
           y: (1 - wy / scene.arena_l) * H };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t)    { return a + (b - a) * t; }

// ── Kinematic projection (differential drive) ─────────────────────────────────
function projectArc(rx, ry, heading, left, right) {
  const DT    = 0.06;
  const steps = Math.round(PROJ_TIME / DT);
  const pts   = [{x: rx, y: ry}];
  let x = rx, y = ry, h = heading;
  for (let i = 0; i < steps; i++) {
    const vL = left  * V_MAX;
    const vR = right * V_MAX;
    const v  = (vL + vR) / 2;
    const w  = (vR - vL) / WHEEL_BASE;
    x += v * Math.cos(h) * DT;
    y += v * Math.sin(h) * DT;
    h += w * DT;
    pts.push({x, y});
  }
  return pts;
}

// Same in robot-view grid coords (robot at centre = GS/2, GS/2)
function projectArcGrid(heading, left, right) {
  const CSX   = VIEW_W / GS;
  const CSY   = VIEW_H / GS;
  const DT    = 0.06;
  const steps = Math.round(PROJ_TIME / DT);
  const half  = GS / 2;
  const pts   = [{col: half, row: half}];
  let col = half, row = half, h = heading;
  for (let i = 0; i < steps; i++) {
    const vL = left  * V_MAX;
    const vR = right * V_MAX;
    const v  = (vL + vR) / 2;
    const w  = (vR - vL) / WHEEL_BASE;
    col +=  v * Math.cos(h) * DT / CSX;
    row += -v * Math.sin(h) * DT / CSY;  // world +Y = canvas -row
    h   +=  w * DT;
    if (row < 0 || row >= GS || col < 0 || col >= GS) break;
    pts.push({col, row});
  }
  return pts;
}

// ── Vehicle-width arc drawing ─────────────────────────────────────────────────
// pts: array of canvas {x, y}
// halfWidthPx: half the robot's physical width in canvas pixels
function drawVehicleArc(ctx, pts, halfWidthPx) {
  if (pts.length < 2) return;

  const left = [], right = [];
  for (let i = 0; i < pts.length; i++) {
    let dx, dy;
    if (i === 0) {
      dx = pts[1].x - pts[0].x; dy = pts[1].y - pts[0].y;
    } else if (i === pts.length - 1) {
      dx = pts[i].x - pts[i-1].x; dy = pts[i].y - pts[i-1].y;
    } else {
      dx = pts[i+1].x - pts[i-1].x; dy = pts[i+1].y - pts[i-1].y;
    }
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;   // perpendicular (CCW)
    const ny =  dx / len;
    left.push( { x: pts[i].x + nx * halfWidthPx, y: pts[i].y + ny * halfWidthPx });
    right.push({ x: pts[i].x - nx * halfWidthPx, y: pts[i].y - ny * halfWidthPx });
  }

  // Gradient from bright-blue at robot to transparent at horizon
  const n    = pts.length;
  const grad = ctx.createLinearGradient(pts[0].x, pts[0].y, pts[n-1].x, pts[n-1].y);
  grad.addColorStop(0,    'rgba(20,140,232,0.70)');
  grad.addColorStop(0.35, 'rgba(20,140,232,0.40)');
  grad.addColorStop(1,    'rgba(20,140,232,0.00)');

  ctx.beginPath();
  ctx.moveTo(left[0].x, left[0].y);
  for (const p of left)  ctx.lineTo(p.x, p.y);
  for (let i = right.length-1; i >= 0; i--) ctx.lineTo(right[i].x, right[i].y);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Bright centreline
  const cGrad = ctx.createLinearGradient(pts[0].x, pts[0].y, pts[n-1].x, pts[n-1].y);
  cGrad.addColorStop(0,   'rgba(88,166,255,0.95)');
  cGrad.addColorStop(0.5, 'rgba(88,166,255,0.50)');
  cGrad.addColorStop(1,   'rgba(88,166,255,0.00)');
  ctx.beginPath();
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = cGrad;
  ctx.lineWidth   = 1.5;
  ctx.lineJoin    = 'round';
  ctx.stroke();
}

// ── Draw robot rectangle (Tesla style) ───────────────────────────────────────
// cx, cy: canvas centre; lPx, wPx: length/width in pixels; heading in radians
function drawRobotRect(ctx, cx, cy, lPx, wPx, heading) {
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-heading);  // canvas Y is flipped, so negate heading

  // Outer glow
  ctx.shadowColor = '#148ce8';
  ctx.shadowBlur  = 18;
  ctx.fillStyle   = 'rgba(20,140,232,0.08)';
  ctx.fillRect(-lPx/2 - 4, -wPx/2 - 4, lPx + 8, wPx + 8);
  ctx.shadowBlur = 0;

  // Body
  ctx.fillStyle   = '#dde1e7';
  ctx.strokeStyle = '#148ce8';
  ctx.lineWidth   = 1.5;
  ctx.fillRect(-lPx/2, -wPx/2, lPx, wPx);
  ctx.strokeRect(-lPx/2, -wPx/2, lPx, wPx);

  // Front blue panel (right side in robot frame = +X = forward)
  ctx.fillStyle = '#148ce8';
  ctx.fillRect(lPx/2 - Math.max(3, lPx*0.08), -wPx/2, Math.max(3, lPx*0.08), wPx);

  ctx.restore();
}

// ── Heatmap colour ────────────────────────────────────────────────────────────
function heatColor(v, lo, hi) {
  const t = clamp((v - lo) / (hi - lo + 1e-9), 0, 1);
  if (t < 0.5) {
    const s = t * 2;
    return `rgb(${Math.round(lerp(10,0,s))},${Math.round(lerp(20,120,s))},${Math.round(lerp(40,160,s))})`;
  }
  const s = (t - 0.5) * 2;
  return `rgb(${Math.round(lerp(0,200,s))},${Math.round(lerp(120,160,s))},${Math.round(lerp(160,20,s))})`;
}

function flatMinMax(arr2d) {
  let mn = Infinity, mx = -Infinity;
  for (const row of arr2d) for (const v of row) { if(v<mn)mn=v; if(v>mx)mx=v; }
  return [mn, mx];
}

// ── Full Arena Canvas ─────────────────────────────────────────────────────────
function drawArena() {
  const wrap = document.getElementById('arenaWrap');
  const canvas = document.getElementById('arenaCanvas');
  const maxW = wrap.clientWidth, maxH = wrap.clientHeight;
  const ratio = scene.arena_w / scene.arena_l;
  let W, H;
  if (maxW / maxH > ratio) { H = maxH; W = H * ratio; }
  else                      { W = maxW; H = W / ratio; }
  if (canvas.width !== Math.floor(W) || canvas.height !== Math.floor(H)) {
    canvas.width  = Math.floor(W);
    canvas.height = Math.floor(H);
  }

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  // ── Background ──
  ctx.fillStyle = '#090d14';
  ctx.fillRect(0, 0, W, H);

  // ── Subtle grid ──
  const GRID_M = 0.5;
  ctx.strokeStyle = 'rgba(255,255,255,0.025)';
  ctx.lineWidth   = 0.5;
  for (let gx = 0; gx <= scene.arena_w; gx += GRID_M) {
    const px = gx / scene.arena_w * W;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
  }
  for (let gy = 0; gy <= scene.arena_l; gy += GRID_M) {
    const py = (1 - gy / scene.arena_l) * H;
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
  }

  // ── Height map (muted) ──
  const [hMin, hMax] = flatMinMax(scene.terrain_h);
  const rows = scene.t_rows, cols = scene.t_cols;
  const cellW = W / cols, cellH = H / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = (scene.terrain_h[r][c] - hMin) / (hMax - hMin + 1e-9);
      const v = Math.round(lerp(6, 28, t));
      ctx.fillStyle = `rgb(${v},${v+1},${v+3})`;
      ctx.fillRect(c * cellW, (rows-1-r) * cellH, cellW+1, cellH+1);
    }
  }

  // ── Zone overlays ──
  const ZC = {
    start:      'rgba(30,100,60,.28)',
    excavation: 'rgba(100,60,0,.22)',
    nav:        'rgba(10,40,90,.22)',
    deposit:    'rgba(70,10,90,.22)',
    berm:       'rgba(180,140,0,.15)',
  };
  const ZL = { start:'START', excavation:'EXCAVATION', nav:'NAV', deposit:'DEPOSIT', berm:'BERM' };
  const ZBorderC = {
    start:'rgba(30,100,60,.5)', excavation:'rgba(140,80,0,.45)',
    nav:'rgba(20,70,160,.45)', deposit:'rgba(100,20,130,.45)', berm:'rgba(200,160,0,.5)'
  };
  for (const [key, color] of Object.entries(ZC)) {
    const z = scene.zones[key];
    const tl = w2c(z.x,       z.y + z.h, W, H);
    const br = w2c(z.x + z.w, z.y,       W, H);
    const zw = br.x - tl.x, zh = br.y - tl.y;

    ctx.fillStyle = color;
    ctx.fillRect(tl.x, tl.y, zw, zh);

    ctx.strokeStyle = ZBorderC[key];
    ctx.lineWidth   = key === 'berm' ? 1.5 : 1;
    if (key === 'berm') ctx.setLineDash([5, 3]);
    ctx.strokeRect(tl.x + .5, tl.y + .5, zw - 1, zh - 1);
    ctx.setLineDash([]);

    ctx.fillStyle   = 'rgba(255,255,255,0.12)';
    ctx.font        = `${clamp(Math.floor(W/40), 7, 11)}px -apple-system,sans-serif`;
    ctx.textAlign   = 'center';
    ctx.textBaseline= 'middle';
    ctx.fillText(ZL[key], (tl.x + br.x)/2, (tl.y + br.y)/2);
  }

  // ── Obstacles ──
  for (const o of scene.obstacles) {
    const p  = w2c(o.x, o.y, W, H);
    const rr = Math.max(3, (o.d / 2) / scene.arena_w * W);
    ctx.beginPath(); ctx.arc(p.x, p.y, rr, 0, 2*Math.PI);
    if (o.k === 'rock') {
      ctx.fillStyle   = 'rgba(120,120,130,0.6)';
      ctx.strokeStyle = 'rgba(160,160,175,0.8)';
    } else {
      ctx.fillStyle   = 'rgba(55,40,40,0.7)';
      ctx.strokeStyle = 'rgba(90,60,60,0.9)';
    }
    ctx.fill(); ctx.lineWidth = 1; ctx.stroke();
  }

  // ── Sensor view rectangle ──
  const vx0 = state.rx - VIEW_W / 2;
  const vy0 = state.ry - VIEW_H / 2;
  const vTL = w2c(vx0,          vy0 + VIEW_H, W, H);
  const vBR = w2c(vx0 + VIEW_W, vy0,          W, H);
  const vRW = vBR.x - vTL.x, vRH = vBR.y - vTL.y;

  ctx.fillStyle   = 'rgba(255,255,255,0.025)';
  ctx.fillRect(vTL.x, vTL.y, vRW, vRH);
  ctx.strokeStyle = 'rgba(255,255,255,0.30)';
  ctx.lineWidth   = 0.75;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(vTL.x + .5, vTL.y + .5, vRW - 1, vRH - 1);
  ctx.setLineDash([]);

  // ── A* world path ──
  if (state.path_world && state.path_world.length > 1) {
    ctx.shadowColor = 'rgba(255,215,0,0.4)';
    ctx.shadowBlur  = 4;
    ctx.beginPath();
    state.path_world.forEach(([wy, wx], i) => {
      const p = w2c(wx, wy, W, H);
      i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y);
    });
    ctx.strokeStyle = 'rgba(212,168,32,0.8)';
    ctx.lineWidth   = 1.5;
    ctx.lineJoin    = 'round';
    ctx.stroke();
    ctx.shadowBlur  = 0;
  }

  // ── Tesla vehicle-width arc ──
  const arcPts = projectArc(state.rx, state.ry, state.heading, state.left, state.right);
  const cArcPts = arcPts.map(p => w2c(p.x, p.y, W, H));
  const halfWPx = (ROBOT_W / scene.arena_w) * W / 2;
  drawVehicleArc(ctx, cArcPts, halfWPx);

  // ── Robot rectangle ──
  const rp  = w2c(state.rx, state.ry, W, H);
  const lPx = (ROBOT_L / scene.arena_l) * H;
  const wPx = (ROBOT_W / scene.arena_w) * W;
  drawRobotRect(ctx, rp.x, rp.y, lPx, wPx, state.heading);
}

// ── Robot view (sensor crop) canvas ──────────────────────────────────────────
function drawCrop() {
  // Fetch fresh crop from server state — server provides it in /api/state
  if (!state.crop_h) return;
  const canvas = document.getElementById('cropCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cW = W / GS, cH = H / GS;
  ctx.clearRect(0, 0, W, H);

  const [hMin, hMax] = flatMinMax(state.crop_h);
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    ctx.fillStyle = heatColor(state.crop_h[r][c], hMin, hMax);
    ctx.fillRect(c*cW, r*cH, cW+.5, cH+.5);
  }
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    const v = state.crop_r[r][c];
    if (v > 0.05) { ctx.fillStyle = `rgba(120,120,135,${v*0.6})`; ctx.fillRect(c*cW,r*cH,cW+.5,cH+.5); }
  }
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    const v = state.crop_c[r][c];
    if (v > 0.05) { ctx.fillStyle = `rgba(50,35,35,${v*0.7})`; ctx.fillRect(c*cW,r*cH,cW+.5,cH+.5); }
  }
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    if (state.crop_w[r][c] > 0.5) { ctx.fillStyle = 'rgba(50,50,60,0.9)'; ctx.fillRect(c*cW,r*cH,cW+.5,cH+.5); }
  }
  for (let r = 0; r < GS; r++) for (let c = 0; c < GS; c++) {
    const v = state.goal_map[r][c];
    if (v > 0.02) { ctx.fillStyle = `rgba(46,160,67,${v*0.45})`; ctx.fillRect(c*cW,r*cH,cW+.5,cH+.5); }
  }

  // A* path on crop (in crop grid coords — convert from full-arena path)
  // Draw vehicle-width arc on crop
  const gridPts = projectArcGrid(state.heading, state.left, state.right);
  const cArcPts = gridPts.map(p => ({ x: (p.col+0.5)*cW, y: (p.row+0.5)*cH }));
  const halfWPx = (GS / VIEW_W * ROBOT_W) / 2 * cW;
  drawVehicleArc(ctx, cArcPts, halfWPx);

  // Robot at centre
  const cx = (GS/2 + 0.5) * cW, cy = (GS/2 + 0.5) * cH;
  const lPx = (ROBOT_L / VIEW_L) * H;
  const wPx = (ROBOT_W / VIEW_W) * W;
  drawRobotRect(ctx, cx, cy, lPx, wPx, state.heading);
}

// ── UI panel updates ──────────────────────────────────────────────────────────
function setMotor(fillId, valId, spdId, v) {
  document.getElementById(fillId).style.width = (v * 100).toFixed(1) + '%';
  document.getElementById(valId).textContent  = v.toFixed(2);
  const rpm = (v * 35).toFixed(1);
  const mps  = (v * V_MAX).toFixed(3);
  document.getElementById(spdId).textContent  = `${rpm} RPM  ·  ${mps} m/s`;
}

function updateUI() {
  document.getElementById('hVel').textContent   = state.v.toFixed(3);
  document.getElementById('hOmega').textContent = state.omega.toFixed(3);
  document.getElementById('hHdg').textContent   = (state.heading * 180/Math.PI).toFixed(1);
  document.getElementById('hPos').textContent   = `(${state.rx.toFixed(2)}, ${state.ry.toFixed(2)}) m`;

  setMotor('motorLFill', 'motorLVal', 'motorLSpd', state.left);
  setMotor('motorRFill', 'motorRVal', 'motorRSpd', state.right);

  const fmt = a => a ? `L=${a[0].toFixed(2)} R=${a[1].toFixed(2)}` : '—';
  document.getElementById('cmpExpert').textContent = fmt(state.expert);
  document.getElementById('cmpModel').textContent  = state.model ? fmt(state.model) : 'no model';

  const badge = document.getElementById('modeBadge');
  if (state.using_model) {
    badge.textContent = 'MODEL'; badge.className = 'badge badge-model';
  } else {
    badge.textContent = 'A* EXPERT'; badge.className = 'badge badge-expert';
  }

  if (scene) {
    const rocks   = scene.obstacles.filter(o=>o.k==='rock').length;
    const craters = scene.obstacles.filter(o=>o.k==='crater').length;
    document.getElementById('iW').textContent     = scene.arena_w.toFixed(2) + ' m';
    document.getElementById('iL').textContent     = scene.arena_l.toFixed(2) + ' m';
    document.getElementById('iSc').textContent    = (scene.arena_scale*100).toFixed(0) + '%';
    document.getElementById('iObs').textContent   = `${rocks}r ${craters}c`;
    document.getElementById('iRx').textContent    = state.rx.toFixed(2) + ' m';
    document.getElementById('iRy').textContent    = state.ry.toFixed(2) + ' m';
    document.getElementById('iPhase').textContent = state.phase;
    document.getElementById('iStep').textContent  = state.step;

    document.getElementById('arenaOverlay').innerHTML =
      `<strong>${scene.arena_w.toFixed(2)}m × ${scene.arena_l.toFixed(2)}m</strong>`+
      `  ·  scale ${(scene.arena_scale*100).toFixed(0)}%`+
      `  ·  scene #${scene.scene_id}`;

    document.getElementById('footerRight').textContent =
      `v_max=${V_MAX} m/s  ·  wheel_base=${WHEEL_BASE} m  ·  diff_limit=${DIFF_LIMIT}`;
  }
}

// ── Controls ──────────────────────────────────────────────────────────────────
async function resetSim() {
  const phase = document.getElementById('phaseSelect').value;
  await fetch(`/api/reset?phase=${phase}`);
}
document.addEventListener('keydown', e => {
  if (e.key === 'r' || e.key === 'R') resetSim();
});

// Kick off first poll immediately
pollState();
</script>
</body>
</html>
"""


# ── HTTP server ────────────────────────────────────────────────────────────────

def _serve(cfg: dict, args: argparse.Namespace, phase_req: list):
    rc = cfg['robot']

    constants = (
        _HTML
        .replace('/*V_MAX*/',      str(rc['v_max']))
        .replace('/*WHEEL_BASE*/', str(rc['wheel_base']))
        .replace('/*ROBOT_W*/',    str(rc['robot_width']))
        .replace('/*ROBOT_L*/',    str(rc['robot_length']))
        .replace('/*PROJ_TIME*/',  str(rc['projection_time']))
        .replace('/*DIFF_LIMIT*/', str(rc['diff_limit']))
        .replace('/*GS*/',         str(cfg['terrain']['grid_size']))
        .replace('/*VIEW_W*/',     str(cfg['terrain']['view_width']))
        .replace('/*VIEW_H*/',     str(cfg['terrain']['view_height']))
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(self.path)
            qs     = parse_qs(parsed.query)
            path   = parsed.path

            if path == '/':
                self._send_html(constants)

            elif path == '/api/state':
                with _sim_lock:
                    s = dict(_sim_state)
                # Add crop data to state
                if s:
                    try:
                        from training_nav.arena import crop_robot_view, build_goal_heatmap
                        from training_nav.arena import generate_arena, build_terrain_maps
                        # We need the current terrain — pull from scene
                        with _sim_lock:
                            sc = dict(_sim_scene)
                        # We can't easily re-get the terrain here; include crop in sim state
                        pass
                    except Exception:
                        pass
                self._send_json(s)

            elif path == '/api/scene':
                with _sim_lock:
                    sc = dict(_sim_scene)
                self._send_json(sc)

            elif path == '/api/reset':
                new_phase = qs.get('phase', ['to_excavation'])[0]
                phase_req[0] = new_phase
                self._send_json({'ok': True})

            else:
                self.send_error(404)

        def _send_html(self, body: str):
            enc = body.encode()
            self.send_response(200)
            self.send_header('Content-Type',   'text/html; charset=utf-8')
            self.send_header('Content-Length', len(enc))
            self.end_headers()
            self.wfile.write(enc)

        def _send_json(self, obj):
            enc = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header('Content-Type',   'application/json')
            self.send_header('Content-Length', len(enc))
            self.send_header('Cache-Control',  'no-cache')
            self.end_headers()
            self.wfile.write(enc)

    port   = args.port
    server = http.server.HTTPServer(('localhost', port), Handler)
    url    = f'http://localhost:{port}'
    print(f'[visualize]  {url}  (press R to reset arena,  Ctrl+C to stop)')
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

    phase_req = [args.phase]

    sim_thread = threading.Thread(
        target=_sim_loop,
        args=(cfg, args.checkpoint, args.seed, phase_req),
        daemon=True,
    )
    sim_thread.start()
    _serve(cfg, args, phase_req)


if __name__ == '__main__':
    main()
