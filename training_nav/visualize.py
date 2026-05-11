"""Navigation debug visualizer.

Generates a random arena, runs A* to get the expert path, runs the trained
model to get predicted actions, and displays both side by side in the browser.

Usage:
  python training_nav/visualize.py
  python training_nav/visualize.py --checkpoint training_nav/checkpoints/best.pt
  python training_nav/visualize.py --seed 42 --phase to_excavation
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
    ArenaConfig, build_goal_heatmap, build_terrain_maps,
    crop_robot_view, generate_arena,
)
from training_nav.dataset import _goal_zone_for_phase, _sample_robot_pose
from training_nav.planner import _build_cost_map, astar, extract_action, plan_action


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _generate_scene(cfg: dict, seed: int, phase: str):
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    arena   = generate_arena(cfg, rng)
    terrain = build_terrain_maps(arena, cfg, np_rng)

    goal_zone       = _goal_zone_for_phase(arena, phase)
    rx, ry, heading = _sample_robot_pose(arena, phase, rng)

    terrain_crop = crop_robot_view(terrain, rx, ry, cfg)
    goal_map     = build_goal_heatmap(arena, goal_zone, rx, ry, cfg)

    cost = _build_cost_map(terrain_crop, cfg)
    gs   = cfg['terrain']['grid_size']
    half = gs // 2
    goal_idx  = np.unravel_index(np.argmax(goal_map), goal_map.shape)
    astar_path = astar(cost, (half, half), (int(goal_idx[0]), int(goal_idx[1])))

    return arena, terrain, terrain_crop, goal_map, astar_path, rx, ry, heading, cost


def _grid_to_json(arr: np.ndarray) -> list:
    return arr.tolist()


def _path_to_json(path) -> list:
    if path is None:
        return []
    return [[r, c] for r, c in path]


def _make_html(cfg: dict, scene_data: dict, model_action=None) -> str:
    gs = cfg['terrain']['grid_size']
    vw = cfg['terrain']['view_width']
    vh = cfg['terrain']['view_height']

    expert = scene_data.get('expert_action')
    pred   = model_action

    expert_str = f"linear={expert[0]:.2f}, angular={expert[1]:.2f}" if expert else "N/A"
    pred_str   = f"linear={pred[0]:.2f}, angular={pred[1]:.2f}"     if pred   else "No model loaded"

    data_js = f"""
const GRID_SIZE   = {gs};
const VIEW_WIDTH  = {vw};
const VIEW_HEIGHT = {vh};
const HEIGHT_MAP  = {json.dumps(scene_data['height'])};
const ROCKS_MAP   = {json.dumps(scene_data['rocks'])};
const CRATERS_MAP = {json.dumps(scene_data['craters'])};
const WALLS_MAP   = {json.dumps(scene_data['walls'])};
const GOAL_MAP    = {json.dumps(scene_data['goal'])};
const COST_MAP    = {json.dumps(scene_data['cost'])};
const ASTAR_PATH  = {json.dumps(scene_data['path'])};
const ROBOT_ROW   = {gs // 2};
const ROBOT_COL   = {gs // 2};
const ROBOT_HEADING = {scene_data['heading']};
const EXPERT_ACTION = {json.dumps(list(expert) if expert else None)};
const MODEL_ACTION  = {json.dumps(list(pred)   if pred   else None)};
const ARENA_WIDTH   = {scene_data['arena_width']:.2f};
const ARENA_LENGTH  = {scene_data['arena_length']:.2f};
const ARENA_SCALE   = {scene_data['arena_scale']:.2f};
const PHASE         = "{scene_data['phase']}";
"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Nav Policy Visualizer</title>
<style>
  body {{ background:#0d0d0d; color:#e0e0e0; font-family:monospace; margin:0; padding:16px; }}
  h1   {{ color:#4fc3f7; margin:0 0 12px; font-size:1.1rem; }}
  .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .panel {{ background:#1a1a1a; border-radius:8px; padding:12px; }}
  canvas {{ display:block; image-rendering:pixelated; }}
  .label  {{ font-size:0.75rem; color:#888; margin-bottom:4px; }}
  .action {{ font-size:0.85rem; margin-top:8px; }}
  .action span {{ color:#4fc3f7; }}
  select, button {{ background:#222; color:#e0e0e0; border:1px solid #444;
                    padding:4px 10px; border-radius:4px; cursor:pointer; }}
  .controls {{ display:flex; gap:8px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }}
  .arrow-canvas {{ display:block; }}
</style>
</head>
<body>
<h1>Navigation Policy Visualizer</h1>
<div class="controls">
  <label>Phase: <select id="phaseSelect">
    <option value="to_excavation">to_excavation</option>
    <option value="to_deposit">to_deposit</option>
  </select></label>
  <label>Seed: <input id="seedInput" type="number" value="0" style="width:70px;background:#222;color:#e0e0e0;border:1px solid #444;border-radius:4px;padding:4px"></label>
  <button onclick="reload()">Regenerate</button>
</div>
<div class="row" id="panels">
  <div class="panel">
    <div class="label">Height Map + A* Path</div>
    <canvas id="heightCanvas" width="300" height="300"></canvas>
    <div class="action">Expert: <span>{expert_str}</span></div>
  </div>
  <div class="panel">
    <div class="label">Obstacle Map (rocks + craters + walls)</div>
    <canvas id="obstacleCanvas" width="300" height="300"></canvas>
  </div>
  <div class="panel">
    <div class="label">Goal Heatmap + Cost</div>
    <canvas id="goalCanvas" width="300" height="300"></canvas>
  </div>
  <div class="panel">
    <div class="label">Model Action vs Expert</div>
    <canvas id="actionCanvas" class="arrow-canvas" width="200" height="200"></canvas>
    <div class="action">Model: <span id="modelActionText">{pred_str}</span></div>
    <div class="action" style="color:#888;font-size:0.7rem">
      Arena: {{ARENA_WIDTH}}m × {{ARENA_LENGTH}}m (scale {{ARENA_SCALE:.2f}}) | Phase: {scene_data['phase']}
    </div>
  </div>
</div>
<script>
{data_js}

function lerp(a,b,t){{ return a+(b-a)*t; }}
function heatColor(v,lo,hi){{
  const t = Math.max(0,Math.min(1,(v-lo)/(hi-lo+1e-9)));
  const r = Math.round(lerp(0,255,t));
  const b = Math.round(lerp(255,0,t));
  return `rgb(${{r}},0,${{b}})`;
}}
function greenRed(v){{
  return `rgb(${{Math.round(255*v)}},0,0)`;
}}

function drawGrid(canvas, grid, colorFn) {{
  const ctx = canvas.getContext('2d');
  const cw = canvas.width / GRID_SIZE;
  const ch = canvas.height / GRID_SIZE;
  let flat = grid.flat();
  let mn = Math.min(...flat), mx = Math.max(...flat);
  for(let r=0;r<GRID_SIZE;r++) for(let c=0;c<GRID_SIZE;c++) {{
    ctx.fillStyle = colorFn(grid[r][c], mn, mx);
    ctx.fillRect(c*cw, r*ch, cw, ch);
  }}
}}

function drawPath(canvas) {{
  const ctx = canvas.getContext('2d');
  const cw  = canvas.width  / GRID_SIZE;
  const ch  = canvas.height / GRID_SIZE;

  // Height map
  drawGrid(canvas, HEIGHT_MAP, heatColor);

  // A* path
  if(ASTAR_PATH.length > 0) {{
    ctx.strokeStyle = '#ffeb3b';
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ASTAR_PATH.forEach(([r,c],i) => {{
      const x = (c+0.5)*cw, y = (r+0.5)*ch;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }});
    ctx.stroke();
  }}

  // Robot marker
  const rx = (ROBOT_COL+0.5)*cw, ry = (ROBOT_ROW+0.5)*ch;
  ctx.fillStyle = '#00e5ff';
  ctx.beginPath(); ctx.arc(rx,ry,cw*0.8,0,2*Math.PI); ctx.fill();

  // Heading arrow
  const hx = rx + Math.cos(-ROBOT_HEADING)*cw*2;
  const hy = ry + Math.sin(-ROBOT_HEADING)*ch*2;
  ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(rx,ry); ctx.lineTo(hx,hy); ctx.stroke();
}}

function drawObstacles(canvas) {{
  const ctx = canvas.getContext('2d');
  const cw  = canvas.width  / GRID_SIZE;
  const ch  = canvas.height / GRID_SIZE;
  for(let r=0;r<GRID_SIZE;r++) for(let c=0;c<GRID_SIZE;c++) {{
    const rock   = ROCKS_MAP[r][c];
    const crater = CRATERS_MAP[r][c];
    const wall   = WALLS_MAP[r][c];
    const combined = Math.min(1, rock*0.8 + crater*0.6 + wall);
    const g = Math.round(255*(1-combined));
    ctx.fillStyle = wall>0.5 ? '#ff5722' : `rgb(0,${{g}},0)`;
    ctx.fillRect(c*cw, r*ch, cw, ch);
  }}
  // Robot
  ctx.fillStyle='#00e5ff';
  ctx.beginPath();
  ctx.arc((ROBOT_COL+0.5)*cw,(ROBOT_ROW+0.5)*ch,cw*0.8,0,2*Math.PI);
  ctx.fill();
}}

function drawGoal(canvas) {{
  const ctx = canvas.getContext('2d');
  const cw  = canvas.width  / GRID_SIZE;
  const ch  = canvas.height / GRID_SIZE;
  let flat = GOAL_MAP.flat(); let mn=Math.min(...flat),mx=Math.max(...flat);
  for(let r=0;r<GRID_SIZE;r++) for(let c=0;c<GRID_SIZE;c++) {{
    const t = (GOAL_MAP[r][c]-mn)/(mx-mn+1e-9);
    ctx.fillStyle = `rgb(0,${{Math.round(lerp(30,200,t))}},0)`;
    ctx.fillRect(c*cw,r*ch,cw,ch);
  }}
  ctx.fillStyle='#00e5ff';
  ctx.beginPath();
  ctx.arc((ROBOT_COL+0.5)*cw,(ROBOT_ROW+0.5)*ch,cw*0.8,0,2*Math.PI);
  ctx.fill();
}}

function drawAction(canvas) {{
  const ctx  = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#111'; ctx.fillRect(0,0,W,H);

  const cx=W/2, cy=H/2, r=70;
  ctx.strokeStyle='#333'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.arc(cx,cy,r,0,2*Math.PI); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx,cy-r-10); ctx.lineTo(cx,cy+r+10); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx-r-10,cy); ctx.lineTo(cx+r+10,cy); ctx.stroke();

  function drawArrow(action, color, label) {{
    if(!action) return;
    const [lin,ang] = action;
    // forward = up in canvas, angular > 0 = turn left
    const dx = -ang * r * 0.8;
    const dy = -lin * r * 0.8;
    ctx.strokeStyle = color; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+dx,cy+dy); ctx.stroke();
    // arrowhead
    const angle = Math.atan2(dy,dx);
    ctx.beginPath();
    ctx.moveTo(cx+dx,cy+dy);
    ctx.lineTo(cx+dx-10*Math.cos(angle-0.4), cy+dy-10*Math.sin(angle-0.4));
    ctx.lineTo(cx+dx-10*Math.cos(angle+0.4), cy+dy-10*Math.sin(angle+0.4));
    ctx.closePath(); ctx.fillStyle=color; ctx.fill();
    ctx.fillStyle=color; ctx.font='11px monospace';
    ctx.fillText(label, cx+dx+6, cy+dy-4);
  }}
  drawArrow(EXPERT_ACTION, '#4fc3f7', 'A*');
  drawArrow(MODEL_ACTION,  '#ff7043', 'Model');
}}

function render() {{
  drawPath(document.getElementById('heightCanvas'));
  drawObstacles(document.getElementById('obstacleCanvas'));
  drawGoal(document.getElementById('goalCanvas'));
  drawAction(document.getElementById('actionCanvas'));
  document.getElementById('modelActionText').textContent =
    MODEL_ACTION ? `linear=${{MODEL_ACTION[0].toFixed(2)}}, angular=${{MODEL_ACTION[1].toFixed(2)}}` : 'No model loaded';
}}

function reload() {{
  const seed  = document.getElementById('seedInput').value;
  const phase = document.getElementById('phaseSelect').value;
  window.location.href = `/?seed=${{seed}}&phase=${{phase}}`;
}}

render();
</script>
</body>
</html>"""


def _run_model(cfg, checkpoint_path, terrain_crop, goal_map, heading):
    try:
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from training_nav.model import NavPolicy
        import numpy as np

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = NavPolicy(cfg).to(device)
        ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()

        terrain_5ch = np.concatenate([terrain_crop, goal_map[None]], axis=0)
        t_in = torch.from_numpy(terrain_5ch).unsqueeze(0).to(device)
        h_in = torch.tensor([[math.sin(heading), math.cos(heading)]], dtype=torch.float32).to(device)

        with torch.inference_mode():
            out = model(t_in, h_in).squeeze(0).cpu().numpy()
        return float(out[0]), float(out[1])
    except Exception as e:
        print(f'[visualize] Model inference failed: {e}')
        return None


def _serve(cfg, args):
    checkpoint = args.checkpoint

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_): pass

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(self.path)
            qs     = parse_qs(parsed.query)
            seed   = int(qs.get('seed',  ['0'])[0])
            phase  = qs.get('phase', ['to_excavation'])[0]

            (arena, terrain, terrain_crop, goal_map,
             astar_path, rx, ry, heading, cost) = _generate_scene(cfg, seed, phase)

            expert_action = plan_action(terrain_crop, goal_map, heading, cfg)
            model_action  = None
            if checkpoint and os.path.exists(checkpoint):
                model_action = _run_model(cfg, checkpoint, terrain_crop, goal_map, heading)

            scene_data = {
                'height':       _grid_to_json(terrain_crop[0]),
                'rocks':        _grid_to_json(terrain_crop[1]),
                'craters':      _grid_to_json(terrain_crop[2]),
                'walls':        _grid_to_json(terrain_crop[3]),
                'goal':         _grid_to_json(goal_map),
                'cost':         _grid_to_json(cost),
                'path':         _path_to_json(astar_path),
                'heading':      heading,
                'expert_action': expert_action,
                'phase':        phase,
                'arena_width':  arena.width,
                'arena_length': arena.length,
                'arena_scale':  arena.scale,
            }

            html = _make_html(cfg, scene_data, model_action)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode())

    port = 8766
    server = http.server.HTTPServer(('localhost', port), Handler)
    print(f'[visualize] http://localhost:{port}  (Ctrl+C to stop)')
    webbrowser.open(f'http://localhost:{port}')
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='training_nav/config.yaml')
    parser.add_argument('--checkpoint', default='training_nav/checkpoints/best.pt')
    parser.add_argument('--seed',       type=int,  default=0)
    parser.add_argument('--phase',      default='to_excavation',
                        choices=['to_excavation', 'to_deposit'])
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    _serve(cfg, args)


if __name__ == '__main__':
    main()
