"""Inference visualizer for TerrainModel.

Usage:
    visualize.bat
    python training/visualize.py
    python training/visualize.py --checkpoint training/checkpoints/best.pt
    python training/visualize.py --episode ep_001234
    python training/visualize.py --gt-only
"""
import os
import sys
import json
import random
import argparse
import math
import webbrowser
import socket
import socketserver
import http.server
import urllib.parse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _all_episodes(data_root: str) -> list[str]:
    from training.dataset import _valid_episodes
    gt_dir = os.path.join(data_root, 'gt')
    all_ids = sorted(
        f.replace('_gt.npz', '')
        for f in os.listdir(gt_dir)
        if f.endswith('_gt.npz')
    )
    return _valid_episodes(data_root, all_ids)


def _resolve_episode(data_root: str, episode_id: str | None) -> str:
    episodes = _all_episodes(data_root)
    if not episodes:
        raise RuntimeError(f"No episodes found in {data_root}/gt")
    if not episode_id:
        return episodes[0]
    if episode_id in episodes:
        return episode_id
    matches = [e for e in episodes if episode_id in e]
    if matches:
        return matches[0]
    raise RuntimeError(f"Episode '{episode_id}' not found")


def _run_inference(checkpoint_path: str, data_root: str, episode_id: str,
                   depth_stats: dict, device_str: str) -> tuple[dict, dict]:
    import torch
    from training.model import TerrainModel
    from training.dataset import TerrainDataset

    device = torch.device(device_str)
    model  = TerrainModel().to(device)
    ckpt   = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ckpt_info = {'epoch': ckpt.get('epoch', '?'), 'val_loss': ckpt.get('val_loss', None)}

    ds = TerrainDataset(data_root, [episode_id], depth_stats, augment=False)
    images, rotation, _ = ds[0]

    with torch.no_grad():
        preds = model(images.unsqueeze(0).to(device),
                      rotation.unsqueeze(0).to(device))

    from training.dataset import HEIGHT_SCALE
    out = {k: v.squeeze(0).cpu().numpy() for k, v in preds.items()}
    out['height'] = out['height'] * HEIGHT_SCALE  # convert back to metres for display
    return out, ckpt_info


def _load_gt(data_root: str, episode_id: str) -> dict:
    from training.dataset import build_object_heatmap, build_wall_mask
    npz     = np.load(os.path.join(data_root, 'gt', f'{episode_id}_gt.npz'))
    objects = npz['objects_gt']
    walls   = npz['walls_gt']
    return {
        'height':      npz['height_gt'],
        'rocks':       build_object_heatmap(objects, class_id=0),
        'craters':     build_object_heatmap(objects, class_id=1),
        'walls':       build_wall_mask(walls),
        'objects_gt':  objects,
        'walls_gt':    walls,
        'robot_roll':  float(npz.get('robot_roll',  np.float32(0.0)).flat[0]),
        'robot_pitch': float(npz.get('robot_pitch', np.float32(0.0)).flat[0]),
        'robot_yaw':   float(npz.get('robot_yaw',   np.float32(0.0)).flat[0]),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

GRID      = 200
CELL      = 0.10
HALF      = GRID * CELL / 2  # 10.0 m  — BEV coverage
FLOOR_EXT = 15.0              # flat floor extends ±15 m around robot

_xs = np.array([i * CELL - HALF for i in range(GRID)])
_ys = np.array([i * CELL - HALF for i in range(GRID)])


def _floor_plane(floor_z: float):
    """Flat luna-gray plane from ±FLOOR_EXT so perimeter walls have ground to stand on."""
    import plotly.graph_objects as go
    e = FLOOR_EXT
    xs = np.array([-e, e])
    ys = np.array([-e, e])
    zs = np.full((2, 2), floor_z)
    return go.Surface(
        x=xs, y=ys, z=zs,
        colorscale=[[0, '#2a2d38'], [1, '#2a2d38']],
        showscale=False, opacity=0.55,
        name='floor', showlegend=False,
        hoverinfo='skip',
    )


def _surface(height: np.ndarray, title: str, colorscale='RdYlGn'):
    import plotly.graph_objects as go
    # Plotly Surface: z[i][j] → (x[j], y[i]), but height[cx, cy] where cx=rx-axis, cy=ry-axis.
    # Transpose so z[ry_idx][rx_idx] = height[rx_idx][ry_idx] → displayed at (_xs[rx_idx], _ys[ry_idx]).
    return go.Surface(
        x=_xs, y=_ys, z=height.T,
        colorscale=colorscale,
        colorbar=dict(title='m', len=0.5, thickness=12),
        name=title,
        showscale=True,
    )


def _rock_circles(objects_gt: np.ndarray, height_map: np.ndarray):
    import plotly.graph_objects as go
    rocks = objects_gt[objects_gt[:, 3] == 0] if len(objects_gt) else np.zeros((0, 4))
    if len(rocks) == 0:
        return []
    theta  = np.linspace(0, 2 * np.pi, 37)
    traces = []
    for i, obj in enumerate(rocks):
        rx, ry, diam = float(obj[0]), float(obj[1]), float(obj[2])
        r   = diam / 2
        xs  = rx + r * np.cos(theta)
        ys  = ry + r * np.sin(theta)
        cxs = np.clip(((xs + HALF) / CELL).astype(int), 0, GRID - 1)
        cys = np.clip(((ys + HALF) / CELL).astype(int), 0, GRID - 1)
        zs  = height_map[cxs, cys] + 0.08
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color='#42a5f5', width=3),
            name='rocks (GT)', showlegend=(i == 0), legendgroup='rocks',
        ))
    return traces


def _crater_circles(objects_gt: np.ndarray, height_map: np.ndarray):
    import plotly.graph_objects as go
    craters = objects_gt[objects_gt[:, 3] == 1] if len(objects_gt) else np.zeros((0, 4))
    if len(craters) == 0:
        return []
    theta  = np.linspace(0, 2 * np.pi, 37)
    traces = []
    for i, obj in enumerate(craters):
        rx, ry, diam = float(obj[0]), float(obj[1]), float(obj[2])
        r   = diam / 2
        xs  = rx + r * np.cos(theta)
        ys  = ry + r * np.sin(theta)
        cxs = np.clip(((xs + HALF) / CELL).astype(int), 0, GRID - 1)
        cys = np.clip(((ys + HALF) / CELL).astype(int), 0, GRID - 1)
        zs  = height_map[cxs, cys] - 0.05
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color='#ef5350', width=3),
            name='craters (GT)', showlegend=(i == 0), legendgroup='craters',
        ))
    return traces


def _wall_lines(walls_gt: np.ndarray, height_map: np.ndarray, floor_z: float):
    import plotly.graph_objects as go
    traces = []
    for i, wall in enumerate(walls_gt):
        rx1, ry1, rx2, ry2 = wall
        n   = 40
        rxs = np.linspace(rx1, rx2, n)
        rys = np.linspace(ry1, ry2, n)
        inside = (rxs >= -HALF) & (rxs <= HALF) & (rys >= -HALF) & (rys <= HALF)
        cxs = np.clip(((rxs + HALF) / CELL).astype(int), 0, GRID - 1)
        cys = np.clip(((rys + HALF) / CELL).astype(int), 0, GRID - 1)
        zs  = np.where(inside, height_map[cxs, cys], floor_z) + 0.20
        traces.append(go.Scatter3d(
            x=rxs, y=rys, z=zs, mode='lines',
            line=dict(color='#ff9800', width=6),
            name='walls (GT)', showlegend=(i == 0), legendgroup='walls',
        ))
    return traces


def _rover_box(height_map: np.ndarray, roll: float, pitch: float):
    """Wireframe rectangular prism for the rover at (0,0) with terrain-following z.
    In BEV space the robot always faces +rx; only roll/pitch tilt the box."""
    import plotly.graph_objects as go

    L, W, H = 0.90, 0.55, 0.40  # approx rover dims in metres (length, width, height)

    z0 = float(height_map[GRID // 2, GRID // 2])

    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1,  0,   0 ], [0, cr, -sr], [0, sr,  cr]])
    Ry = np.array([[cp, 0,  sp ], [0,  1,   0], [-sp, 0, cp]])
    R  = Rx @ Ry

    raw = np.array([
        [-L/2, -W/2, 0], [ L/2, -W/2, 0], [ L/2,  W/2, 0], [-L/2,  W/2, 0],
        [-L/2, -W/2, H], [ L/2, -W/2, H], [ L/2,  W/2, H], [-L/2,  W/2, H],
    ])
    v = (R @ raw.T).T + np.array([0, 0, z0])

    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [v[a,0], v[b,0], None]
        ys += [v[a,1], v[b,1], None]
        zs += [v[a,2], v[b,2], None]

    body = go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                        line=dict(color='#eeeeee', width=3),
                        name='rover', showlegend=True)

    # Forward arrow: yellow line from front face centre to ahead
    mid_front = (R @ np.array([L/2, 0, H/2])) + [0, 0, z0]
    tip        = (R @ np.array([L/2 + 0.35, 0, H/2])) + [0, 0, z0]
    arrow = go.Scatter3d(x=[mid_front[0], tip[0]], y=[mid_front[1], tip[1]],
                         z=[mid_front[2], tip[2]], mode='lines',
                         line=dict(color='#ffeb3b', width=6),
                         name='forward', showlegend=False)

    return [body, arrow]


def _confidence_heatmap(conf: np.ndarray, title: str, colorscale: str):
    import plotly.graph_objects as go
    return go.Heatmap(
        z=conf.T, x=_xs, y=_ys,
        colorscale=colorscale, zmin=0, zmax=1,
        colorbar=dict(title='conf', len=0.4, thickness=10),
        name=title,
    )


def build_figure(pred: dict | None, gt: dict) -> 'plotly.graph_objects.Figure':
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_pred = pred is not None
    cols     = 2 if has_pred else 1

    row1 = [{'type': 'scene'}, {'type': 'scene'}] if has_pred else [{'type': 'scene'}]
    row2 = [{'type': 'xy'},    {'type': 'xy'}]    if has_pred else [{'type': 'xy'}]

    fig = make_subplots(
        rows=2, cols=cols,
        specs=[row1, row2],
        subplot_titles=(['Height Map'] * cols + ['Rock / Crater Confidence'] * cols),
        row_heights=[0.65, 0.35],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    objects = gt.get('objects_gt',  np.zeros((0, 4), dtype=np.float32))
    walls   = gt.get('walls_gt',    np.zeros((0, 4), dtype=np.float32))
    roll    = gt.get('robot_roll',  0.0)
    pitch   = gt.get('robot_pitch', 0.0)

    def _add_scene(height, col, show_rover=False):
        floor_z = float(height.mean())
        fig.add_trace(_floor_plane(floor_z), row=1, col=col)
        fig.add_trace(_surface(height, ''), row=1, col=col)
        for t in _rock_circles(objects, height):
            fig.add_trace(t, row=1, col=col)
        for t in _crater_circles(objects, height):
            fig.add_trace(t, row=1, col=col)
        for w in _wall_lines(walls, height, floor_z):
            fig.add_trace(w, row=1, col=col)
        if show_rover:
            for t in _rover_box(height, roll, pitch):
                fig.add_trace(t, row=1, col=col)

    def _add_conf(rocks, craters, col):
        fig.add_trace(_confidence_heatmap(rocks,   'rocks',   'Reds'),  row=2, col=col)
        fig.add_trace(_confidence_heatmap(craters, 'craters', 'Blues'), row=2, col=col)

    if has_pred:
        _add_scene(pred['height'], col=1)
        _add_conf(pred['rocks'], pred['craters'], col=1)

    _add_scene(gt['height'], col=cols, show_rover=True)
    _add_conf(gt['rocks'], gt['craters'], col=cols)

    scene_cfg = dict(
        xaxis_title='rx (fwd m)',
        yaxis_title='ry (lat m)',
        zaxis_title='height m',
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.10),
        bgcolor='rgba(0,0,0,0)',
    )
    layout_kw = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Roboto, sans-serif', color='#c4c6d0', size=12),
        legend=dict(bgcolor='rgba(26,29,36,0.8)', bordercolor='#44474f', borderwidth=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=880,
        template='plotly_dark',
        scene=scene_cfg,
    )
    if has_pred:
        layout_kw['scene2'] = scene_cfg
    fig.update_layout(**layout_kw)

    fig.update_xaxes(constrain='domain', row=2)
    fig.update_yaxes(scaleanchor='x',  scaleratio=1, constrain='domain', row=2, col=1)
    if has_pred:
        fig.update_yaxes(scaleanchor='x2', scaleratio=1, constrain='domain', row=2, col=2)

    for ann in fig.layout.annotations:
        ann.font = dict(size=13, color='#8e9099', family='Roboto, sans-serif')

    return fig


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TerrainModel — TMPL_EPISODE</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg:       #0f1117;
      --surface:  #1a1d24;
      --surf-var: #252930;
      --primary:  #80cbc4;
      --on-surf:  #e2e2e6;
      --on-var:   #8e9099;
      --outline:  #44474f;
      --pred-col: #80cbc4;
      --gt-col:   #aed581;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Roboto', sans-serif; background: var(--bg); color: var(--on-surf); min-height: 100vh; }

    /* top bar */
    .top-bar {
      background: var(--surface); border-bottom: 1px solid var(--outline);
      padding: 0 20px; height: 60px;
      display: flex; align-items: center; justify-content: space-between;
      position: sticky; top: 0; z-index: 100; gap: 16px;
    }
    .top-bar-left { display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
    .app-icon {
      width: 34px; height: 34px; border-radius: 10px;
      background: linear-gradient(135deg, #4db6ac 0%, #26a69a 100%);
      display: flex; align-items: center; justify-content: center;
      font-size: 17px; flex-shrink: 0; user-select: none;
    }
    .title-block h1 { font-size: 17px; font-weight: 500; letter-spacing: 0.1px; line-height: 1; }
    .title-block .ep { font-family: 'Roboto Mono', monospace; font-size: 11px; color: var(--on-var); margin-top: 3px; }

    /* episode picker */
    .ep-form { display: flex; align-items: center; gap: 8px; flex: 1; max-width: 520px; }
    .ep-input {
      flex: 1; height: 36px; padding: 0 14px;
      background: var(--surf-var); border: 1px solid var(--outline);
      border-radius: 20px; color: var(--on-surf);
      font-family: 'Roboto Mono', monospace; font-size: 13px;
      outline: none; transition: border-color 0.15s;
    }
    .ep-input:focus { border-color: var(--primary); }
    .ep-input::placeholder { color: var(--on-var); }
    .load-btn {
      height: 36px; padding: 0 18px; border-radius: 20px; border: none;
      background: var(--primary); color: #003731;
      font-family: 'Roboto', sans-serif; font-size: 13px; font-weight: 500;
      cursor: pointer; flex-shrink: 0; transition: opacity 0.15s;
    }
    .load-btn:hover { opacity: 0.88; }
    .ep-count { font-size: 12px; color: var(--on-var); flex-shrink: 0; white-space: nowrap; }

    /* content */
    .content { padding: 18px 20px 28px; }
    .info-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
    .badge { padding: 4px 13px; border-radius: 20px; font-size: 12px; font-weight: 500; letter-spacing: 0.4px; border: 1px solid transparent; }
    .badge-mode-pred { background: rgba(128,203,196,0.1); color: var(--pred-col); border-color: rgba(128,203,196,0.25); }
    .badge-mode-gt   { background: rgba(174,213,129,0.1); color: var(--gt-col);   border-color: rgba(174,213,129,0.25); }
    .badge-stat { background: var(--surf-var); color: var(--on-var); border-color: var(--outline); }

    .legend { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
    .chip { display: flex; align-items: center; gap: 7px; padding: 5px 13px; border-radius: 8px; background: var(--surf-var); border: 1px solid var(--outline); font-size: 12px; font-weight: 500; color: var(--on-var); }
    .dot      { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
    .dot-ring { width: 9px; height: 9px; border-radius: 50%; border: 2px solid #42a5f5; flex-shrink: 0; }
    .dash-icon { width: 16px; height: 3px; border-radius: 2px; flex-shrink: 0; background: #ff9800; }

    .col-header-row { display: flex; gap: 12px; margin-bottom: 8px; }
    .col-header { flex: 1; text-align: center; padding: 8px 12px; border-radius: 10px; font-size: 12px; font-weight: 500; letter-spacing: 0.8px; text-transform: uppercase; border: 1px solid transparent; }
    .col-header.predicted { color: var(--pred-col); background: rgba(128,203,196,0.07); border-color: rgba(128,203,196,0.2); }
    .col-header.gt        { color: var(--gt-col);   background: rgba(174,213,129,0.07); border-color: rgba(174,213,129,0.2); }

    .notice { margin-bottom: 14px; padding: 10px 16px; background: var(--surf-var); border: 1px solid var(--outline); border-radius: 10px; font-size: 13px; color: var(--on-var); display: flex; align-items: center; gap: 10px; }
    .notice-icon { font-size: 16px; flex-shrink: 0; }

    .plot-card { background: var(--surface); border-radius: 16px; border: 1px solid var(--outline); overflow: hidden; padding: 6px 4px 4px; }

    /* loading overlay */
    #overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(15,17,23,0.85); z-index: 1000;
      align-items: center; justify-content: center; flex-direction: column; gap: 16px;
    }
    #overlay.active { display: flex; }
    .spinner { width: 36px; height: 36px; border-radius: 50%; border: 3px solid rgba(128,203,196,0.2); border-top-color: #80cbc4; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .overlay-text { font-size: 14px; color: #c4c6d0; }
  </style>
</head>
<body>
  <div id="overlay"><div class="spinner"></div><p class="overlay-text">Loading episode…</p></div>

  <div class="top-bar">
    <div class="top-bar-left">
      <div class="app-icon">🗺</div>
      <div class="title-block">
        <h1>TerrainModel Visualizer</h1>
        <div class="ep">TMPL_EPISODE</div>
      </div>
    </div>
    <form class="ep-form" action="/" method="get" id="ep-form">
      <input class="ep-input" type="text" name="episode"
             placeholder="Episode ID…" value="TMPL_EPISODE"
             list="ep-list" autocomplete="off" spellcheck="false">
      <datalist id="ep-list">TMPL_EP_OPTIONS</datalist>
      <button class="load-btn" type="submit">Load</button>
      <span class="ep-count">TMPL_EP_COUNT episodes</span>
    </form>
  </div>

  <div class="content">
    <div class="info-row">
      TMPL_MODE_BADGE
      TMPL_STAT_BADGES
    </div>
    <div class="legend">
      <div class="chip"><span class="dot-ring"></span>Rocks (GT)</div>
      <div class="chip"><span class="dot" style="background:#ef5350"></span>Craters (GT)</div>
      <div class="chip"><span class="dash-icon"></span>Walls (GT)</div>
    </div>
    TMPL_NOTICE
    TMPL_COL_HEADERS
    <div class="plot-card">TMPL_PLOT_DIV</div>
  </div>

  <script>
    document.getElementById('ep-form').addEventListener('submit', function() {
      document.getElementById('overlay').classList.add('active');
    });
  </script>
</body>
</html>
"""


# ── page builder ──────────────────────────────────────────────────────────────

def _build_page(episode_id: str, pred: dict | None, gt: dict,
                ckpt_info: dict | None, all_episodes: list[str]) -> str:
    import plotly.io as pio

    fig      = build_figure(pred, gt)
    plot_div = pio.to_html(fig, include_plotlyjs='cdn', full_html=False,
                           config={'responsive': True, 'displayModeBar': True,
                                   'modeBarButtonsToRemove': ['sendDataToCloud']})

    has_pred = pred is not None

    if has_pred:
        epoch_str = f"epoch {ckpt_info['epoch']}" if ckpt_info else ''
        val_str   = (f"  val_loss={ckpt_info['val_loss']:.4f}"
                     if ckpt_info and ckpt_info['val_loss'] else '')
        mode_badge = f'<span class="badge badge-mode-pred">AI Prediction ({epoch_str}{val_str})</span>'
    else:
        mode_badge = '<span class="badge badge-mode-gt">Ground Truth Only</span>'

    objects   = gt.get('objects_gt', np.zeros((0, 4)))
    walls_arr = gt.get('walls_gt',   np.zeros((0, 4)))
    n_rocks   = int((objects[:, 3] == 0).sum()) if len(objects) else 0
    n_craters = int((objects[:, 3] == 1).sum()) if len(objects) else 0
    n_walls   = len(walls_arr)
    stat_badges = (
        f'<span class="badge badge-stat">{n_rocks} rock{"s" if n_rocks != 1 else ""}</span>'
        f'<span class="badge badge-stat">{n_craters} crater{"s" if n_craters != 1 else ""}</span>'
        f'<span class="badge badge-stat">{n_walls} wall{"s" if n_walls != 1 else ""}</span>'
    )

    notice = (
        '<div class="notice"><span class="notice-icon">ℹ</span>'
        'No checkpoint loaded — showing ground truth only. '
        'The AI Prediction column will appear once <code>training/checkpoints/best.pt</code> exists.</div>'
    ) if not has_pred else ''

    if has_pred:
        col_headers = ('<div class="col-header-row">'
                       '<div class="col-header predicted">▶ AI Prediction</div>'
                       '<div class="col-header gt">Ground Truth</div>'
                       '</div>')
    else:
        col_headers = ('<div class="col-header-row">'
                       '<div class="col-header gt">Ground Truth</div>'
                       '</div>')

    ep_options = ''.join(f'<option value="{e}">' for e in all_episodes)

    html = _HTML
    html = html.replace('TMPL_EPISODE',    episode_id)
    html = html.replace('TMPL_EP_OPTIONS', ep_options)
    html = html.replace('TMPL_EP_COUNT',   str(len(all_episodes)))
    html = html.replace('TMPL_MODE_BADGE', mode_badge)
    html = html.replace('TMPL_STAT_BADGES', stat_badges)
    html = html.replace('TMPL_NOTICE',     notice)
    html = html.replace('TMPL_COL_HEADERS', col_headers)
    html = html.replace('TMPL_PLOT_DIV',   plot_div)
    return html


# ── local server ──────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _make_handler(data_root: str, depth_stats: dict | None,
                  checkpoint: str, gt_only: bool, all_episodes: list[str]):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == '/favicon.ico':
                self.send_response(404); self.end_headers(); return

            params     = urllib.parse.parse_qs(parsed.query)
            episode_id = params.get('episode', [all_episodes[0]])[0]

            # resolve partial match
            if episode_id not in all_episodes:
                matches = [e for e in all_episodes if episode_id in e]
                episode_id = matches[0] if matches else all_episodes[0]

            try:
                print(f"  Loading {episode_id}...", end=' ', flush=True)
                gt = _load_gt(data_root, episode_id)

                pred      = None
                ckpt_info = None
                if not gt_only and depth_stats and os.path.exists(checkpoint):
                    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
                    pred, ckpt_info = _run_inference(
                        checkpoint, data_root, episode_id, depth_stats, device)

                html = _build_page(episode_id, pred, gt, ckpt_info, all_episodes)
                print("done")

                body = html.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                import traceback
                tb   = traceback.format_exc()
                body = (f'<html><body style="background:#111;color:#ef5350;'
                        f'font-family:monospace;padding:2em">'
                        f'<h2>Error loading {episode_id}</h2><pre>{tb}</pre>'
                        f'</body></html>').encode('utf-8')
                self.send_response(500)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_):
            pass  # suppress default request logging

    return Handler


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='training/config.yaml')
    parser.add_argument('--checkpoint', default='training/checkpoints/best.pt')
    parser.add_argument('--episode',    default=None)
    parser.add_argument('--gt-only',    action='store_true')
    args = parser.parse_args()

    cfg       = _load_config(args.config)
    data_root = cfg['data']['root']
    episodes  = _all_episodes(data_root)

    if not episodes:
        raise RuntimeError(f"No episodes found in {data_root}/gt")

    default_ep = _resolve_episode(data_root, args.episode)

    depth_stats = None
    if not args.gt_only:
        stats_file = cfg['data']['depth_stats_file']
        if os.path.exists(stats_file):
            with open(stats_file) as f:
                depth_stats = json.load(f)
        elif os.path.exists(args.checkpoint):
            depth_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}

    port    = _free_port()
    Handler = _make_handler(data_root, depth_stats, args.checkpoint,
                            args.gt_only, episodes)

    httpd = socketserver.TCPServer(('localhost', port), Handler)
    url   = f'http://localhost:{port}/?episode={default_ep}'

    print(f"\nTerrainModel Visualizer  →  {url}")
    print(f"  {len(episodes)} episodes available")
    print("  Press Ctrl+C to stop.\n")

    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()


if __name__ == '__main__':
    main()
