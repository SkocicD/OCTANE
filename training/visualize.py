"""Inference visualizer for TerrainModel.

Usage:
    python training/visualize.py
    python training/visualize.py --checkpoint training/checkpoints/best.pt
    python training/visualize.py --episode ep_001234
    python training/visualize.py --episode ep_001234 --gt-only   # skip model, just show GT
"""
import os
import sys
import json
import random
import argparse
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def _pick_episode(data_root: str, episode_id: str | None) -> str:
    gt_dir = os.path.join(data_root, 'gt')
    episodes = sorted(
        f.replace('_gt.npz', '')
        for f in os.listdir(gt_dir)
        if f.endswith('_gt.npz')
    )
    if not episodes:
        raise RuntimeError(f"No episodes found in {gt_dir}")

    if episode_id:
        if episode_id in episodes:
            return episode_id
        # Allow partial match (e.g. "1234" matches "ep_001234")
        matches = [e for e in episodes if episode_id in e]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"Ambiguous — matches: {matches[:5]}{'...' if len(matches)>5 else ''}")
            return matches[0]
        raise RuntimeError(f"Episode '{episode_id}' not found")

    # Interactive prompt
    print(f"Available: {len(episodes)} episodes  (e.g. {episodes[0]}, {episodes[-1]})")
    raw = input("Episode ID (press Enter for random): ").strip()
    if not raw:
        ep = random.choice(episodes)
        print(f"  → picked {ep}")
        return ep
    return _pick_episode(data_root, raw)


def _run_inference(checkpoint_path: str, data_root: str, episode_id: str,
                   depth_stats: dict, device_str: str) -> dict:
    import torch
    from training.model import TerrainModel
    from training.dataset import TerrainDataset

    device = torch.device(device_str)
    model  = TerrainModel().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Checkpoint epoch {ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', '?')}")

    ds = TerrainDataset(data_root, [episode_id], depth_stats, augment=False)
    images, rotation, gt = ds[0]

    with torch.no_grad():
        preds = model(images.unsqueeze(0).to(device),
                      rotation.unsqueeze(0).to(device))

    return {k: v.squeeze(0).cpu().numpy() for k, v in preds.items()}


def _load_gt(data_root: str, episode_id: str) -> dict:
    from training.dataset import build_object_heatmap, build_wall_mask
    npz     = np.load(os.path.join(data_root, 'gt', f'{episode_id}_gt.npz'))
    objects = npz['objects_gt']
    walls   = npz['walls_gt']
    return {
        'height':  npz['height_gt'],
        'rocks':   build_object_heatmap(objects, class_id=0),
        'craters': build_object_heatmap(objects, class_id=1),
        'walls':   build_wall_mask(walls),
        'objects_gt': objects,
        'walls_gt':   walls,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

GRID   = 200
CELL   = 0.05
HALF   = GRID * CELL / 2   # 5.0 m

def _cell_to_world(cell_idx: int) -> float:
    return cell_idx * CELL - HALF


_xs = np.array([_cell_to_world(i) for i in range(GRID)])  # world coords along rx axis
_ys = np.array([_cell_to_world(i) for i in range(GRID)])  # world coords along ry axis


def _surface(height: np.ndarray, title: str, colorscale='RdYlGn'):
    import plotly.graph_objects as go
    return go.Surface(
        x=_xs, y=_ys, z=height,
        colorscale=colorscale,
        colorbar=dict(title='m', len=0.5, thickness=12),
        name=title,
        showscale=True,
    )


def _rock_scatter(objects_gt: np.ndarray, pred_map: np.ndarray | None = None):
    """Red markers at GT rock positions; size scaled by confidence if pred given."""
    import plotly.graph_objects as go
    rocks = objects_gt[objects_gt[:, 3] == 0] if len(objects_gt) else np.zeros((0, 4))
    if len(rocks) == 0:
        return None
    rx, ry, diam = rocks[:, 0], rocks[:, 1], rocks[:, 2]
    # Sample height at rock position for z placement
    cx = np.clip(((rx + HALF) / CELL).astype(int), 0, GRID - 1)
    cy = np.clip(((ry + HALF) / CELL).astype(int), 0, GRID - 1)
    z  = pred_map[cx, cy] + 0.1 if pred_map is not None else np.zeros_like(rx) + 0.1
    return go.Scatter3d(
        x=rx, y=ry, z=z,
        mode='markers',
        marker=dict(size=np.clip(diam * 8, 4, 20), color='red', opacity=0.8),
        name='rocks (GT)',
    )


def _crater_scatter(objects_gt: np.ndarray, pred_map: np.ndarray | None = None):
    import plotly.graph_objects as go
    craters = objects_gt[objects_gt[:, 3] == 1] if len(objects_gt) else np.zeros((0, 4))
    if len(craters) == 0:
        return None
    rx, ry, diam = craters[:, 0], craters[:, 1], craters[:, 2]
    cx = np.clip(((rx + HALF) / CELL).astype(int), 0, GRID - 1)
    cy = np.clip(((ry + HALF) / CELL).astype(int), 0, GRID - 1)
    z  = pred_map[cx, cy] - 0.05 if pred_map is not None else np.zeros_like(rx)
    return go.Scatter3d(
        x=rx, y=ry, z=z,
        mode='markers',
        marker=dict(size=np.clip(diam * 8, 4, 20), color='dodgerblue',
                    symbol='circle-open', opacity=0.9),
        name='craters (GT)',
    )


def _wall_lines(walls_gt: np.ndarray, height_map: np.ndarray):
    import plotly.graph_objects as go
    traces = []
    for wall in walls_gt:
        rx1, ry1, rx2, ry2 = wall
        # Sample a few z points along the wall so it follows the terrain
        n   = 10
        rxs = np.linspace(rx1, rx2, n)
        rys = np.linspace(ry1, ry2, n)
        cxs = np.clip(((rxs + HALF) / CELL).astype(int), 0, GRID - 1)
        cys = np.clip(((rys + HALF) / CELL).astype(int), 0, GRID - 1)
        zs  = height_map[cxs, cys] + 0.15
        traces.append(go.Scatter3d(
            x=rxs, y=rys, z=zs,
            mode='lines',
            line=dict(color='orange', width=6),
            name='walls (GT)',
            showlegend=len(traces) == 0,
        ))
    return traces


def _confidence_heatmap(conf: np.ndarray, title: str, colorscale: str):
    import plotly.graph_objects as go
    return go.Heatmap(
        z=conf.T,
        x=_xs, y=_ys,
        colorscale=colorscale,
        zmin=0, zmax=1,
        colorbar=dict(title='conf', len=0.4, thickness=10),
        name=title,
    )


def build_figure(pred: dict | None, gt: dict, episode_id: str) -> 'plotly.graph_objects.Figure':
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_pred = pred is not None
    cols     = 2 if has_pred else 1
    col_titles = (['Predicted', 'Ground Truth'] if has_pred else ['Ground Truth'])

    fig = make_subplots(
        rows=2, cols=cols,
        specs=[[{'type': 'scene'}, {'type': 'scene'}] if has_pred else [{'type': 'scene'}],
               [{'type': 'xy'},    {'type': 'xy'}]    if has_pred else [{'type': 'xy'}]],
        subplot_titles=(
            [f'Height Map — {t}' for t in col_titles] +
            [f'Rock / Crater Confidence — {t}' for t in col_titles]
        ),
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    objects = gt.get('objects_gt', np.zeros((0, 4), dtype=np.float32))
    walls   = gt.get('walls_gt',   np.zeros((0, 4), dtype=np.float32))

    def _add_scene(height, col):
        fig.add_trace(_surface(height, col_titles[col - 1]), row=1, col=col)
        r = _rock_scatter(objects, height)
        c = _crater_scatter(objects, height)
        if r: fig.add_trace(r, row=1, col=col)
        if c: fig.add_trace(c, row=1, col=col)
        for w in _wall_lines(walls, height):
            fig.add_trace(w, row=1, col=col)

    def _add_conf(rocks, craters, col):
        # Overlay rocks (red) and craters (blue) as separate heatmaps
        fig.add_trace(_confidence_heatmap(rocks,   'rocks',   'Reds'),   row=2, col=col)
        fig.add_trace(_confidence_heatmap(craters, 'craters', 'Blues'),  row=2, col=col)

    if has_pred:
        _add_scene(pred['height'], col=1)
        _add_conf(pred['rocks'], pred['craters'], col=1)

    _add_scene(gt['height'], col=cols)
    _add_conf(gt['rocks'], gt['craters'], col=cols)

    camera = dict(eye=dict(x=1.4, y=1.4, z=1.0))
    scene_cfg = dict(
        xaxis_title='rx (fwd m)',
        yaxis_title='ry (lat m)',
        zaxis_title='height m',
        camera=camera,
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.35),
    )
    fig.update_layout(
        title=dict(text=f'TerrainModel — episode: {episode_id}', font_size=16),
        scene=scene_cfg,
        **(dict(scene2=scene_cfg) if has_pred else {}),
        height=900,
        template='plotly_dark',
    )
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='training/config.yaml')
    parser.add_argument('--checkpoint', default='training/checkpoints/best.pt')
    parser.add_argument('--episode',    default=None)
    parser.add_argument('--gt-only',    action='store_true',
                        help='Skip model inference, show ground truth only')
    args = parser.parse_args()

    cfg       = _load_config(args.config)
    data_root = cfg['data']['root']

    episode_id = _pick_episode(data_root, args.episode)
    print(f"\nEpisode: {episode_id}")

    gt = _load_gt(data_root, episode_id)
    print(f"  GT — rocks: {int((gt['objects_gt'][:,3]==0).sum())}  "
          f"craters: {int((gt['objects_gt'][:,3]==1).sum())}  "
          f"walls: {len(gt['walls_gt'])}")

    pred = None
    if not args.gt_only:
        if not os.path.exists(args.checkpoint):
            print(f"  Checkpoint not found at {args.checkpoint} — showing GT only")
        else:
            stats_file = cfg['data']['depth_stats_file']
            if os.path.exists(stats_file):
                with open(stats_file) as f:
                    depth_stats = json.load(f)
            else:
                depth_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}

            device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
            print(f"  Running inference on {device}...")
            pred = _run_inference(args.checkpoint, data_root, episode_id, depth_stats, device)
            print(f"  height range: [{pred['height'].min():.3f}, {pred['height'].max():.3f}]  "
                  f"rocks max: {pred['rocks'].max():.3f}  "
                  f"craters max: {pred['craters'].max():.3f}  "
                  f"walls max: {pred['walls'].max():.3f}")

    print("\nBuilding visualization...")
    fig = build_figure(pred, gt, episode_id)
    fig.show()


if __name__ == '__main__':
    main()
