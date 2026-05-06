"""Evaluation script — MAE, RMSE, mIoU, mAP@0.5.

Usage:
    python -m training.terrain_mapping.evaluate \
        --data_dir /path/to/data --checkpoint /path/to/best_model.pt
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import TerrainDataset
from model import TerrainMappingModel
from train import collate_fn


def circle_iou(px, py, pd, gx, gy, gd):
    dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    pr, gr = pd / 2, gd / 2
    if dist >= pr + gr:
        return 0.0
    if dist <= abs(pr - gr):
        return min(pr, gr) ** 2 / max(pr, gr) ** 2
    inter = min(pr, gr) ** 2 * np.pi
    return inter / (pr ** 2 * np.pi + gr ** 2 * np.pi - inter)


def evaluate(data_dir: str, checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TerrainDataset(data_dir)
    loader = DataLoader(dataset, batch_size=16, num_workers=2, collate_fn=collate_fn)

    model = TerrainMappingModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    height_errors, all_ious = [], {c: [] for c in range(4)}
    det_tp = {0: 0, 1: 0}
    det_fp = {0: 0, 1: 0}
    det_fn = {0: 0, 1: 0}

    with torch.no_grad():
        for bev, height_gt, semantic_gt, objects_gt, occupancy in loader:
            bev = bev.to(device)
            height_pred, semantic_pred, detect_pred = model(bev)
            occ = occupancy.bool()
            diff = (height_pred.squeeze(1).cpu() - height_gt)[occ]
            height_errors.extend(diff.abs().numpy().tolist())
            labels_pred = semantic_pred.argmax(dim=1).cpu()
            for c in range(4):
                inter = ((labels_pred == c) & (semantic_gt == c)).float().sum().item()
                union = ((labels_pred == c) | (semantic_gt == c)).float().sum().item()
                if union > 0:
                    all_ious[c].append(inter / union)
            for b_idx, b_gt in enumerate(objects_gt):
                confs = detect_pred[b_idx, :, 3].sigmoid().cpu().numpy()
                preds_xyz = detect_pred[b_idx, :, :3].cpu().numpy()
                for obj in b_gt:
                    t = int(obj["type"])
                    matched = any(
                        confs[s] >= 0.5 and circle_iou(
                            preds_xyz[s,0], preds_xyz[s,1], preds_xyz[s,2],
                            obj["x"], obj["y"], obj["diameter"]
                        ) >= 0.5
                        for s in range(len(confs))
                    )
                    (det_tp if matched else det_fn)[t] += 1

    mae = np.mean(height_errors)
    rmse = np.sqrt(np.mean(np.array(height_errors) ** 2))
    miou_all = np.mean([np.mean(v) for v in all_ious.values() if v])
    miou_obs = np.mean([np.mean(all_ious[c]) for c in [1, 2, 3] if all_ious[c]])
    print(f"Height MAE:  {mae:.4f} m  (target <0.03)")
    print(f"Height RMSE: {rmse:.4f} m  (target <0.05)")
    print(f"mIoU all:    {miou_all:.4f}  (target >0.70)")
    print(f"mIoU obstacles: {miou_obs:.4f}  (target >0.60)")
    for t, name in [(0, "rock"), (1, "crater")]:
        tp, fn = det_tp[t], det_fn[t]
        rec = tp / (tp + fn + 1e-6)
        print(f"Recall@0.5 {name}: {rec:.4f}  (target >0.75)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    evaluate(args.data_dir, args.checkpoint)
