from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HeightmapLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        # pred: (B,1,H,W)  target: (B,H,W)  occupancy: (B,H,W) float
        diff = (pred.squeeze(1) - target) ** 2
        return (diff * occupancy).sum() / (occupancy.sum() + 1e-6)


_SEMANTIC_WEIGHTS = torch.tensor([0.3, 2.0, 2.0, 2.0])  # free, rock, crater, wall


class SemanticLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B,4,H,W) logits  target: (B,H,W) long
        return F.cross_entropy(pred, target, weight=_SEMANTIC_WEIGHTS.to(pred.device))


class DetectionLoss(nn.Module):
    def forward(self, pred: torch.Tensor, gt_objects: list[list[dict]]) -> torch.Tensor:
        # pred: (B,110,5) [x,y,diam,conf,type]
        # gt_objects: list of B lists of dicts {type,x,y,diameter}
        total = pred.new_zeros(())
        for b_pred, b_gt in zip(pred, gt_objects):
            conf_target = torch.zeros(b_pred.shape[0], device=pred.device)
            if len(b_gt) == 0:
                total = total + F.binary_cross_entropy_with_logits(b_pred[:, 3], conf_target)
                continue
            gt_tensor = torch.tensor(
                [[o["x"], o["y"], o["diameter"], 1.0, float(o["type"])] for o in b_gt],
                dtype=torch.float32, device=pred.device,
            )
            n_gt = len(b_gt)
            with torch.no_grad():
                cost = np.linalg.norm(
                    b_pred[:n_gt, :2].cpu().numpy()[:, None] - gt_tensor[:, :2].cpu().numpy()[None, :],
                    axis=-1,
                )
                row_ind, col_ind = linear_sum_assignment(cost)
            matched_pred = b_pred[row_ind]
            matched_gt = gt_tensor[col_ind]
            reg_loss = F.l1_loss(matched_pred[:, :3], matched_gt[:, :3])
            conf_target[row_ind] = 1.0
            conf_loss = F.binary_cross_entropy_with_logits(b_pred[:, 3], conf_target)
            total = total + reg_loss + conf_loss
        return total / pred.shape[0]


class TotalLoss(nn.Module):
    def __init__(self, w_height: float = 1.0, w_semantic: float = 0.5, w_detect: float = 0.3):
        super().__init__()
        self.w_height = w_height
        self.w_semantic = w_semantic
        self.w_detect = w_detect
        self.height_loss = HeightmapLoss()
        self.semantic_loss = SemanticLoss()
        self.detect_loss = DetectionLoss()

    def forward(self, preds, targets) -> torch.Tensor:
        height_pred, semantic_pred, detect_pred = preds
        height_gt, semantic_gt, objects_gt, occupancy = targets
        l_h = self.height_loss(height_pred, height_gt, occupancy)
        l_s = self.semantic_loss(semantic_pred, semantic_gt)
        l_d = self.detect_loss(detect_pred, objects_gt)
        return self.w_height * l_h + self.w_semantic * l_s + self.w_detect * l_d
