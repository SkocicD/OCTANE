import pytest
import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from losses import HeightmapLoss, SemanticLoss, DetectionLoss, TotalLoss


def test_heightmap_loss_ignores_empty():
    loss_fn = HeightmapLoss()
    pred = torch.ones(2, 1, 200, 200)
    target = torch.zeros(2, 200, 200)
    occupancy = torch.zeros(2, 200, 200)
    assert loss_fn(pred, target, occupancy).item() == 0.0


def test_heightmap_loss_nonzero_when_occupied():
    loss_fn = HeightmapLoss()
    pred = torch.ones(1, 1, 200, 200)
    target = torch.zeros(1, 200, 200)
    occupancy = torch.ones(1, 200, 200)
    assert loss_fn(pred, target, occupancy).item() > 0.0


def test_semantic_loss_scalar():
    loss_fn = SemanticLoss()
    loss = loss_fn(torch.randn(2, 4, 200, 200), torch.randint(0, 4, (2, 200, 200)))
    assert loss.shape == () and loss.item() > 0.0


def test_detection_loss_no_gt_no_nan():
    loss_fn = DetectionLoss()
    pred = torch.randn(2, 110, 5)
    assert not torch.isnan(loss_fn(pred, [[], []]))


def test_total_loss_scalar_no_nan():
    criterion = TotalLoss()
    height_pred = torch.randn(2, 1, 200, 200)
    semantic_pred = torch.randn(2, 4, 200, 200)
    detect_pred = torch.randn(2, 110, 5)
    height_gt = torch.randn(2, 200, 200)
    semantic_gt = torch.randint(0, 4, (2, 200, 200))
    occupancy = torch.ones(2, 200, 200)
    objects_gt = [[{"type": 0, "x": 1.0, "y": 0.5, "diameter": 0.35}], []]
    loss = criterion(
        (height_pred, semantic_pred, detect_pred),
        (height_gt, semantic_gt, objects_gt, occupancy),
    )
    assert loss.shape == () and not torch.isnan(loss)
