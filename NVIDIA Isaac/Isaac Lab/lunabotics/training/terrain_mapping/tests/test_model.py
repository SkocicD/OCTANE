import pytest
import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from model import TerrainMappingModel


def test_output_shapes():
    model = TerrainMappingModel()
    model.eval()
    with torch.no_grad():
        height, semantic, detections = model(torch.zeros(2, 6, 200, 200))
    assert height.shape == (2, 1, 200, 200)
    assert semantic.shape == (2, 4, 200, 200)
    assert detections.shape == (2, 110, 5)


def test_no_nan():
    model = TerrainMappingModel()
    model.eval()
    with torch.no_grad():
        height, semantic, detections = model(torch.rand(1, 6, 200, 200))
    assert not torch.isnan(height).any()
    assert not torch.isnan(semantic).any()
    assert not torch.isnan(detections).any()
