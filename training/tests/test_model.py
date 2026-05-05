import os, sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.model import TerrainModel


def test_output_shapes():
    model = TerrainModel()
    images   = torch.randn(2, 12, 3, 224, 224)
    rotation = torch.randn(2, 6)
    preds = model(images, rotation)
    assert preds['height'].shape  == (2, 200, 200)
    assert preds['rocks'].shape   == (2, 200, 200)
    assert preds['craters'].shape == (2, 200, 200)
    assert preds['walls'].shape   == (2, 200, 200)


def test_sigmoid_outputs_bounded():
    model = TerrainModel()
    images   = torch.randn(1, 12, 3, 224, 224)
    rotation = torch.randn(1, 6)
    preds = model(images, rotation)
    for key in ('rocks', 'craters', 'walls'):
        assert preds[key].min().item() >= 0.0 - 1e-6
        assert preds[key].max().item() <= 1.0 + 1e-6


def test_height_has_no_activation():
    model = TerrainModel()
    torch.manual_seed(0)
    images   = torch.randn(1, 12, 3, 224, 224)
    rotation = torch.randn(1, 6)
    preds = model(images, rotation)
    height_range = preds['height'].max().item() - preds['height'].min().item()
    assert height_range > 0.0


def test_batch_size_one():
    model = TerrainModel()
    images   = torch.randn(1, 12, 3, 224, 224)
    rotation = torch.randn(1, 6)
    preds = model(images, rotation)
    assert preds['height'].shape == (1, 200, 200)


def test_pose_tensor_registered_as_buffer():
    model = TerrainModel()
    assert 'pose_tensor' in dict(model.named_buffers())


def test_parameter_count_under_20m():
    model = TerrainModel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 20_000_000, f"Too many params: {n_params:,}"
