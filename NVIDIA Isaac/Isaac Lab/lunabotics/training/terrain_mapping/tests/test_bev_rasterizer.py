import numpy as np
import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from bev_rasterizer import rasterize_bev

def _pack(r, g, b):
    return np.array((r << 16) | (g << 8) | b, dtype=np.uint32).view(np.float32)

def test_output_shape():
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, 3] = _pack(255, 0, 0)
    assert rasterize_bev(pts).shape == (6, 200, 200)

def test_empty_cloud():
    assert rasterize_bev(np.zeros((0, 4), dtype=np.float32)).sum() == 0.0

def test_single_point_center():
    pts = np.array([[0.0, 0.0, 1.5, _pack(255, 0, 0)]], dtype=np.float32)
    g = rasterize_bev(pts)
    cx = cy = 100
    assert g[0, cx, cy] == pytest.approx(1.5)
    assert g[5, cx, cy] == pytest.approx(1.0)

def test_out_of_range_ignored():
    pts = np.array([[100.0, 0.0, 1.0, _pack(0, 255, 0)]], dtype=np.float32)
    assert rasterize_bev(pts).sum() == 0.0
