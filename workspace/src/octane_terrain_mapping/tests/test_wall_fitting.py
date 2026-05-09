import numpy as np
import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from octane_terrain_mapping.wall_fitting import fit_walls


def test_detects_horizontal_wall():
    semantic = np.zeros((200, 200), dtype=np.int64)
    semantic[100, 50:150] = 3
    walls = fit_walls(semantic, cell_size=0.05)
    assert len(walls) == 1
    length = np.hypot(walls[0]["x2"] - walls[0]["x1"], walls[0]["y2"] - walls[0]["y1"])
    assert length > 2.0


def test_empty_returns_no_walls():
    assert fit_walls(np.zeros((200, 200), dtype=np.int64)) == []
