import numpy as np
import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from octane_terrain_mapping.instance_extraction import extract_instances


def test_detects_single_rock():
    semantic = np.zeros((200, 200), dtype=np.int64)
    semantic[97:104, 97:104] = 1
    instances = extract_instances(semantic, cell_size=0.05)
    rocks = [i for i in instances if i["type"] == 0]
    assert len(rocks) == 1
    assert abs(rocks[0]["x"]) < 0.5
    assert abs(rocks[0]["y"]) < 0.5
    assert 0.2 < rocks[0]["diameter"] < 0.6


def test_detects_crater_and_rock_separately():
    semantic = np.zeros((200, 200), dtype=np.int64)
    semantic[50:56, 50:56] = 2
    semantic[150:157, 150:157] = 1
    instances = extract_instances(semantic, cell_size=0.05)
    assert sum(1 for i in instances if i["type"] == 0) == 1
    assert sum(1 for i in instances if i["type"] == 1) == 1


def test_empty_semantic_returns_empty():
    semantic = np.zeros((200, 200), dtype=np.int64)
    assert extract_instances(semantic) == []
