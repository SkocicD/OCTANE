import numpy as np
import sys, pathlib
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "source" / "lunabotics"))

from lunabotics.terrains.crater import carve_craters, CraterCfg

def test_carve_craters_returns_records():
    cfg = CraterCfg(
        horizontal_scale=0.05,
        crater_count_range=(2, 2),
        crater_diameter_range=(0.4, 0.4),
        crater_depth_ratio=0.25,
    )
    hf = np.zeros((200, 200), dtype=np.float32)
    updated_hf, records = carve_craters(hf, cfg, arena_size=(10.0, 10.0), rng=np.random.default_rng(42))
    assert len(records) == 2
    for r in records:
        assert set(r.keys()) == {"cx", "cy", "diameter", "depth"}
        assert 0.39 < r["diameter"] < 0.41
        assert r["depth"] == pytest.approx(r["diameter"] * 0.25)

def test_carve_craters_depresses_height_field():
    cfg = CraterCfg(
        horizontal_scale=0.05,
        crater_count_range=(1, 1),
        crater_diameter_range=(0.5, 0.5),
        crater_depth_ratio=0.25,
    )
    hf = np.zeros((200, 200), dtype=np.float32)
    updated_hf, records = carve_craters(hf, cfg, arena_size=(10.0, 10.0), rng=np.random.default_rng(0))
    r = records[0]
    cx_idx = int(r["cx"] / 0.05)
    cy_idx = int(r["cy"] / 0.05)
    assert updated_hf[cx_idx, cy_idx] < -0.05
