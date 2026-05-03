"""Load TerrainMappingModel — isolated so sys.path manipulation stays contained."""
import pathlib, sys


def load_terrain_model(checkpoint: str = ""):
    candidates = [
        pathlib.Path(__file__).parents[5] / "training" / "terrain_mapping",
        pathlib.Path("/mnt/c/Users/adam.carbone/source/repos/ros_intro")
            / "NVIDIA Isaac" / "Isaac Lab" / "lunabotics" / "training" / "terrain_mapping",
    ]
    for p in candidates:
        if (p / "model.py").exists():
            sys.path.insert(0, str(p))
            break

    import torch
    from model import TerrainMappingModel  # type: ignore[import]
    model = TerrainMappingModel()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model
