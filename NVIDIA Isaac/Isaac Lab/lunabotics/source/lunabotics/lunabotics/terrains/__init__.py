"""Lunabotics custom terrain generators."""

try:
    from .regolith import RegolithTerrainCfg, regolith_terrain
    __all__ = ["regolith_terrain", "RegolithTerrainCfg"]
except ModuleNotFoundError:
    __all__ = []  # Isaac Lab not available (e.g. standalone unit tests)
