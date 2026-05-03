import numpy as np
import pytest
import sys, pathlib
from unittest.mock import MagicMock

# Mock ROS2 packages that are unavailable outside a ROS2 environment
sys.modules.setdefault("sensor_msgs", MagicMock())
sys.modules.setdefault("sensor_msgs.msg", MagicMock())
sys.modules.setdefault("sensor_msgs_py", MagicMock())
sys.modules.setdefault("sensor_msgs_py.point_cloud2", MagicMock())

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from octane_terrain_mapping.bev_rasterizer import rasterize_from_ros_points


def _pack(r, g, b):
    return np.array((r << 16) | (g << 8) | b, dtype=np.uint32).view(np.float32)


def test_shape():
    pts = np.zeros((50, 4), dtype=np.float32)
    pts[:, 3] = _pack(0, 255, 0)
    assert rasterize_from_ros_points(pts).shape == (6, 200, 200)


def test_empty():
    assert rasterize_from_ros_points(np.zeros((0, 4), dtype=np.float32)).sum() == 0.0
