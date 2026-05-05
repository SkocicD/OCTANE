# training/tests/test_cameras.py
import math
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from training.cameras import POSE_TENSOR, INPUT_SLOTS


def test_pose_tensor_shape():
    assert POSE_TENSOR.shape == (12, 9)


def test_pose_tensor_dtype():
    assert POSE_TENSOR.dtype == np.float32


def test_rgb_slots_have_is_rgb_flag():
    for i in range(6):
        assert POSE_TENSOR[i, 7] == pytest.approx(1.0)
        assert POSE_TENSOR[i, 8] == pytest.approx(0.0)


def test_depth_slots_have_is_depth_flag():
    for i in range(6, 12):
        assert POSE_TENSOR[i, 7] == pytest.approx(0.0)
        assert POSE_TENSOR[i, 8] == pytest.approx(1.0)


def test_left_front_rgb_position():
    assert POSE_TENSOR[0, 0] == pytest.approx(0.472)
    assert POSE_TENSOR[0, 1] == pytest.approx(0.232)
    assert POSE_TENSOR[0, 2] == pytest.approx(0.376)


def test_left_front_rgb_and_depth_share_pose():
    assert np.allclose(POSE_TENSOR[0, :7], POSE_TENSOR[6, :7])


def test_input_slots_length():
    assert len(INPUT_SLOTS) == 12


def test_orbbec_slots():
    assert INPUT_SLOTS[5][0] == 'orbbec_depth'
    assert INPUT_SLOTS[11][0] == 'orbbec_depth'
    assert INPUT_SLOTS[5][1] == 'rgb'
    assert INPUT_SLOTS[11][1] == 'depth'


def test_left_front_rgb_angles():
    # near_rgb_left_front: pitch=135 deg, yaw=45 deg
    assert POSE_TENSOR[0, 3] == pytest.approx(math.sin(math.radians(135.0)), abs=1e-5)
    assert POSE_TENSOR[0, 4] == pytest.approx(math.cos(math.radians(135.0)), abs=1e-5)
    assert POSE_TENSOR[0, 5] == pytest.approx(math.sin(math.radians(45.0)),  abs=1e-5)
    assert POSE_TENSOR[0, 6] == pytest.approx(math.cos(math.radians(45.0)),  abs=1e-5)


def test_orbbec_angles():
    # orbbec_depth: pitch=120 deg, yaw=0 deg (different pitch from near cameras)
    assert POSE_TENSOR[5, 3] == pytest.approx(math.sin(math.radians(120.0)), abs=1e-5)
    assert POSE_TENSOR[5, 4] == pytest.approx(math.cos(math.radians(120.0)), abs=1e-5)
    assert POSE_TENSOR[5, 5] == pytest.approx(math.sin(math.radians(0.0)),   abs=1e-5)
    assert POSE_TENSOR[5, 6] == pytest.approx(math.cos(math.radians(0.0)),   abs=1e-5)
