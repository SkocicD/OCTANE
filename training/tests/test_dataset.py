import math
import os
import sys
import tempfile
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.dataset import build_object_heatmap, build_wall_mask, TerrainDataset

GRID = 200
CELL = 0.05
HALF = GRID * CELL / 2  # 5.0 m


def _cell(v):
    return int((v + HALF) / CELL)


def test_heatmap_peak_at_object_center():
    objects = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    h = build_object_heatmap(objects, class_id=0)
    assert h.shape == (GRID, GRID)
    cx, cy = _cell(0.0), _cell(0.0)
    assert h[cx, cy] == pytest.approx(1.0, abs=0.01)


def test_heatmap_decays_from_center():
    objects = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    h = build_object_heatmap(objects, class_id=0)
    cx, cy = _cell(0.0), _cell(0.0)
    assert h[cx, cy + 15] < h[cx, cy]


def test_heatmap_filters_wrong_class():
    objects = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    h = build_object_heatmap(objects, class_id=0)
    assert h.max() == pytest.approx(0.0)


def test_heatmap_empty_objects():
    h = build_object_heatmap(np.zeros((0, 4), dtype=np.float32), class_id=0)
    assert h.shape == (GRID, GRID)
    assert h.max() == pytest.approx(0.0)


def test_wall_mask_center_row():
    walls = np.array([[-2.0, 0.0, 2.0, 0.0]], dtype=np.float32)
    m = build_wall_mask(walls)
    assert m.shape == (GRID, GRID)
    cy = _cell(0.0)
    cx1, cx2 = _cell(-2.0), _cell(2.0)
    assert m[cx1:cx2, cy - 1:cy + 2].max() > 0


def test_wall_mask_empty():
    m = build_wall_mask(np.zeros((0, 4), dtype=np.float32))
    assert m.max() == pytest.approx(0.0)


def _make_fake_episode(root, ep_id):
    serials_rgb = ['left_front', 'left_side', 'right_front', 'right_side',
                   'back_rear', 'depth_cam_rgb']
    serials_depth = ['left_front', 'left_side', 'right_front', 'right_side', 'back_rear']
    rgb_sizes = {
        'left_front': (480, 360), 'left_side': (480, 360),
        'right_front': (480, 360), 'right_side': (480, 360),
        'back_rear': (480, 360), 'depth_cam_rgb': (640, 480),
    }

    for serial in serials_rgb:
        w, h = rgb_sizes[serial]
        folder = os.path.join(root, 'images', serial)
        os.makedirs(folder, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        img.save(os.path.join(folder, f'{ep_id}.jpg'))

    folder = os.path.join(root, 'images', 'depth_cam_d')
    os.makedirs(folder, exist_ok=True)
    img = Image.fromarray(np.random.randint(0, 255, (480, 640), dtype=np.uint8))
    img.save(os.path.join(folder, f'{ep_id}.png'))

    for serial in serials_depth:
        folder = os.path.join(root, 'depth', serial)
        os.makedirs(folder, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8))
        img.save(os.path.join(folder, f'{ep_id}.png'))

    gt_folder = os.path.join(root, 'gt')
    os.makedirs(gt_folder, exist_ok=True)
    np.savez_compressed(
        os.path.join(gt_folder, f'{ep_id}_gt.npz'),
        height_gt=np.random.randn(200, 200).astype(np.float32),
        objects_gt=np.array([[0.5, 0.5, 1.0, 0.0], [-1.0, -1.0, 0.8, 1.0]], dtype=np.float32),
        walls_gt=np.array([[-2.0, 1.0, 2.0, 1.0]], dtype=np.float32),
        robot_yaw=np.float32(0.3),
        robot_pitch=np.float32(0.05),
        robot_roll=np.float32(-0.02),
    )


def test_dataset_getitem_shapes():
    with tempfile.TemporaryDirectory() as root:
        ep_id = 'ep_000000'
        _make_fake_episode(root, ep_id)
        depth_stats = {
            'mean': [0.485, 0.456, 0.406],
            'std':  [0.229, 0.224, 0.225],
        }
        ds = TerrainDataset(root, [ep_id], depth_stats=depth_stats, augment=False)
        assert len(ds) == 1
        images, rotation, gt = ds[0]
        assert images.shape == (12, 3, 224, 224)
        assert rotation.shape == (6,)
        assert gt['height'].shape  == (200, 200)
        assert gt['rocks'].shape   == (200, 200)
        assert gt['craters'].shape == (200, 200)
        assert gt['walls'].shape   == (200, 200)


def test_dataset_gt_values_bounded():
    with tempfile.TemporaryDirectory() as root:
        ep_id = 'ep_000000'
        _make_fake_episode(root, ep_id)
        depth_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}
        ds = TerrainDataset(root, [ep_id], depth_stats=depth_stats)
        _, _, gt = ds[0]
        assert gt['rocks'].min() >= 0.0
        assert gt['rocks'].max() <= 1.0
        assert gt['craters'].min() >= 0.0
        assert gt['craters'].max() <= 1.0
        assert gt['walls'].min() >= 0.0
        assert gt['walls'].max() <= 1.0


def test_dataset_rotation_encoding():
    with tempfile.TemporaryDirectory() as root:
        ep_id = 'ep_000000'
        _make_fake_episode(root, ep_id)
        depth_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}
        ds = TerrainDataset(root, [ep_id], depth_stats=depth_stats)
        _, rotation, _ = ds[0]
        for i in range(0, 6, 2):
            assert rotation[i]**2 + rotation[i+1]**2 == pytest.approx(1.0, abs=1e-5)
