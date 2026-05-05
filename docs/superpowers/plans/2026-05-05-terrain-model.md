# Terrain Mapping Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyTorch training pipeline that takes 12 camera images + robot rotation and predicts height, rock, crater, and wall BEV maps of the terrain.

**Architecture:** Shared EfficientNet-B0 backbone encodes each of the 12 images independently. Camera pose embeddings (from known extrinsics) are added to each image's features. All 12 feature maps are channel-concatenated and fused, robot rotation is embedded and added, then a BEV decoder upsamples to 200×200 with four output heads.

**Tech Stack:** Python 3.10+, PyTorch 2.x, torchvision, numpy, opencv-python, Pillow, pyyaml, tqdm, pytest

---

## File Map

```
training/
  cameras.py          fixed camera pose constants for all 12 input slots
  dataset.py          TerrainDataset + GT heatmap builders + depth stats
  model.py            TerrainModel (encoder, pose embed, fusion, decoder, heads)
  train.py            training + validation loop, checkpointing, splits
  export.py           ONNX + TensorRT export
  config.yaml         all hyperparameters and paths
  tests/
    test_cameras.py
    test_dataset.py
    test_model.py
    test_train.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `training/config.yaml`
- Create: `training/requirements.txt`
- Create: `training/cameras.py`
- Create: `training/tests/__init__.py`
- Create: `training/tests/test_cameras.py`

- [ ] **Step 1: Write `training/requirements.txt`**

```
torch>=2.0.0
torchvision>=0.15.0
numpy
opencv-python
Pillow
pyyaml
tqdm
pytest
```

- [ ] **Step 2: Write `training/config.yaml`**

```yaml
data:
  root: "E:/terrain_data"
  splits_file: "training/splits.json"
  depth_stats_file: "training/depth_stats.json"

training:
  batch_size: 16
  epochs: 100
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  val_ratio: 0.2
  seed: 42

checkpoints:
  dir: "training/checkpoints"
  save_every: 10

model:
  feat_proj_dim: 128
  pose_embed_dim: 32
  fusion_dim: 512
```

- [ ] **Step 3: Write failing test for cameras.py**

```python
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
    # Slots 0-5 are RGB; last two values are [is_rgb=1, is_depth=0]
    for i in range(6):
        assert POSE_TENSOR[i, 7] == pytest.approx(1.0)
        assert POSE_TENSOR[i, 8] == pytest.approx(0.0)


def test_depth_slots_have_is_depth_flag():
    # Slots 6-11 are depth; last two values are [is_rgb=0, is_depth=1]
    for i in range(6, 12):
        assert POSE_TENSOR[i, 7] == pytest.approx(0.0)
        assert POSE_TENSOR[i, 8] == pytest.approx(1.0)


def test_left_front_rgb_position():
    # Input slot 0 = near_rgb_left_front, pos=(0.472, 0.232, 0.376)
    assert POSE_TENSOR[0, 0] == pytest.approx(0.472)
    assert POSE_TENSOR[0, 1] == pytest.approx(0.232)
    assert POSE_TENSOR[0, 2] == pytest.approx(0.376)


def test_left_front_rgb_and_depth_share_pose():
    # Slot 0 (left_front RGB) and slot 6 (left_front depth) share x,y,z,pitch,yaw
    assert np.allclose(POSE_TENSOR[0, :7], POSE_TENSOR[6, :7])


def test_input_slots_length():
    assert len(INPUT_SLOTS) == 12


def test_orbbec_slots():
    # Slot 5 = orbbec RGB, slot 11 = orbbec depth — same camera
    assert INPUT_SLOTS[5][0] == 'orbbec_depth'
    assert INPUT_SLOTS[11][0] == 'orbbec_depth'
    assert INPUT_SLOTS[5][1] == 'rgb'
    assert INPUT_SLOTS[11][1] == 'depth'
```

- [ ] **Step 4: Run test to confirm it fails**

```
cd C:\Users\adam.carbone\source\repos\ros_intro
python -m pytest training/tests/test_cameras.py -v
```
Expected: `ModuleNotFoundError: No module named 'training.cameras'`

- [ ] **Step 5: Write `training/cameras.py`**

```python
import math
import numpy as np

# Physical cameras in order used for the 12 input slots.
# Poses from workspace/src/octane/octane/config/cameras.yaml (dev branch).
# pos: meters from base_link. rot: degrees Euler (roll always 0).

def _r(deg):
    return math.radians(deg)

_POSES = {
    'near_rgb_left_front':  (0.472,  0.232, 0.376, 135.0,  45.0),
    'near_rgb_left_side':   (0.000,  0.321, 0.376, 135.0,  90.0),
    'near_rgb_right_front': (0.472, -0.232, 0.376, 135.0, -45.0),
    'near_rgb_right_side':  (0.000, -0.321, 0.376, 135.0, -90.0),
    'near_rgb_back_rear':   (-0.570, 0.000, 0.467, 135.0, 180.0),
    'orbbec_depth':         (0.442,  0.038, 0.661, 120.0,   0.0),
}

# Each input slot: (camera_name, modality)
INPUT_SLOTS = [
    ('near_rgb_left_front',  'rgb'),    # 0
    ('near_rgb_left_side',   'rgb'),    # 1
    ('near_rgb_right_front', 'rgb'),    # 2
    ('near_rgb_right_side',  'rgb'),    # 3
    ('near_rgb_back_rear',   'rgb'),    # 4
    ('orbbec_depth',         'rgb'),    # 5
    ('near_rgb_left_front',  'depth'),  # 6
    ('near_rgb_left_side',   'depth'),  # 7
    ('near_rgb_right_front', 'depth'),  # 8
    ('near_rgb_right_side',  'depth'),  # 9
    ('near_rgb_back_rear',   'depth'),  # 10
    ('orbbec_depth',         'depth'),  # 11
]

# Swap pairs for horizontal-flip augmentation: (left, right) camera index pairs
FLIP_SWAP_PAIRS = [(0, 2), (1, 3), (6, 8), (7, 9)]

def _make_pose_tensor():
    rows = []
    for cam_name, modality in INPUT_SLOTS:
        x, y, z, pitch_deg, yaw_deg = _POSES[cam_name]
        row = [
            x, y, z,
            math.sin(_r(pitch_deg)), math.cos(_r(pitch_deg)),
            math.sin(_r(yaw_deg)),   math.cos(_r(yaw_deg)),
            1.0 if modality == 'rgb' else 0.0,
            1.0 if modality == 'depth' else 0.0,
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float32)

# Fixed (12, 9) pose tensor: [x, y, z, sin_pitch, cos_pitch, sin_yaw, cos_yaw, is_rgb, is_depth]
POSE_TENSOR = _make_pose_tensor()
```

- [ ] **Step 6: Run tests to confirm they pass**

```
python -m pytest training/tests/test_cameras.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 7: Commit**

```
git add training/cameras.py training/config.yaml training/requirements.txt training/tests/__init__.py training/tests/test_cameras.py
git commit -m "feat: terrain model scaffold — cameras, config, requirements"
```

---

## Task 2: GT Heatmap Builders + Dataset

**Files:**
- Create: `training/dataset.py`
- Create: `training/tests/test_dataset.py`

- [ ] **Step 1: Write failing tests for GT builders**

```python
# training/tests/test_dataset.py
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


# ── GT builder tests ─────────────────────────────────────────────────────────

def test_heatmap_peak_at_object_center():
    # Rock at robot center (0, 0), diameter 1.0 m → σ = 10 cells
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
    # Only craters (class=1), asking for rocks (class=0) → zero map
    objects = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    h = build_object_heatmap(objects, class_id=0)
    assert h.max() == pytest.approx(0.0)


def test_heatmap_empty_objects():
    h = build_object_heatmap(np.zeros((0, 4), dtype=np.float32), class_id=0)
    assert h.shape == (GRID, GRID)
    assert h.max() == pytest.approx(0.0)


def test_wall_mask_center_row():
    # Horizontal wall from (-2, 0) to (2, 0) — runs along rx axis at ry=0
    walls = np.array([[-2.0, 0.0, 2.0, 0.0]], dtype=np.float32)
    m = build_wall_mask(walls)
    assert m.shape == (GRID, GRID)
    # Column at ry=0 should be nonzero in the rx=-2 to rx=2 range
    cy = _cell(0.0)
    cx1, cx2 = _cell(-2.0), _cell(2.0)
    assert m[cx1:cx2, cy - 1:cy + 2].max() > 0


def test_wall_mask_empty():
    m = build_wall_mask(np.zeros((0, 4), dtype=np.float32))
    assert m.max() == pytest.approx(0.0)


# ── Dataset tests ─────────────────────────────────────────────────────────────

def _make_fake_episode(root, ep_id):
    """Write minimal fake image files and NPZ for one episode."""
    serials_rgb = ['left_front', 'left_side', 'right_front', 'right_side',
                   'back_rear', 'depth_cam_rgb']
    serials_depth = ['left_front', 'left_side', 'right_front', 'right_side', 'back_rear']
    rgb_sizes = {
        'left_front': (480, 360), 'left_side': (480, 360),
        'right_front': (480, 360), 'right_side': (480, 360),
        'back_rear': (480, 360), 'depth_cam_rgb': (640, 480),
    }
    depth_cam_d_size = (640, 480)

    for serial in serials_rgb:
        w, h = rgb_sizes[serial]
        folder = os.path.join(root, 'images', serial)
        os.makedirs(folder, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        img.save(os.path.join(folder, f'{ep_id}.jpg'))

    folder = os.path.join(root, 'images', 'depth_cam_d')
    os.makedirs(folder, exist_ok=True)
    h, w = depth_cam_d_size[1], depth_cam_d_size[0]
    img = Image.fromarray(np.random.randint(0, 255, (h, w), dtype=np.uint8))
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
        # rotation is [sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw]
        # Each sin/cos pair should satisfy sin²+cos² = 1
        for i in range(0, 6, 2):
            assert rotation[i]**2 + rotation[i+1]**2 == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest training/tests/test_dataset.py -v
```
Expected: `ModuleNotFoundError: No module named 'training.dataset'`

- [ ] **Step 3: Write `training/dataset.py`**

```python
import os
import json
import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

GRID_SIZE = 200
CELL_SIZE = 0.05
BEV_HALF  = GRID_SIZE * CELL_SIZE / 2  # 5.0 m

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _to_cell(v: float) -> int:
    return int((v + BEV_HALF) / CELL_SIZE)


def build_object_heatmap(objects_gt: np.ndarray, class_id: int,
                          grid_size: int = GRID_SIZE,
                          cell_size: float = CELL_SIZE) -> np.ndarray:
    """Gaussian heatmap for one object class. objects_gt shape: (N, 4) [rx, ry, diameter, class]."""
    half = grid_size * cell_size / 2
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    for obj in objects_gt:
        rx, ry, diameter, cls = float(obj[0]), float(obj[1]), float(obj[2]), int(obj[3])
        if cls != class_id:
            continue
        cx = int((rx + half) / cell_size)
        cy = int((ry + half) / cell_size)
        if not (0 <= cx < grid_size and 0 <= cy < grid_size):
            continue
        sigma = max(1.0, (diameter / 2.0) / cell_size)
        radius = int(sigma * 3)
        x0, x1 = max(0, cx - radius), min(grid_size, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(grid_size, cy + radius + 1)
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        blob = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        heatmap[x0:x1, y0:y1] = np.maximum(heatmap[x0:x1, y0:y1], blob)
    return heatmap


def build_wall_mask(walls_gt: np.ndarray,
                    grid_size: int = GRID_SIZE,
                    cell_size: float = CELL_SIZE) -> np.ndarray:
    """Binary mask with wall segments rasterized as 3-pixel-thick lines.
    walls_gt shape: (M, 4) [rx1, ry1, rx2, ry2]."""
    half = grid_size * cell_size / 2
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    for wall in walls_gt:
        rx1, ry1, rx2, ry2 = float(wall[0]), float(wall[1]), float(wall[2]), float(wall[3])
        # BEV: axis 0 = rx (fwd), axis 1 = ry (lateral)
        # cv2.line uses (col, row) = (axis1, axis0)
        px1 = int((rx1 + half) / cell_size)
        py1 = int((ry1 + half) / cell_size)
        px2 = int((rx2 + half) / cell_size)
        py2 = int((ry2 + half) / cell_size)
        cv2.line(mask, (py1, px1), (py2, px2), 1.0, thickness=3)
    return mask


def compute_depth_stats(data_root: str, episode_ids: list,
                         max_episodes: int = 500) -> dict:
    """Sample up to max_episodes to compute per-channel mean/std for depth images."""
    serials = ['left_front', 'left_side', 'right_front', 'right_side', 'back_rear']
    sample_ids = episode_ids[:max_episodes]
    pixels = []
    for ep_id in sample_ids:
        for serial in serials:
            path = os.path.join(data_root, 'depth', serial, f'{ep_id}.png')
            if not os.path.exists(path):
                continue
            img = np.array(Image.open(path).convert('RGB').resize((224, 224))) / 255.0
            pixels.append(img.reshape(-1, 3))
    if not pixels:
        return {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}
    all_pixels = np.concatenate(pixels, axis=0)
    return {
        'mean': all_pixels.mean(axis=0).tolist(),
        'std':  all_pixels.std(axis=0).clip(1e-6).tolist(),
    }


# Image file paths for each of the 12 input slots.
# (folder_under_images_or_depth, extension, is_in_depth_dir)
_SLOT_PATHS = [
    ('images/left_front',    '.jpg', False),
    ('images/left_side',     '.jpg', False),
    ('images/right_front',   '.jpg', False),
    ('images/right_side',    '.jpg', False),
    ('images/back_rear',     '.jpg', False),
    ('images/depth_cam_rgb', '.jpg', False),
    ('depth/left_front',     '.png', False),
    ('depth/left_side',      '.png', False),
    ('depth/right_front',    '.png', False),
    ('depth/right_side',     '.png', False),
    ('depth/back_rear',      '.png', False),
    ('images/depth_cam_d',   '.png', False),
]

# Indices 0-5 use ImageNet norm; 6-11 use depth_stats norm
_RGB_SLOTS   = set(range(6))
_DEPTH_SLOTS = set(range(6, 12))

# Horizontal-flip augmentation: swap these index pairs, then flip all images
_FLIP_SWAP_PAIRS = [(0, 2), (1, 3), (6, 8), (7, 9)]


class TerrainDataset(Dataset):
    def __init__(self, data_root: str, episode_ids: list,
                 depth_stats: dict, augment: bool = False):
        self.root        = data_root
        self.episode_ids = episode_ids
        self.augment     = augment

        self._rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self._depth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=depth_stats['mean'], std=depth_stats['std']),
        ])

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        ep_id = self.episode_ids[idx]
        images   = self._load_images(ep_id)    # (12, 3, 224, 224)
        rotation = self._load_rotation(ep_id)  # (6,)
        gt       = self._load_gt(ep_id)        # dict of (200,200) tensors

        if self.augment and torch.rand(1).item() < 0.5:
            images, gt = self._hflip(images, gt)

        return images, rotation, gt

    def _load_images(self, ep_id: str) -> torch.Tensor:
        tensors = []
        for i, (rel_path, ext, _) in enumerate(_SLOT_PATHS):
            path = os.path.join(self.root, rel_path, f'{ep_id}{ext}')
            img  = Image.open(path).convert('RGB')
            t    = self._rgb_transform(img) if i in _RGB_SLOTS else self._depth_transform(img)
            tensors.append(t)
        return torch.stack(tensors, dim=0)  # (12, 3, 224, 224)

    def _load_rotation(self, ep_id: str) -> torch.Tensor:
        npz = np.load(os.path.join(self.root, 'gt', f'{ep_id}_gt.npz'))
        roll  = float(npz.get('robot_roll',  np.float32(0.0)))
        pitch = float(npz.get('robot_pitch', np.float32(0.0)))
        yaw   = float(npz['robot_yaw'])
        return torch.tensor([
            math.sin(roll),  math.cos(roll),
            math.sin(pitch), math.cos(pitch),
            math.sin(yaw),   math.cos(yaw),
        ], dtype=torch.float32)

    def _load_gt(self, ep_id: str) -> dict:
        npz = np.load(os.path.join(self.root, 'gt', f'{ep_id}_gt.npz'))
        height  = torch.from_numpy(npz['height_gt'])
        objects = npz['objects_gt']
        walls   = npz['walls_gt']
        rocks   = torch.from_numpy(build_object_heatmap(objects, class_id=0))
        craters = torch.from_numpy(build_object_heatmap(objects, class_id=1))
        wall_m  = torch.from_numpy(build_wall_mask(walls))
        return {'height': height, 'rocks': rocks, 'craters': craters, 'walls': wall_m}

    def _hflip(self, images: torch.Tensor, gt: dict):
        # Flip all images horizontally
        images = torch.flip(images, dims=[-1])
        # Swap left/right camera pairs
        for a, b in _FLIP_SWAP_PAIRS:
            images[[a, b]] = images[[b, a]]
        # Flip GT maps on the lateral axis (axis 1 = ry dimension)
        gt = {k: torch.flip(v, dims=[-1]) for k, v in gt.items()}
        return images, gt
```

- [ ] **Step 4: Run tests**

```
python -m pytest training/tests/test_dataset.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```
git add training/dataset.py training/tests/test_dataset.py
git commit -m "feat: dataset GT builders and TerrainDataset"
```

---

## Task 3: Model

**Files:**
- Create: `training/model.py`
- Create: `training/tests/test_model.py`

- [ ] **Step 1: Write failing model tests**

```python
# training/tests/test_model.py
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
    # Height can be negative (terrain below robot level)
    model = TerrainModel()
    torch.manual_seed(0)
    images   = torch.randn(1, 12, 3, 224, 224)
    rotation = torch.randn(1, 6)
    preds = model(images, rotation)
    # With random weights the height head will produce values outside [0,1]
    height_range = preds['height'].max().item() - preds['height'].min().item()
    assert height_range > 0.0  # not constant


def test_batch_size_one():
    model = TerrainModel()
    images   = torch.randn(1, 12, 3, 224, 224)
    rotation = torch.randn(1, 6)
    preds = model(images, rotation)
    assert preds['height'].shape == (1, 200, 200)


def test_pose_tensor_registered_as_buffer():
    model = TerrainModel()
    # Buffers are moved with model.to(device), parameters are not
    assert 'pose_tensor' in dict(model.named_buffers())


def test_parameter_count_under_20m():
    model = TerrainModel()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 20_000_000, f"Too many params: {n_params:,}"
```

- [ ] **Step 2: Run tests to confirm failure**

```
python -m pytest training/tests/test_model.py -v
```
Expected: `ModuleNotFoundError: No module named 'training.model'`

- [ ] **Step 3: Write `training/model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.cameras import POSE_TENSOR


class PoseEmbedding(nn.Module):
    """Projects fixed 9-dim pose vectors to embed_dim feature space."""
    def __init__(self, pose_dim: int = 9, embed_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(pose_dim, embed_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.proj(pose)  # (12, embed_dim)


class UpBlock(nn.Module):
    """Bilinear upsample to target_size then double conv."""
    def __init__(self, in_ch: int, out_ch: int, target_size: int):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return self.conv(x)


class TerrainModel(nn.Module):
    """
    Inputs:
        images:   (B, 12, 3, 224, 224)
        rotation: (B, 6)  — [sin_r, cos_r, sin_p, cos_p, sin_y, cos_y]
    Outputs:
        dict with keys 'height', 'rocks', 'craters', 'walls' — each (B, 200, 200)
    """

    FEAT_DIM      = 1280   # EfficientNet-B0 output channels
    PROJ_DIM      = 128    # per-camera projection dim
    POSE_DIM      = 9
    POSE_EMBED    = 32
    FUSION_DIM    = 512
    ROT_EMBED     = 32

    def __init__(self):
        super().__init__()

        # Register pose tensor as non-trainable buffer (moves with .to(device))
        self.register_buffer('pose_tensor',
                             torch.from_numpy(POSE_TENSOR))  # (12, 9)

        # Shared image encoder — EfficientNet-B0 features
        # For 224×224 input: outputs (B, 1280, 7, 7)
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.encoder = backbone.features  # (B, 1280, 7, 7)

        # Per-camera feature projection: 1280 → PROJ_DIM
        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.FEAT_DIM, self.PROJ_DIM, 1, bias=False),
            nn.BatchNorm2d(self.PROJ_DIM),
            nn.ReLU(inplace=True),
        )

        # Camera pose embedding: 9-dim fixed → POSE_EMBED-dim learned
        self.pose_emb = PoseEmbedding(self.POSE_DIM, self.POSE_EMBED)
        # Project pose embedding to the same feature space as feat_proj output
        self.pose_to_feat = nn.Conv2d(self.POSE_EMBED, self.PROJ_DIM, 1)

        # Fusion: 12 cameras × PROJ_DIM channels → FUSION_DIM
        self.fusion = nn.Sequential(
            nn.Conv2d(12 * self.PROJ_DIM, self.FUSION_DIM, 1, bias=False),
            nn.BatchNorm2d(self.FUSION_DIM),
            nn.ReLU(inplace=True),
        )

        # Robot rotation embedding: 6 → ROT_EMBED
        self.rot_mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.ROT_EMBED),
        )
        self.rot_to_feat = nn.Conv2d(self.ROT_EMBED, self.FUSION_DIM, 1)

        # BEV decoder: 7×7 → 14 → 25 → 50 → 100 → 200
        self.decoder = nn.Sequential(
            UpBlock(self.FUSION_DIM, 256, 14),
            UpBlock(256, 128, 25),
            UpBlock(128, 64,  50),
            UpBlock(64,  32,  100),
            UpBlock(32,  32,  200),
        )

        # Output heads
        self.height_head  = nn.Conv2d(32, 1, 1)
        self.rocks_head   = nn.Conv2d(32, 1, 1)
        self.craters_head = nn.Conv2d(32, 1, 1)
        self.walls_head   = nn.Conv2d(32, 1, 1)

    def forward(self, images: torch.Tensor, rotation: torch.Tensor) -> dict:
        B = images.shape[0]

        # Encode all 12 images in a single batch pass through shared backbone
        imgs_flat = images.view(B * 12, 3, 224, 224)
        feats = self.encoder(imgs_flat)       # (B*12, 1280, 7, 7)
        feats = self.feat_proj(feats)         # (B*12, 128, 7, 7)

        # Camera pose embeddings: fixed 9-dim → 32-dim → add to features
        pose_emb = self.pose_emb(self.pose_tensor)        # (12, 32)
        pose_emb = pose_emb.unsqueeze(-1).unsqueeze(-1)   # (12, 32, 1, 1)
        # Expand to (B*12, 32, 1, 1)
        pose_emb = pose_emb.unsqueeze(0).expand(B, -1, -1, 1, 1).reshape(B * 12, self.POSE_EMBED, 1, 1)
        feats = feats + self.pose_to_feat(pose_emb)       # broadcast add

        # Fuse all 12 camera features
        feats = feats.view(B, 12 * self.PROJ_DIM, 7, 7)  # (B, 1536, 7, 7)
        bev   = self.fusion(feats)                        # (B, 512, 7, 7)

        # Robot rotation embedding
        rot_emb = self.rot_mlp(rotation)                  # (B, 32)
        rot_emb = rot_emb.unsqueeze(-1).unsqueeze(-1)     # (B, 32, 1, 1)
        bev = bev + self.rot_to_feat(rot_emb)             # broadcast add

        # Decode to BEV
        bev = self.decoder(bev)                           # (B, 32, 200, 200)

        return {
            'height':  self.height_head(bev).squeeze(1),
            'rocks':   torch.sigmoid(self.rocks_head(bev)).squeeze(1),
            'craters': torch.sigmoid(self.craters_head(bev)).squeeze(1),
            'walls':   torch.sigmoid(self.walls_head(bev)).squeeze(1),
        }
```

- [ ] **Step 4: Run tests**

```
python -m pytest training/tests/test_model.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```
git add training/model.py training/tests/test_model.py
git commit -m "feat: TerrainModel — EfficientNet encoder, pose embedding, BEV decoder"
```

---

## Task 4: Training Loop

**Files:**
- Create: `training/train.py`
- Create: `training/tests/test_train.py`

- [ ] **Step 1: Write smoke test for training**

```python
# training/tests/test_train.py
import os, sys, json, math, tempfile
import numpy as np
import torch
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.train import compute_loss, create_splits
from training.model import TerrainModel
from training.dataset import TerrainDataset
from training.tests.test_dataset import _make_fake_episode


def _fake_preds(B=2):
    return {
        'height':  torch.randn(B, 200, 200),
        'rocks':   torch.sigmoid(torch.randn(B, 200, 200)),
        'craters': torch.sigmoid(torch.randn(B, 200, 200)),
        'walls':   torch.sigmoid(torch.randn(B, 200, 200)),
    }


def _fake_targets(B=2):
    return {
        'height':  torch.randn(B, 200, 200),
        'rocks':   torch.rand(B, 200, 200),
        'craters': torch.rand(B, 200, 200),
        'walls':   (torch.rand(B, 200, 200) > 0.8).float(),
    }


def test_compute_loss_returns_scalar():
    loss = compute_loss(_fake_preds(), _fake_targets())
    assert loss.shape == torch.Size([])
    assert loss.item() > 0


def test_compute_loss_decreases_on_perfect_preds():
    targets = _fake_targets()
    perfect_preds = {k: v.clone() for k, v in targets.items()}
    # Heights need exact match; heatmaps/masks are already 0-1
    # Perfect predictions should yield lower loss than random predictions
    loss_random  = compute_loss(_fake_preds(), targets)
    loss_perfect = compute_loss(perfect_preds, targets)
    assert loss_perfect.item() < loss_random.item()


def test_create_splits():
    with tempfile.TemporaryDirectory() as root:
        gt_dir = os.path.join(root, 'gt')
        os.makedirs(gt_dir)
        for i in range(10):
            ep_id = f'ep_{i:06d}'
            np.savez(os.path.join(gt_dir, f'{ep_id}_gt.npz'),
                     height_gt=np.zeros((200, 200), dtype=np.float32),
                     objects_gt=np.zeros((0, 4), dtype=np.float32),
                     walls_gt=np.zeros((0, 4), dtype=np.float32),
                     robot_yaw=np.float32(0.0),
                     robot_pitch=np.float32(0.0),
                     robot_roll=np.float32(0.0))
        splits_file = os.path.join(root, 'splits.json')
        train_ids, val_ids = create_splits(root, splits_file, val_ratio=0.2, seed=42)
        assert len(train_ids) == 8
        assert len(val_ids)   == 2
        assert set(train_ids) | set(val_ids) == {f'ep_{i:06d}' for i in range(10)}
        # Check idempotent — reloads same splits
        train2, val2 = create_splits(root, splits_file, val_ratio=0.2, seed=42)
        assert train2 == train_ids
        assert val2   == val_ids


def test_single_training_step_updates_weights():
    with tempfile.TemporaryDirectory() as root:
        ep_id = 'ep_000000'
        _make_fake_episode(root, ep_id)
        depth_stats = {'mean': [0.5, 0.5, 0.5], 'std': [0.25, 0.25, 0.25]}
        ds = TerrainDataset(root, [ep_id], depth_stats=depth_stats)
        loader = torch.utils.data.DataLoader(ds, batch_size=1)

        model = TerrainModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        before = [p.clone() for p in model.parameters()]

        images, rotation, gt = next(iter(loader))
        preds = model(images, rotation)
        loss  = compute_loss(preds, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        changed = any(not torch.equal(a, b)
                      for a, b in zip(before, model.parameters()))
        assert changed, "No parameters were updated"
```

- [ ] **Step 2: Run tests to confirm failure**

```
python -m pytest training/tests/test_train.py -v
```
Expected: `ModuleNotFoundError: No module named 'training.train'`

- [ ] **Step 3: Write `training/train.py`**

```python
import os
import json
import random
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.model   import TerrainModel
from training.dataset import TerrainDataset, compute_depth_stats


def create_splits(data_root: str, splits_file: str,
                  val_ratio: float = 0.2, seed: int = 42):
    """Load splits from file, or create and save them if missing."""
    if os.path.exists(splits_file):
        with open(splits_file) as f:
            splits = json.load(f)
        return splits['train'], splits['val']

    gt_dir = os.path.join(data_root, 'gt')
    ep_ids = sorted(
        f.replace('_gt.npz', '')
        for f in os.listdir(gt_dir)
        if f.endswith('_gt.npz')
    )
    rng = random.Random(seed)
    rng.shuffle(ep_ids)
    n_val   = max(1, int(len(ep_ids) * val_ratio))
    val_ids = ep_ids[:n_val]
    train_ids = ep_ids[n_val:]

    os.makedirs(os.path.dirname(splits_file) or '.', exist_ok=True)
    with open(splits_file, 'w') as f:
        json.dump({'train': train_ids, 'val': val_ids}, f, indent=2)
    print(f"[train] Splits saved: {len(train_ids)} train / {len(val_ids)} val")
    return train_ids, val_ids


def compute_loss(preds: dict, targets: dict) -> torch.Tensor:
    height_loss  = F.l1_loss(preds['height'],  targets['height'])
    rocks_loss   = F.mse_loss(preds['rocks'],   targets['rocks'])
    craters_loss = F.mse_loss(preds['craters'], targets['craters'])
    walls_loss   = F.binary_cross_entropy(preds['walls'], targets['walls'])
    return height_loss + rocks_loss + craters_loss + walls_loss


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, rotation, gt in tqdm(loader, desc='train', leave=False):
        images, rotation, gt = _to_device(images, device), _to_device(rotation, device), _to_device(gt, device)
        preds = model(images, rotation)
        loss  = compute_loss(preds, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def val_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, rotation, gt in tqdm(loader, desc='val', leave=False):
            images, rotation, gt = _to_device(images, device), _to_device(rotation, device), _to_device(gt, device)
            preds = model(images, rotation)
            total += compute_loss(preds, gt).item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='training/config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    data_root    = cfg['data']['root']
    splits_file  = cfg['data']['splits_file']
    stats_file   = cfg['data']['depth_stats_file']

    train_ids, val_ids = create_splits(
        data_root, splits_file,
        val_ratio=cfg['training']['val_ratio'],
        seed=cfg['training']['seed'],
    )

    # Compute depth normalization stats once
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            depth_stats = json.load(f)
        print("[train] Loaded depth stats from cache")
    else:
        print("[train] Computing depth stats (first run)...")
        depth_stats = compute_depth_stats(data_root, train_ids)
        os.makedirs(os.path.dirname(stats_file) or '.', exist_ok=True)
        with open(stats_file, 'w') as f:
            json.dump(depth_stats, f, indent=2)
        print(f"[train] Depth stats saved to {stats_file}")

    train_ds = TerrainDataset(data_root, train_ids, depth_stats, augment=True)
    val_ds   = TerrainDataset(data_root, val_ids,   depth_stats, augment=False)

    bs = cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = TerrainModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['epochs']
    )

    ckpt_dir  = cfg['checkpoints']['dir']
    save_every = cfg['checkpoints']['save_every']
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = val_epoch(model, val_loader, device)
        scheduler.step()

        print(f"[epoch {epoch:03d}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': val_loss},
                       os.path.join(ckpt_dir, 'best.pt'))
            print(f"  -> best checkpoint saved (val={val_loss:.4f})")

        if epoch % save_every == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict()},
                       os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt'))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests**

```
python -m pytest training/tests/test_train.py -v
```
Expected: all 4 tests PASS. Note `test_single_training_step_updates_weights` may be slow (~30s on CPU).

- [ ] **Step 5: Commit**

```
git add training/train.py training/tests/test_train.py
git commit -m "feat: training loop with splits, checkpointing, loss"
```

---

## Task 5: Export

**Files:**
- Create: `training/export.py`

- [ ] **Step 1: Write `training/export.py`**

No dedicated test — ONNX/TRT export is verified by loading and running inference on the exported model. Run manually after training produces a checkpoint.

```python
"""Export trained TerrainModel to ONNX and optionally TensorRT.

Usage:
    python training/export.py --checkpoint training/checkpoints/best.pt --onnx terrain.onnx
    python training/export.py --checkpoint training/checkpoints/best.pt --onnx terrain.onnx --trt terrain.engine --fp16
"""
import argparse
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.model import TerrainModel


def export_onnx(checkpoint_path: str, onnx_path: str) -> None:
    model = TerrainModel()
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy_images   = torch.randn(1, 12, 3, 224, 224)
    dummy_rotation = torch.randn(1, 6)

    torch.onnx.export(
        model,
        (dummy_images, dummy_rotation),
        onnx_path,
        input_names=['images', 'rotation'],
        output_names=['height', 'rocks', 'craters', 'walls'],
        dynamic_axes={
            'images':   {0: 'batch'},
            'rotation': {0: 'batch'},
            'height':   {0: 'batch'},
            'rocks':    {0: 'batch'},
            'craters':  {0: 'batch'},
            'walls':    {0: 'batch'},
        },
        opset_version=17,
    )
    print(f"[export] ONNX saved to {onnx_path}")

    # Quick verification
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(onnx_path)
    out  = sess.run(None, {
        'images':   dummy_images.numpy(),
        'rotation': dummy_rotation.numpy(),
    })
    print(f"[export] ONNX verified. Output shapes: {[o.shape for o in out]}")


def export_trt(onnx_path: str, engine_path: str, fp16: bool = True) -> None:
    import tensorrt as trt
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[export] FP16 enabled")

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    print(f"[export] TRT engine saved to {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--onnx',       required=True)
    parser.add_argument('--trt',        default=None,
                        help='Path for TRT .engine output (optional)')
    parser.add_argument('--fp16',       action='store_true')
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.onnx)
    if args.trt:
        export_trt(args.onnx, args.trt, fp16=args.fp16)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test export with a random (untrained) model**

```
python training/export.py --checkpoint training/checkpoints/best.pt --onnx training/terrain_test.onnx
```

Skip if no checkpoint exists yet — run after Task 4 produces a checkpoint. Expected output:
```
[export] ONNX saved to training/terrain_test.onnx
[export] ONNX verified. Output shapes: [(1, 200, 200), (1, 200, 200), (1, 200, 200), (1, 200, 200)]
```

- [ ] **Step 3: Commit**

```
git add training/export.py
git commit -m "feat: ONNX and TensorRT export for TerrainModel"
```

---

## Running Training

Once all tasks are complete and the terrain data is at `E:\terrain_data`:

```
cd C:\Users\adam.carbone\source\repos\ros_intro
pip install -r training/requirements.txt
python training/train.py --config training/config.yaml
```

Monitor: look for `train=` and `val=` losses decreasing over epochs. If val loss plateaus within 10 epochs, check that images are loading correctly by printing a sample batch shape. If val loss diverges from train loss (overfitting) after ~30 epochs, reduce `learning_rate` to `5e-5` in `config.yaml`.

---

## Self-Review

**Spec coverage:**
- ✅ 12 image inputs (6 RGB + 6 depth) → dataset.py `_SLOT_PATHS`
- ✅ Robot rotation (3 axes, sin/cos encoded) → `_load_rotation`
- ✅ Camera pose embeddings from cameras.yaml values → cameras.py + model.py
- ✅ Height regression head → `height_head`, L1 loss
- ✅ Rock/crater Gaussian heatmap heads → `build_object_heatmap`, MSE loss
- ✅ Wall binary mask head → `build_wall_mask`, BCE loss
- ✅ EfficientNet-B0 pretrained backbone → `efficientnet_b0(weights=...)`
- ✅ BEV decoder 7×7 → 200×200 → `UpBlock` chain
- ✅ Horizontal flip augmentation with left/right camera swap → `_hflip`
- ✅ Depth normalization separate from RGB → `compute_depth_stats`
- ✅ 80/20 train/val split → `create_splits`
- ✅ Cosine annealing scheduler
- ✅ Checkpointing best val loss + every N epochs
- ✅ ONNX + TRT export
- ✅ Semantic derivation noted (inference only, no training head)

**Note on spec vs implementation:** The spec document says "14×14×320 feature maps" — this is incorrect. EfficientNet-B0 outputs 7×7×1280 for 224×224 input. The plan uses the correct values. The spec document does not need to be updated (it is high-level).
