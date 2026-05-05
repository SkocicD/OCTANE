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
    """Sample up to max_episodes to compute per-channel mean/std for depth images.
    Covers all 6 depth input slots: depth/{serial}/ (slots 6-10) + images/depth_cam_d/ (slot 11)."""
    _DEPTH_SOURCES = [
        ('depth', 'left_front'),
        ('depth', 'left_side'),
        ('depth', 'right_front'),
        ('depth', 'right_side'),
        ('depth', 'back_rear'),
        ('images', 'depth_cam_d'),
    ]
    sample_ids = episode_ids[:max_episodes]
    pixels = []
    for ep_id in sample_ids:
        for subdir, serial in _DEPTH_SOURCES:
            path = os.path.join(data_root, subdir, serial, f'{ep_id}.png')
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
# (relative_path_under_root, extension)
_SLOT_PATHS = [
    ('images/left_front',    '.jpg'),
    ('images/left_side',     '.jpg'),
    ('images/right_front',   '.jpg'),
    ('images/right_side',    '.jpg'),
    ('images/back_rear',     '.jpg'),
    ('images/depth_cam_rgb', '.jpg'),
    ('depth/left_front',     '.png'),
    ('depth/left_side',      '.png'),
    ('depth/right_front',    '.png'),
    ('depth/right_side',     '.png'),
    ('depth/back_rear',      '.png'),
    ('images/depth_cam_d',   '.png'),
]

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
        npz = np.load(os.path.join(self.root, 'gt', f'{ep_id}_gt.npz'))
        images   = self._load_images(ep_id)
        rotation = self._load_rotation(npz)
        gt       = self._load_gt(npz)

        if self.augment and torch.rand(1).item() < 0.5:
            images, gt = self._hflip(images, gt)

        return images, rotation, gt

    def _load_images(self, ep_id: str) -> torch.Tensor:
        tensors = []
        for i, (rel_path, ext) in enumerate(_SLOT_PATHS):
            path = os.path.join(self.root, rel_path, f'{ep_id}{ext}')
            img  = Image.open(path).convert('RGB')
            t    = self._rgb_transform(img) if i in _RGB_SLOTS else self._depth_transform(img)
            tensors.append(t)
        return torch.stack(tensors, dim=0)  # (12, 3, 224, 224)

    def _load_rotation(self, npz) -> torch.Tensor:
        roll  = float(npz.get('robot_roll',  np.float32(0.0)))
        pitch = float(npz.get('robot_pitch', np.float32(0.0)))
        yaw   = float(npz['robot_yaw'])
        return torch.tensor([
            math.sin(roll),  math.cos(roll),
            math.sin(pitch), math.cos(pitch),
            math.sin(yaw),   math.cos(yaw),
        ], dtype=torch.float32)

    def _load_gt(self, npz) -> dict:
        height  = torch.from_numpy(npz['height_gt'])
        objects = npz['objects_gt']
        walls   = npz['walls_gt']
        rocks   = torch.from_numpy(build_object_heatmap(objects, class_id=0))
        craters = torch.from_numpy(build_object_heatmap(objects, class_id=1))
        wall_m  = torch.from_numpy(build_wall_mask(walls))
        return {'height': height, 'rocks': rocks, 'craters': craters, 'walls': wall_m}

    def _hflip(self, images: torch.Tensor, gt: dict):
        images = torch.flip(images, dims=[-1])
        for a, b in _FLIP_SWAP_PAIRS:
            images[[a, b]] = images[[b, a]]
        gt = {k: torch.flip(v, dims=[-1]) for k, v in gt.items()}
        return images, gt
