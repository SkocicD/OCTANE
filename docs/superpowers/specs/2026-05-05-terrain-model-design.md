# Terrain Mapping Model — Design Spec
**Date:** 2026-05-05  
**Branch:** `ml/terrain_model`  
**Status:** Approved

---

## Overview

A supervised learning model that takes 12 camera images (6 RGB + 6 depth) and robot rotation as input and predicts a top-down BEV terrain map: continuous height, rock confidence, crater confidence, and wall mask. Trained offline on Windows, deployed on Jetson Orin.

---

## Data

### Source
`E:\terrain_data\` (3500 episodes collected from Isaac Sim)

```
terrain_data/
  images/
    left_front/      ep_XXXXXX.jpg   (480×360 RGB)
    left_side/       ep_XXXXXX.jpg
    right_front/     ep_XXXXXX.jpg
    right_side/      ep_XXXXXX.jpg
    back_rear/       ep_XXXXXX.jpg
    depth_cam_rgb/   ep_XXXXXX.jpg   (640×480 RGB)
    depth_cam_d/     ep_XXXXXX.png   (640×480 grayscale inverted depth)
  depth/
    left_front/      ep_XXXXXX.png   (480×360 DA3 colorized heatmap)
    left_side/       ep_XXXXXX.png
    right_front/     ep_XXXXXX.png
    right_side/      ep_XXXXXX.png
    back_rear/       ep_XXXXXX.png
  gt/
    ep_XXXXXX_gt.npz
```

### NPZ Contents
| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `height_gt` | (200, 200) | float32 | Height in meters, robot-yaw-aligned BEV |
| `semantic_gt` | (200, 200) | uint8 | 0=ground, 1=rock, 2=crater, 3=wall |
| `objects_gt` | (N, 4) | float32 | [rx, ry, diameter, class] per object (variable N) |
| `walls_gt` | (M, 4) | float32 | [rx1, ry1, rx2, ry2] per wall segment (variable M) |
| `robot_yaw` | (1,) | float32 | Radians |
| `robot_pitch` | (1,) | float32 | Radians |
| `robot_roll` | (1,) | float32 | Radians |

### BEV Grid
- 200×200 cells, 5cm resolution → 10m×10m centered on robot
- Robot at center, yaw-aligned (forward = top of grid)

---

## Inputs

12 images total, all resized to **224×224** at load time:

| Index | Source file | Description |
|-------|-------------|-------------|
| 0 | `images/left_front` | Near RGB |
| 1 | `images/left_side` | Near RGB |
| 2 | `images/right_front` | Near RGB |
| 3 | `images/right_side` | Near RGB |
| 4 | `images/back_rear` | Near RGB |
| 5 | `images/depth_cam_rgb` | Orbbec RGB |
| 6 | `depth/left_front` | DA3 depth heatmap |
| 7 | `depth/left_side` | DA3 depth heatmap |
| 8 | `depth/right_front` | DA3 depth heatmap |
| 9 | `depth/right_side` | DA3 depth heatmap |
| 10 | `depth/back_rear` | DA3 depth heatmap |
| 11 | `images/depth_cam_d` | Orbbec real depth (grayscale, loaded as RGB) |

**Normalization:**
- Indices 0–5 (RGB): ImageNet mean/std
- Indices 6–11 (depth): per-channel mean/std computed from training set, saved to `training/depth_stats.json`

**Robot rotation:** encoded as `[sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw)]` — 6 floats. Avoids angle discontinuities.

---

## Architecture

### Encoder
- **Backbone:** EfficientNet-B0 pretrained on ImageNet (~5M params)
- Each image processed independently through shared backbone
- Output per image: (14×14×320) feature map (stride 16 from 224×224 input)
- **Camera ID embedding:** 12 learned embeddings of dim 32, broadcast-added to each image's features after a projection conv → gives the model positional/directional context per camera without requiring explicit geometric projection

### Fusion
- Stack all 12 feature maps channel-wise: (14×14×3840)
- 1×1 conv + BN + ReLU: (14×14×3840) → (14×14×512)
- **Rotation embedding:** 6 rotation values → MLP (6→64→32) → reshape to (1×1×32), broadcast-added to fused features

### BEV Decoder
U-Net style progressive upsampling with skip-free transposed convolutions:

| Stage | Shape |
|-------|-------|
| Input | 14×14×512 |
| Up 1 | 25×25×256 |
| Up 2 | 50×50×128 |
| Up 3 | 100×100×64 |
| Up 4 | 200×200×32 |

Each stage: `ConvTranspose2d` + `BN` + `ReLU` + `Conv2d` + `BN` + `ReLU`

### Output Heads
Four independent `Conv2d(32, 1, 1×1)` heads at 200×200:

| Head | Activation | Target | Loss |
|------|-----------|--------|------|
| Height | none | `height_gt` | L1 |
| Rocks | sigmoid | Gaussian heatmap from `objects_gt` (class=0) | MSE |
| Craters | sigmoid | Gaussian heatmap from `objects_gt` (class=1) | MSE |
| Walls | sigmoid | Rasterized binary mask from `walls_gt` | BCE |

**GT preprocessing (done in dataset, not stored):**
- Rocks/craters: for each object at (rx, ry) with diameter d, stamp a Gaussian with σ = max(1, d/2 / cell_size) cells centered at the object's BEV cell
- Walls: rasterize each [rx1,ry1,rx2,ry2] segment onto the 200×200 grid, 3-pixel thick line

### Semantic Derivation (inference only)
No separate training head. At inference, derive from thresholded outputs:
```
semantic[rocks > 0.5]   = 1
semantic[craters > 0.5] = 2
semantic[walls > 0.5]   = 3
# remainder             = 0 (ground)
```

---

## Training

### Split
- 80/20 random split on episode IDs: ~2800 train, ~700 val
- Split computed once, saved to `training/splits.json`

### Hyperparameters
```yaml
batch_size: 16
learning_rate: 1e-4
scheduler: cosine_annealing (T_max=100)
epochs: 100
optimizer: AdamW (weight_decay=1e-4)
```

### Loss
```
total_loss = height_loss + rocks_loss + craters_loss + walls_loss
```
Equal weighting to start. Tune if one head dominates.

### Augmentation (training only)
- Random horizontal flip (all 12 images + GT flipped consistently)
- Random brightness/contrast jitter on RGB images only (indices 0–5), not depth
- No rotation augmentation — cameras have fixed real-world positions

### Hardware
- Training: Windows, NVIDIA RTX 3060 (12GB VRAM)
- ~16 images × 12 frames × 224×224 fits comfortably within 12GB

### Checkpointing
- Save best val loss checkpoint to `training/checkpoints/best.pt`
- Save every 10 epochs to `training/checkpoints/epoch_N.pt`

---

## Deployment

### Export
- PyTorch → ONNX → TensorRT engine for Jetson Orin
- `training/export.py` handles both ONNX and TRT conversion
- Fixed input shape: (1, 12, 3, 224, 224) + (1, 6) rotation

### Live Inference on Rover
- Input: 12 camera frames from live perception pipeline + robot IMU rotation
- Output: 200×200 height map + rock/crater/wall confidence maps
- Plugs into existing `terrain_inference_node.py` in the ROS `octane_mapping` package
- That node already subscribes to perception topics and publishes terrain maps

---

## File Structure

```
training/
  dataset.py          # TerrainDataset: loads images + NPZ, builds GT heatmaps
  model.py            # TerrainModel: encoder + fusion + decoder + heads
  train.py            # training loop, validation, checkpointing
  export.py           # ONNX + TensorRT export
  config.yaml         # all hyperparameters and data paths
  depth_stats.json    # depth normalization stats (computed on first run)
  splits.json         # train/val episode ID split
  checkpoints/        # saved model weights
```

---

## Future Enhancements
- Add camera pose embeddings derived from `cameras.yaml` extrinsics (geometric prior)
- Semantic refinement head: lightweight Conv on top of the 4 output heads
- Iterative refinement: feed predicted outputs back as additional input channels
- Increase dataset to 10K–20K episodes for improved generalization
