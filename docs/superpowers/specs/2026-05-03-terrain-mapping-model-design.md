# Terrain Mapping Model — Design Spec

**Date:** 2026-05-03
**Branch:** `isaac_lab/terrain_mapping_model`
**Status:** Approved

---

## 1. Problem Statement

Replace the NvBlox/Depth Anything pipeline with a trained neural network that produces a real-time 3D terrain map from the robot's existing combined point cloud. The model must run on the Jetson Orin AGX 64GB at 50W and output a map that a downstream navigation NN can consume directly.

---

## 2. System Overview

Four sequential stages:

| Stage | What it does |
|-------|-------------|
| 1. Isaac Sim Environment Generator | Procedurally spawns craters, rocks, walls, textures each episode |
| 2. Data Collection Pipeline | Runs real ROS perception stack (DA3 + Orbbec) inside Isaac Sim, pairs point cloud with ground truth |
| 3. Model Training | Trains ResNet-34 U-Net offline on collected dataset, exports TensorRT FP16 for Orin |
| 4. ROS2 Inference Node | Subscribes to `/mapping/point_cloud/combined`, publishes terrain map topics |

---

## 3. Environment Generator

Each training episode generates a completely new environment from scratch. The robot spawns at a random pose with random yaw. One data sample is collected per episode.

### 3.1 Config Parameters

```python
@dataclass
class TerrainMappingEnvCfg:
    # Arena
    arena_size: tuple = (10.0, 10.0)            # meters, X×Y
    regolith_noise_amplitude: float = 0.02       # subtle surface roughness (m)

    # Rocks
    rock_count_range: tuple = (5, 20)
    rock_diameter_range: tuple = (0.30, 0.40)    # meters

    # Craters (carved depressions in the height field)
    crater_count_range: tuple = (5, 20)
    crater_diameter_range: tuple = (0.40, 0.50)  # meters
    crater_depth_ratio: float = 0.25             # depth = diameter × ratio

    # Walls
    wall_count_range: tuple = (0, 4)
    wall_length_range: tuple = (0.5, 3.0)        # meters
    wall_height: float = 0.6                     # meters
    wall_materials: tuple = ("concrete", "glass", "metal", "wood")

    # Ground
    ground_texture_variants: int = 8             # PBR texture variants

    # Robot spawn
    spawn_margin: float = 1.0                    # meters from arena edge
```

### 3.2 Craters

Craters are carved directly into the height field as inverted paraboloids (negative Gaussian depressions). The crater center position, diameter, and depth are recorded before carving so ground truth is exact. They are not separate collision prims.

### 3.3 Rocks

Scaled sphere/capsule collision prims placed at random XY positions on the terrain surface. Material: random PBR rock texture.

### 3.4 Walls

Extruded rectangular prims along a random line segment `{x1, y1, x2, y2}`. Height is fixed at `wall_height` (tall enough to be unambiguous as a barrier). Material randomized from `wall_materials`. Up to 4 walls per episode.

---

## 4. Data Collection Pipeline

### 4.1 Approach

Run the real ROS perception stack inside Isaac Sim rather than using Isaac Sim's perfect depth sensor. This means DA3 depth estimation noise is baked into the training data, matching real deployment conditions exactly. No synthetic noise augmentation is needed.

### 4.2 Flow

```
Isaac Sim
  → renders camera images via existing ROS2 OmniGraphs
  → publishes on existing camera topics
        ↓
Existing ROS perception stack (running in same process via rclpy)
  → 5× DA3 metric depth estimation @ 10Hz
  → 1× Orbbec native structured-light depth @ 640×480
  → point cloud projection + TF transform per camera
  → /mapping/point_cloud/combined (PointCloud2, base_link frame)
        ↓
Data collection node (rclpy + Isaac Sim Python API)
  → subscribes to /mapping/point_cloud/combined
  → reads ground truth from Isaac Sim scene API
      (terrain height field, rock positions/diameters, crater positions/diameters, wall segments)
  → timestamp-syncs point cloud with sim state
  → saves paired .npz: {point_cloud, height_gt, semantic_gt, objects_gt, walls_gt}
```

### 4.3 Target Dataset Size

~50,000 episodes to start. Each `.npz` contains one BEV grid + three ground truth arrays.

---

## 5. BEV Rasterization

Runs as a preprocessing step inside the inference node (and data collection node). Not a learned component.

- **Grid:** 200×200 cells, 5 cm/cell → covers ±5m X and Y from robot origin
- **Channels per cell (6 total):** `height_max`, `height_mean`, `r`, `g`, `b`, `occupancy`
- **Empty cells:** all zeros, `occupancy=0` — loss ignores these cells during training
- **Input:** `/mapping/point_cloud/combined` — XYZ + packed RGB, base_link frame, metric meters

---

## 6. Model Architecture

### 6.1 Overview

ResNet-34 encoder (ImageNet pretrained) with a U-Net dense decoder and a lightweight detection head off the bottleneck.

```
200×200×6 BEV grid
        ↓
ResNet-34 Encoder (shared weights)
        ↓
    Bottleneck
   /     |      \
Dense  Dense   Detection
Decoder Decoder   Head
   ↓       ↓        ↓
Head 1   Head 2   Head 3
Height   Semantic  Objects
map      grid      list
```

### 6.2 Output Heads

| Head | Output shape | Description |
|------|-------------|-------------|
| Heightmap | 200×200×1 float32 | Terrain height in meters per BEV cell |
| Semantic grid | 200×200×4 float32 | Per-cell logits: free / rock / crater / wall |
| Detection | 110×5 float32 | 50 rock slots + 60 crater slots, each `{x, y, diameter, confidence, type}`, zero-padded |

### 6.3 Semantic Grid

A top-down 2D label map. Each 5cm×5cm cell gets one label:
- `0` free/drivable
- `1` rock
- `2` crater
- `3` wall

This is the unified obstacle map consumed by the navigation NN.

### 6.4 Detection Head

Lightweight MLP off the bottleneck. Predicts up to 50 rocks + 60 craters. Output sorted by distance from robot origin so zero-padding is always at the end. Confidence threshold (default 0.5) applied at inference time.

---

## 7. Loss Functions

```
L_total = L_height + 0.5 × L_semantic + 0.3 × L_detect
```

| Head | Loss | Notes |
|------|------|-------|
| Heightmap | MSE, occupied cells only | Ignores empty BEV cells |
| Semantic | Weighted cross-entropy | Upweight rock/crater/wall — rare vs free space |
| Detection | L1 (x, y, diameter) + BCE (confidence) | Hungarian matching to ground truth objects |

Loss weights are tunable via config without retraining the architecture.

---

## 8. Training Setup

- **Hardware:** Workstation GPU (training), Jetson Orin AGX 64GB (inference)
- **Optimizer:** AdamW, lr=3e-4, weight_decay=1e-4
- **Scheduler:** CosineAnnealingLR
- **Batch size:** 32
- **Deployment export:** PyTorch → ONNX → TensorRT FP16
- **Target inference rate:** ≥5 Hz on Orin at 50W (matches input topic rate)

---

## 9. Evaluation Metrics

| Metric | Head | Target |
|--------|------|--------|
| MAE (m) | Heightmap | < 0.03m |
| RMSE (m) | Heightmap | < 0.05m |
| mIoU | Semantic (all classes) | > 0.70 |
| mIoU | Semantic (rock/crater/wall only) | > 0.60 |
| mAP@0.5 | Rock detection (circle IoU) | > 0.75 |
| mAP@0.5 | Crater detection (circle IoU) | > 0.75 |

---

## 10. ROS2 Inference Node

### 10.1 Input

- `/mapping/point_cloud/combined` — `sensor_msgs/PointCloud2`, 5Hz, base_link frame

### 10.2 Pipeline

```
/mapping/point_cloud/combined
        ↓
BEV rasterizer (numpy/CUDA kernel)
200×200×6 grid
        ↓
TensorRT FP16 engine
        ↓
Post-processing
├── Heightmap → mesh triangulation (marching squares)
├── Semantic grid → RANSAC wall line segment fitting
└── Detection slots → confidence threshold filter
        ↓
Published topics
```

### 10.3 Published Topics

| Topic | Message type | Content |
|-------|-------------|---------|
| `/terrain/heightmap` | `sensor_msgs/Image` (32FC1) | Float32 height in meters per BEV cell |
| `/terrain/semantic` | `nav_msgs/OccupancyGrid` | Label per cell (0=free,1=rock,2=crater,3=wall) |
| `/terrain/objects` | `custom TerrainObjects.msg` | Rock + crater list with positions and diameters |
| `/terrain/walls` | `custom WallSegments.msg` | Detected wall line segments |

### 10.4 Custom Messages

```
# TerrainObject.msg
uint8 type        # 0=rock, 1=crater
float32 x
float32 y
float32 diameter
float32 confidence

# TerrainObjects.msg
Header header
TerrainObject[] objects

# WallSegment.msg
float32 x1
float32 y1
float32 x2
float32 y2

# WallSegments.msg
Header header
WallSegment[] walls
```

---

## 11. Future Work (Out of Scope for This Spec)

- Navigation NN that subscribes to `/terrain/semantic` and `/terrain/heightmap`
- Multi-frame temporal accumulation into a persistent global map
- Fine-tuning on real arena data after initial sim-trained deployment
