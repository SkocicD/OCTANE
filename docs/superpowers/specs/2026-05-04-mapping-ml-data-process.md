# ROS ML Data Processing Pipeline — Implementation Spec

**Branch to create:** `octane/mapping_ml_process`  
**Base branch:** `mapping_model/data_collection` (has the launch_system.sh and data_collection.launch.py patterns to follow)  
**Date:** 2026-05-04

---

## Goal

Build an offline batch pipeline that reads the per-episode camera images collected by
Isaac Sim, runs them through the real DA3 + point cloud stack, and exports one
point cloud file per episode.  The output pairs with the existing `ep_XXXXXX_gt.npz`
ground-truth files for neural network training.

---

## Input (written by Isaac Sim collection)

All paths are WSL-mounted.  The Windows paths `E:\terrain_data\...` mount as
`/mnt/e/terrain_data/...` in WSL2.

```
/mnt/e/terrain_data/
├── gt/
│   ├── ep_000000_gt.npz       ← ground truth (height, semantic, objects, walls)
│   ├── ep_000000.ready        ← sentinel: episode is fully written
│   ├── ep_000001_gt.npz
│   ├── ep_000001.ready
│   └── ...
└── images/
    ├── left_front/
    │   ├── ep_000000.jpg      ← uint8 RGB JPEG, 480×360
    │   └── ...
    ├── left_side/             ← same, 480×360
    ├── right_front/           ← same, 480×360
    ├── right_side/            ← same, 480×360
    ├── back_rear/             ← same, 480×360
    ├── depth_cam_rgb/         ← uint8 RGB JPEG, 640×480  (Orbbec RGB)
    └── depth_cam_d/
        ├── ep_000000.npy      ← float32 numpy array, metres, 640×480  (Orbbec native depth)
        └── ...
```

**Serial → camera mapping:**

| Serial (folder name) | ROS camera name | Camera type |
|----------------------|-----------------|-------------|
| `left_front`         | `near/left_front`   | Innomaker RGB → DA3 depth |
| `left_side`          | `near/left_side`    | Innomaker RGB → DA3 depth |
| `right_front`        | `near/right_front`  | Innomaker RGB → DA3 depth |
| `right_side`         | `near/right_side`   | Innomaker RGB → DA3 depth |
| `back_rear`          | `near/back_rear`    | Innomaker RGB → DA3 depth |
| `depth_cam_rgb`      | `depth_camera`      | Orbbec RGB (color only, not fed to DA3) |
| `depth_cam_d`        | `depth_camera`      | Orbbec native structured-light depth |

**Important:** Only the 5 Innomaker cameras go through DA3.  The Orbbec depth
(`depth_cam_d`) is native structured-light depth and is used directly.

---

## Output (produced by this pipeline)

```
/mnt/e/terrain_data/
└── point_clouds/
    ├── ep_000000_pc.npz       ← per-episode combined point cloud
    ├── ep_000001_pc.npz
    └── ...
```

Each `ep_XXXXXX_pc.npz` contains:
```
points   (N, 6)  float32   columns: x, y, z, r, g, b
                            xyz in metres, base_link frame
                            rgb in 0–255 uint8 packed as float32
frame_id  str              "base_link"
episode   int              episode number
```

If you prefer a different point cloud format (PCD, PLY, or separate xyz/rgb arrays),
adapt as needed — just document the format clearly so the training code can load it.

---

## Architecture

```
disk images
    │
    ▼
image_replay_node          (NEW)
  Reads episode images from disk one at a time.
  Publishes each camera image to the standard perception topics.
  Waits for the combined point cloud to arrive before moving to next episode.
  Signals episode ID via /data_process/current_episode (std_msgs/Int32).
    │
    ├─► perception/camera/near/left_front/rgb/frame  ─┐
    ├─► perception/camera/near/left_side/rgb/frame    │
    ├─► perception/camera/near/right_front/rgb/frame  ├─► depth_estimation_node (DA3, existing)
    ├─► perception/camera/near/right_side/rgb/frame   │       produces depth topics for each
    ├─► perception/camera/near/back_rear/rgb/frame    ┘
    │
    ├─► perception/camera/depth_camera/rgb/frame       (Orbbec RGB, color passthrough)
    └─► perception/camera/depth_camera/depth/frame     (Orbbec depth, 16UC1 → float32 metres)
                                                           ↓
                                              point_cloud_mux_node (existing)
                                              combines all cameras → /mapping/point_cloud/combined
                                                           ↓
                                              point_cloud_exporter_node   (NEW)
                                              subscribes to combined cloud +
                                              /data_process/current_episode,
                                              saves ep_XXXXXX_pc.npz to disk
```

---

## New Nodes to Create

### 1. `image_replay_node` — `octane_mapping`

**File:** `workspace/src/octane_mapping/octane_mapping/nodes/image_replay_node.py`

**Parameters:**
- `images_dir` (string, default `/mnt/e/terrain_data/images`) — root of the images folder
- `gt_dir` (string, default `/mnt/e/terrain_data/gt`) — where `.ready` files are
- `output_dir` (string, default `/mnt/e/terrain_data/point_clouds`) — to skip already-exported episodes
- `publish_delay_sec` (float, default `0.5`) — seconds to wait after publishing images before expecting point cloud
- `timeout_sec` (float, default `10.0`) — max wait for point cloud per episode before skipping

**Behavior:**
1. On startup, scan `gt_dir` for all `ep_XXXXXX.ready` files to build the episode list.
2. For each episode, check if `output_dir/ep_XXXXXX_pc.npz` already exists — skip if so (resumable).
3. Load all 7 images for the episode from disk.
4. Publish all images simultaneously with the same ROS timestamp to the standard perception topics.
5. Publish the episode number to `/data_process/current_episode` (std_msgs/Int32).
6. Wait `publish_delay_sec` then check for a response on `/data_process/episode_exported` (std_msgs/Int32).
7. Repeat up to timeout; warn and move to next episode if no response.
8. Print progress: `[ImageReplay] ep 42/5000 — published, waiting...` and `[ImageReplay] ep 42 — exported.`

**Topic publishing (match existing perception topic names exactly):**

| Serial | Published topic | Message type | Encoding |
|--------|----------------|--------------|---------|
| `left_front` | `perception/camera/near/left_front/rgb/frame` | sensor_msgs/Image | `rgb8` |
| `left_side` | `perception/camera/near/left_side/rgb/frame` | sensor_msgs/Image | `rgb8` |
| `right_front` | `perception/camera/near/right_front/rgb/frame` | sensor_msgs/Image | `rgb8` |
| `right_side` | `perception/camera/near/right_side/rgb/frame` | sensor_msgs/Image | `rgb8` |
| `back_rear` | `perception/camera/near/back_rear/rgb/frame` | sensor_msgs/Image | `rgb8` |
| `depth_cam_rgb` | `perception/camera/depth_camera/rgb/frame` | sensor_msgs/Image | `rgb8` |
| `depth_cam_d` | `perception/camera/depth_camera/depth/frame` | sensor_msgs/Image | `16UC1` (mm) |

Check `cameras.yaml` to confirm exact topic names — that file is the source of truth.

The Orbbec depth image (`depth_cam_d`) is stored as a raw float32 `.npy` file
in metres.  Load with `np.load("ep_XXXXXX.npy")`, multiply by 1000, clip to
uint16, then publish as `16UC1` — or publish as `32FC1` metres directly if the
point cloud mux accepts that encoding (check `point_cloud_mux_node`).

---

### 2. `point_cloud_exporter_node` — `octane_mapping`

**File:** `workspace/src/octane_mapping/octane_mapping/nodes/point_cloud_exporter_node.py`

**Parameters:**
- `output_dir` (string, default `/mnt/e/terrain_data/point_clouds`) — where to write NPZ files

**Behavior:**
1. Subscribe to `/mapping/point_cloud/combined` (sensor_msgs/PointCloud2).
2. Subscribe to `/data_process/current_episode` (std_msgs/Int32).
3. When a new episode ID arrives, wait for the next combined cloud message.
4. Extract XYZ + RGB from the PointCloud2 (use `sensor_msgs_py.point_cloud2.read_points` or `ros2_numpy`).
5. Save as `ep_XXXXXX_pc.npz` with arrays `points (N,6) float32` and scalar `episode`.
6. Publish the exported episode number to `/data_process/episode_exported` (std_msgs/Int32)
   so `image_replay_node` knows to advance.
7. Log: `[PCExporter] ep 42 — saved 18432 points → ep_000042_pc.npz`

---

## Launch File to Create

**File:** `workspace/src/octane/octane/launch/data_process.launch.py`

Model it on `data_collection.launch.py` (on branch `mapping_model/data_collection`).

```python
"""Launch file for offline ML training data processing.

Reads Isaac Sim camera images, runs DA3 + point cloud pipeline, exports
per-episode point cloud NPZ files paired with ground-truth NPZ files.

Usage (via launch_system.sh):
    ./launch_system.sh --data_process

Or directly:
    ros2 launch octane data_process.launch.py
    ros2 launch octane data_process.launch.py images_dir:=/mnt/e/terrain_data/images output_dir:=/mnt/e/terrain_data/point_clouds
"""
```

**Should launch:**
1. `depth_estimation_node` (DA3) — from `octane_perception`, configured for the 5 Innomaker RGB topics only (not Orbbec)
2. `point_cloud_mux_node` — from `octane_mapping`, existing node
3. `image_replay_node` — NEW, from `octane_mapping`
4. `point_cloud_exporter_node` — NEW, from `octane_mapping`

**Parameters to expose:**
- `images_dir` — passed to `image_replay_node`
- `gt_dir` — passed to `image_replay_node`
- `output_dir` — passed to both `image_replay_node` and `point_cloud_exporter_node`

**Do NOT launch:** real camera drivers (`astra_depth_node`, `rgb_camera_node`), `terrain_inference_node`, supervisor, network.

---

## launch_system.sh Changes

Add `--data_process` flag.  Follow the existing pattern in `launch_system.sh`
(on branch `mapping_model/data_collection`).

```bash
# New usage line at top:
# Usage: ./launch_system.sh [--collect] [--data_process]

# Add to argument parsing section:
if [[ "$*" == *"--data_process"* ]]; then
    echo "=== Launching ML Data Processing Pipeline ==="
    ros2 launch octane data_process.launch.py \
        images_dir:=/mnt/e/terrain_data/images \
        gt_dir:=/mnt/e/terrain_data/gt \
        output_dir:=/mnt/e/terrain_data/point_clouds
    exit 0
fi
```

Make it an early-exit mode (like `--collect`) — when `--data_process` is passed,
launch only the data processing pipeline and nothing else (no supervisor, no network,
no mapping inference).

---

## Registration

Register both new nodes in `octane_mapping`'s `setup.py` (or `CMakeLists.txt`
depending on build system — check how existing nodes like `point_cloud_mux_node`
are registered and follow the same pattern).

---

## Key Things to Verify in the Existing Codebase

Before implementing, confirm:

1. **DA3 input topics**: Check `depth_estimation_node.py` — what parameter controls
   which RGB topics it subscribes to?  Make sure `data_process.launch.py` configures
   it for exactly the 5 Innomaker topics, not Orbbec RGB.

2. **Orbbec depth units**: Check `point_cloud_mux_node.py` — does it expect Orbbec
   depth in metres (float32) or millimetres (uint16)?  The saved depth PNGs are
   uint16 millimetres.  If the mux expects metres, `image_replay_node` should
   convert before publishing (divide by 1000, publish as `32FC1`).

3. **Topic names**: Confirm all topic names against `cameras.yaml` — that file is
   the authoritative source.  The table above may be slightly off.

4. **Point cloud RGB packing**: The existing `point_cloud_mux_node` may pack RGB
   as a float32 with packed int encoding (standard PCL convention) or as separate
   fields.  Match whatever format it outputs.

5. **TF frames**: The point cloud mux transforms clouds to `base_link`.  For offline
   replay, TF may not be published.  Check if `point_cloud_mux_node` needs a live
   TF tree or reads static transforms from a config.  If it needs TF, add a
   `static_transform_publisher` in the launch file or a robot_state_publisher
   with the URDF.

---

## Resumability

Both new nodes must be resumable:
- `image_replay_node`: skip episodes where `output_dir/ep_XXXXXX_pc.npz` already exists
- `point_cloud_exporter_node`: overwrite is fine (exporter doesn't need to track state)

Print a summary on startup: `[ImageReplay] Found 4832 episodes, 1201 already exported, processing 3631 remaining.`
