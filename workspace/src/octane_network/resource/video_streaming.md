# OCTANE Network — Video Streaming

On-demand camera and map streaming from rover to GUI over UDP.

---

## Overview

The GUI requests a stream over TCP; the rover sends JPEG frames over UDP.
Control traffic (telemetry, commands, heartbeat) stays on TCP and is never
blocked by video.

```
GUI ──[TCP :5000]──▶ stream request (type V, 8 bytes)
GUI ◀─[UDP :5002]── JPEG chunks   (~1400 bytes each)
```

---

## Stream Request — TCP type `V`

Wire format: `[O][V][4][source_id][variant][scale][fps][CRC]` — always **8 bytes**.

| Field | Size | Values |
|---|---|---|
| `source_id` | 1B | 0–5 = individual camera, 6 = mosaic, 7 = map, 255 = stop all |
| `variant` | 1B | `R` (0x52) = RGB, `D` (0x44) = depth heatmap |
| `scale` | 1B | 1–100 (% of native resolution); **0 = use server default** |
| `fps` | 1B | 1–30 (target frame rate) |

Rover responds with a standard `A` (ACK) frame.

Sending `source_id=255` stops all active streams immediately.

---

## Source IDs

| ID | Camera | Native resolution |
|---|---|---|
| 0 | orbbec_depth | 640×480 |
| 1 | near_rgb_left_side | 480×360 |
| 2 | near_rgb_left_front | 480×360 |
| 3 | near_rgb_right_side | 480×360 |
| 4 | near_rgb_right_front | 480×360 |
| 5 | near_rgb_back_rear | 480×360 |
| 6 | mosaic (all 6 tiled 3×2) | 480×240 |
| 7 | nvblox ESDF map slice | variable |
| 255 | stop all | — |

IDs match declaration order in `octane/config/cameras.yaml`.

---

## UDP Video Frame Format

Each UDP packet is ≤ 1400 bytes:

```
Byte 0:   0x4F  (magic 'O')
Byte 1:   0x56  (type 'V')
Byte 2:   source_id
Byte 3:   variant  ('R' or 'D')
Byte 4:   seq_hi   ─┐ 16-bit frame sequence number
Byte 5:   seq_lo   ─┘
Byte 6:   chunk_idx   (0-based index of this chunk)
Byte 7:   chunk_total (total chunks for this frame)
Bytes 8…: JPEG payload (up to 1392 bytes)
```

**Reassembly (GUI side):**
1. Group packets by `seq` number.
2. Once all `chunk_total` chunks for a `seq` are received, concatenate in `chunk_idx` order.
3. Decode the concatenated bytes as JPEG.
4. If a new `seq` arrives before the previous one is complete, discard the incomplete frame.

---

## Depth Heatmap Rendering

When `variant = D`, the rover renders depth before JPEG encoding:

- **DA3 depth** (32FC1, float meters): clipped at `depth_max_m` (default 8 m), normalized 0–1
- **Orbbec depth** (16UC1, uint16 mm): clipped at `orbbec_depth_max_m` (default 5 m), normalized 0–1
- Normalized map → `cv2.applyColorMap(COLORMAP_INFERNO)` → BGR → JPEG

---

## Mosaic Layout (source_id = 6)

3×2 grid of 160×120 thumbnails:

```
┌──────────────┬──────────────┬──────────────┐
│  orbbec (0)  │ left_side(1) │left_front(2) │
├──────────────┼──────────────┼──────────────┤
│right_side(3) │right_front(4)│ back_rear(5) │
└──────────────┴──────────────┴──────────────┘
```

Cameras with no data yet show as black cells.

---

## Bandwidth Estimates (JPEG quality 70)

| Mode | Resolution after scale | Est. KB/frame | @ 10 fps |
|---|---|---|---|
| Single RGB 100% | 480×360 | ~20 KB | ~200 KB/s |
| Single RGB 50% | 240×180 | ~6 KB | ~60 KB/s |
| Single RGB 25% | 120×90 | ~2 KB | ~20 KB/s |
| Depth heatmap 50% | 240×180 | ~5 KB | ~50 KB/s |
| Mosaic 50% | 240×120 | ~6 KB | ~60 KB/s |
| Map | variable | ~3 KB | ~3 KB/s @ 1fps |

Lower `jpeg_quality` and `scale` are tunable per launch argument or `network_params.yaml`.

---

## Server Parameters (`video_stream_node`)

| Parameter | Default | Description |
|---|---|---|
| `udp_port` | 5002 | UDP port for outbound video |
| `default_scale` | 50 | Scale % used when GUI sends scale=0 |
| `jpeg_quality` | 70 | JPEG encode quality (1–100) |
| `depth_max_m` | 8.0 | DA3 depth clip distance (m) |
| `orbbec_depth_max_m` | 5.0 | Orbbec depth clip distance (m) |

Override at launch: `ros2 launch octane network.launch.py default_scale:=25 jpeg_quality:=60`

---

## ROS2 Topics (internal)

| Topic | Type | Direction | Purpose |
|---|---|---|---|
| `/network/client_ip` | String | comm→video | GUI IP when connected, empty on disconnect |
| `/network/stream_request` | String | comm→video | `"source_id,variant,scale,fps"` |
