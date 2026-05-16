# OCTANE Network — Video Streaming

On-demand camera and terrain map streaming from rover to GUI over UDP.

---

## Overview

The GUI requests a stream over TCP; the rover sends data over UDP.
Camera/mosaic streams use JPEG-encoded chunks. The terrain map (source_id=7)
uses a raw binary payload (see [Terrain Map Stream](#terrain-map-stream-source_id-7)).
Control traffic (telemetry, commands, heartbeat) stays on TCP and is never
blocked by video.

```
GUI ──[TCP :5000]──▶ stream request (type V, 8 bytes)
GUI ◀─[UDP :5002]── JPEG chunks   (~1400 bytes each)   ← cameras / mosaic
GUI ◀─[UDP :5002]── raw binary chunks (~1400 bytes each) ← terrain map
```

---

## Stream Request — TCP type `V`

Wire format: `[O][V][4][source_id][variant][scale][fps][CRC]` — always **8 bytes**.

| Field | Size | Values |
|---|---|---|
| `source_id` | 1B | 0–5 = individual camera, 6 = mosaic, 7 = terrain map, 8–11 = far ESP32 cameras, 255 = stop all |
| `variant` | 1B | `R` (0x52) = RGB, `D` (0x44) = depth heatmap, `T` (0x54) = terrain raw |
| `scale` | 1B | 1–100 (% of native resolution); **0 = use server default** (ignored for terrain) |
| `fps` | 1B | 1–30 (target frame rate; 1 Hz recommended for terrain) |

Rover responds with a standard `A` (ACK) frame.

Sending `source_id=255` stops all active streams immediately.

---

## Source IDs

| ID | Source | Native resolution |
|---|---|---|
| 0 | orbbec_depth | 640×480 |
| 1 | near_rgb_left_side | 480×360 |
| 2 | near_rgb_left_front | 480×360 |
| 3 | near_rgb_right_side | 480×360 |
| 4 | near_rgb_right_front | 480×360 |
| 5 | near_rgb_back_rear | 480×360 |
| 6 | mosaic (all 6 tiled 3×2) | 480×240 |
| 7 | terrain map (nexus model output) | 200×200 grid, raw binary |
| 8 | far_front (ESP32, localization) | varies (JPEG from Pi) |
| 9 | far_right (ESP32, localization) | varies (JPEG from Pi) |
| 10 | far_back (ESP32, localization) | varies (JPEG from Pi) |
| 11 | far_left (ESP32, localization) | varies (JPEG from Pi) |
| 255 | stop all | — |

IDs 0–5 match declaration order in `octane/config/cameras.yaml`.
IDs 8–11 subscribe to `perception/camera/far/{front,right,back,left}/frame` published by `octane_localization/far_camera_receiver_node`. **RGB only** — depth variant is not supported for far cameras.

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

## Terrain Map Stream (source_id = 7)

Request with `variant='T'` (0x54). The rover subscribes to the nexus model output topics and
streams raw binary data — **no JPEG encoding**. The receiver gets float32 height values and
uint8 hazard probabilities suitable for direct 3D mesh construction.

### UDP Packet Format

Same 8-byte header as camera streams:
```
Byte 0:  0x4F  magic
Byte 1:  0x56  'V'
Byte 2:  0x07  source_id = 7
Byte 3:  0x54  variant   = 'T'
Byte 4-5: seq  (16-bit frame sequence)
Byte 6:  chunk_idx
Byte 7:  chunk_total
Bytes 8+: raw binary payload chunk (up to 1392 bytes)
```

### Assembled Payload (after chunk reassembly + zlib decompress)

```
Offset  Field         Type       Value / Notes
──────  ─────         ────       ─────────────
0       version       uint8      = 1
1       flags         uint8      bit0 = has_terrain
                                 bit1 = has_pose      (reserved, future)
                                 bit2 = has_nav       (reserved, future)
                                 bit7 = zlib_compressed (always set)
2-3     reserved      uint16     = 0

── terrain section (flags & 0x01) ─────────────────────────────────────────
4       width         uint32     = 200   (grid columns)
8       height        uint32     = 200   (grid rows)
12      cell_m        float32    = 0.05  (metres per cell; grid covers ±5 m)
16      height_map    float32×40000   metres, row-major; [100,100] = robot origin
160016  rocks         uint8×40000     0-255 scaled from 0-1 probability
200016  craters       uint8×40000     0-255 scaled from 0-1 probability
240016  walls         uint8×40000     0-255 scaled from 0-1 probability

── pose section (flags & 0x02, not yet sent) ───────────────────────────────
        pos_x/y/z     float32×3  metres (ROS: X=fwd, Y=left, Z=up)
        roll/pitch/yaw float32×3 radians

── nav section (flags & 0x04, not yet sent) ────────────────────────────────
        left_motor    float32    tanh ∈ [-1, 1]
        right_motor   float32    tanh ∈ [-1, 1]
        bucket        uint8      0=UP  1=COLLECT  2=DUMP
        _pad          uint8×3    = 0
```

**Total uncompressed size:** ~280 KB. zlib level 6 typically reduces to 80–150 KB.

### Receiver Pseudocode

```python
# Reassemble chunks by seq, then:
raw = zlib.decompress(payload)
version, flags = raw[0], raw[1]
assert flags & 0x80  # always compressed

if flags & 0x01:  # terrain
    width, height, cell_m = struct.unpack_from('<IIf', raw, 4)
    n = width * height
    off = 16
    height_map = np.frombuffer(raw, np.float32, n, off).reshape(height, width)
    off += n * 4
    rocks   = np.frombuffer(raw, np.uint8, n, off).reshape(height, width) / 255.0; off += n
    craters = np.frombuffer(raw, np.uint8, n, off).reshape(height, width) / 255.0; off += n
    walls   = np.frombuffer(raw, np.uint8, n, off).reshape(height, width) / 255.0; off += n
```

---

## Bandwidth Estimates (JPEG quality 70)

| Mode | Resolution after scale | Est. KB/frame | @ 10 fps |
|---|---|---|---|
| Single RGB 100% | 480×360 | ~20 KB | ~200 KB/s |
| Single RGB 50% | 240×180 | ~6 KB | ~60 KB/s |
| Single RGB 25% | 120×90 | ~2 KB | ~20 KB/s |
| Depth heatmap 50% | 240×180 | ~5 KB | ~50 KB/s |
| Mosaic 50% | 240×120 | ~6 KB | ~60 KB/s |
| Terrain map (compressed) | 200×200 grid | ~80–150 KB | ~80–150 KB/s @ 1fps |

Lower `jpeg_quality` and `scale` are tunable per launch argument or `network_params.yaml`.
Terrain `scale` is ignored — the grid is always 200×200.

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
