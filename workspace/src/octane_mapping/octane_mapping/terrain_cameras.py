import math
import numpy as np

# Camera poses from octane/config/cameras.yaml.
# pos: metres from base_link (ROS: X=forward, Y=left, Z=up).
# rot: degrees Euler — pitch=tilt-down (90°=horizontal, 135°=45° below), yaw=azimuth from forward.

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

INPUT_SLOTS = [
    ('near_rgb_left_front',  'rgb'),
    ('near_rgb_left_side',   'rgb'),
    ('near_rgb_right_front', 'rgb'),
    ('near_rgb_right_side',  'rgb'),
    ('near_rgb_back_rear',   'rgb'),
    ('orbbec_depth',         'rgb'),
    ('near_rgb_left_front',  'depth'),
    ('near_rgb_left_side',   'depth'),
    ('near_rgb_right_front', 'depth'),
    ('near_rgb_right_side',  'depth'),
    ('near_rgb_back_rear',   'depth'),
    ('orbbec_depth',         'depth'),
]

# Camera name per slot (for intrinsics / ray-map lookup)
SLOT_CAMERAS = [cam for cam, _ in INPUT_SLOTS]

# Intrinsics per slot (sim capture resolution; scaled at runtime by compute_ray_maps)
_INTR_RGB    = dict(fx=131.9, fy=158.3, cx=240.0, cy=162.0, w=480, h=360)
_INTR_ORBBEC = dict(fx=570.0, fy=570.0, cx=320.0, cy=240.0, w=640, h=480)
SLOT_INTRINSICS = [
    _INTR_RGB,    # slot  0 — left_front rgb
    _INTR_RGB,    # slot  1 — left_side rgb
    _INTR_RGB,    # slot  2 — right_front rgb
    _INTR_RGB,    # slot  3 — right_side rgb
    _INTR_RGB,    # slot  4 — back_rear rgb
    _INTR_ORBBEC, # slot  5 — depth_cam rgb
    _INTR_RGB,    # slot  6 — left_front depth
    _INTR_RGB,    # slot  7 — left_side depth
    _INTR_RGB,    # slot  8 — right_front depth
    _INTR_RGB,    # slot  9 — right_side depth
    _INTR_RGB,    # slot 10 — back_rear depth
    _INTR_ORBBEC, # slot 11 — depth_cam depth
]


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


POSE_TENSOR = _make_pose_tensor()


def _cam_to_robot_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """3x3 rotation: camera optical frame -> robot body frame (ROS: X fwd, Y left, Z up).

    Camera optical: Z forward, X right, Y down.
    pitch_deg=90 -> horizontal; pitch_deg=135 -> 45 deg below horizontal.
    """
    tilt = _r(pitch_deg - 90.0)
    yaw  = _r(yaw_deg)
    C = np.array([[ 0,  0, 1],
                  [-1,  0, 0],
                  [ 0, -1, 0]], dtype=np.float32)
    Ry = np.array([[ math.cos(tilt), 0, math.sin(tilt)],
                   [ 0,              1, 0             ],
                   [-math.sin(tilt), 0, math.cos(tilt)]], dtype=np.float32)
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0,              0,             1]], dtype=np.float32)
    return Rz @ Ry @ C


def compute_ray_maps(img_size: int = 224) -> np.ndarray:
    """Per-pixel unit ray directions in robot body frame for all 12 input slots.

    Returns (12, 3, img_size, img_size) float32 -- (dx, dy, dz) per pixel.
    Fixed camera geometry; compute once and store as a model buffer.
    """
    uu, vv = np.meshgrid(np.arange(img_size), np.arange(img_size))
    ray_maps = np.zeros((12, 3, img_size, img_size), dtype=np.float32)
    for slot, (cam_name, intr) in enumerate(zip(SLOT_CAMERAS, SLOT_INTRINSICS)):
        fx = intr['fx'] * img_size / intr['w']
        fy = intr['fy'] * img_size / intr['h']
        cx = intr['cx'] * img_size / intr['w']
        cy = intr['cy'] * img_size / intr['h']
        rx = (uu - cx) / fx
        ry = (vv - cy) / fy
        rz = np.ones_like(rx)
        norm = np.sqrt(rx**2 + ry**2 + rz**2)
        rays_cam = np.stack([rx / norm, ry / norm, rz / norm], axis=0)  # (3, H, W)
        _, _, _, pitch_deg, yaw_deg = _POSES[cam_name]
        R = _cam_to_robot_rotation(yaw_deg, pitch_deg)
        ray_maps[slot] = (R @ rays_cam.reshape(3, -1)).reshape(3, img_size, img_size)
    return ray_maps
