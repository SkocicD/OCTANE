import math
import numpy as np

# Camera poses from workspace/src/octane/octane/config/cameras.yaml (dev branch).
# pos: meters from base_link (ROS: X=forward, Y=left, Z=up).
# rot: degrees Euler — roll always 0 for all cameras; pitch=tilt-down; yaw=azimuth from robot forward.

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

POSE_TENSOR = _make_pose_tensor()
