"""Configuration for the CSU Lunabotics 6-wheel skid-steer rover."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ── USD path resolution ───────────────────────────────────────────────────────
# This file lives at:
#   <project>/source/lunabotics/lunabotics/assets/lunabotics.py
# The USD file lives at:
#   <repo>/NVIDIA Isaac/Isaac Sim/USD/CSU_Lunabotics_Isaac_Lab_Model.usd
#
# Directory tree relative to this file:
#   assets/               ← os.path.dirname(__file__)
#   lunabotics/ (inner)   ← up 1
#   lunabotics/ (source)  ← up 2
#   source/               ← up 3
#   lunabotics/ (project) ← up 4  (project root)
#   Isaac Lab/            ← up 5
#   NVIDIA Isaac/         ← up 6  (_NVIDIA_ISAAC_ROOT)
_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_NVIDIA_ISAAC_ROOT = os.path.normpath(os.path.join(_ASSETS_DIR, "..", "..", "..", "..", "..", ".."))
_USD_PATH = os.path.join(_NVIDIA_ISAAC_ROOT, "Isaac Sim", "USD", "CSU_Lunabotics_Isaac_Lab_Model.usd")
ARENA_USD_PATH = os.path.join(
    _NVIDIA_ISAAC_ROOT, "Isaac Sim", "Lunabotics Arenas", "Artemis Arena", "ksc_artemis_arena.usd"
)

_WHEEL_JOINTS = [
    "Left_Front_Wheel", "Left_Center_Wheel", "Left_Rear_Wheel",
    "Right_Front_Wheel", "Right_Center_Wheel", "Right_Rear_Wheel",
]

LUNABOTICS_DIRECT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        scale=(0.1, 0.1, 0.1),  # USD authored in mm; metersPerUnit=0.01 tag is wrong — true unit is mm → scale=0.1 gives correct geometry
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=0.5,  # low limit prevents explosive spawn launches
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
            fix_root_link=False,  # override USD fixed-root if present — robot must be free-floating
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),  # scale=0.1 — RE-RUN find_resting_height.py to verify exact resting height
        joint_pos={j: 0.0 for j in _WHEEL_JOINTS},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # DCMotorCfg = explicit actuator: computes effort in Python, immune to USD DriveAPI bugs.
        "wheels": DCMotorCfg(
            joint_names_expr=_WHEEL_JOINTS,
            saturation_effort=25.0,      # 25 Nm motor limit
            effort_limit=25.0,
            velocity_limit=3.665,        # 35 RPM = 3.665 rad/s
            stiffness=0.0,
            damping=50.0,               # 50 × 3.665 = 183 Nm → saturates at 25 Nm; hits max torque at ~0.5 rad/s error
            friction=0.0,
        ),
    },
)
"""DirectRLEnv config: DCMotorCfg explicit actuator, bypasses USD drives entirely."""
