import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


# ── Arena zone definitions ────────────────────────────────────────────────────
# Rectangles defined by two opposite corners (x1,y1) → (x2,y2).
# Coordinate frame: origin at SW corner of arena, X east (away from berm),
# Y north.  Units: meters.
#
# UPDATE THESE from the NASA field specification PDF before each competition.
# These values are placeholders based on a typical Lunabotics arena layout.
#
#  ┌────────────────────────────────────────────┐  Y = 3.81m
#  │  start / deposition  │     nav     │ excav │
#  │   (includes berm)    │             │       │
#  └────────────────────────────────────────────┘  Y = 0.00m
#  X=0                   X=1.50       X=3.50   X=7.62
#
ZONES = {
    # Robot starts here; also where it returns to dump regolith.
    'start': {
        'x1': 0.00, 'y1': 0.00,
        'x2': 1.50, 'y2': 3.81,
    },
    # Transition corridor — robot drives through here between start and digging.
    'nav': {
        'x1': 1.50, 'y1': 0.00,
        'x2': 3.50, 'y2': 3.81,
    },
    # Active digging area — regolith is here.
    'excavation': {
        'x1': 3.50, 'y1': 0.00,
        'x2': 7.62, 'y2': 3.81,
    },
    # Area directly in front of the berm where the bucket is raised and dumped.
    'deposition': {
        'x1': 0.00, 'y1': 0.75,
        'x2': 1.00, 'y2': 3.06,
    },
    # The physical berm structure itself.
    'berm': {
        'x1': 0.00, 'y1': 1.40,
        'x2': 0.40, 'y2': 2.41,
    },
}

# Nav2 goal waypoints — center of each target zone, used by the supervisor to
# send goals.  Derived from ZONES above; update if ZONES change.
ZONE_CENTERS = {
    name: {
        'x': (z['x1'] + z['x2']) / 2.0,
        'y': (z['y1'] + z['y2']) / 2.0,
    }
    for name, z in ZONES.items()
}


def generate_launch_description():
    nodes = [
        LogInfo(msg='Starting mapping subsystem'),
        # Stub identity odom→base_link until octane_localization is ready
        # (April tags + IMU fusion).  Replace this node with the real source then.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
            output='log',
        ),
    ]

    # Load camera config
    try:
        config_file = os.path.join(
            get_package_share_directory('octane'), 'config', 'cameras.yaml'
        )
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f'[WARN] Could not load cameras.yaml: {e} — skipping mapping nodes')
        nodes.append(LogInfo(msg='Mapping subsystem skipped (cameras.yaml unavailable)'))
        return LaunchDescription(nodes)

    # ── Camera frame splitters (one per camera) ──────────────────────────────
    # Unpacks each CameraFrame into Image + CameraInfo on nav2/AI-compatible
    # topic names, and broadcasts the static TF base_link → <cam>_frame.
    for cam_name, cam_cfg in cfg['cameras'].items():
        nodes.append(
            Node(
                package='octane_mapping',
                executable='camera_frame_splitter',
                name=f'{cam_name}_rgb_splitter',
                output='log',
                parameters=[{
                    'camera_name': cam_name,
                    'input_topic': cam_cfg['rgb_topic'],
                    'output_namespace': f'mapping/{cam_name}/rgb',
                    'frame_id': f'{cam_name}_frame',
                    'parent_frame': 'base_link',
                    'config_file': config_file,
                }],
            )
        )
        nodes.append(
            Node(
                package='octane_mapping',
                executable='camera_frame_splitter',
                name=f'{cam_name}_depth_splitter',
                output='log',
                parameters=[{
                    'camera_name': cam_name,
                    'input_topic': cam_cfg['depth_topic'],
                    'output_namespace': f'mapping/{cam_name}/depth',
                    'frame_id': f'{cam_name}_frame',
                    'parent_frame': 'base_link',
                    'config_file': config_file,
                }],
            )
        )

    # ── Nav2 (path planning against static arena map) ────────────────────────
    # Publishes /plan continuously as the AI policy's path hint.
    # Zone goal waypoints (ZONE_CENTERS above) are sent by the supervisor.
    #
    # map_server and nav2_bringup are expected to be launched separately via
    # the nav2 bringup package once the static arena map .pgm/.yaml is ready.
    # Add those nodes here when the map file is available.

    nodes.append(LogInfo(msg='Mapping subsystem online'))
    return LaunchDescription(nodes)
