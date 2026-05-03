"""Launch file for terrain mapping data collection (WSL2 side).

Start this BEFORE running collect_terrain_data.bat on Windows.
It launches the full perception stack + terrain data collector node.

Usage:
    ros2 launch octane data_collection.launch.py
    ros2 launch octane data_collection.launch.py gt_dir:=/mnt/e/terrain_data/gt output_dir:=/mnt/e/terrain_data
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    octane_share = get_package_share_directory("octane")

    return LaunchDescription([
        DeclareLaunchArgument("gt_dir",     default_value="/mnt/e/terrain_data/gt",
                              description="Directory where Isaac Sim writes ground truth files"),
        DeclareLaunchArgument("output_dir", default_value="/mnt/e/terrain_data",
                              description="Directory to save paired .npz training samples"),

        # Full perception stack (DA3 depth estimation + cameras)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(octane_share, "launch", "perception.launch.py")
            ),
        ),

        # Point cloud mux — merges all camera clouds into /mapping/point_cloud/combined
        Node(
            package="octane_mapping",
            executable="point_cloud_mux_node",
            name="point_cloud_mux_node",
            output="screen",
        ),

        # Data collector — pairs point clouds with Isaac Sim ground truth
        Node(
            package="octane_mapping",
            executable="terrain_data_collector_node",
            name="terrain_data_collector_node",
            output="screen",
            parameters=[{
                "gt_dir":     LaunchConfiguration("gt_dir"),
                "output_dir": LaunchConfiguration("output_dir"),
                "grid_size":  200,
                "cell_size":  0.05,
            }],
        ),
    ])
