"""Terrain data collector — WSL2 side of the data collection pipeline.

Watches a ground truth directory for .ready sentinels written by the
Isaac Sim collect_terrain_data.py script.  For each new episode it:
  1. Reads the paired _gt.npz (height_gt, semantic_gt, objects_gt, walls_gt)
  2. Waits for the next /mapping/point_cloud/combined message
  3. Rasterises the point cloud into a BEV grid
  4. Saves the paired training sample to output_dir

Launch:
    ros2 run octane_mapping terrain_data_collector_node \\
        --ros-args \\
        -p gt_dir:=/mnt/e/terrain_data/gt \\
        -p output_dir:=/mnt/e/terrain_data
"""
from __future__ import annotations
import pathlib
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from octane_mapping.nodes.bev_rasterizer import rasterize_from_ros_points, unpack_point_cloud2


class TerrainDataCollectorNode(Node):
    def __init__(self):
        super().__init__("terrain_data_collector")

        self.declare_parameter("gt_dir",     "/mnt/e/terrain_data/gt")
        self.declare_parameter("output_dir", "/mnt/e/terrain_data")
        self.declare_parameter("grid_size",  200)
        self.declare_parameter("cell_size",  0.05)

        self._gt_dir     = pathlib.Path(self.get_parameter("gt_dir").value)
        self._output_dir = pathlib.Path(self.get_parameter("output_dir").value)
        self._grid_size  = self.get_parameter("grid_size").value
        self._cell_size  = self.get_parameter("cell_size").value

        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._processed: set[str] = set()
        self._pending_ep: str | None = None
        self._pending_gt: dict | None = None
        self._saved = 0

        self._sub = self.create_subscription(
            PointCloud2, "/mapping/point_cloud/combined", self._cloud_cb, 10
        )
        self.create_timer(0.5, self._check_ready_files)

        self.get_logger().info(
            f"TerrainDataCollectorNode ready\n"
            f"  gt_dir     : {self._gt_dir}\n"
            f"  output_dir : {self._output_dir}"
        )

    # ── timer: scan for new .ready sentinels ──────────────────────────────

    def _check_ready_files(self):
        if self._pending_ep is not None:
            return  # still waiting for a point cloud

        for ready in sorted(self._gt_dir.glob("ep_*.ready")):
            ep_id = ready.stem  # e.g. ep_000042
            if ep_id in self._processed:
                continue
            gt_file = self._gt_dir / f"{ep_id}_gt.npz"
            if not gt_file.exists():
                continue
            self._pending_ep = ep_id
            self._pending_gt = self._load_gt(gt_file)
            self.get_logger().info(f"Episode {ep_id} — waiting for point cloud")
            return  # handle one episode per timer tick

    # ── point cloud callback ──────────────────────────────────────────────

    def _cloud_cb(self, msg: PointCloud2):
        if self._pending_ep is None:
            return

        ep_id = self._pending_ep
        gt    = self._pending_gt
        self._pending_ep = None
        self._pending_gt = None

        points = unpack_point_cloud2(msg)
        bev    = rasterize_from_ros_points(points, self._grid_size, self._cell_size)

        np.savez_compressed(
            str(self._output_dir / f"{ep_id}.npz"),
            bev=bev,
            height_gt=gt["height_gt"],
            semantic_gt=gt["semantic_gt"],
            objects_gt=gt["objects_gt"],
            walls_gt=gt["walls_gt"],
            robot_pos=gt["robot_pos"],
            robot_yaw=gt["robot_yaw"],
        )

        self._processed.add(ep_id)
        self._saved += 1
        if self._saved % 100 == 0:
            self.get_logger().info(f"{self._saved} episodes saved")

    # ── helpers ───────────────────────────────────────────────────────────

    def _load_gt(self, path: pathlib.Path) -> dict:
        raw = np.load(str(path), allow_pickle=True)
        return {
            "height_gt":   raw["height_gt"],
            "semantic_gt": raw["semantic_gt"],
            "objects_gt":  raw["objects_gt"],
            "walls_gt":    raw["walls_gt"],
            "robot_pos":   raw["robot_pos"],
            "robot_yaw":   raw["robot_yaw"],
        }


def main(args=None):
    rclpy.init(args=args)
    node = TerrainDataCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
