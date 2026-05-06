"""ROS2 terrain mapping inference node.

Subscribes to /mapping/point_cloud/combined, runs TensorRT or PyTorch model,
publishes terrain heightmap, semantic grid, object list, and wall segments.

Launch:
    ros2 run octane_terrain_mapping inference_node \
        --ros-args -p engine_path:=/path/to/terrain_mapping.trt
"""
from __future__ import annotations
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from octane_terrain_mapping.bev_rasterizer import rasterize_from_ros_points, unpack_point_cloud2
from octane_terrain_mapping.instance_extraction import extract_instances
from octane_terrain_mapping.wall_fitting import fit_walls


class TerrainMappingNode(Node):
    def __init__(self):
        super().__init__("terrain_mapping")

        self.declare_parameter("engine_path", "")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("grid_size", 200)
        self.declare_parameter("cell_size", 0.05)

        self.conf_thresh = self.get_parameter("confidence_threshold").value
        self.grid_size = self.get_parameter("grid_size").value
        self.cell_size = self.get_parameter("cell_size").value
        engine_path = self.get_parameter("engine_path").value

        self._load_model(engine_path)

        self.sub = self.create_subscription(
            PointCloud2, "/mapping/point_cloud/combined", self._on_cloud, 10
        )
        self.pub_height = self.create_publisher(Image, "/terrain/heightmap", 10)
        self.pub_semantic = self.create_publisher(OccupancyGrid, "/terrain/semantic", 10)

        from octane_terrain_mapping.msg import TerrainObjects, WallSegments
        self.pub_objects = self.create_publisher(TerrainObjects, "/terrain/objects", 10)
        self.pub_walls = self.create_publisher(WallSegments, "/terrain/walls", 10)

        self.get_logger().info("TerrainMappingNode ready")

    def _load_model(self, engine_path: str):
        if engine_path:
            try:
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f:
                    runtime = trt.Runtime(logger)
                    self._engine = runtime.deserialize_cuda_engine(f.read())
                self._context = self._engine.create_execution_context()
                self._backend = "trt"
                self.get_logger().info(f"TensorRT engine loaded from {engine_path}")
                return
            except Exception as e:
                self.get_logger().warn(f"TRT load failed ({e}), falling back to PyTorch")

        import torch
        import pathlib, sys
        sys.path.insert(0, str(pathlib.Path(__file__).parents[4] /
                                "NVIDIA Isaac" / "Isaac Lab" / "lunabotics" /
                                "training" / "terrain_mapping"))
        from model import TerrainMappingModel
        self._torch_model = TerrainMappingModel()
        self._torch_model.eval()
        self._backend = "pytorch"
        self.get_logger().info("PyTorch fallback model loaded (no weights — random)")

    def _infer(self, bev: np.ndarray):
        if self._backend == "pytorch":
            import torch
            with torch.no_grad():
                x = torch.from_numpy(bev).unsqueeze(0)
                h, s, d = self._torch_model(x)
            return h.squeeze(0, 1).numpy(), s.squeeze(0).numpy(), d.squeeze(0).numpy()
        raise NotImplementedError("TRT inference: implement with pycuda bindings")

    def _on_cloud(self, msg: PointCloud2):
        points = unpack_point_cloud2(msg)
        bev = rasterize_from_ros_points(points, self.grid_size, self.cell_size)
        height_map, semantic_logits, detections = self._infer(bev)
        semantic_labels = semantic_logits.argmax(axis=0).astype(np.int8)

        header = Header()
        header.stamp = msg.header.stamp
        header.frame_id = "base_link"

        self._publish_heightmap(height_map, header)
        self._publish_semantic(semantic_labels, header)
        self._publish_objects(detections, semantic_labels, header)
        self._publish_walls(semantic_labels, header)

    def _publish_heightmap(self, height_map: np.ndarray, header: Header):
        from cv_bridge import CvBridge
        img_msg = CvBridge().cv2_to_imgmsg(height_map, encoding="32FC1")
        img_msg.header = header
        self.pub_height.publish(img_msg)

    def _publish_semantic(self, labels: np.ndarray, header: Header):
        msg = OccupancyGrid()
        msg.header = header
        msg.info.resolution = self.cell_size
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -(self.grid_size * self.cell_size) / 2.0
        msg.info.origin.position.y = -(self.grid_size * self.cell_size) / 2.0
        msg.data = labels.flatten().tolist()
        self.pub_semantic.publish(msg)

    def _publish_objects(self, detections: np.ndarray, semantic: np.ndarray, header: Header):
        from octane_terrain_mapping.msg import TerrainObjects, TerrainObject
        import torch
        msg = TerrainObjects()
        msg.header = header
        for inst in extract_instances(semantic, self.cell_size):
            obj = TerrainObject()
            obj.type, obj.x, obj.y, obj.diameter, obj.confidence = (
                inst["type"], inst["x"], inst["y"], inst["diameter"], 1.0
            )
            msg.objects.append(obj)
        confs = torch.tensor(detections[:, 3]).sigmoid().numpy()
        for i, conf in enumerate(confs):
            if conf >= self.conf_thresh:
                obj = TerrainObject()
                obj.type = int(round(float(detections[i, 4])))
                obj.x, obj.y, obj.diameter, obj.confidence = (
                    float(detections[i, 0]), float(detections[i, 1]),
                    float(detections[i, 2]), float(conf)
                )
                msg.objects.append(obj)
        self.pub_objects.publish(msg)

    def _publish_walls(self, semantic: np.ndarray, header: Header):
        from octane_terrain_mapping.msg import WallSegments, WallSegment
        msg = WallSegments()
        msg.header = header
        for w in fit_walls(semantic, self.cell_size):
            seg = WallSegment()
            seg.x1, seg.y1, seg.x2, seg.y2 = w["x1"], w["y1"], w["x2"], w["y2"]
            msg.walls.append(seg)
        self.pub_walls.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainMappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
