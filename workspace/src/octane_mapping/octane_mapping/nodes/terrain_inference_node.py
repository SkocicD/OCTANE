"""Terrain mapping inference node.

Subscribes to /mapping/point_cloud/combined, runs TensorRT or PyTorch model,
publishes heightmap, semantic grid, object list, and wall segments.

Launch:
    ros2 run octane_mapping terrain_inference_node \
        --ros-args -p engine_path:=/path/to/terrain_mapping.trt
"""
from __future__ import annotations
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from cv_bridge import CvBridge

from octane_mapping.nodes.bev_rasterizer import rasterize_from_ros_points, unpack_point_cloud2
from octane_mapping.nodes.instance_extraction import extract_instances
from octane_mapping.nodes.wall_fitting import fit_walls


class TerrainInferenceNode(Node):
    def __init__(self):
        super().__init__("terrain_inference")

        self.declare_parameter("engine_path",          "")
        self.declare_parameter("model_checkpoint",     "")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("grid_size",            200)
        self.declare_parameter("cell_size",            0.05)

        self._conf_thresh = self.get_parameter("confidence_threshold").value
        self._grid_size   = self.get_parameter("grid_size").value
        self._cell_size   = self.get_parameter("cell_size").value
        self._bridge      = CvBridge()

        self._load_model(
            self.get_parameter("engine_path").value,
            self.get_parameter("model_checkpoint").value,
        )

        self._sub = self.create_subscription(
            PointCloud2, "/mapping/point_cloud/combined", self._on_cloud, 10
        )
        self._pub_height   = self.create_publisher(Image,         "/terrain/heightmap", 10)
        self._pub_semantic = self.create_publisher(OccupancyGrid,  "/terrain/semantic",  10)

        from octane_msgs.msg import TerrainObjects, WallSegments
        self._pub_objects = self.create_publisher(TerrainObjects, "/terrain/objects", 10)
        self._pub_walls   = self.create_publisher(WallSegments,   "/terrain/walls",   10)

        self.get_logger().info("TerrainInferenceNode ready")

    def _load_model(self, engine_path: str, checkpoint: str):
        if engine_path:
            try:
                import tensorrt as trt
                logger = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f:
                    self._engine  = trt.Runtime(logger).deserialize_cuda_engine(f.read())
                self._context = self._engine.create_execution_context()
                self._backend = "trt"
                self.get_logger().info(f"TRT engine loaded: {engine_path}")
                return
            except Exception as e:
                self.get_logger().warn(f"TRT load failed ({e}), falling back to PyTorch")

        from octane_mapping.nodes._model_import import load_terrain_model
        self._torch_model = load_terrain_model(checkpoint)
        self._torch_model.eval()
        self._backend = "pytorch"
        status = f"checkpoint={checkpoint}" if checkpoint else "no weights (random)"
        self.get_logger().info(f"PyTorch fallback loaded — {status}")

    def _infer(self, bev: np.ndarray):
        if self._backend == "pytorch":
            import torch
            with torch.no_grad():
                x = torch.from_numpy(bev).unsqueeze(0)
                h, s, d = self._torch_model(x)
            return h.squeeze().numpy(), s.squeeze(0).numpy(), d.squeeze(0).numpy()
        raise NotImplementedError("TRT inference — use engine_path on Jetson")

    def _on_cloud(self, msg: PointCloud2):
        points = unpack_point_cloud2(msg)
        bev    = rasterize_from_ros_points(points, self._grid_size, self._cell_size)
        height_map, semantic_logits, detections = self._infer(bev)
        semantic = semantic_logits.argmax(axis=0).astype(np.int8)

        header = Header()
        header.stamp    = msg.header.stamp
        header.frame_id = "base_link"

        self._publish_heightmap(height_map, header)
        self._publish_semantic(semantic, header)
        self._publish_objects(detections, semantic, header)
        self._publish_walls(semantic, header)

    def _publish_heightmap(self, height_map: np.ndarray, header: Header):
        img = self._bridge.cv2_to_imgmsg(height_map, encoding="32FC1")
        img.header = header
        self._pub_height.publish(img)

    def _publish_semantic(self, labels: np.ndarray, header: Header):
        msg = OccupancyGrid()
        msg.header          = header
        msg.info.resolution = self._cell_size
        msg.info.width      = self._grid_size
        msg.info.height     = self._grid_size
        half = self._grid_size * self._cell_size / 2.0
        msg.info.origin.position.x = -half
        msg.info.origin.position.y = -half
        msg.data = labels.flatten().tolist()
        self._pub_semantic.publish(msg)

    def _publish_objects(self, detections: np.ndarray, semantic: np.ndarray, header: Header):
        from octane_msgs.msg import TerrainObjects, TerrainObject
        import torch
        msg = TerrainObjects()
        msg.header = header
        for inst in extract_instances(semantic, self._cell_size):
            obj = TerrainObject()
            obj.type, obj.x, obj.y, obj.diameter, obj.confidence = (
                inst["type"], inst["x"], inst["y"], inst["diameter"], 1.0
            )
            msg.objects.append(obj)
        confs = torch.tensor(detections[:, 3]).sigmoid().numpy()
        for i, conf in enumerate(confs):
            if conf >= self._conf_thresh:
                obj = TerrainObject()
                obj.type       = int(round(float(detections[i, 4])))
                obj.x          = float(detections[i, 0])
                obj.y          = float(detections[i, 1])
                obj.diameter   = float(detections[i, 2])
                obj.confidence = float(conf)
                msg.objects.append(obj)
        self._pub_objects.publish(msg)

    def _publish_walls(self, semantic: np.ndarray, header: Header):
        from octane_msgs.msg import WallSegments, WallSegment
        msg = WallSegments()
        msg.header = header
        for w in fit_walls(semantic, self._cell_size):
            seg = WallSegment()
            seg.x1, seg.y1, seg.x2, seg.y2 = w["x1"], w["y1"], w["x2"], w["y2"]
            msg.walls.append(seg)
        self._pub_walls.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TerrainInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
