"""Export trained model to TensorRT FP16 engine for Orin deployment.

Run ON the Jetson Orin (TRT engines are device-specific):
    python export_trt.py --checkpoint /path/to/best_model.pt --output terrain_mapping.trt

ONNX export works on any machine with PyTorch. TRT build requires tensorrt on the Orin.
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import torch
from model import TerrainMappingModel


def export(checkpoint: str, output: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TerrainMappingModel().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    dummy = torch.zeros(1, 6, 200, 200, device=device)
    onnx_path = pathlib.Path(output).with_suffix(".onnx")

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["bev"],
        output_names=["height", "semantic", "detections"],
        dynamic_axes={"bev": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX saved to {onnx_path}")

    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        engine_bytes = builder.build_serialized_network(network, config)
        with open(output, "wb") as f:
            f.write(engine_bytes)
        print(f"TensorRT FP16 engine saved to {output}")
    except ImportError:
        print("tensorrt not found — ONNX export only. Install tensorrt on the Orin to build engine.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="terrain_mapping.trt")
    args = p.parse_args()
    export(args.checkpoint, args.output)
