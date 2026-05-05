"""Export trained TerrainModel to ONNX and optionally TensorRT.

Usage:
    python training/export.py --checkpoint training/checkpoints/best.pt --onnx terrain.onnx
    python training/export.py --checkpoint training/checkpoints/best.pt --onnx terrain.onnx --trt terrain.engine --fp16
"""
import argparse
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from training.model import TerrainModel


def export_onnx(checkpoint_path: str, onnx_path: str) -> None:
    model = TerrainModel()
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy_images   = torch.randn(1, 12, 3, 224, 224)
    dummy_rotation = torch.randn(1, 6)

    torch.onnx.export(
        model,
        (dummy_images, dummy_rotation),
        onnx_path,
        input_names=['images', 'rotation'],
        output_names=['height', 'rocks', 'craters', 'walls'],
        dynamic_axes={
            'images':   {0: 'batch'},
            'rotation': {0: 'batch'},
            'height':   {0: 'batch'},
            'rocks':    {0: 'batch'},
            'craters':  {0: 'batch'},
            'walls':    {0: 'batch'},
        },
        opset_version=17,
    )
    print(f"[export] ONNX saved to {onnx_path}")

    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(onnx_path)
    out  = sess.run(None, {
        'images':   dummy_images.numpy(),
        'rotation': dummy_rotation.numpy(),
    })
    print(f"[export] ONNX verified. Output shapes: {[o.shape for o in out]}")


def export_trt(onnx_path: str, engine_path: str, fp16: bool = True) -> None:
    import tensorrt as trt
    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[export] FP16 enabled")

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    print(f"[export] TRT engine saved to {engine_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--onnx',       required=True)
    parser.add_argument('--trt',        default=None,
                        help='Path for TRT .engine output (optional)')
    parser.add_argument('--fp16',       action='store_true')
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.onnx)
    if args.trt:
        export_trt(args.onnx, args.trt, fp16=args.fp16)


if __name__ == '__main__':
    main()
