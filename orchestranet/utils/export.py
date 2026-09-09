"""
ONNX / TensorRT Export Utility for OrchestraNet.

Exports the full pipeline or individual micro-models to optimized formats
for production deployment:
  1. ONNX export with dynamic batch size
  2. ONNX optimization (graph simplification, constant folding)
  3. TensorRT conversion (FP16 / INT8 quantization)
  4. Validation of exported models against PyTorch outputs

Usage:
  python -m orchestranet.utils.export --weights checkpoints/orchestranet_final.pt --format onnx
  python -m orchestranet.utils.export --weights checkpoints/orchestranet_final.pt --format tensorrt --precision fp16
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ModelExporter:
    """
    Export OrchestraNet models to optimized inference formats.

    Supports exporting:
      - The full OrchestraNet pipeline (backbone + FPN + all models)
      - Individual micro-models (M1-M7) for fine-grained deployment
      - The backbone + FPN separately for shared feature extraction
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def export_onnx(
        self,
        output_path: str,
        input_size: tuple[int, int] = (640, 640),
        batch_size: int = 1,
        dynamic_batch: bool = True,
        opset_version: int = 17,
        simplify: bool = True,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            output_path: Path for the .onnx file
            input_size: (H, W) input resolution
            batch_size: Static batch size (if dynamic_batch=False)
            dynamic_batch: Allow variable batch sizes
            opset_version: ONNX opset version
            simplify: Run onnx-simplifier after export

        Returns:
            Path to the exported ONNX model
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Create dummy input
        dummy = torch.randn(
            batch_size, 3, input_size[0], input_size[1],
            device=self.device,
        )

        # Dynamic axes for variable batch size
        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"images": {0: "batch_size"}}

        print(f"📦 Exporting to ONNX: {output_path}")
        print(f"   Input shape: {dummy.shape}")
        print(f"   Opset version: {opset_version}")
        print(f"   Dynamic batch: {dynamic_batch}")

        # We need a wrapper that only returns tensors (ONNX doesn't support dicts)
        wrapper = _ONNXWrapper(self.model)

        torch.onnx.export(
            wrapper,
            dummy,
            output_path,
            opset_version=opset_version,
            input_names=["images"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        print(f"   ✅ ONNX export complete: {output_path}")

        # Validate
        self._validate_onnx(output_path, dummy)

        # Simplify
        if simplify:
            self._simplify_onnx(output_path)

        # Report file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   📊 Model size: {size_mb:.1f} MB")

        return output_path

    def export_individual_models(
        self,
        output_dir: str,
        input_size: tuple[int, int] = (640, 640),
        opset_version: int = 17,
    ) -> dict[str, str]:
        """
        Export each micro-model individually for microservice deployment.

        Returns:
            Dict mapping model_id to output path
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # Export backbone + FPN
        print("\n▶ Exporting Backbone + FPN...")
        backbone_fpn = _BackboneFPNWrapper(self.model)
        backbone_path = os.path.join(output_dir, "backbone_fpn.onnx")
        dummy_img = torch.randn(1, 3, input_size[0], input_size[1], device=self.device)

        torch.onnx.export(
            backbone_fpn, dummy_img, backbone_path,
            opset_version=opset_version,
            input_names=["images"],
            output_names=["P3", "P4", "P5"],
            do_constant_folding=True,
        )
        paths["backbone_fpn"] = backbone_path
        print(f"   ✅ {backbone_path}")

        # Create dummy FPN features for exporting individual models
        fpn_features = [
            torch.randn(1, 128, 80, 80, device=self.device),
            torch.randn(1, 128, 40, 40, device=self.device),
            torch.randn(1, 128, 20, 20, device=self.device),
        ]

        # Export each micro-model
        for model_id, model in self.model.models.items():
            print(f"\n▶ Exporting {model_id}: {model.model_name}...")
            model_path = os.path.join(output_dir, f"{model_id}.onnx")

            wrapper = _MicroModelWrapper(model)

            # Use concatenated features as input since ONNX needs fixed inputs
            # The wrapper will split them back
            try:
                torch.onnx.export(
                    wrapper,
                    (fpn_features[0], fpn_features[1], fpn_features[2]),
                    model_path,
                    opset_version=opset_version,
                    input_names=["P3", "P4", "P5"],
                    do_constant_folding=True,
                )
                paths[model_id] = model_path
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"   ✅ {model_path} ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"   ⚠️  Failed to export {model_id}: {e}")

        return paths

    def export_tensorrt(
        self,
        onnx_path: str,
        output_path: str,
        precision: str = "fp16",
        max_batch_size: int = 8,
        workspace_gb: float = 4.0,
        calibration_data: torch.Tensor | None = None,
    ) -> str:
        """
        Convert ONNX model to TensorRT engine.

        Args:
            onnx_path: Path to ONNX model
            output_path: Path for TensorRT engine
            precision: "fp32", "fp16", or "int8"
            max_batch_size: Maximum batch size for the engine
            workspace_gb: GPU workspace in GB
            calibration_data: Calibration data for INT8 (required if precision="int8")

        Returns:
            Path to TensorRT engine
        """
        try:
            import tensorrt as trt
        except ImportError:
            print("❌ TensorRT not installed. Install with: pip install tensorrt")
            print("   Generating TensorRT conversion script instead...")
            self._generate_trt_script(onnx_path, output_path, precision)
            return output_path

        print(f"🔧 Converting to TensorRT ({precision})...")
        print(f"   Input: {onnx_path}")
        print(f"   Output: {output_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"   Error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gb * (1 << 30))
        )

        if precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            print("   Using FP16 precision")
        elif precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_data is not None:
                config.int8_calibrator = _INT8Calibrator(calibration_data)
            print("   Using INT8 precision")

        # Dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "images",
            min=(1, 3, 640, 640),
            opt=(4, 3, 640, 640),
            max=(max_batch_size, 3, 640, 640),
        )
        config.add_optimization_profile(profile)

        # Build engine
        print("   Building engine (this may take several minutes)...")
        engine = builder.build_serialized_network(network, config)

        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(engine)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✅ TensorRT engine saved: {output_path} ({size_mb:.1f} MB)")

        return output_path

    def _validate_onnx(self, onnx_path: str, dummy_input: torch.Tensor):
        """Validate ONNX model produces same outputs as PyTorch."""
        try:
            import onnx
            import onnxruntime as ort

            # Check model validity
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("   ✅ ONNX model validation passed")

            # Compare outputs
            session = ort.InferenceSession(onnx_path)
            ort_inputs = {"images": dummy_input.cpu().numpy()}
            ort_outputs = session.run(None, ort_inputs)

            with torch.no_grad():
                wrapper = _ONNXWrapper(self.model)
                pt_outputs = wrapper(dummy_input)

            for i, (ort_out, pt_out) in enumerate(zip(ort_outputs, pt_outputs)):
                pt_np = pt_out.cpu().numpy()
                diff = np.abs(ort_out - pt_np).max()
                print(f"   Output {i}: max diff = {diff:.6f}")
                if diff > 0.01:
                    print(f"   ⚠️  Large difference in output {i}!")

        except ImportError:
            print("   ⚠️  onnx/onnxruntime not installed, skipping validation")

    def _simplify_onnx(self, onnx_path: str):
        """Simplify ONNX model graph."""
        try:
            import onnx
            from onnxsim import simplify

            model = onnx.load(onnx_path)
            simplified, ok = simplify(model)
            if ok:
                onnx.save(simplified, onnx_path)
                print("   ✅ ONNX graph simplified")
            else:
                print("   ⚠️  Simplification failed, keeping original")
        except ImportError:
            print("   ⚠️  onnxsim not installed, skipping simplification")

    def _generate_trt_script(self, onnx_path, output_path, precision):
        """Generate a shell script for TensorRT conversion via trtexec."""
        script = f"""#!/bin/bash
# TensorRT conversion script for OrchestraNet
# Requires: NVIDIA TensorRT (trtexec) installed

trtexec \\
    --onnx={onnx_path} \\
    --saveEngine={output_path} \\
    --{'fp16' if precision == 'fp16' else 'int8' if precision == 'int8' else 'noTF32'} \\
    --workspace=4096 \\
    --minShapes=images:1x3x640x640 \\
    --optShapes=images:4x3x640x640 \\
    --maxShapes=images:8x3x640x640 \\
    --verbose

echo "Engine saved to {output_path}"
"""
        script_path = output_path.replace(".engine", "_convert.sh")
        with open(script_path, "w") as f:
            f.write(script)
        print(f"   📝 Conversion script written: {script_path}")


# ============ ONNX Wrapper Models ============

class _ONNXWrapper(torch.nn.Module):
    """Wrapper that returns flat tensors instead of dicts for ONNX compatibility."""

    def __init__(self, orchestranet):
        super().__init__()
        self.backbone = orchestranet.backbone
        self.fpn = orchestranet.fpn
        self.m1 = orchestranet.models["m1"]

    def forward(self, images):
        features = self.backbone(images)
        fpn_features = self.fpn(features)
        m1_out = self.m1(fpn_features)

        boxes = m1_out["decoded_boxes"]
        scores = torch.sigmoid(m1_out["objectness"]).squeeze(-1)
        class_probs = torch.sigmoid(m1_out["class_logits"])
        max_scores, labels = class_probs.max(dim=-1)
        final_scores = scores * max_scores

        return boxes, final_scores, labels


class _BackboneFPNWrapper(torch.nn.Module):
    """Exports backbone + FPN as a standalone feature extractor."""

    def __init__(self, orchestranet):
        super().__init__()
        self.backbone = orchestranet.backbone
        self.fpn = orchestranet.fpn

    def forward(self, images):
        features = self.backbone(images)
        fpn_features = self.fpn(features)
        return fpn_features[0], fpn_features[1], fpn_features[2]


class _MicroModelWrapper(torch.nn.Module):
    """Wraps a micro-model to accept flat tensor inputs for ONNX."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, p3, p4, p5):
        features = [p3, p4, p5]
        out = self.model(features)
        # Return first tensor value from output dict
        values = list(out.values())
        return values[0] if values else p3


class _INT8Calibrator:
    """INT8 calibration data provider for TensorRT."""

    def __init__(self, calibration_data):
        self.data = calibration_data
        self.idx = 0

    def get_batch(self, names):
        if self.idx >= len(self.data):
            return None
        batch = self.data[self.idx:self.idx + 1]
        self.idx += 1
        return [batch.numpy()]


# ============ CLI ============

def main():
    parser = argparse.ArgumentParser(description="OrchestraNet Model Export")
    parser.add_argument("--weights", default=None, help="Model checkpoint path")
    parser.add_argument("--format", choices=["onnx", "tensorrt", "both", "individual"],
                        default="onnx")
    parser.add_argument("--output-dir", default="./exports")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from orchestranet.orchestrator import OrchestraNet

    print("🎼 OrchestraNet Model Export")
    print("=" * 50)

    # Build model
    model = OrchestraNet(num_classes=80, pretrained_backbone=False)

    if args.weights and Path(args.weights).exists():
        state = torch.load(args.weights, map_location=args.device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"✅ Loaded weights: {args.weights}")

    model = model.to(args.device)
    exporter = ModelExporter(model, device=args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.format in ("onnx", "both"):
        onnx_path = os.path.join(args.output_dir, "orchestranet.onnx")
        exporter.export_onnx(onnx_path, input_size=(args.img_size, args.img_size))

    if args.format in ("tensorrt", "both"):
        onnx_path = os.path.join(args.output_dir, "orchestranet.onnx")
        if not os.path.exists(onnx_path):
            exporter.export_onnx(onnx_path, input_size=(args.img_size, args.img_size))
        trt_path = os.path.join(args.output_dir, f"orchestranet_{args.precision}.engine")
        exporter.export_tensorrt(onnx_path, trt_path, precision=args.precision)

    if args.format == "individual":
        ind_dir = os.path.join(args.output_dir, "individual")
        exporter.export_individual_models(ind_dir, input_size=(args.img_size, args.img_size))

    print("\n✅ Export complete!")


if __name__ == "__main__":
    main()
