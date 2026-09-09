"""
Speed Benchmark for OrchestraNet.

Measures inference latency and FPS across different:
- Routing profiles (simple/medium/complex)
- Hardware (GPU/CPU)
- Batch sizes
- Precision modes (FP32/FP16)

Usage:
  python inference/benchmark.py --device cuda --batch-size 1
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestranet.orchestrator import OrchestraNet


def benchmark_model(model, device, batch_size=1, img_size=640,
                    num_runs=200, warmup=50, fp16=False):
    """Benchmark model inference speed."""
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    if fp16 and device == "cuda":
        model = model.half()
        dummy = dummy.half()

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"  Benchmarking ({num_runs} iterations)...")
    times = []
    routing_counts = {"simple": 0, "medium": 0, "complex": 0}

    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
            routing_counts[outputs["routing"]["routing_level"]] += 1

    times = np.array(times)
    return {
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "min_ms": times.min(),
        "max_ms": times.max(),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "fps": 1000.0 / times.mean(),
        "routing_distribution": routing_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-runs", type=int, default=200)
    args = parser.parse_args()

    print("🎼 OrchestraNet Speed Benchmark")
    print("=" * 60)

    model = OrchestraNet(
        num_classes=80,
        pretrained_backbone=False,
    ).to(args.device)

    params = model.count_all_parameters()
    print(f"\n📊 Total Parameters: {params['TOTAL']['total']:,}")
    print(f"   Device: {args.device}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Image Size: {args.img_size}\n")

    # FP32 benchmark
    print("▶ FP32 Benchmark:")
    fp32_results = benchmark_model(
        model, args.device, args.batch_size, args.img_size, args.num_runs
    )
    print(f"  Mean: {fp32_results['mean_ms']:.2f}ms")
    print(f"  Median: {fp32_results['median_ms']:.2f}ms")
    print(f"  P95: {fp32_results['p95_ms']:.2f}ms")
    print(f"  FPS: {fp32_results['fps']:.1f}")
    print(f"  Routing: {fp32_results['routing_distribution']}")

    # FP16 benchmark (GPU only)
    if args.device == "cuda":
        print("\n▶ FP16 Benchmark:")
        model_fp16 = OrchestraNet(num_classes=80, pretrained_backbone=False).to(args.device)
        fp16_results = benchmark_model(
            model_fp16, args.device, args.batch_size, args.img_size,
            args.num_runs, fp16=True
        )
        print(f"  Mean: {fp16_results['mean_ms']:.2f}ms")
        print(f"  Median: {fp16_results['median_ms']:.2f}ms")
        print(f"  P95: {fp16_results['p95_ms']:.2f}ms")
        print(f"  FPS: {fp16_results['fps']:.1f}")
        print(f"  Speedup: {fp32_results['mean_ms'] / fp16_results['mean_ms']:.2f}x")

    # Per-model latency breakdown
    print("\n▶ Per-Model Latency Breakdown:")
    dummy_features = [
        torch.randn(1, 128, 80, 80, device=args.device),
        torch.randn(1, 128, 40, 40, device=args.device),
        torch.randn(1, 128, 20, 20, device=args.device),
    ]
    for model_id, micro_model in model.models.items():
        latency = micro_model.estimate_latency(dummy_features, num_runs=100)
        print(f"  {model_id}: {latency['mean_ms']:.2f}ms ± {latency['std_ms']:.2f}ms")

    print("\n✅ Benchmark complete.")


if __name__ == "__main__":
    main()
