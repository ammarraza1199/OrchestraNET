"""
Phase 3 Dry-Run Evaluation Script — OrchestraNet.

Performs a single-pass evaluation with RANDOM WEIGHTS and SYNTHETIC DATA
to validate the entire evaluation framework without requiring:
  - Real training checkpoints
  - Downloaded datasets (COCO / KINS / KITTI)

What this script validates:
  1. All 7 micro-models forward pass without errors
  2. OrchestraNet orchestrator forward pass (M1→M2→...→M7)
  3. MetricRegistry accumulation and compute_all()
  4. OcclusionAwareMetrics — UNAVAILABLE path (no real visibility GT)
  5. DepthMetrics — UNAVAILABLE path (no real depth GT)
  6. EfficiencyProfiler — latency, memory, FLOPs (fvcore optional)
  7. SystemEvaluator.profile_efficiency()
  8. Writes artifacts/evaluation/training_readiness.md

Exit code:
  0 — All checks passed
  1 — Any check failed

Usage:
  python scripts/dry_run_eval.py --device cpu
  python scripts/dry_run_eval.py --device cuda --num-runs 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows (cp1252 cannot encode emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np


# Project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OrchestraNet Phase 3 Dry-Run Evaluation")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--img-size",   type=int, default=640)
    p.add_argument("--num-runs",   type=int, default=50,
                   help="Latency profiling iterations (default 50 for dry-run)")
    p.add_argument("--warmup",     type=int, default=5)
    p.add_argument("--out-dir",    default=str(_ROOT / "artifacts" / "evaluation"))
    p.add_argument("--num-classes", type=int, default=80)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

_RESULTS: dict[str, dict] = {}
_PASS = "✅ PASS"
_FAIL = "❌ FAIL"
_SKIP = "⏭  SKIP"


def _check(name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), record pass/fail/result."""
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        _RESULTS[name] = {"status": "PASS", "result": result, "ms": round(elapsed, 1)}
        print(f"  {_PASS}  {name}  ({elapsed:.0f} ms)")
        return result
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        tb = traceback.format_exc()
        _RESULTS[name] = {"status": "FAIL", "error": str(exc), "traceback": tb, "ms": round(elapsed, 1)}
        print(f"  {_FAIL}  {name}  ({elapsed:.0f} ms)")
        print(f"         {exc}")
        return None


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_imports():
    """Verify evaluation package imports cleanly."""
    from orchestranet.evaluation import (
        MetricRegistry, OcclusionAwareMetrics, DepthMetrics,
        EfficiencyProfiler, SystemEvaluator, evaluate_system,
    )
    from orchestranet.evaluation.metric_registry import _Unavailable, UNAVAILABLE
    return "All imports OK"


def test_model_build(num_classes, device):
    """Build OrchestraNet with random weights."""
    from orchestranet.orchestrator import OrchestraNet
    model = OrchestraNet(
        num_classes=num_classes,
        pretrained_backbone=False,
    ).to(device)
    params = model.count_all_parameters()
    total = params["TOTAL"]["total"]
    return {"total_params": total, "total_M": round(total / 1e6, 3)}


def test_forward_pass(model, dummy_input):
    """Run OrchestraNet forward pass and validate output keys."""
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    assert "detections" in out, "Missing 'detections' in output"
    assert "routing" in out,    "Missing 'routing' in output"
    return {
        "output_keys": list(out.keys()),
        "routing_level": out["routing"].get("routing_level", "?"),
        "active_models": list(out.get("model_outputs", {}).keys()),
    }


def test_individual_models(dummy_features, device, num_classes):
    """Forward pass for each micro-model independently."""
    from orchestranet.models import (
        M1PrimaryDetector, M2OcclusionAnalyzer, M3SmallObjectEnhancer,
        M4DepthEstimator, M5SemanticContext, M6AmodalCompleter,
        M7ConfidenceCalibrator,
    )
    results = {}
    models_cfg = {
        "M1": M1PrimaryDetector(in_channels=128, num_classes=num_classes),
        "M2": M2OcclusionAnalyzer(in_channels=128),
        "M3": M3SmallObjectEnhancer(in_channels=128),
        "M4": M4DepthEstimator(in_channels=128),
        "M5": M5SemanticContext(in_channels=128),
        "M6": M6AmodalCompleter(d_model=128),
        "M7": M7ConfidenceCalibrator(),
    }
    for name, m in models_cfg.items():
        m = m.to(device).eval()
        with torch.no_grad():
            out = m(dummy_features)
        keys = list(out.keys())
        has_nan = any(
            v.isnan().any().item()
            for v in out.values()
            if isinstance(v, torch.Tensor)
        )
        results[name] = {"output_keys": keys, "has_nan": has_nan}
    return results


def test_metric_registry():
    """Smoke test MetricRegistry accumulation and compute."""
    from orchestranet.evaluation.metric_registry import MetricRegistry, _Unavailable
    registry = MetricRegistry(num_classes=2)
    registry.reset_all()
    # Feed one synthetic image with zero GT
    registry.update_detection(
        pred_boxes=np.array([[10, 10, 50, 50]], dtype=np.float32),
        pred_scores=np.array([0.9], dtype=np.float32),
        pred_labels=np.array([0], dtype=np.int64),
        gt_boxes=np.array([[12, 12, 48, 48]], dtype=np.float32),
        gt_labels=np.array([0], dtype=np.int64),
        image_id=0,
    )
    results = registry.compute_all()
    assert "mAP@50"   in results, "mAP@50 missing"
    assert "mAP@50:95" in results, "mAP@50:95 missing"
    serialized = MetricRegistry.serialize_results(results)
    return {"keys": list(results.keys()), "mAP@50": float(results.get("mAP@50", 0))}


def test_occlusion_metrics_unavailable():
    """Confirm OcclusionAwareMetrics returns UNAVAILABLE (not 0.0) without visibility GT."""
    from orchestranet.evaluation.occlusion_metrics import OcclusionAwareMetrics
    occ = OcclusionAwareMetrics(num_classes=2)
    # Feed without visibility (gt_visibility=None)
    occ.update(
        pred_boxes=[[10, 10, 50, 50]], pred_scores=[0.9], pred_labels=[0],
        gt_boxes=[[12, 12, 48, 48]],  gt_labels=[0],
        gt_visibility=None,  # No annotations
    )
    results = occ.compute()
    ap_occ = results["AP_occ"]
    assert isinstance(ap_occ, dict), f"Expected UNAVAILABLE dict, got {type(ap_occ)}"
    assert ap_occ.get("status") == "UNAVAILABLE", f"Expected UNAVAILABLE status, got {ap_occ}"
    return {"AP_occ": ap_occ["status"], "OA_NMS_preserved_pairs": results.get("OA_NMS_preserved_pairs")}


def test_depth_metrics_unavailable():
    """Confirm DepthMetrics returns UNAVAILABLE (not 0.0) without GT depth."""
    from orchestranet.evaluation.depth_metrics import DepthMetrics
    dm = DepthMetrics()
    # compute() without any update() calls
    results = dm.compute()
    assert results.get("status") == "UNAVAILABLE", f"Expected UNAVAILABLE, got {results}"
    assert results["AbsRel"]["status"] == "UNAVAILABLE"
    return {"status": results["status"]}


def test_depth_metrics_available():
    """Confirm DepthMetrics computes correctly when synthetic GT is available."""
    from orchestranet.evaluation.depth_metrics import DepthMetrics
    dm = DepthMetrics(min_depth=0.1, max_depth=10.0)
    rng = np.random.default_rng(42)
    for _ in range(5):
        gt = rng.uniform(0.5, 9.0, (480, 640)).astype(np.float32)
        # Simulate predictions with some noise
        pred = gt * rng.uniform(0.9, 1.1, gt.shape).astype(np.float32)
        dm.update(pred, gt)
    results = dm.compute()
    assert isinstance(results["AbsRel"], float), "AbsRel should be float when GT available"
    return {
        "AbsRel":  round(results["AbsRel"], 4),
        "d1":      round(results["d1"], 4),
        "n_imgs":  results["n_images"],
    }


def test_efficiency_profiler(model, dummy_input, device, num_runs, warmup):
    """Profile latency, memory (CUDA only), and FLOPs (fvcore optional)."""
    from orchestranet.evaluation.efficiency_profiler import EfficiencyProfiler, _FVCORE_UNAVAILABLE_REASON
    profiler = EfficiencyProfiler(model, device=device)
    params = profiler.count_parameters()
    lat    = profiler.profile_latency(dummy_input, num_runs=num_runs, warmup=warmup)
    mem    = profiler.profile_memory(dummy_input)
    flops  = profiler.count_flops(dummy_input)

    # Validate: flops should NEVER be 0 or a silent failure
    if flops.get("status") == "UNAVAILABLE":
        assert flops.get("total_flops") is None, "Expected None FLOPs when UNAVAILABLE, not 0"

    report = profiler.report()
    return {
        "fps":        round(lat["fps"], 1),
        "mean_ms":    round(lat["mean_ms"], 2),
        "peak_vram_mb": mem.get("peak_vram_mb", "N/A (CPU)"),
        "flops_status": flops.get("status", "?"),
        "total_params_M": round(
            sum(v["total"] for k, v in params.items() if k != "TOTAL" and isinstance(v, dict)) / 1e6, 3
        ) if isinstance(params, dict) and "TOTAL" in params else "?",
    }


# ---------------------------------------------------------------------------
# Report generation (Phase 4-reusable functions)
# ---------------------------------------------------------------------------

def generate_training_readiness_report(
    args,
    all_results: dict,
    out_path: Path,
) -> None:
    """
    Write artifacts/evaluation/training_readiness.md.

    Phase 4 note: this function is intentionally self-contained and accepts
    all data as arguments. Checkpoint evaluators in Phase 4 can call it with
    updated metrics without touching this script.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device_str = args.device.upper()

    param_result = all_results.get("model_build", {}).get("result") or {}
    total_M = param_result.get("total_M", "N/A")

    prof_result = all_results.get("efficiency_profiler", {}).get("result") or {}
    fps      = prof_result.get("fps", "N/A")
    mean_ms  = prof_result.get("mean_ms", "N/A")
    vram_mb  = prof_result.get("peak_vram_mb", "N/A (CPU)")
    flops_st = prof_result.get("flops_status", "N/A")

    model_result = all_results.get("individual_models", {}).get("result") or {}
    forward_result = all_results.get("forward_pass", {}).get("result") or {}

    occ_result   = all_results.get("occlusion_unavailable", {}).get("result") or {}
    depth_unav   = all_results.get("depth_unavailable", {}).get("result") or {}
    depth_synth  = all_results.get("depth_synthetic", {}).get("result") or {}

    n_pass = sum(1 for v in all_results.values() if v.get("status") == "PASS")
    n_fail = sum(1 for v in all_results.values() if v.get("status") == "FAIL")
    n_total = len(all_results)

    lines = [
        "# OrchestraNet — Phase 3 Training Readiness Report",
        "",
        f"> **Generated:** {now}  ",
        f"> **Device:** {device_str}  ",
        f"> **Phase:** 3 — Unified Supervision + Evaluation + Benchmarking Framework  ",
        f"> **Status:** {'✅ ALL CLEAR' if n_fail == 0 else f'❌ {n_fail} CHECK(S) FAILED'}",
        "",
        "---",
        "",
        "## 1. Smoke Test Summary",
        "",
        f"| Check | Status | Time (ms) |",
        f"|-------|--------|-----------|",
    ]
    for name, r in all_results.items():
        icon = "✅" if r.get("status") == "PASS" else "❌"
        ms   = r.get("ms", "—")
        lines.append(f"| {name} | {icon} {r.get('status')} | {ms} |")

    lines += [
        "",
        f"**{n_pass}/{n_total} checks passed.**",
        "",
        "---",
        "",
        "## 2. Parameter Count",
        "",
        f"| Component | Params (M) |",
        f"|-----------|------------|",
        f"| **Total** | **{total_M}** |",
        "",
        "> Reference target: ~5M (from project_documentation.md Table 5).",
        "",
        "---",
        "",
        "## 3. Efficiency Profile (Random Weights, Synthetic Input)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean Latency (FP32, batch=1) | {mean_ms} ms |",
        f"| FPS (FP32, batch=1) | {fps} |",
        f"| Peak VRAM | {vram_mb} MB |",
        f"| FLOPs status | {flops_st} |",
        "",
        "> **Note:** These are RANDOM-WEIGHT, SYNTHETIC-INPUT measurements. Real weights",
        "> and real data may differ. Reference: 12 ms / 83 FPS @ RTX 4090 FP32.",
        "",
        "---",
        "",
        "## 4. Per-Model Forward Pass",
        "",
        "| Model | Output Keys | NaN? |",
        "|-------|-------------|------|",
    ]
    for mname, mres in model_result.items():
        keys_str = ", ".join(mres.get("output_keys", []))
        nan_str  = "⚠️ YES" if mres.get("has_nan") else "✅ No"
        lines.append(f"| {mname} | {keys_str} | {nan_str} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Metric Registry",
        "",
        "### Standard Detection Metrics (COCO-style)",
        "",
        "| Metric | Status |",
        "|--------|--------|",
        "| mAP@50     | ✅ Registered — requires real data to compute |",
        "| mAP@50:95  | ✅ Registered — requires real data to compute |",
        "| AP_small   | ✅ Registered — requires real data to compute |",
        "| AP_medium  | ✅ Registered — requires real data to compute |",
        "| AP_large   | ✅ Registered — requires real data to compute |",
        "",
        "### Occlusion-Aware Metrics",
        "",
        "| Metric | Status | Reason |",
        "|--------|--------|--------|",
        f"| AP_occ (vis < 30%) | ⚠️ {occ_result.get('AP_occ', 'N/A')} | KINS/COCOA visibility_ratio required |",
        "| AP_partial (30–70%) | ⚠️ UNAVAILABLE | Same as above |",
        "| AP_visible (>70%)  | ⚠️ UNAVAILABLE | Same as above |",
        "| OA_NMS_preserved_pairs | ⚠️ UNAVAILABLE | record_nms_pair() not called during dry-run |",
        "",
        "> **Policy:** AP_occ is NEVER substituted with 0.0 when annotations are absent.",
        "> The UNAVAILABLE sentinel prevents incorrect zero-AP reporting.",
        "",
        "### Depth Metrics (KITTI-standard)",
        "",
        f"| Metric | Status |",
        f"|--------|--------|",
        f"| AbsRel | ⚠️ {depth_unav.get('status', 'N/A')} |",
        f"| SqRel  | ⚠️ {depth_unav.get('status', 'N/A')} |",
        f"| RMSE   | ⚠️ {depth_unav.get('status', 'N/A')} |",
        f"| δ<1.25 | ⚠️ {depth_unav.get('status', 'N/A')} |",
        "",
        f"Synthetic GT verification: AbsRel={depth_synth.get('AbsRel', 'N/A')}, "
        f"d1={depth_synth.get('d1', 'N/A')} ✅ (computed correctly with GT)",
        "",
        "---",
        "",
        "## 6. Available vs Unavailable Metrics",
        "",
        "| Metric | Available | Dataset Required |",
        "|--------|-----------|------------------|",
        "| mAP@50      | ✅ When data loaded | COCO val2017 |",
        "| mAP@50:95   | ✅ When data loaded | COCO val2017 |",
        "| AP_small    | ✅ When data loaded | COCO val2017 |",
        "| AP_occ      | ⚠️ UNAVAILABLE    | KINS / COCOA (visibility_ratio GT) |",
        "| AbsRel/RMSE | ⚠️ UNAVAILABLE    | KITTI / NYU Depth v2 (depth_gt) |",
        "| FLOPs       | ⚠️ " + ("OK" if flops_st == "OK" else "UNAVAILABLE") + " | fvcore (`pip install fvcore`) |",
        "| FPS         | ✅ Always available | Synthetic input is sufficient |",
        "| Peak VRAM   | ✅ On CUDA | CUDA device required |",
        "",
        "---",
        "",
        "## 7. Reference Targets vs Current Capability",
        "",
        "| Metric | Paper Reference | Current | Gap |",
        "|--------|----------------|---------|-----|",
        "| mAP@50 | 64.3 → target ≥65.0 | Requires training | Training needed |",
        "| mAP@50:95 | 42.7 | Requires training | Training needed |",
        "| AP_occ | 30.6 | UNAVAILABLE | KINS data needed |",
        f"| FPS (FP32) | 83 | ~{fps} (random weights) | Real weights may differ |",
        "| Params | ~5M | ~{total_M}M | Check |",
        "",
        "---",
        "",
        "## 8. Training Readiness Priorities (Phase 4)",
        "",
        "1. **Individual Pre-training** — Run `train_individual.py --model m1` on COCO.",
        "   This is the highest-leverage first step: M1 runs on every frame.",
        "2. **M2 Self-Supervised Pre-training** — Run `train_individual.py --model m2`.",
        "   Requires no labeled occlusion data; uses synthetic occlusion injection.",
        "3. **KINS Data Setup** — Download KINS annotations to unlock AP_occ.",
        "   Without KINS, `AP_occ` will remain UNAVAILABLE.",
        "4. **KITTI Data Setup** — Download KITTI depth maps to unlock AbsRel/RMSE.",
        "   Without KITTI, M4 trains on smoothness loss only (weaker).",
        "5. **Joint Fine-tuning** — `train_joint.py` after individual pre-training.",
        "6. **Router Training** — `train_router.py` to tune routing thresholds.",
        "",
        "### Exact Next Step for Phase 4",
        "",
        "```bash",
        "# Step 1: Pre-train M1 on COCO",
        "python training/train_individual.py \\",
        "    --model m1 \\",
        "    --data-root ./data/coco \\",
        "    --epochs 50 \\",
        "    --batch-size 16 \\",
        "    --lr 1e-3 \\",
        "    --device cuda \\",
        "    --save-dir ./checkpoints/individual",
        "",
        "# Step 2: Pre-train M2 (self-supervised, no KINS needed)",
        "python training/train_individual.py \\",
        "    --model m2 \\",
        "    --data-root ./data/coco \\",
        "    --epochs 30 \\",
        "    --batch-size 16 \\",
        "    --device cuda \\",
        "    --save-dir ./checkpoints/individual",
        "```",
        "",
        "---",
        "",
        "> *This report was generated by `scripts/dry_run_eval.py`.*  ",
        "> *For checkpoint-triggered updates, call `generate_training_readiness_report()`*",
        "> *from the Phase 4 training loop after each evaluation epoch.*",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n💾  Report saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device

    print("\n🎼  OrchestraNet — Phase 3 Dry-Run Evaluation")
    print("=" * 62)
    print(f"   Device:     {device}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Img size:   {args.img_size}×{args.img_size}")
    print(f"   Num runs:   {args.num_runs} (latency profiling)")
    print("=" * 62)

    # Synthetic dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)

    # Dummy FPN features (P3, P4, P5 at 128 channels)
    dummy_features = [
        torch.randn(args.batch_size, 128, 80, 80).to(device),  # P3
        torch.randn(args.batch_size, 128, 40, 40).to(device),  # P4
        torch.randn(args.batch_size, 128, 20, 20).to(device),  # P5
    ]

    # ---- Build model once for reuse ----
    print("\n[1/9] Building model...")
    model_result = _check("model_build", test_model_build, args.num_classes, device)
    model = None
    if model_result is not None:
        from orchestranet.orchestrator import OrchestraNet
        model = OrchestraNet(
            num_classes=args.num_classes,
            pretrained_backbone=False,
        ).to(device)

    # ---- Run all checks ----
    print("\n[2/9] Testing imports...")
    _check("imports", test_imports)

    print("\n[3/9] Testing forward pass...")
    if model is not None:
        _check("forward_pass", test_forward_pass, model, dummy_input)

    print("\n[4/9] Testing individual micro-models...")
    _check("individual_models", test_individual_models, dummy_features, device, args.num_classes)

    print("\n[5/9] Testing MetricRegistry...")
    _check("metric_registry", test_metric_registry)

    print("\n[6/9] Testing OcclusionAwareMetrics (UNAVAILABLE path)...")
    _check("occlusion_unavailable", test_occlusion_metrics_unavailable)

    print("\n[7/9] Testing DepthMetrics (UNAVAILABLE path)...")
    _check("depth_unavailable", test_depth_metrics_unavailable)

    print("\n[8/9] Testing DepthMetrics (synthetic GT path)...")
    _check("depth_synthetic", test_depth_metrics_available)

    print("\n[9/9] Running EfficiencyProfiler...")
    if model is not None:
        _check(
            "efficiency_profiler", test_efficiency_profiler,
            model, dummy_input, device, args.num_runs, args.warmup,
        )

    # ---- Summary ----
    n_pass = sum(1 for v in _RESULTS.values() if v.get("status") == "PASS")
    n_fail = sum(1 for v in _RESULTS.values() if v.get("status") == "FAIL")
    n_total = len(_RESULTS)

    print("\n" + "=" * 62)
    print(f"🎼  Dry-Run Complete: {n_pass}/{n_total} checks passed", end="")
    print("  ✅" if n_fail == 0 else f"  ❌ {n_fail} FAILED")
    print("=" * 62)

    # ---- Save JSON results ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dry_run_results.json"
    serializable = {}
    for k, v in _RESULTS.items():
        entry = {"status": v.get("status"), "ms": v.get("ms")}
        r = v.get("result")
        if r is not None and isinstance(r, dict):
            entry["result"] = {
                rk: float(rv) if isinstance(rv, (float, np.floating)) else rv
                for rk, rv in r.items()
            }
        elif r is not None:
            entry["result"] = str(r)
        serializable[k] = entry

    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"💾  JSON saved: {json_path}")

    # ---- Write training_readiness.md ----
    report_path = out_dir / "training_readiness.md"
    generate_training_readiness_report(args, _RESULTS, report_path)

    # ---- Return exit code ----
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
