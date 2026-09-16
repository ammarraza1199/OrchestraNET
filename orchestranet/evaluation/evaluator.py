"""
Unified SystemEvaluator and evaluate_system() Interface — Phase 3.

Provides:
  1. evaluate_system() — top-level function that wraps the training/evaluate.py
     loop and adds occlusion-aware metrics + depth metrics on top.
  2. SystemEvaluator — class-based interface for repeated evaluation cycles
     (Phase 4 checkpoint integration).

Design:
  - Reuses training/evaluate.py's evaluate() loop internals (no duplication).
  - Delegates detection accumulation to MetricRegistry.
  - Delegates occlusion-specific AP to OcclusionAwareMetrics.
  - Delegates depth metrics to DepthMetrics.
  - All UNAVAILABLE decisions propagate from the sub-metrics.

Usage:
    from orchestranet.evaluation.evaluator import evaluate_system, SystemEvaluator

    # Functional interface (one-shot)
    results = evaluate_system(model, loader, device="cuda")

    # Class interface (Phase 4 — repeated calls per checkpoint)
    evaluator = SystemEvaluator(model, device="cuda")
    evaluator.reset()
    results = evaluator.run(loader)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from orchestranet.evaluation.metric_registry import MetricRegistry, _Unavailable
from orchestranet.evaluation.occlusion_metrics import OcclusionAwareMetrics
from orchestranet.evaluation.depth_metrics import DepthMetrics
from orchestranet.evaluation.efficiency_profiler import EfficiencyProfiler


# ---------------------------------------------------------------------------
# Top-level functional interface
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_system(
    model: "torch.nn.Module",
    loader: DataLoader,
    device: str = "cuda",
    conf_thresh: float = 0.25,
    ablate: str | None = None,
    num_images: int | None = None,
    compute_depth: bool = False,
    compute_occlusion: bool = False,
    num_classes: int = 80,
) -> dict[str, Any]:
    """
    Full OrchestraNet evaluation pipeline.

    Args:
        model:            OrchestraNet (or any model returning {"detections": ...}).
        loader:           DataLoader yielding (images, targets) batches.
        device:           "cuda" or "cpu".
        conf_thresh:      Detection confidence threshold (default 0.25).
        ablate:           Model ID to disable for ablation study (e.g., "m2").
        num_images:       Cap total images evaluated (None = full dataset).
        compute_depth:    If True, attempt to compute KITTI depth metrics using
                          targets["depth_gt"]. Falls back to UNAVAILABLE if absent.
        compute_occlusion: If True, attempt to compute AP_occ using
                          targets["visibility_ratio"]. Falls back to UNAVAILABLE.
        num_classes:      Number of detection classes.

    Returns:
        Unified results dict containing:
          - Standard detection: mAP@50, mAP@50:95, AP_small/medium/large
          - Occlusion (if available): AP_occ, AP_partial, AP_visible, detail
          - Depth (if available): AbsRel, SqRel, RMSE, RMSElog, d1/d2/d3
          - Speed: mean_ms, median_ms, p95_ms, fps
          - Routing: routing_distribution
          - Meta: num_images, ablation
    """
    evaluator = SystemEvaluator(model, device=device, num_classes=num_classes)
    return evaluator.run(
        loader,
        conf_thresh=conf_thresh,
        ablate=ablate,
        num_images=num_images,
        compute_depth=compute_depth,
        compute_occlusion=compute_occlusion,
    )


# ---------------------------------------------------------------------------
# Class-based interface for Phase 4 reuse
# ---------------------------------------------------------------------------

class SystemEvaluator:
    """
    Stateful evaluation orchestrator for OrchestraNet.

    Maintains metric accumulators so that Phase 4 can:
      1. Call reset() at the start of each validation epoch.
      2. Feed batches incrementally via update().
      3. Call finalize() to get the full results dict.

    Or use the simpler run() for one-shot evaluation.

    Args:
        model:       OrchestraNet or compatible module.
        device:      "cuda" or "cpu".
        num_classes: Number of detection classes (80 for COCO, 8 for KINS).

    Phase 4 note:
        The report() method on EfficiencyProfiler and finalize() here
        are intentionally kept separate. Call profile_efficiency() once
        (at first evaluation) and reuse the result across checkpoints;
        re-run finalize() after each validation epoch.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        device: str = "cuda",
        num_classes: int = 80,
    ):
        self.model       = model
        self.device      = device
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Clear all metric accumulators. Call before each evaluation cycle."""
        self._registry  = MetricRegistry(num_classes=self.num_classes)
        self._occ_met   = OcclusionAwareMetrics(num_classes=self.num_classes)
        self._depth_met = DepthMetrics()
        self._latencies: list[float] = []
        self._route_dist: dict[str, int] = {"simple": 0, "medium": 0, "complex": 0}
        self._n_images: int = 0

    @torch.no_grad()
    def run(
        self,
        loader: DataLoader,
        conf_thresh: float = 0.25,
        ablate: str | None = None,
        num_images: int | None = None,
        compute_depth: bool = False,
        compute_occlusion: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate the model over the full data loader and return results.

        This is the primary entry point for one-shot evaluation.
        """
        self.reset()
        self.model.eval()

        # Optionally disable an ablated model
        if ablate and hasattr(self.model, "models") and ablate in self.model.models:
            self.model.models[ablate].is_active = False
            print(f"⚠️  ABLATION: {ablate} disabled")

        use_cuda = (self.device == "cuda" and torch.cuda.is_available())
        limit    = num_images or len(loader)

        for images, targets in loader:
            if self._n_images >= limit:
                break

            images = images.to(self.device)

            # ---- Timed inference ----
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = self.model(images)
            if use_cuda:
                torch.cuda.synchronize()
            self._latencies.append((time.perf_counter() - t0) * 1000.0)

            # ---- Routing distribution ----
            if "routing" in outputs and "routing_level" in outputs["routing"]:
                lvl = outputs["routing"]["routing_level"]
                if lvl in self._route_dist:
                    self._route_dist[lvl] += 1

            # ---- Detection accumulation ----
            detections = outputs.get("detections", [])
            self._accumulate_detections(
                detections, targets, conf_thresh,
                compute_occlusion=compute_occlusion,
            )

            # ---- Depth accumulation ----
            if compute_depth and "depth_gt" in targets:
                depth_preds = outputs.get("model_outputs", {}).get("m4", {})
                if "depth_map" in depth_preds:
                    self._depth_met.update(
                        depth_preds["depth_map"].cpu(),
                        targets["depth_gt"].cpu(),
                    )

            self._n_images += images.shape[0]

        return self.finalize(
            compute_depth=compute_depth,
            compute_occlusion=compute_occlusion,
            ablate=ablate,
        )

    def finalize(
        self,
        compute_depth: bool = False,
        compute_occlusion: bool = False,
        ablate: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute and return the full results dict from accumulated state.

        Phase 4: call this after all validation batches have been fed via
        update() to get the epoch-level results without re-running the loop.
        """
        # Standard detection metrics
        results: dict[str, Any] = self._registry.compute_all()

        # Occlusion-aware metrics
        if compute_occlusion:
            occ_results = self._occ_met.compute()
            results["occlusion"] = occ_results
        else:
            results["occlusion"] = _Unavailable(
                "compute_occlusion=False. Re-run with compute_occlusion=True "
                "and a dataset that provides per-box visibility_ratio targets."
            ).to_dict()

        # Depth metrics
        if compute_depth:
            results["depth"] = self._depth_met.compute()
        else:
            results["depth"] = _Unavailable(
                "compute_depth=False. Re-run with compute_depth=True and a "
                "dataset that provides depth_gt targets (KITTI or NYU Depth v2)."
            ).to_dict()

        # Speed
        if self._latencies:
            arr = np.array(self._latencies)
            results["speed"] = {
                "mean_ms":   float(arr.mean()),
                "median_ms": float(np.median(arr)),
                "p95_ms":    float(np.percentile(arr, 95)),
                "fps":       float(1000.0 / arr.mean()),
            }

        results["routing_distribution"] = self._route_dist
        results["num_images"]   = self._n_images
        results["ablation"]     = ablate

        return results

    # ------------------------------------------------------------------
    # Profile efficiency (separate from metric evaluation)
    # ------------------------------------------------------------------

    def profile_efficiency(
        self,
        dummy_input: torch.Tensor,
        num_runs: int = 200,
        warmup: int = 10,
    ) -> dict[str, Any]:
        """
        Run EfficiencyProfiler and return JSON-serializable efficiency report.

        Profiles: FP32 latency, FP16 latency, peak VRAM, FLOPs (optional),
        and per-component parameter count.

        Phase 4: call once per training phase and persist the result.
        """
        profiler = EfficiencyProfiler(self.model, device=self.device)
        profiler.count_parameters()
        profiler.profile_latency(dummy_input, num_runs=num_runs, warmup=warmup, use_amp=False)
        if self.device == "cuda" and torch.cuda.is_available():
            profiler.profile_latency(dummy_input, num_runs=num_runs, warmup=warmup, use_amp=True)
            profiler.profile_memory(dummy_input)
        profiler.count_flops(dummy_input)
        return profiler.report()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accumulate_detections(
        self,
        detections: list[dict] | dict,
        targets: dict,
        conf_thresh: float,
        compute_occlusion: bool = False,
    ) -> None:
        """Extract per-image preds + GTs and feed both metric accumulators."""
        if isinstance(detections, dict):
            # Batch-level dict from training mode
            B = targets["boxes"].shape[0]
            detections = [detections] * B  # Fallback — treat same for all images

        if not isinstance(detections, list):
            return

        for b, det in enumerate(detections):
            img_id = self._n_images + b

            # GT
            n = int(targets["num_objects"][b].item()) if "num_objects" in targets else \
                targets["boxes"].shape[1]
            gt_boxes_t = targets["boxes"][b][:n]
            gt_labels_t = targets["labels"][b][:n]
            gt_boxes  = gt_boxes_t.detach().cpu().numpy() if isinstance(gt_boxes_t, torch.Tensor) else np.asarray(gt_boxes_t)
            gt_labels = gt_labels_t.detach().cpu().numpy() if isinstance(gt_labels_t, torch.Tensor) else np.asarray(gt_labels_t)
            gt_vis    = None
            if compute_occlusion and "visibility_ratio" in targets:
                gt_vis_t = targets["visibility_ratio"][b][:n]
                gt_vis = gt_vis_t.detach().cpu().numpy() if isinstance(gt_vis_t, torch.Tensor) else np.asarray(gt_vis_t)

            # Predictions
            if det is None or det.get("boxes") is None or len(det["boxes"]) == 0:
                pred_boxes, pred_scores, pred_labels = [], [], []
            else:
                mask        = det["scores"] > conf_thresh
                pred_boxes  = det["boxes"][mask].detach().cpu().numpy()
                pred_scores = det["scores"][mask].detach().cpu().numpy()
                pred_labels = det["labels"][mask].detach().cpu().numpy()

            # Accumulate into registry
            self._registry.update_detection_with_visibility(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels,
                gt_visibility=gt_vis,
                image_id=img_id,
            )

            # Accumulate into occlusion-aware evaluator
            if compute_occlusion:
                self._occ_met.update(
                    pred_boxes, pred_scores, pred_labels,
                    gt_boxes, gt_labels,
                    gt_visibility=gt_vis,
                    image_id=img_id,
                )
