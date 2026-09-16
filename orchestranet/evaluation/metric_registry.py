"""
Centralized Metric Registry for OrchestraNet — Phase 3.

Provides a named registry of evaluation metric functions that can be
registered, accumulated, computed, and reset in a unified interface.

Design principles:
  - Reuses DetectionMetrics from orchestranet/utils/metrics.py (no duplication)
  - Metric functions are decoupled: register any callable(predictions, targets) -> float
  - Supports batch-level accumulation before final compute()
  - Phase 4 can extend this by adding new metric registrations without
    changing the registry interface itself.

Usage:
    registry = MetricRegistry(num_classes=80)
    registry.reset_all()
    for preds, targets in loader:
        registry.accumulate(preds, targets)
    results = registry.compute_all()
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any, Callable

import numpy as np

# Re-use the existing DetectionMetrics — no code duplication
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from orchestranet.utils.metrics import DetectionMetrics


# ---------------------------------------------------------------------------
# Sentinel for "metric data unavailable" — never substitutes 0.0 for missing GT
# ---------------------------------------------------------------------------

class _Unavailable:
    """
    Sentinel type returned when a metric cannot be computed due to missing
    ground-truth annotations.  Serializes cleanly to JSON as a structured dict.

    Attributes:
        reason: Human-readable explanation of why the metric is unavailable.
    """

    def __init__(self, reason: str = "Annotations not available for this evaluation split"):
        self.reason = reason

    def __repr__(self) -> str:
        return f"UNAVAILABLE({self.reason!r})"

    def to_dict(self) -> dict[str, str]:
        return {"status": "UNAVAILABLE", "reason": self.reason}

    def __float__(self):
        # Prevent silent coercion to 0.0 — always raise explicitly
        raise TypeError(
            f"Cannot convert UNAVAILABLE metric to float. "
            f"Reason: {self.reason}. "
            "Check .status == 'UNAVAILABLE' before using this value numerically."
        )


UNAVAILABLE = _Unavailable  # Export the class so callers can isinstance-check


# ---------------------------------------------------------------------------
# MetricRegistry
# ---------------------------------------------------------------------------

class MetricRegistry:
    """
    Centralized registry of OrchestraNet evaluation metrics.

    Standard detection metrics (mAP@50, mAP@50:95, AP_small/medium/large)
    are pre-registered and backed by DetectionMetrics.

    Additional task-specific metrics can be registered with register().

    Args:
        num_classes: Number of detection classes (default 80 for COCO).

    Phase 4 note:
        To auto-update after each checkpoint, call reset_all() then
        re-accumulate via accumulate() and call compute_all(). The interface
        is intentionally stateless between compute cycles.
    """

    # Standard IOu thresholds for mAP@50:95
    _IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes

        # Core detection accumulator (reused from utils/metrics.py)
        self._det_metrics = DetectionMetrics(
            num_classes=num_classes,
            iou_thresholds=self._IOU_THRESHOLDS,
        )

        # Registry of custom metric functions:
        #   name -> callable(predictions_list, targets_list) -> float | _Unavailable
        self._custom_metrics: dict[str, Callable] = {}

        # Raw accumulation buffers for custom metrics
        self._custom_preds: list[Any] = []
        self._custom_targets: list[Any] = []

        # Running counts
        self._n_images: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        metric_fn: Callable[[list, list], Any],
        overwrite: bool = False,
    ) -> None:
        """
        Register a named metric function.

        Args:
            name: Unique metric name (e.g., "AP_occ", "AbsRel").
            metric_fn: Callable(predictions_list, targets_list) -> float | _Unavailable.
                       Will be called with all accumulated preds and targets at compute time.
            overwrite: If False (default), raises if name already registered.
        """
        if name in self._custom_metrics and not overwrite:
            raise ValueError(
                f"Metric '{name}' is already registered. "
                "Use overwrite=True to replace it."
            )
        self._custom_metrics[name] = metric_fn

    def registered_names(self) -> list[str]:
        """Return names of all registered custom metrics."""
        standard = ["mAP@50", "mAP@50:95", "AP_small", "AP_medium", "AP_large"]
        return standard + list(self._custom_metrics.keys())

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def update_detection(
        self,
        pred_boxes,
        pred_scores,
        pred_labels,
        gt_boxes,
        gt_labels,
        image_id: int | None = None,
    ) -> None:
        """
        Accumulate standard detection results for one image.

        Delegates directly to DetectionMetrics.update() — same signature.
        """
        self._det_metrics.update(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, image_id=image_id,
        )
        self._n_images += 1

    def update_detection_with_visibility(
        self,
        pred_boxes,
        pred_scores,
        pred_labels,
        gt_boxes,
        gt_labels,
        gt_visibility=None,
        image_id: int | None = None,
    ) -> None:
        """
        Accumulate detection results with per-box visibility ratios.

        Calls DetectionMetrics.update_with_visibility() so OcclusionAwareMetrics
        downstream can filter by visibility < 0.3 for AP_occ computation.

        Args:
            gt_visibility: (N,) array-like of float32 in [0,1], one per GT box.
                           If None, falls back to update_detection().
        """
        if gt_visibility is not None and hasattr(self._det_metrics, "update_with_visibility"):
            self._det_metrics.update_with_visibility(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels, gt_visibility,
                image_id=image_id,
            )
        else:
            self._det_metrics.update(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels, image_id=image_id,
            )
        self._n_images += 1

    def accumulate_custom(self, predictions: Any, targets: Any) -> None:
        """
        Buffer one batch for custom metric functions.

        Custom metric_fn callables receive the full list at compute time.
        """
        self._custom_preds.append(predictions)
        self._custom_targets.append(targets)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute_all(self) -> dict[str, Any]:
        """
        Run all registered metrics and return a unified results dict.

        Standard metrics always present:
            mAP@50, mAP@50:95, AP_small, AP_medium, AP_large

        Custom metrics appended with their registered names.

        Values are either:
          - float  — valid metric
          - _Unavailable.to_dict() — structured UNAVAILABLE sentinel
        """
        # Standard detection metrics
        results: dict[str, Any] = self._det_metrics.compute()
        results["n_images_evaluated"] = self._n_images

        # Custom metrics
        for name, fn in self._custom_metrics.items():
            try:
                val = fn(self._custom_preds, self._custom_targets)
                if isinstance(val, _Unavailable):
                    results[name] = val.to_dict()
                else:
                    results[name] = val
            except Exception as exc:
                warnings.warn(
                    f"Metric '{name}' raised during compute: {exc}",
                    stacklevel=2,
                )
                results[name] = _Unavailable(
                    f"Computation failed: {exc}"
                ).to_dict()

        return results

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_all(self) -> None:
        """
        Clear all accumulators.

        Call this before each evaluation cycle (e.g., at start of each
        checkpoint evaluation in Phase 4).
        """
        self._det_metrics.reset()
        self._custom_preds.clear()
        self._custom_targets.clear()
        self._n_images = 0

    # ------------------------------------------------------------------
    # Serialization helpers (Phase 4 checkpoint integration)
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_results(results: Any) -> Any:
        """
        Convert results to a fully JSON-serializable form recursively.

        Handles nested dicts, lists, numpy floats/ints/ndarrays, and UNAVAILABLE sentinels.
        """
        if isinstance(results, dict):
            return {k: MetricRegistry.serialize_results(v) for k, v in results.items()}
        elif isinstance(results, (list, tuple)):
            return [MetricRegistry.serialize_results(v) for v in results]
        elif isinstance(results, (np.floating, np.integer)):
            return float(results)
        elif isinstance(results, np.ndarray):
            return results.tolist()
        return results

    def __repr__(self) -> str:
        return (
            f"MetricRegistry("
            f"num_classes={self.num_classes}, "
            f"n_images={self._n_images}, "
            f"custom_metrics={list(self._custom_metrics.keys())})"
        )
