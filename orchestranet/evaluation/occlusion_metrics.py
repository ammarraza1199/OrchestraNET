"""
Occlusion-Aware Detection Metrics — Phase 3.

Extends DetectionMetrics with visibility-filtered AP computation:
  - AP_occ:     heavily occluded objects (visibility < 30%)
  - AP_partial: partially occluded (30% ≤ visibility < 70%)
  - AP_visible: mostly visible (visibility ≥ 70%)
  - OA_NMS_preserved_pairs: fraction of valid occlusion pairs kept

Policy (Q3 decision):
  - When per-instance visibility annotations are NOT available → return
    structured UNAVAILABLE result, NEVER 0.0.
  - When annotations ARE available (e.g., KINS), compute full AP50/AP50:95,
    precision, recall, F1, AR, and instance count.
  - Do NOT fabricate visibility annotations.

Usage:
    from orchestranet.evaluation.occlusion_metrics import OcclusionAwareMetrics

    occ = OcclusionAwareMetrics(num_classes=8)  # KINS has 8 classes
    for image in dataset:
        occ.update(pred_boxes, pred_scores, pred_labels,
                   gt_boxes, gt_labels, gt_visibility=image['visibility_ratios'])
    results = occ.compute()
    # results['AP_occ'] is either a float or an UNAVAILABLE dict
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from orchestranet.utils.metrics import compute_iou, compute_ap
from orchestranet.evaluation.metric_registry import UNAVAILABLE, _Unavailable


# Documented heavily-occluded threshold (project_documentation.md, Table 9)
HEAVY_OCC_THRESHOLD = 0.30   # visibility < 30% → "heavily occluded"
PARTIAL_OCC_LOW    = 0.30    # 30% ≤ vis < 70%
PARTIAL_OCC_HIGH   = 0.70
VISIBLE_THRESHOLD  = 0.70    # visibility ≥ 70% → "mostly visible"

_NO_VIS_REASON = (
    "No per-instance visibility annotations available for this evaluation split. "
    "Per-box visibility ratios (float in [0,1]) are required to compute AP_occ. "
    "These are provided by KINS (via kins_dataset.py) but not by plain COCO."
)


class OcclusionAwareMetrics:
    """
    Occlusion-aware AP evaluator.

    Accumulates per-image predictions + GT boxes + per-GT-box visibility ratios,
    then computes AP for three visibility strata.

    Args:
        num_classes: Number of detection classes.
        iou_thresholds: IoU thresholds for mAP sweep (default: COCO 0.5:0.95).

    Notes:
        If no image is updated with valid visibility annotations, all
        occlusion-stratified metrics return UNAVAILABLE sentinels (not 0.0).
    """

    _IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def __init__(self, num_classes: int = 80, iou_thresholds: list[float] | None = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or self._IOU_THRESHOLDS
        self._has_visibility: bool = False
        self.reset()

    def reset(self) -> None:
        """Clear all accumulators."""
        # Per image: each entry is a dict with boxes/scores/labels/visibility
        self._preds: dict[int, dict] = {}
        self._gts: dict[int, dict] = {}
        self._n_images: int = 0
        self._has_visibility = False

        # OA-NMS pair tracking
        self._total_occ_pairs: int = 0
        self._preserved_pairs: int = 0

    def update(
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
        Accumulate results for one image.

        Args:
            pred_boxes: (N, 4) predicted boxes, xyxy format.
            pred_scores: (N,) confidence scores.
            pred_labels: (N,) predicted class labels.
            gt_boxes: (M, 4) GT boxes, xyxy format.
            gt_labels: (M,) GT class labels.
            gt_visibility: (M,) float in [0,1] — fraction of object that is
                           visible.  If None, occlusion-stratified metrics
                           will return UNAVAILABLE.
            image_id: Optional unique image identifier.
        """
        key = image_id if image_id is not None else self._n_images

        pred_boxes   = np.array(pred_boxes,   dtype=np.float32) if len(pred_boxes)  else np.empty((0, 4), np.float32)
        pred_scores  = np.array(pred_scores,  dtype=np.float32) if len(pred_scores) else np.empty((0,), np.float32)
        pred_labels  = np.array(pred_labels,  dtype=np.int64)   if len(pred_labels) else np.empty((0,), np.int64)
        gt_boxes     = np.array(gt_boxes,     dtype=np.float32) if len(gt_boxes)    else np.empty((0, 4), np.float32)
        gt_labels    = np.array(gt_labels,    dtype=np.int64)   if len(gt_labels)   else np.empty((0,), np.int64)

        vis_arr = None
        if gt_visibility is not None:
            vis_arr = np.array(gt_visibility, dtype=np.float32)
            if vis_arr.shape[0] == gt_boxes.shape[0]:
                self._has_visibility = True
            else:
                # Shape mismatch — treat as no visibility
                vis_arr = None

        self._preds[key] = {
            "boxes": pred_boxes,
            "scores": pred_scores,
            "labels": pred_labels,
        }
        self._gts[key] = {
            "boxes": gt_boxes,
            "labels": gt_labels,
            "visibility": vis_arr,  # None if unavailable
        }
        self._n_images += 1

    def record_nms_pair(self, total_pairs: int, preserved_pairs: int) -> None:
        """
        Accumulate OA-NMS occlusion pair statistics across images.

        Args:
            total_pairs: Number of valid occlusion pairs detected (IoU > 0.5,
                         depth difference significant).
            preserved_pairs: Number of those pairs that OA-NMS kept (both boxes
                             survived suppression).
        """
        self._total_occ_pairs += total_pairs
        self._preserved_pairs += preserved_pairs

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self) -> dict[str, Any]:
        """
        Compute all occlusion-aware metrics.

        Returns a dict where each value is either:
          - float  — computed metric
          - dict   — UNAVAILABLE sentinel (status, reason, n_instances)
          - dict   — full occlusion AP breakdown (AP50, AP50:95, P, R, F1, AR, n)

        Never returns 0.0 for a metric that cannot be computed due to missing GT.
        """
        results: dict[str, Any] = {}

        if not self._has_visibility:
            # No visibility annotations available — all occlusion metrics UNAVAILABLE
            _unav = _Unavailable(_NO_VIS_REASON).to_dict()
            results["AP_occ"]     = _unav
            results["AP_partial"] = _unav
            results["AP_visible"] = _unav
            results["AP_occ_detail"] = _unav
            results["n_heavily_occluded"] = _Unavailable(_NO_VIS_REASON).to_dict()
        else:
            results["AP_occ"]     = self._compute_stratum_map("heavy")
            results["AP_partial"] = self._compute_stratum_map("partial")
            results["AP_visible"] = self._compute_stratum_map("visible")
            results["AP_occ_detail"] = self._compute_detailed_occ()
            results["n_heavily_occluded"] = self._count_stratum("heavy")

        # OA-NMS pair preservation (independent of visibility annotations)
        if self._total_occ_pairs > 0:
            results["OA_NMS_preserved_pairs"] = (
                self._preserved_pairs / self._total_occ_pairs
            )
        else:
            results["OA_NMS_preserved_pairs"] = _Unavailable(
                "No occlusion pairs were recorded via record_nms_pair(). "
                "Call record_nms_pair(total, preserved) during OA-NMS evaluation."
            ).to_dict()

        results["n_images"] = self._n_images
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _visibility_mask(self, visibility: np.ndarray, stratum: str) -> np.ndarray:
        """Return boolean mask for the requested visibility stratum."""
        if stratum == "heavy":
            return visibility < HEAVY_OCC_THRESHOLD
        elif stratum == "partial":
            return (visibility >= PARTIAL_OCC_LOW) & (visibility < PARTIAL_OCC_HIGH)
        elif stratum == "visible":
            return visibility >= VISIBLE_THRESHOLD
        raise ValueError(f"Unknown stratum: {stratum!r}")

    def _count_stratum(self, stratum: str) -> int:
        """Count total GT instances in this visibility stratum."""
        count = 0
        for gt in self._gts.values():
            if gt["visibility"] is not None:
                count += int(self._visibility_mask(gt["visibility"], stratum).sum())
        return count

    def _compute_stratum_map(self, stratum: str) -> float | dict:
        """Compute mAP@50 for a visibility stratum across all classes."""
        n_instances = self._count_stratum(stratum)
        if n_instances == 0:
            return _Unavailable(
                f"No GT instances found for stratum='{stratum}' "
                f"(threshold: {HEAVY_OCC_THRESHOLD}). "
                + _NO_VIS_REASON
            ).to_dict()
        ap = self._compute_filtered_map(stratum, iou_thresh=0.5)
        return float(ap)

    def _compute_detailed_occ(self) -> dict[str, Any]:
        """
        Full breakdown for AP_occ (heavily occluded, visibility < 30%):
          - AP50, AP50:95, precision@50, recall@50, F1@50, AR@50:95
          - n_instances
        """
        n_instances = self._count_stratum("heavy")
        if n_instances == 0:
            return _Unavailable(_NO_VIS_REASON).to_dict()

        ap50 = self._compute_filtered_map("heavy", iou_thresh=0.5)
        ap5095 = np.mean([
            self._compute_filtered_map("heavy", iou_thresh=t)
            for t in self.iou_thresholds
        ])
        prec, rec = self._compute_pr_at_iou("heavy", iou_thresh=0.5)
        f1 = (2 * prec * rec / (prec + rec + 1e-9)) if (prec + rec) > 0 else 0.0
        ar = self._compute_ar("heavy")

        return {
            "AP50": float(ap50),
            "AP50:95": float(ap5095),
            "precision": float(prec),
            "recall": float(rec),
            "F1": float(f1),
            "AR": float(ar),
            "n_instances": n_instances,
            "visibility_threshold": f"< {HEAVY_OCC_THRESHOLD:.0%}",
        }

    def _compute_filtered_map(self, stratum: str, iou_thresh: float) -> float:
        """Compute mean AP over all classes, filtering GT by visibility stratum."""
        aps = []
        for cls in range(self.num_classes):
            ap = self._class_ap_filtered(cls, stratum, iou_thresh)
            if ap is not None:
                aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0

    def _class_ap_filtered(self, cls: int, stratum: str, iou_thresh: float) -> float | None:
        """AP for a single class with GT filtered to the visibility stratum."""
        all_scores = []
        all_matches = []
        total_gt = 0

        for key in self._gts:
            gt = self._gts[key]
            pred = self._preds.get(key, {
                "boxes": np.empty((0, 4), np.float32),
                "scores": np.empty((0,), np.float32),
                "labels": np.empty((0,), np.int64),
            })

            # Filter GT by class
            gt_cls_mask = gt["labels"] == cls
            gt_boxes    = gt["boxes"][gt_cls_mask]
            visibility  = gt["visibility"]

            # Filter GT by visibility stratum
            if visibility is not None:
                gt_vis = visibility[gt_cls_mask]
                stratum_mask = self._visibility_mask(gt_vis, stratum)
                gt_boxes = gt_boxes[stratum_mask]

            total_gt += len(gt_boxes)
            if len(gt_boxes) == 0:
                continue

            # Filter predictions by class
            pred_cls_mask = pred["labels"] == cls
            pred_boxes  = pred["boxes"][pred_cls_mask]
            pred_scores = pred["scores"][pred_cls_mask]

            if len(pred_boxes) == 0:
                continue

            ious = compute_iou(pred_boxes, gt_boxes)
            matched_gt = set()
            order = np.argsort(-pred_scores)
            for i in order:
                best_iou = ious[i].max() if ious.shape[1] > 0 else 0.0
                best_gt  = ious[i].argmax() if ious.shape[1] > 0 else 0
                if best_iou >= iou_thresh and best_gt not in matched_gt:
                    all_scores.append(pred_scores[i])
                    all_matches.append(True)
                    matched_gt.add(best_gt)
                else:
                    all_scores.append(pred_scores[i])
                    all_matches.append(False)

        if total_gt == 0:
            return None

        if len(all_scores) == 0:
            return 0.0

        order = np.argsort(-np.asarray(all_scores, dtype=np.float32))
        matches = np.asarray(all_matches, dtype=np.bool_)[order]
        tp = np.cumsum(matches, dtype=np.int64)
        fp = np.cumsum(np.logical_not(matches), dtype=np.int64)
        recall    = tp / total_gt
        precision = tp / (tp + fp + 1e-9)
        return compute_ap(recall, precision)

    def _compute_pr_at_iou(self, stratum: str, iou_thresh: float) -> tuple[float, float]:
        """Compute mean precision and recall across classes at a specific IoU threshold."""
        precisions, recalls = [], []
        for cls in range(self.num_classes):
            ap = self._class_ap_filtered(cls, stratum, iou_thresh)
            if ap is None:
                continue
            # Re-derive last P/R from class accumulation
            all_scores, all_matches, total_gt = self._class_pr_arrays(cls, stratum, iou_thresh)
            if total_gt > 0 and len(all_scores) > 0:
                order = np.argsort(-np.asarray(all_scores, dtype=np.float32))
                matches = np.asarray(all_matches, dtype=np.bool_)[order]
                tp = np.cumsum(matches, dtype=np.int64)
                fp = np.cumsum(np.logical_not(matches), dtype=np.int64)
                recalls.append(float(tp[-1] / total_gt))
                precisions.append(float(tp[-1] / (tp[-1] + fp[-1] + 1e-9)))
        mean_p = float(np.mean(precisions)) if precisions else 0.0
        mean_r = float(np.mean(recalls))    if recalls    else 0.0
        return mean_p, mean_r

    def _class_pr_arrays(
        self, cls: int, stratum: str, iou_thresh: float
    ) -> tuple[list, list, int]:
        """Return (all_scores, all_matches, total_gt) for one class/stratum."""
        all_scores, all_matches, total_gt = [], [], 0
        for key in self._gts:
            gt = self._gts[key]
            pred = self._preds.get(key, {
                "boxes": np.empty((0, 4), np.float32),
                "scores": np.empty((0,), np.float32),
                "labels": np.empty((0,), np.int64),
            })
            gt_cls_mask = gt["labels"] == cls
            gt_boxes    = gt["boxes"][gt_cls_mask]
            visibility  = gt["visibility"]
            if visibility is not None:
                gt_vis = visibility[gt_cls_mask]
                stratum_mask = self._visibility_mask(gt_vis, stratum)
                gt_boxes = gt_boxes[stratum_mask]
            total_gt += len(gt_boxes)
            if len(gt_boxes) == 0:
                continue
            pred_cls_mask = pred["labels"] == cls
            pred_boxes  = pred["boxes"][pred_cls_mask]
            pred_scores = pred["scores"][pred_cls_mask]
            if len(pred_boxes) == 0:
                continue
            ious = compute_iou(pred_boxes, gt_boxes)
            matched_gt = set()
            order = np.argsort(-pred_scores)
            for i in order:
                best_iou = ious[i].max() if ious.shape[1] > 0 else 0.0
                best_gt  = ious[i].argmax() if ious.shape[1] > 0 else 0
                if best_iou >= iou_thresh and best_gt not in matched_gt:
                    all_scores.append(pred_scores[i])
                    all_matches.append(True)
                    matched_gt.add(best_gt)
                else:
                    all_scores.append(pred_scores[i])
                    all_matches.append(False)
        return all_scores, all_matches, total_gt

    def _compute_ar(self, stratum: str) -> float:
        """Compute Average Recall across IoU thresholds (COCO AR@50:95)."""
        recalls = []
        for iou_thresh in self.iou_thresholds:
            cls_recalls = []
            for cls in range(self.num_classes):
                scores_m, matches_m, tgt_m = self._class_pr_arrays(cls, stratum, iou_thresh)
                if tgt_m == 0 or len(scores_m) == 0:
                    continue
                order = np.argsort(-np.asarray(scores_m, dtype=np.float32))
                matches_arr = np.asarray(matches_m, dtype=np.bool_)[order]
                tp = np.cumsum(matches_arr, dtype=np.int64)
                cls_recalls.append(float(tp[-1] / tgt_m))
            if cls_recalls:
                recalls.append(float(np.mean(cls_recalls)))
        return float(np.mean(recalls)) if recalls else 0.0
