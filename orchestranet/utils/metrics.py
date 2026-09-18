"""
Metrics for OrchestraNet evaluation.

Includes standard detection metrics (mAP) plus occlusion-specific metrics.
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Any


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes (xyxy format)."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision using 101-point interpolation (COCO-style)."""
    if len(recall) == 0 or len(precision) == 0 or np.all(recall == 0):
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Ensure precision is monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        prec_at_r = mpre[mrec >= r]
        ap += prec_at_r.max() if len(prec_at_r) > 0 else 0.0

    return ap / 101.0


def compute_amodal_metrics(
    predictions: dict[str, torch.Tensor | np.ndarray],
    targets: dict[str, torch.Tensor | np.ndarray],
) -> dict[str, Any]:
    """
    Compute amodal completion evaluation metrics:
      1. Amodal Mask IoU: Mean intersection-over-union between predicted 28x28
         amodal masks and ground-truth amodal masks.
      2. Bbox MAE: Mean Absolute Error between predicted amodal bbox offsets
         and ground-truth amodal bbox offsets.

    Evaluates exclusively on valid ground-truth objects (up to num_objects).
    """
    pred_masks = predictions.get("amodal_masks")
    pred_boxes = predictions.get("amodal_bbox_offset")
    gt_masks = targets.get("amodal_masks")
    gt_boxes = targets.get("amodal_boxes")
    num_objs = targets.get("num_objects")

    if pred_masks is None or gt_masks is None or pred_boxes is None or gt_boxes is None:
        return {
            "amodal_mask_iou": 0.0,
            "amodal_bbox_mae": 0.0,
            "ious": [],
            "maes": [],
        }

    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.detach().cpu()
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.detach().cpu()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.detach().cpu()

    if isinstance(pred_masks, np.ndarray):
        pred_masks = torch.from_numpy(pred_masks)
    if isinstance(pred_boxes, np.ndarray):
        pred_boxes = torch.from_numpy(pred_boxes)
    if isinstance(gt_masks, np.ndarray):
        gt_masks = torch.from_numpy(gt_masks)
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes)

    B = pred_masks.shape[0]
    mask_ious = []
    bbox_maes = []

    for b in range(B):
        if num_objs is not None:
            n_raw = num_objs[b].item() if isinstance(num_objs[b], torch.Tensor) else num_objs[b]
            n_b = min(int(n_raw), pred_masks.shape[1], gt_masks.shape[1])
        else:
            n_b = min(pred_masks.shape[1], gt_masks.shape[1])

        for i in range(n_b):
            p_mask = pred_masks[b, i] > 0.5
            g_mask = gt_masks[b, i] > 0.5
            inter = (p_mask & g_mask).sum().item()
            union = (p_mask | g_mask).sum().item()
            iou = inter / (union + 1e-6) if union > 0 else 1.0
            mask_ious.append(iou)

            mae = torch.abs(pred_boxes[b, i] - gt_boxes[b, i]).mean().item()
            bbox_maes.append(mae)

    return {
        "amodal_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "amodal_bbox_mae": float(np.mean(bbox_maes)) if bbox_maes else 0.0,
        "ious": mask_ious,
        "maes": bbox_maes,
    }


class DetectionMetrics:
    """
    Accumulates detection results and computes COCO-style metrics.

    Tracks mAP at different IoU thresholds and object size categories,
    with additional occlusion-aware metrics.
    """

    def __init__(self, num_classes: int = 80, iou_thresholds: list[float] | None = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.reset()

    def reset(self):
        self.predictions = defaultdict(list)
        self.ground_truths = defaultdict(list)

    def update(self, pred_boxes, pred_scores, pred_labels,
               gt_boxes, gt_labels, image_id=None):
        """Add predictions and ground truths for one image."""
        key = image_id if image_id is not None else len(self.predictions)
        self.predictions[key] = {
            "boxes": np.array(pred_boxes),
            "scores": np.array(pred_scores),
            "labels": np.array(pred_labels),
        }
        self.ground_truths[key] = {
            "boxes": np.array(gt_boxes),
            "labels": np.array(gt_labels),
        }

    def update_with_visibility(self, pred_boxes, pred_scores, pred_labels,
                                gt_boxes, gt_labels, gt_visibility, image_id=None):
        """
        Add predictions and ground truths for one image, including per-GT-box
        visibility ratios for downstream occlusion-aware metric computation.

        Args:
            gt_visibility: (N,) array-like of float32 in [0, 1] — fraction of
                           the object that is visible.  One entry per GT box.
                           Used by OcclusionAwareMetrics to compute AP_occ
                           (visibility < 30%), AP_partial, and AP_visible.

        All other args are identical to update().  Backward-compatible: callers
        using update() are unaffected.
        """
        key = image_id if image_id is not None else len(self.predictions)
        self.predictions[key] = {
            "boxes": np.array(pred_boxes),
            "scores": np.array(pred_scores),
            "labels": np.array(pred_labels),
        }
        self.ground_truths[key] = {
            "boxes": np.array(gt_boxes),
            "labels": np.array(gt_labels),
            "visibility": np.array(gt_visibility, dtype=np.float32),
        }

    def compute(self) -> dict[str, float]:
        """Compute all metrics."""
        results = {}

        # mAP@50
        results["mAP@50"] = self._compute_map(iou_threshold=0.5)

        # mAP@50:95
        aps = [self._compute_map(t) for t in self.iou_thresholds]
        results["mAP@50:95"] = np.mean(aps)

        # Size-specific AP (using COCO area thresholds)
        results["AP_small"] = self._compute_map(0.5, max_area=32**2)
        results["AP_medium"] = self._compute_map(0.5, min_area=32**2, max_area=96**2)
        results["AP_large"] = self._compute_map(0.5, min_area=96**2)

        return results

    def _compute_map(self, iou_threshold=0.5, min_area=0, max_area=float("inf")):
        """Compute mAP at a specific IoU threshold."""
        aps = []
        for cls in range(self.num_classes):
            ap = self._compute_class_ap(cls, iou_threshold, min_area, max_area)
            if ap is not None:
                aps.append(ap)
        return np.mean(aps) if aps else 0.0

    def _compute_class_ap(self, cls, iou_threshold, min_area, max_area):
        """Compute AP for a single class."""
        all_scores = []
        all_matches = []
        total_gt = 0

        for key in self.ground_truths:
            gt = self.ground_truths[key]
            pred = self.predictions.get(key, {"boxes": np.empty((0,4)), "scores": np.empty(0), "labels": np.empty(0)})

            gt_mask = gt["labels"] == cls
            pred_mask = pred["labels"] == cls

            gt_boxes = gt["boxes"][gt_mask]
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            # Filter by area
            if len(gt_boxes) > 0:
                areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                area_mask = (areas >= min_area) & (areas < max_area)
                gt_boxes = gt_boxes[area_mask]

            total_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            if len(gt_boxes) == 0:
                all_scores.extend(pred_scores.tolist())
                all_matches.extend([False] * len(pred_scores))
                continue

            ious = compute_iou(pred_boxes, gt_boxes)
            matched_gt = set()

            order = np.argsort(-pred_scores)
            for i in order:
                best_iou = ious[i].max()
                best_gt = ious[i].argmax()
                if best_iou >= iou_threshold and best_gt not in matched_gt:
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
        recall = tp / total_gt
        precision = tp / (tp + fp + 1e-9)

        return compute_ap(recall, precision)
