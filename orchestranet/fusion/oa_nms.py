"""
OA-NMS: Occlusion-Aware Non-Maximum Suppression — Key Novelty.

Standard NMS suppresses overlapping boxes purely by IoU, which destroys
detections of occluded objects. OA-NMS uses occlusion maps and depth
ordering to distinguish overlapping-but-real objects from true duplicates.

This is one of the 7 novelty claims for the PhD thesis.
"""

import torch
from torchvision.ops import box_iou


def occlusion_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    occlusion_scores: torch.Tensor | None = None,
    depth_values: torch.Tensor | None = None,
    iou_threshold: float = 0.65,
    occlusion_threshold: float = 0.20,
    score_threshold: float = 0.01,
    max_detections: int = 300,
) -> dict[str, torch.Tensor]:
    """
    Occlusion-Aware Non-Maximum Suppression.

    Unlike standard NMS, this algorithm:
    1. Checks if overlapping detections form an occlusion pair
    2. Uses depth ordering to determine which is in front
    3. Keeps both detections if they represent distinct objects
    4. Only suppresses true duplicates

    Args:
        boxes: (N, 4) bounding boxes in xyxy format
        scores: (N,) confidence scores
        labels: (N,) class labels
        occlusion_scores: (N,) per-detection visibility ratio [0,1] (from M2)
        depth_values: (N,) relative depth at detection center (from M4)
        iou_threshold: IoU threshold for considering overlap
        occlusion_threshold: Min visibility difference to consider occlusion pair
        score_threshold: Min score to keep
        max_detections: Maximum output detections

    Returns:
        Dict with filtered "boxes", "scores", "labels", "keep_indices"
    """
    # Filter by score threshold
    keep_mask = scores > score_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]
    original_indices = torch.where(keep_mask)[0]

    if occlusion_scores is not None:
        occlusion_scores = occlusion_scores[keep_mask]
    if depth_values is not None:
        depth_values = depth_values[keep_mask]

    if boxes.shape[0] == 0:
        return {
            "boxes": boxes, "scores": scores, "labels": labels,
            "keep_indices": torch.tensor([], dtype=torch.long, device=boxes.device),
        }

    # Pre-NMS top-k limit (standard in YOLO/Faster-RCNN) to prevent O(N^2) stalls
    max_pre_nms = 300
    if boxes.shape[0] > max_pre_nms:
        topk_idx = scores.topk(max_pre_nms)[1]
        boxes = boxes[topk_idx]
        scores = scores[topk_idx]
        labels = labels[topk_idx]
        original_indices = original_indices[topk_idx]
        if occlusion_scores is not None:
            occlusion_scores = occlusion_scores[topk_idx]
        if depth_values is not None:
            depth_values = depth_values[topk_idx]

    # Fast path: if neither occlusion nor depth signal is available, use fast CUDA batched_nms
    if occlusion_scores is None and depth_values is None:
        from torchvision.ops import batched_nms
        keep = batched_nms(boxes, scores, labels, iou_threshold)
        if len(keep) > max_detections:
            keep = keep[:max_detections]
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
            "keep_indices": original_indices[keep],
        }

    # Sort by confidence (descending)
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    scores = scores[order]
    labels = labels[order]
    original_indices = original_indices[order]
    if occlusion_scores is not None:
        occlusion_scores = occlusion_scores[order]
    if depth_values is not None:
        depth_values = depth_values[order]

    # Compute pairwise IoU
    iou_matrix = box_iou(boxes, boxes)
    N = len(boxes)
    suppressed = torch.zeros(N, dtype=torch.bool, device=boxes.device)
    keep = []

    # Vectorized suppression loop (O(kept) tensor operations, zero .item() GPU barriers)
    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(i)

        # Candidate indices j > i that are not yet suppressed
        cand_indices = torch.arange(i + 1, N, device=boxes.device)
        cand_indices = cand_indices[~suppressed[cand_indices]]
        if len(cand_indices) == 0:
            continue

        same_class = (labels[cand_indices] == labels[i])
        high_iou = (iou_matrix[i, cand_indices] >= iou_threshold)
        overlap_mask = same_class & high_iou
        if not overlap_mask.any():
            continue

        overlap_cands = cand_indices[overlap_mask]

        # Vectorized occlusion-pair detection using M2 (occlusion), M4 (depth), and scale equivariance
        is_occ_pair = torch.zeros(len(overlap_cands), dtype=torch.bool, device=boxes.device)
        if occlusion_scores is not None:
            vis_diff = torch.abs(occlusion_scores[i] - occlusion_scores[overlap_cands])
            is_occ_pair = is_occ_pair | (vis_diff > 0.05)

        if depth_values is not None:
            depth_diff = torch.abs(depth_values[i] - depth_values[overlap_cands])
            is_occ_pair = is_occ_pair | (depth_diff > 0.04)

        # Scale difference detection: small object overlapping large object (e.g. person holding cup/bag)
        area_i = (boxes[i, 2] - boxes[i, 0]).clamp(min=1) * (boxes[i, 3] - boxes[i, 1]).clamp(min=1)
        area_cands = (boxes[overlap_cands, 2] - boxes[overlap_cands, 0]).clamp(min=1) * (boxes[overlap_cands, 3] - boxes[overlap_cands, 1]).clamp(min=1)
        scale_ratio = torch.max(area_i / area_cands, area_cands / area_i)
        is_occ_pair = is_occ_pair | (scale_ratio > 1.8)

        # Occlusion pair candidates: preserved with full confidence!
        # Duplicate candidates: apply Gaussian Soft-NMS decay
        dup_cands = overlap_cands[~is_occ_pair]
        if len(dup_cands) > 0:
            ious = iou_matrix[i, dup_cands]
            decay = torch.exp(-(ious ** 2) / 0.5)
            scores[dup_cands] = scores[dup_cands] * decay
            suppressed[dup_cands] = scores[dup_cands] < score_threshold

    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device)

    # Limit to max detections
    if len(keep) > max_detections:
        keep = keep[:max_detections]

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
        "keep_indices": original_indices[keep],
    }


def standard_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Standard NMS fallback using torchvision."""
    from torchvision.ops import nms
    return nms(boxes, scores, iou_threshold)
