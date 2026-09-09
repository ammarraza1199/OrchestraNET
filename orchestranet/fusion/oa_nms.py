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
    iou_threshold: float = 0.5,
    occlusion_threshold: float = 0.3,
    score_threshold: float = 0.25,
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

    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)

        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue

            iou = iou_matrix[i, j].item()
            if iou < iou_threshold:
                continue  # No significant overlap

            # Same class check — different classes shouldn't suppress each other
            if labels[i] != labels[j]:
                continue

            # === OCCLUSION-AWARE LOGIC (Novel) ===
            is_occlusion_pair = False

            if occlusion_scores is not None:
                vis_i = occlusion_scores[i].item()
                vis_j = occlusion_scores[j].item()
                vis_diff = abs(vis_i - vis_j)

                # If one object is significantly more visible, it's likely
                # an occlusion relationship (one behind the other)
                if vis_diff > occlusion_threshold:
                    is_occlusion_pair = True

            if depth_values is not None and not is_occlusion_pair:
                depth_i = depth_values[i].item()
                depth_j = depth_values[j].item()
                depth_diff = abs(depth_i - depth_j)

                # Significant depth difference → different objects at different depths
                if depth_diff > 0.15:
                    is_occlusion_pair = True

            if is_occlusion_pair:
                # DON'T suppress — these are two real objects, one occluding the other
                # Adjust the occluded object's score based on visibility
                if occlusion_scores is not None:
                    scores[j] = scores[j] * max(0.3, occlusion_scores[j].item())
            else:
                # True duplicate — suppress lower confidence detection
                suppressed[j] = True

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
