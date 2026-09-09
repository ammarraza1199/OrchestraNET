"""
M1: Primary Detector — YOLO-Nano Detection Head.

Always-active detection model producing raw bounding boxes, class logits,
and objectness scores using depthwise-separable convolutions.

Loss computation uses SimOTA-lite anchor-target assignment with:
  - GIoU loss for bounding box regression
  - Focal loss for classification
  - BCE for objectness
"""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseMicroModel


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 + BN + SiLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))


class DetectionHead(nn.Module):
    """Single-scale detection head: bboxes + objectness + classes."""
    def __init__(self, in_channels, hidden_channels, num_anchors, num_classes, num_convs=2):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        ch = in_channels
        for _ in range(num_convs):
            layers.append(DepthwiseSeparableBlock(ch, hidden_channels))
            ch = hidden_channels
        self.convs = nn.Sequential(*layers)
        self.pred = nn.Conv2d(hidden_channels, num_anchors * (4 + 1 + num_classes), 1, bias=True)

    def forward(self, x):
        return self.pred(self.convs(x))


# ============ Loss Utilities ============

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes (xyxy format).
    Args:
        boxes1: (N, 4), boxes2: (M, 4)
    Returns:
        (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes (xyxy format).
    Returns (N,) GIoU values for paired boxes (boxes1[i] vs boxes2[i]).
    """
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    union = area1 + area2 - inter

    iou = inter / (union + 1e-7)

    # Enclosing box
    enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_x2 - enc_x1).clamp(0) * (enc_y2 - enc_y1).clamp(0)

    giou = iou - (enc_area - union) / (enc_area + 1e-7)
    return giou


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for dense classification.
    Args:
        logits: (N, C) raw class logits
        targets: (N,) integer class labels
    """
    num_classes = logits.shape[-1]
    # One-hot encode targets
    target_one_hot = F.one_hot(targets.long(), num_classes).float()

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target_one_hot, reduction="none")
    p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)
    alpha_t = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * ce).sum(-1).mean()


class M1PrimaryDetector(BaseMicroModel):
    """
    M1: Primary Object Detector — Always Active (~1.2M params).
    YOLO-Nano-style multi-scale detection with 3 anchors per level.

    Loss computation uses SimOTA-lite assignment:
      1. For each GT box, find top-k predicted boxes by IoU
      2. Assign the best available prediction to each GT
      3. Compute GIoU loss (bbox), focal loss (cls), BCE (objectness)
    """
    DEFAULT_ANCHORS = {
        0: [[10, 13], [16, 30], [33, 23]],
        1: [[30, 61], [62, 45], [59, 119]],
        2: [[116, 90], [156, 198], [373, 326]],
    }

    def __init__(self, in_channels=128, hidden_channels=64, num_classes=80,
                 num_anchors=3, num_convs=2, anchors=None,
                 bbox_loss_weight=5.0, cls_loss_weight=1.0, obj_loss_weight=1.0):
        super().__init__(model_id="m1", model_name="Primary Detector")
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchors = anchors or self.DEFAULT_ANCHORS
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight

        for lvl, anc in self.anchors.items():
            self.register_buffer(f"anchors_{lvl}", torch.tensor(anc, dtype=torch.float32))
        self.heads = nn.ModuleList([
            DetectionHead(in_channels, hidden_channels, num_anchors, num_classes, num_convs)
            for _ in range(3)
        ])

    def forward(self, features, context=None):
        all_boxes, all_obj, all_cls = [], [], []
        for lvl, (feat, head) in enumerate(zip(features, self.heads)):
            raw = head(feat)
            B, _, H, W = raw.shape
            raw = raw.view(B, self.num_anchors, -1, H, W)
            bbox_pred = raw[:, :, :4, :, :]
            obj_pred = raw[:, :, 4:5, :, :]
            cls_pred = raw[:, :, 5:, :, :]
            decoded = self._decode_boxes(bbox_pred, lvl, H, W, feat.device)
            all_boxes.append(decoded.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
            all_obj.append(obj_pred.permute(0, 1, 3, 4, 2).reshape(B, -1, 1))
            all_cls.append(cls_pred.permute(0, 1, 3, 4, 2).reshape(B, -1, self.num_classes))
        return {
            "decoded_boxes": torch.cat(all_boxes, 1),
            "objectness": torch.cat(all_obj, 1),
            "class_logits": torch.cat(all_cls, 1),
        }

    def _decode_boxes(self, bbox_pred, level_idx, H, W, device):
        stride = 640 // H
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
        anchors = getattr(self, f"anchors_{level_idx}").to(device)
        tx, ty, tw, th = bbox_pred.split(1, dim=2)
        bx = (2 * torch.sigmoid(tx) - 0.5 + gx) * stride
        by = (2 * torch.sigmoid(ty) - 0.5 + gy) * stride
        bw = (2 * torch.sigmoid(tw)) ** 2 * anchors[:, 0].view(1, -1, 1, 1, 1)
        bh = (2 * torch.sigmoid(th)) ** 2 * anchors[:, 1].view(1, -1, 1, 1, 1)
        return torch.cat([bx - bw/2, by - bh/2, bx + bw/2, by + bh/2], dim=2)

    def get_loss(self, predictions, targets):
        """
        Compute detection loss with SimOTA-lite anchor-target matching.

        Matches GT boxes to predicted boxes using IoU-based top-k assignment,
        then computes GIoU (bbox), focal (cls), and BCE (objectness) losses.
        """
        device = predictions["decoded_boxes"].device
        pred_boxes = predictions["decoded_boxes"]   # (B, N_pred, 4)
        pred_obj = predictions["objectness"]         # (B, N_pred, 1)
        pred_cls = predictions["class_logits"]       # (B, N_pred, C)

        B = pred_boxes.shape[0]
        total_bbox_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(B):
            # Get GT for this image
            gt_boxes = targets["boxes"][b]      # (max_obj, 4)
            gt_labels = targets["labels"][b]    # (max_obj,)

            # Get actual number of objects (filter padding)
            if "num_objects" in targets:
                n_gt = targets["num_objects"][b].item()
            else:
                # Infer from non-zero boxes
                valid = (gt_boxes[:, 2] - gt_boxes[:, 0]) > 0
                n_gt = valid.sum().item()

            if n_gt == 0:
                # No GT — all objectness targets are 0
                obj_target = torch.zeros_like(pred_obj[b, :, 0])
                total_obj_loss = total_obj_loss + F.binary_cross_entropy_with_logits(
                    pred_obj[b, :, 0], obj_target, reduction="mean"
                )
                continue

            gt_boxes_valid = gt_boxes[:n_gt].to(device)   # (n_gt, 4)
            gt_labels_valid = gt_labels[:n_gt].to(device)  # (n_gt,)

            b_pred_boxes = pred_boxes[b]  # (N_pred, 4)
            b_pred_obj = pred_obj[b, :, 0]  # (N_pred,)
            b_pred_cls = pred_cls[b]       # (N_pred, C)

            # === SimOTA-Lite Assignment ===
            # Compute IoU between all predictions and all GT
            with torch.no_grad():
                cost_iou = box_iou(b_pred_boxes.detach(), gt_boxes_valid)  # (N_pred, n_gt)

                # Cost = -IoU + cls cost
                cls_cost = torch.zeros_like(cost_iou)
                pred_cls_prob = torch.sigmoid(b_pred_cls.detach())
                for g in range(n_gt):
                    label = gt_labels_valid[g].long()
                    if label < self.num_classes:
                        cls_cost[:, g] = -pred_cls_prob[:, label]

                cost_matrix = -cost_iou + 0.5 * cls_cost  # (N_pred, n_gt)

                # Top-k selection per GT (k = min(10, N_pred))
                k = min(10, b_pred_boxes.shape[0])
                matched_pred_indices = []
                matched_gt_indices = []
                used_preds = set()

                for g in range(n_gt):
                    costs = cost_matrix[:, g]
                    topk_vals, topk_idxs = costs.topk(k, largest=False)
                    # Find best unused prediction
                    for idx in topk_idxs:
                        idx_item = idx.item()
                        if idx_item not in used_preds:
                            matched_pred_indices.append(idx_item)
                            matched_gt_indices.append(g)
                            used_preds.add(idx_item)
                            break

            n_matched = len(matched_pred_indices)
            if n_matched == 0:
                obj_target = torch.zeros_like(b_pred_obj)
                total_obj_loss = total_obj_loss + F.binary_cross_entropy_with_logits(
                    b_pred_obj, obj_target, reduction="mean"
                )
                continue

            pred_idx = torch.tensor(matched_pred_indices, dtype=torch.long, device=device)
            gt_idx = torch.tensor(matched_gt_indices, dtype=torch.long, device=device)

            # === Bbox Loss (GIoU) ===
            matched_pred_boxes = b_pred_boxes[pred_idx]    # (n_matched, 4)
            matched_gt_boxes = gt_boxes_valid[gt_idx]      # (n_matched, 4)
            giou = generalized_box_iou(matched_pred_boxes, matched_gt_boxes)
            bbox_loss = (1.0 - giou).mean()
            total_bbox_loss = total_bbox_loss + bbox_loss

            # === Classification Loss (Focal) ===
            matched_pred_cls = b_pred_cls[pred_idx]        # (n_matched, C)
            matched_gt_labels = gt_labels_valid[gt_idx]    # (n_matched,)
            # Clamp labels to valid range
            matched_gt_labels = matched_gt_labels.clamp(0, self.num_classes - 1)
            cls_loss = focal_loss(matched_pred_cls, matched_gt_labels)
            total_cls_loss = total_cls_loss + cls_loss

            # === Objectness Loss (BCE) ===
            obj_target = torch.zeros_like(b_pred_obj)
            # Positive targets: IoU with matched GT
            obj_target[pred_idx] = giou.detach().clamp(0, 1)
            obj_loss = F.binary_cross_entropy_with_logits(
                b_pred_obj, obj_target, reduction="mean"
            )
            total_obj_loss = total_obj_loss + obj_loss

            num_pos += n_matched

        # Average across batch
        total_bbox_loss = total_bbox_loss / max(B, 1)
        total_cls_loss = total_cls_loss / max(B, 1)
        total_obj_loss = total_obj_loss / max(B, 1)

        total_loss = (
            self.bbox_loss_weight * total_bbox_loss
            + self.cls_loss_weight * total_cls_loss
            + self.obj_loss_weight * total_obj_loss
        )

        return {
            "bbox_loss": total_bbox_loss,
            "cls_loss": total_cls_loss,
            "obj_loss": total_obj_loss,
            "total_loss": total_loss,
            "num_pos": torch.tensor(num_pos, device=device, dtype=torch.float32),
        }

    def required_levels(self):
        return ["P3", "P4", "P5"]
