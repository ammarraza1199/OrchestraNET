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
        self._init_bias()

    def _init_bias(self):
        # Initialize objectness and class logits bias to -4.595 (prior prob ~0.01)
        # to prevent massive false-positive saturation and early loss explosion.
        with torch.no_grad():
            b = self.pred.bias.view(self.num_anchors, -1)
            b[:, 4].fill_(-4.595)    # objectness prior = 0.01
            b[:, 5:].fill_(-4.595)   # class logits prior = 0.01

    def forward(self, x):
        return self.pred(self.convs(x))


# ============ Loss Utilities ============

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes (xyxy format) in FP32.
    Args:
        boxes1: (N, 4), boxes2: (M, 4)
    Returns:
        (N, M) IoU matrix
    """
    with torch.amp.autocast("cuda", enabled=False):
        b1 = boxes1.float()
        b2 = boxes2.float()
        area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
        area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

        inter_x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
        inter_y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
        inter_x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
        inter_y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-7)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes (xyxy format) in FP32.
    Returns (N,) GIoU values for paired boxes (boxes1[i] vs boxes2[i]).
    """
    with torch.amp.autocast("cuda", enabled=False):
        b1 = boxes1.float()
        b2 = boxes2.float()

        x1 = torch.max(b1[:, 0], b2[:, 0])
        y1 = torch.max(b1[:, 1], b2[:, 1])
        x2 = torch.min(b1[:, 2], b2[:, 2])
        y2 = torch.min(b1[:, 3], b2[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
        area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
        union = area1 + area2 - inter

        iou = inter / (union + 1e-7)

        # Enclosing box
        enc_x1 = torch.min(b1[:, 0], b2[:, 0])
        enc_y1 = torch.min(b1[:, 1], b2[:, 1])
        enc_x2 = torch.max(b1[:, 2], b2[:, 2])
        enc_y2 = torch.max(b1[:, 3], b2[:, 3])
        enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

        giou = iou - (enc_area - union) / (enc_area + 1e-7)
        return torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for dense classification.

    The loss is explicitly computed in FP32 so that AMP does not
    create dtype mismatches between FP16 logits and FP32 targets.

    Args:
        logits: (N, C) raw class logits
        targets: (N,) integer class labels
    """
    num_classes = logits.shape[-1]

    with torch.amp.autocast("cuda", enabled=False):
        logits_fp32 = logits.float()

        target_one_hot = F.one_hot(
            targets.long(),
            num_classes
        ).float()

        p = torch.sigmoid(logits_fp32)

        ce = F.binary_cross_entropy_with_logits(
            logits_fp32,
            target_one_hot,
            reduction="none"
        )

        p_t = (
            p * target_one_hot
            + (1 - p) * (1 - target_one_hot)
        )

        alpha_t = (
            alpha * target_one_hot
            + (1 - alpha) * (1 - target_one_hot)
        )

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
        with torch.amp.autocast("cuda", enabled=False):
            bbox_pred_fp32 = torch.nan_to_num(bbox_pred.float(), nan=0.0, posinf=10.0, neginf=-10.0)
            stride = 640 // H
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
            anchors = getattr(self, f"anchors_{level_idx}").to(device=device, dtype=torch.float32)
            tx, ty, tw, th = bbox_pred_fp32.split(1, dim=2)
            tx = tx.clamp(-10.0, 10.0)
            ty = ty.clamp(-10.0, 10.0)
            tw = tw.clamp(-5.0, 5.0)
            th = th.clamp(-5.0, 5.0)
            bx = (2 * torch.sigmoid(tx) - 0.5 + gx) * stride
            by = (2 * torch.sigmoid(ty) - 0.5 + gy) * stride
            bw = (2 * torch.sigmoid(tw)) ** 2 * anchors[:, 0].view(1, -1, 1, 1, 1)
            bh = (2 * torch.sigmoid(th)) ** 2 * anchors[:, 1].view(1, -1, 1, 1, 1)
            decoded = torch.cat([bx - bw/2, by - bh/2, bx + bw/2, by + bh/2], dim=2)
            return torch.nan_to_num(decoded, nan=0.0, posinf=640.0, neginf=0.0)

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
                num_obj_b = targets["num_objects"][b]
                n_gt = int(num_obj_b.item() if isinstance(num_obj_b, torch.Tensor) else num_obj_b)
            else:
                # Infer from non-zero boxes
                valid = (gt_boxes[:, 2] - gt_boxes[:, 0]) > 0
                n_gt = int(valid.sum().item())

            if n_gt == 0:
                with torch.amp.autocast("cuda", enabled=False):
                    obj_pred = pred_obj[b, :, 0].float()
                    total_obj_loss = total_obj_loss + F.binary_cross_entropy_with_logits(
                        obj_pred,
                        torch.zeros_like(obj_pred),
                        reduction="mean"
                    )

                continue

            gt_boxes_valid = gt_boxes[:n_gt].to(device=device, dtype=torch.float32)   # (n_gt, 4)
            gt_labels_valid = gt_labels[:n_gt].to(device)  # (n_gt,)

            b_pred_boxes = pred_boxes[b].float()  # (N_pred, 4)
            b_pred_obj = pred_obj[b, :, 0]  # (N_pred,)
            b_pred_cls = pred_cls[b]       # (N_pred, C)

            # === SimOTA-Lite Assignment ===
            # Compute IoU between all predictions and all GT
            with torch.no_grad():
                cost_iou = box_iou(b_pred_boxes.detach(), gt_boxes_valid)  # (N_pred, n_gt)

                # Center prior: predictions whose center is inside or near the GT box
                pred_cx = (b_pred_boxes[:, 0] + b_pred_boxes[:, 2]) * 0.5
                pred_cy = (b_pred_boxes[:, 1] + b_pred_boxes[:, 3]) * 0.5

                gt_w = (gt_boxes_valid[:, 2] - gt_boxes_valid[:, 0]).clamp(min=1.0)
                gt_h = (gt_boxes_valid[:, 3] - gt_boxes_valid[:, 1]).clamp(min=1.0)

                # Tolerant center bounding: within [x1 - 0.2*w, x2 + 0.2*w]
                in_gt = (
                    (pred_cx[:, None] >= gt_boxes_valid[None, :, 0] - 0.2 * gt_w[None, :])
                    & (pred_cx[:, None] <= gt_boxes_valid[None, :, 2] + 0.2 * gt_w[None, :])
                    & (pred_cy[:, None] >= gt_boxes_valid[None, :, 1] - 0.2 * gt_h[None, :])
                    & (pred_cy[:, None] <= gt_boxes_valid[None, :, 3] + 0.2 * gt_h[None, :])
                )  # (N_pred, n_gt)

                # Cost = -IoU - 0.5 * cls cost (lower cost is better for matching)
                labels_clamped = gt_labels_valid.long().clamp(0, self.num_classes - 1)
                cls_probs = torch.sigmoid(b_pred_cls[:, labels_clamped].detach().float())

                cost_matrix = -cost_iou - 0.5 * cls_probs  # (N_pred, n_gt)
                cost_matrix = cost_matrix + (~in_gt).float() * 1000.0

                # Vectorized top-k per GT column in a single GPU call
                k = min(15, b_pred_boxes.shape[0])
                topk_vals, topk_idxs = cost_matrix.topk(k, dim=0, largest=False)  # (k, n_gt)
                topk_cand_list = topk_idxs.t().tolist()  # (n_gt, k)

                # Match up to top-3 anchors per GT box (standard multi-anchor matching)
                top_k_per_gt = 3
                matched_pred_indices = []
                matched_gt_indices = []
                used_preds = set()

                for g, candidates in enumerate(topk_cand_list):
                    matched_count = 0
                    for idx_item in candidates:
                        if idx_item not in used_preds:
                            # Prioritize candidates inside GT; fallback to at least 1 match if none inside
                            if in_gt[idx_item, g] or matched_count == 0:
                                matched_pred_indices.append(idx_item)
                                matched_gt_indices.append(g)
                                used_preds.add(idx_item)
                                matched_count += 1
                                if matched_count >= top_k_per_gt:
                                    break

            n_matched = len(matched_pred_indices)
            if n_matched == 0:
                with torch.amp.autocast("cuda", enabled=False):
                    obj_pred = b_pred_obj.float()
                    total_obj_loss = total_obj_loss + F.binary_cross_entropy_with_logits(
                        obj_pred,
                        torch.zeros_like(obj_pred),
                        reduction="mean"
                    )

                continue

            pred_idx = torch.tensor(matched_pred_indices, dtype=torch.long, device=device)
            gt_idx = torch.tensor(matched_gt_indices, dtype=torch.long, device=device)

            # === Bbox Loss (GIoU) ===
            matched_pred_boxes = torch.nan_to_num(b_pred_boxes[pred_idx].float(), nan=0.0, posinf=640.0, neginf=0.0)
            matched_gt_boxes = gt_boxes_valid[gt_idx].float()      # (n_matched, 4)
            giou = generalized_box_iou(matched_pred_boxes, matched_gt_boxes)
            giou = torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
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
            # Balanced foreground and background BCE in FP32 so foreground
            # gradients are not diluted by N_pred (100k or 25k).
            with torch.amp.autocast("cuda", enabled=False):
                obj_pred = b_pred_obj.float()
                pos_target = torch.ones_like(obj_pred[pred_idx])
                pos_loss = F.binary_cross_entropy_with_logits(
                    obj_pred[pred_idx],
                    pos_target,
                    reduction="mean"
                )
                neg_mask = torch.ones(b_pred_obj.shape[0], dtype=torch.bool, device=device)
                neg_mask[pred_idx] = False
                neg_loss = F.binary_cross_entropy_with_logits(
                    obj_pred[neg_mask],
                    torch.zeros_like(obj_pred[neg_mask]),
                    reduction="mean"
                )
                obj_loss = pos_loss + neg_loss
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
