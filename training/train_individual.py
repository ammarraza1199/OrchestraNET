"""
Individual Micro-Model Pre-training Script for OrchestraNet.

Pre-trains each micro-model on its dedicated task before joint fine-tuning.
This establishes strong per-task baselines and speeds up joint convergence.

Supported models:
  m1 — Object detection on COCO
  m2 — Occlusion prediction on COCOA/KINS
  m3 — Small object detection on VisDrone
  m4 — Depth estimation on NYU Depth V2
  m5 — Scene classification on Places365
  m6 — Amodal completion on COCOA
  m7 — Confidence calibration (post-hoc, no pre-training needed)

Usage:
  python training/train_individual.py --model m1 --data-root ./data/coco --epochs 50
  python training/train_individual.py --model m2 --data-root ./data/cocoa --epochs 30
"""

import argparse
from collections import defaultdict
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestranet.backbone import MobileNetV4Backbone, LightweightFPN
from orchestranet.models import (
    M1PrimaryDetector, M2OcclusionAnalyzer, M3SmallObjectEnhancer,
    M4DepthEstimator, M5SemanticContext, M6AmodalCompleter,
)
from orchestranet.data.datasets import COCODetectionDataset
from orchestranet.data.synthetic_occlusion import SyntheticOcclusionGenerator
from orchestranet.data.transforms import get_train_transforms, get_val_transforms
from orchestranet.losses.self_supervised_loss import (
    SelfSupervisedOcclusionLoss, OcclusionMaskGenerator,
)
from orchestranet.utils.config import Config
from orchestranet.utils.ema import ModelEMA
from orchestranet.utils.logger import TrainingLogger, AverageMeter
from orchestranet.utils.metrics import DetectionMetrics, compute_amodal_metrics
from orchestranet.evaluation.depth_metrics import DepthMetrics
from orchestranet.data.kitti_depth_dataset import KITTIDepthDataset


MODEL_REGISTRY = {
    "m1": {"cls": M1PrimaryDetector, "task": "Object Detection", "dataset": "COCO"},
    "m2": {"cls": M2OcclusionAnalyzer, "task": "Occlusion Prediction", "dataset": "COCOA"},
    "m3": {"cls": M3SmallObjectEnhancer, "task": "Small Object Enhancement", "dataset": "VisDrone"},
    "m4": {"cls": M4DepthEstimator, "task": "Depth Estimation", "dataset": "NYU Depth V2"},
    "m5": {"cls": M5SemanticContext, "task": "Scene Classification", "dataset": "Places365"},
    "m6": {"cls": M6AmodalCompleter, "task": "Amodal Completion", "dataset": "KINS"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="OrchestraNet Individual Model Training")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()),
                        help="Which model to train")
    parser.add_argument("--config", default="configs/training/individual_pretrain.yaml")
    parser.add_argument("--data-root", default="./data/coco")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="./checkpoints/individual")
    parser.add_argument("--log-dir", default="./logs/individual")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone during training")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--val-freq", type=int, default=1,
                        help="Frequency (in epochs) to run validation (default: 1)")
    parser.add_argument("--val-images", type=int, default=0,
                        help="Limit number of validation images (0 or unset = all validation images)")
    parser.add_argument("--val-num-images", type=int, default=None,
                        help="Alias for --val-images")
    parser.add_argument("--val-root", default=None,
                        help="Path to validation images directory (defaults to <data-root>/val2017)")
    parser.add_argument("--val-ann-file", default=None,
                        help="Path to validation annotations JSON (defaults to <data-root>/annotations/instances_val2017.json)")
    parser.add_argument("--dataset", default=None,
                        help="Explicit dataset name override (e.g., kitti, kins, coco)")
    parser.add_argument("--raw-root", default=None,
                        help="Explicit path to KITTI raw sequence images")
    parser.add_argument("--drive-save-dir", default=None,
                        help="Google Drive directory to synchronise checkpoints to")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run validation only on the checkpoint and exit")
    parser.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "none"],
                        help="AMP precision mode: 'bf16' (recommended for A100/Ampere), 'fp16', or 'none' (FP32)")
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Confidence threshold for detection evaluation (default: 0.25)")
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for NMS during evaluation (default: 0.5)")
    return parser.parse_args()


class IndividualTrainer:
    """
    Trains a single micro-model with shared backbone feature extraction.

    Architecture during individual training:
      Input → Backbone → FPN → [Single Micro-Model] → Task Loss
    """

    def __init__(self, model_id, device, fpn_channels=128, freeze_backbone=False):
        self.model_id = model_id
        self.device = device
        info = MODEL_REGISTRY[model_id]

        # Shared backbone + FPN
        self.backbone = MobileNetV4Backbone(pretrained=True).to(device)
        backbone_channels = self.backbone.get_out_channels()
        self.fpn = LightweightFPN(
            in_channels=backbone_channels, out_channels=fpn_channels
        ).to(device)

        # Target micro-model
        model_cls: Any = info["cls"]
        if model_id == "m6":
            self.model = model_cls(d_model=fpn_channels).to(device)
        else:
            self.model = model_cls(in_channels=fpn_channels).to(device)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("   Backbone frozen")

        # Collect all trainable parameters
        self.all_params = (
            list(self.backbone.parameters())
            + list(self.fpn.parameters())
            + list(self.model.parameters())
        )

        # For M2: self-supervised components
        self.ss_loss = None
        self.mask_gen = None
        if model_id == "m2":
            self.ss_loss = SelfSupervisedOcclusionLoss().to(device)
            self.mask_gen = OcclusionMaskGenerator().to(device)

    def train_step(self, images, targets):
        """Single training step."""
        images = images.to(self.device, non_blocking=True)

        # Feature extraction
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)

        # Model forward
        predictions = self.model(fpn_features)

        # Task loss (keep num_objects on CPU to avoid CUDA sync when slicing)
        targets_device = {
            k: v.to(self.device, non_blocking=True) if (isinstance(v, torch.Tensor) and k != "num_objects") else v
            for k, v in targets.items()
        }
        losses = self.model.get_loss(predictions, targets_device)

        # Self-supervised loss for M2
        if self.ss_loss is not None and self.mask_gen is not None:
            masked_feats, gt_mask = self.mask_gen(fpn_features[0])
            ss_pred = self.model([masked_feats] + fpn_features[1:])
            ss_losses = self.ss_loss(ss_pred["occlusion_map"], gt_mask)
            losses["ss_loss"] = ss_losses["total"]
            losses["total_loss"] = losses["total_loss"] + 0.5 * ss_losses["total"]

        return losses

    def count_parameters(self):
        total = sum(p.numel() for p in self.all_params)
        trainable = sum(p.numel() for p in self.all_params if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        fpn = sum(p.numel() for p in self.fpn.parameters())
        model = sum(p.numel() for p in self.model.parameters())
        return {
            "total": total, "trainable": trainable,
            "backbone": backbone, "fpn": fpn, "model_head": model,
        }


@torch.no_grad()
def validate_m1(
    trainer: IndividualTrainer,
    val_loader: DataLoader,
    device: str,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
    max_detections: int = 300,
    num_images: int | None = None,
    amp_dtype: str = "bf16",
) -> dict[str, Any]:
    """
    Evaluate M1 on COCO val2017 using DetectionMetrics with full diagnostics.

    Returns:
        dict containing:
          - mAP@50
          - mAP@50:95
          - AP_small
          - AP_medium
          - AP_large
          - diagnostics (gt_count, raw_preds, after_conf, after_nms, score_stats, box_ranges)
    """
    trainer.backbone.eval()
    trainer.fpn.eval()
    trainer.model.eval()

    metrics = DetectionMetrics(num_classes=80)
    count = 0

    total_gt = 0
    total_raw = 0
    total_conf = 0
    total_nms = 0

    all_raw_scores = []
    all_conf_scores = []
    raw_box_min = [float("inf")] * 4
    raw_box_max = [float("-inf")] * 4
    kept_box_min = [float("inf")] * 4
    kept_box_max = [float("-inf")] * 4

    for images, targets in val_loader:
        if num_images is not None and count >= num_images:
            break

        images = images.to(device, non_blocking=True)
        autocast_enabled = (device == "cuda" and amp_dtype != "none")
        autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype):
            backbone_feats = trainer.backbone(images)
            fpn_feats = trainer.fpn(backbone_feats)
            predictions = trainer.model(fpn_feats)

        pred_boxes_b = predictions["decoded_boxes"]
        pred_obj_b = torch.sigmoid(predictions["objectness"]).squeeze(-1)
        pred_cls_b = torch.sigmoid(predictions["class_logits"])

        B = images.shape[0]
        for b in range(B):
            img_id = count + b
            boxes = pred_boxes_b[b].clone()
            # Clamp box coordinates to 640x640 canvas boundaries
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, 640)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, 640)

            obj = pred_obj_b[b]
            cls_probs = pred_cls_b[b]
            max_cls, labels = cls_probs.max(dim=-1)
            scores = obj * max_cls

            total_raw += boxes.shape[0]

            # Collect raw score stats & raw box bounds
            if scores.numel() > 0:
                all_raw_scores.append(scores.detach().cpu())
                b_min = boxes.min(dim=0)[0].tolist()
                b_max = boxes.max(dim=0)[0].tolist()
                for i_coord in range(4):
                    raw_box_min[i_coord] = min(raw_box_min[i_coord], b_min[i_coord])
                    raw_box_max[i_coord] = max(raw_box_max[i_coord], b_max[i_coord])

            valid_box = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            mask = (scores > conf_thresh) & valid_box
            n_conf = int(mask.sum().item())
            total_conf += n_conf

            if n_conf == 0:
                pred_boxes, pred_scores, pred_labels = [], [], []
            else:
                f_boxes = boxes[mask]
                f_scores = scores[mask]
                f_labels = labels[mask]

                all_conf_scores.append(f_scores.detach().cpu())

                if f_boxes.shape[0] > 1000:
                    topk_idx = f_scores.topk(1000)[1]
                    f_boxes = f_boxes[topk_idx]
                    f_scores = f_scores[topk_idx]
                    f_labels = f_labels[topk_idx]

                keep = batched_nms(f_boxes, f_scores, f_labels, iou_thresh)
                if len(keep) > max_detections:
                    keep = keep[:max_detections]

                total_nms += len(keep)

                # Collect kept box range
                k_min = f_boxes[keep].min(dim=0)[0].tolist()
                k_max = f_boxes[keep].max(dim=0)[0].tolist()
                for i_coord in range(4):
                    kept_box_min[i_coord] = min(kept_box_min[i_coord], k_min[i_coord])
                    kept_box_max[i_coord] = max(kept_box_max[i_coord], k_max[i_coord])

                pred_boxes = f_boxes[keep].float().cpu().numpy()
                pred_scores = f_scores[keep].float().cpu().numpy()
                pred_labels = f_labels[keep].cpu().numpy()

            # Extract ground truth
            if "num_objects" in targets:
                n_gt = int(targets["num_objects"][b].item())
            else:
                valid_gt = (targets["boxes"][b][:, 2] - targets["boxes"][b][:, 0]) > 0
                n_gt = int(valid_gt.sum().item())

            total_gt += n_gt

            gt_boxes_t = targets["boxes"][b][:n_gt]
            gt_labels_t = targets["labels"][b][:n_gt]

            gt_boxes = gt_boxes_t.float().cpu().numpy() if isinstance(gt_boxes_t, torch.Tensor) else np.asarray(gt_boxes_t, dtype=np.float32)
            gt_labels = gt_labels_t.cpu().numpy() if isinstance(gt_labels_t, torch.Tensor) else np.asarray(gt_labels_t)

            metrics.update(
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                image_id=img_id,
            )

        count += B

    # Compute diagnostics
    if all_raw_scores:
        cat_raw = torch.cat(all_raw_scores)
        raw_min = float(cat_raw.min().item())
        raw_mean = float(cat_raw.mean().item())
        raw_max = float(cat_raw.max().item())
    else:
        raw_min = raw_mean = raw_max = 0.0

    if all_conf_scores:
        cat_conf = torch.cat(all_conf_scores)
        conf_min = float(cat_conf.min().item())
        conf_mean = float(cat_conf.mean().item())
        conf_max = float(cat_conf.max().item())
    else:
        conf_min = conf_mean = conf_max = 0.0

    raw_box_str = (
        f"[{raw_box_min[0]:.1f}, {raw_box_min[1]:.1f}, {raw_box_min[2]:.1f}, {raw_box_max[3]:.1f}]"
        if total_raw > 0 else "N/A"
    )
    kept_box_str = (
        f"[{kept_box_min[0]:.1f}, {kept_box_min[1]:.1f}, {kept_box_min[2]:.1f}, {kept_box_max[3]:.1f}]"
        if total_nms > 0 else "N/A (0 kept)"
    )

    print(f"\n  [M1 Validation Diagnostics - {count} images]")
    print(f"     - Ground Truth Boxes:             {total_gt}")
    print(f"     - Raw Predictions (Decoded):       {total_raw:,}")
    print(f"     - Predictions After Conf (> {conf_thresh:.2f}): {total_conf:,}")
    print(f"     - Predictions After NMS (Kept):    {total_nms:,}")
    print(f"     - Raw Score Stats:                min={raw_min:.4f}, mean={raw_mean:.4f}, max={raw_max:.4f}")
    if total_conf > 0:
        print(f"     - Conf-Filtered Score Stats:      min={conf_min:.4f}, mean={conf_mean:.4f}, max={conf_max:.4f}")
    print(f"     - Raw Box Coord Bounds:           {raw_box_str}")
    print(f"     - Kept Box Coord Bounds:          {kept_box_str}\n")

    results = metrics.compute()
    results["diagnostics"] = {
        "gt_count": total_gt,
        "raw_preds": total_raw,
        "after_conf": total_conf,
        "after_nms": total_nms,
        "raw_score_min": raw_min,
        "raw_score_mean": raw_mean,
        "raw_score_max": raw_max,
        "conf_score_min": conf_min,
        "conf_score_mean": conf_mean,
        "conf_score_max": conf_max,
        "raw_box_bounds": [raw_box_min, raw_box_max] if total_raw > 0 else None,
        "kept_box_bounds": [kept_box_min, kept_box_max] if total_nms > 0 else None,
    }
    return results


@torch.no_grad()
def validate_loss(
    trainer: IndividualTrainer,
    val_loader: DataLoader,
    device: str,
    num_images: int | None = None,
    amp_dtype: str = "bf16",
) -> dict[str, float]:
    """
    Evaluate validation loss for non-M1 micro-models (M2, M3, M4, M6).
    """
    trainer.backbone.eval()
    trainer.fpn.eval()
    trainer.model.eval()

    total_loss_sum = 0.0
    comp_sums = defaultdict(float)
    count = 0

    for images, targets in val_loader:
        if num_images is not None and count >= num_images:
            break

        B = images.shape[0]
        images = images.to(device, non_blocking=True)
        autocast_enabled = (device == "cuda" and amp_dtype != "none")
        autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype):
            losses = trainer.train_step(images, targets)

        total_loss_sum += losses["total_loss"].item() * B
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and k != "total_loss":
                comp_sums[k] += v.item() * B

        count += B

    avg_total = total_loss_sum / max(1, count)
    res = {"val_loss": avg_total}
    for k, v in comp_sums.items():
        res[f"val_{k}"] = v / max(1, count)
    return res


@torch.no_grad()
def validate_m4(
    trainer: IndividualTrainer,
    val_loader: DataLoader,
    device: str,
    num_images: int | None = None,
    amp_dtype: str = "bf16",
) -> dict[str, float]:
    """
    Evaluate validation loss and canonical depth metrics for M4:
      - val_loss, val_depth_loss, val_smoothness_loss
      - AbsRel, SqRel, RMSE, RMSElog, SILog, log10, d1, d2, d3
    """
    trainer.backbone.eval()
    trainer.fpn.eval()
    trainer.model.eval()

    total_loss_sum = 0.0
    comp_sums = defaultdict(float)
    count = 0
    depth_metrics = DepthMetrics(min_depth=1e-3, max_depth=80.0)

    for images, targets in val_loader:
        if num_images is not None and count >= num_images:
            break

        B = images.shape[0]
        images = images.to(device, non_blocking=True)
        targets_device = {
            k: v.to(device, non_blocking=True) if (isinstance(v, torch.Tensor) and k != "num_objects") else v
            for k, v in targets.items()
        }

        autocast_enabled = (device == "cuda" and amp_dtype != "none")
        autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype):
            backbone_features = trainer.backbone(images)
            fpn_features = trainer.fpn(backbone_features)
            predictions = trainer.model(fpn_features)
            losses = trainer.model.get_loss(predictions, targets_device)

        total_loss_sum += losses["total_loss"].item() * B
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and k != "total_loss":
                comp_sums[k] += v.item() * B

        # Compute depth metrics against metric ground truth
        if "depth_gt" in targets:
            pred_map = predictions["depth_map"]
            # Convert normalized pred to meters [0, 80.0]
            pred_meters = pred_map * 80.0
            if "depth_meters" in targets:
                gt_meters = targets["depth_meters"]
            else:
                gt_meters = targets["depth_gt"] * 80.0

            if pred_meters.shape[-2:] != gt_meters.shape[-2:]:
                pred_meters = F.interpolate(
                    pred_meters, size=gt_meters.shape[-2:],
                    mode="bilinear", align_corners=False
                )

            for b in range(B):
                depth_metrics.update(pred_meters[b, 0], gt_meters[b, 0])

        count += B

    avg_total = total_loss_sum / max(1, count)
    res = {"val_loss": avg_total}
    for k, v in comp_sums.items():
        res[f"val_{k}"] = v / max(1, count)

    metrics_dict = depth_metrics.compute()
    for m_k in ["AbsRel", "SqRel", "RMSE", "RMSElog", "SILog", "log10", "d1", "d2", "d3"]:
        v = metrics_dict.get(m_k)
        if isinstance(v, (int, float)):
            res[f"val_{m_k}"] = float(v)

    return res


def _check_finiteness(
    tensor: torch.Tensor,
    name: str,
    batch_idx: int,
    image_ids: list | None = None,
) -> None:
    """
    Check finiteness of a tensor during M6 evaluation.
    On non-finite tensor:
      - prints batch index
      - prints dataset image IDs if available
      - prints tensor name, shape, dtype, NaN count, Inf count, finite min/max/mean
      - raises descriptive RuntimeError stopping evaluation immediately.
    """
    if not torch.isfinite(tensor).all():
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        finite_mask = torch.isfinite(tensor)
        if finite_mask.any():
            fin_vals = tensor[finite_mask].float()
            fin_min = float(fin_vals.min().item())
            fin_max = float(fin_vals.max().item())
            fin_mean = float(fin_vals.mean().item())
            stats_str = f"min={fin_min:.6f}, max={fin_max:.6f}, mean={fin_mean:.6f}"
        else:
            stats_str = "no finite values"

        id_str = f", image_ids={image_ids}" if image_ids is not None else ""
        print("\n" + "!" * 70, flush=True)
        print("[NON-FINITE TENSOR DETECTED IN M6 EVALUATION]", flush=True)
        print(f"  Batch index:    {batch_idx}{id_str}", flush=True)
        print(f"  Tensor name:    {name}", flush=True)
        print(f"  Shape:          {list(tensor.shape)}", flush=True)
        print(f"  Dtype:          {tensor.dtype}", flush=True)
        print(f"  NaN count:      {nan_count}", flush=True)
        print(f"  Inf count:      {inf_count}", flush=True)
        print(f"  Finite stats:   {stats_str}", flush=True)
        print("!" * 70 + "\n", flush=True)

        raise RuntimeError(
            f"Non-finite tensor detected in M6 evaluation: '{name}' "
            f"at batch {batch_idx}{id_str} (NaNs={nan_count}, Infs={inf_count}, finite stats: {stats_str})"
        )


@torch.no_grad()
def validate_m6(
    trainer: IndividualTrainer,
    val_loader: DataLoader,
    device: str,
    num_images: int | None = None,
) -> dict[str, float]:
    """
    Evaluate validation loss and amodal completion metrics for M6:
      - val_loss: Total validation loss
      - val_amodal_mask_iou: Mean IoU between predicted 28x28 and GT amodal mask
      - val_amodal_bbox_mae: Mean Absolute Error between predicted and GT amodal bbox offsets
      - val_amodal_bbox_loss, val_amodal_mask_loss, val_amodal_conf_loss
    """
    trainer.backbone.eval()
    trainer.fpn.eval()
    trainer.model.eval()

    print("[INFO] M6 evaluation precision: FP32", flush=True)

    total_loss_sum = 0.0
    comp_sums = defaultdict(float)
    mask_ious = []
    bbox_maes = []
    count = 0

    for batch_idx, (images, targets) in enumerate(val_loader):
        if num_images is not None and count >= num_images:
            break

        B = images.shape[0]
        images = images.to(device, non_blocking=True)
        targets_device = {
            k: v.to(device, non_blocking=True) if (isinstance(v, torch.Tensor) and k != "num_objects") else v
            for k, v in targets.items()
        }

        # Extract image IDs if available from targets
        image_ids = None
        for id_k in ("image_id", "image_ids", "id", "img_id"):
            if id_k in targets:
                v = targets[id_k]
                image_ids = v.tolist() if isinstance(v, torch.Tensor) else list(v)
                break

        # Diagnostic FP32 forward path: CUDA autocast explicitly disabled
        with torch.amp.autocast("cuda", enabled=False):
            # 1. Backbone features
            backbone_features = trainer.backbone(images)
            for i_feat, feat in enumerate(backbone_features):
                _check_finiteness(feat, f"backbone_features[{i_feat}]", batch_idx, image_ids)

            # 2. FPN features
            fpn_features = trainer.fpn(backbone_features)
            for i_feat, feat in enumerate(fpn_features):
                _check_finiteness(feat, f"fpn_features[{i_feat}]", batch_idx, image_ids)

            # M6 forward
            predictions = trainer.model(fpn_features)

        # 3. M6 transformer output / decoded tensor
        if "decoded" in predictions and predictions["decoded"] is not None:
            _check_finiteness(predictions["decoded"], "M6 transformer output / decoded tensor", batch_idx, image_ids)

        # 4. amodal_bbox_offset
        _check_finiteness(predictions["amodal_bbox_offset"], "amodal_bbox_offset", batch_idx, image_ids)

        # 5. raw amodal_mask logits BEFORE sigmoid
        if "raw_mask_logits" in predictions and predictions["raw_mask_logits"] is not None:
            _check_finiteness(predictions["raw_mask_logits"], "raw amodal_mask logits BEFORE sigmoid", batch_idx, image_ids)

        # 6. amodal_masks AFTER sigmoid
        _check_finiteness(predictions["amodal_masks"], "amodal_masks AFTER sigmoid", batch_idx, image_ids)

        # 7. completion_confidence
        _check_finiteness(predictions["completion_confidence"], "completion_confidence", batch_idx, image_ids)

        losses = trainer.model.get_loss(predictions, targets_device)

        # Print compact runtime diagnostic on the first validation batch
        if count == 0:
            p_mask = predictions["amodal_masks"]
            p_conf = predictions["completion_confidence"]
            g_mask = targets.get("amodal_masks")
            g_occ = targets.get("is_occluded")
            g_bbox = targets.get("amodal_boxes")
            n_objs = targets.get("num_objects")

            print("\n" + "=" * 60)
            print("  [M6 RUNTIME EVALUATION DIAGNOSTIC - BATCH 0]")
            print("=" * 60)
            print("M6 MASK PRED:")
            print(f"  shape:    {list(p_mask.shape)}")
            print(f"  dtype:    {p_mask.dtype}")
            print(f"  min:      {p_mask.min().item():.6f}")
            print(f"  max:      {p_mask.max().item():.6f}")
            print(f"  mean:     {p_mask.mean().item():.6f}")
            print(f"  finite:   {bool(torch.isfinite(p_mask).all().item())}")
            print(f"  in_range: {bool((p_mask >= 0.0).all().item() and (p_mask <= 1.0).all().item())}")

            print("\nM6 CONF PRED:")
            print(f"  shape:    {list(p_conf.shape)}")
            print(f"  dtype:    {p_conf.dtype}")
            print(f"  min:      {p_conf.min().item():.6f}")
            print(f"  max:      {p_conf.max().item():.6f}")
            print(f"  mean:     {p_conf.mean().item():.6f}")
            print(f"  finite:   {bool(torch.isfinite(p_conf).all().item())}")
            print(f"  in_range: {bool((p_conf >= 0.0).all().item() and (p_conf <= 1.0).all().item())}")

            if g_mask is not None:
                gm = g_mask.float()
                print("\nKINS MASK GT:")
                print(f"  shape:    {list(g_mask.shape)}")
                print(f"  dtype:    {g_mask.dtype}")
                print(f"  min:      {gm.min().item():.6f}")
                print(f"  max:      {gm.max().item():.6f}")
                print(f"  mean:     {gm.mean().item():.6f}")
                print(f"  finite:   {bool(torch.isfinite(gm).all().item())}")
                print(f"  in_range: {bool((gm >= 0.0).all().item() and (gm <= 1.0).all().item())}")

            if g_occ is not None:
                go = g_occ.float()
                print("\nKINS OCC GT:")
                print(f"  shape:    {list(g_occ.shape)}")
                print(f"  dtype:    {g_occ.dtype}")
                print(f"  min:      {go.min().item():.6f}")
                print(f"  max:      {go.max().item():.6f}")
                print(f"  mean:     {go.mean().item():.6f}")
                print(f"  finite:   {bool(torch.isfinite(go).all().item())}")
                print(f"  in_range: {bool((go >= 0.0).all().item() and (go <= 1.0).all().item())}")

            if g_bbox is not None:
                gb = g_bbox.float()
                print("\nKINS BBOX GT:")
                print(f"  shape:    {list(g_bbox.shape)}")
                print(f"  dtype:    {g_bbox.dtype}")
                print(f"  min:      {gb.min().item():.6f}")
                print(f"  max:      {gb.max().item():.6f}")
                print(f"  finite:   {bool(torch.isfinite(gb).all().item())}")

            if n_objs is not None:
                n_objs_f = n_objs.float()
                n_valid = int(n_objs.clamp(max=50).sum().item())
                n_padded = int((50 - n_objs.clamp(max=50)).clamp(min=0).sum().item())
                print(f"\nnum_objects: min={int(n_objs.min().item())}, max={int(n_objs.max().item())}, mean={n_objs_f.mean().item():.2f}")
                print(f"valid objects evaluated: {n_valid}")
                print(f"padded objects ignored:  {n_padded}")
            print("=" * 60 + "\n")

        total_loss_sum += losses["total_loss"].item() * B
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and k != "total_loss":
                comp_sums[k] += v.item() * B

        # Compute Amodal Mask IoU & Bbox MAE on valid GT objects
        metrics = compute_amodal_metrics(predictions, targets)
        mask_ious.extend(metrics["ious"])
        bbox_maes.extend(metrics["maes"])

        count += B

    avg_total = total_loss_sum / max(1, count)
    res = {"val_loss": avg_total}
    for k, v in comp_sums.items():
        res[f"val_{k}"] = v / max(1, count)

    res["val_amodal_mask_iou"] = float(np.mean(mask_ious)) if mask_ious else 0.0
    res["val_amodal_bbox_mae"] = float(np.mean(bbox_maes)) if bbox_maes else 0.0
    return res


def save_checkpoint_atomic(ckpt_data: dict, target_path: str | Path) -> Path:
    """
    Save checkpoint atomically:
      1. Write to target_path.tmp
      2. Flush and fsync
      3. Atomically rename to target_path
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")

    with open(tmp_path, "wb") as f:
        torch.save(ckpt_data, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, IOError):
            pass

    os.replace(tmp_path, target_path)
    return target_path


def sync_file_to_drive(
    src_path: str | Path,
    drive_dir: str | Path,
    logger: TrainingLogger | None = None,
) -> str | None:
    """
    Safely copy/sync a file to Google Drive using atomic write:
      1. Copy to drive_dest.tmp
      2. Flush and fsync if practical
      3. Atomically rename to drive_dest
      4. Return drive_dest path on success, or None on failure without raising.
    """
    if not drive_dir:
        return None
    try:
        src_path = Path(src_path)
        drive_dir = Path(drive_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)

        dest_path = drive_dir / src_path.name
        tmp_dest = drive_dir / f"{src_path.name}.tmp"

        shutil.copyfile(src_path, tmp_dest)

        try:
            with open(tmp_dest, "a+b") as f:
                f.flush()
                os.fsync(f.fileno())
        except (OSError, IOError):
            pass

        os.replace(tmp_dest, dest_path)
        if logger:
            logger.info(f"  ☁️ Drive sync complete: {dest_path}")
        return str(dest_path)
    except Exception as e:
        msg = f"Drive synchronisation failed for {src_path} -> {drive_dir}: {type(e).__name__}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(f"[ERROR] {msg}", file=sys.stderr)
        return None


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Load config (merge with CLI args)
    cfg = Config.from_yaml(args.config).merge_args(args)

    info = MODEL_REGISTRY[args.model]

    # Setup logger
    log_dir = os.path.join(args.log_dir, args.model)
    logger = TrainingLogger(log_dir=log_dir, tb_enabled=True)

    is_kins = (args.dataset == "kins") or ("kins" in args.data_root.lower())
    is_kitti = (args.dataset == "kitti") or (args.model == "m4" and "kitti" in args.data_root.lower())

    if is_kitti:
        dataset_name = "KITTI"
    elif is_kins:
        dataset_name = "KINS"
    else:
        dataset_name = info["dataset"]

    logger.info("🎼 OrchestraNet — Individual Model Training")
    logger.info("=" * 60)
    logger.info(f"   Model:   {args.model} ({info['task']})")
    logger.info(f"   Dataset: {dataset_name}")
    logger.info(f"   Device:  {args.device}")
    if args.model == "m6" and not is_kins:
        logger.warning(
            "⚠️  M6 requires amodal ground truth (e.g. KINS). "
            "Dataset does not appear to be KINS, so amodal loss may evaluate to zero."
        )

    # Build trainer
    trainer = IndividualTrainer(
        args.model, args.device,
        freeze_backbone=args.freeze_backbone,
    )
    params = trainer.count_parameters()
    logger.info(f"\n📊 Parameters:")
    logger.info(f"   Backbone:   {params['backbone']:>10,}")
    logger.info(f"   FPN:        {params['fpn']:>10,}")
    logger.info(f"   Model Head: {params['model_head']:>10,}")
    logger.info(f"   Trainable:  {params['trainable']:>10,}")

    # Dataset
    occ_aug = SyntheticOcclusionGenerator() if args.model == "m2" else None
    train_transforms = get_train_transforms(img_size=640)
    if is_kitti:
        dataset = KITTIDepthDataset(
            root=args.data_root,
            split="train",
            img_size=640,
            raw_root=args.raw_root,
            transforms=train_transforms,
        )
        if len(dataset) == 0:
            logger.error(
                "\n" + "!" * 70 + "\n"
                "❌ No KITTI training RGB/depth pairs found.\n"
                f"Scanned: {args.data_root}/train/*/proj_depth/groundtruth/\n"
                "The KITTI depth benchmark contains depth files, but raw RGB sequence images\n"
                "must be downloaded/extracted separately (or specified via --raw-root).\n"
                "!" * 70 + "\n"
            )
            sys.exit(1)
    elif is_kins:
        train_ann_candidates = [
            os.path.join(args.data_root, "update_train_2020.json"),
            os.path.join(args.data_root, "annotations", "update_train_2020.json"),
            os.path.join(args.data_root, "instances_train.json"),
        ]
        ann_path = next((p for p in train_ann_candidates if os.path.exists(p)), train_ann_candidates[0])
        train_root_candidates = [
            os.path.join(args.data_root, "training/image_2"),
            os.path.join(args.data_root, "image_2"),
            args.data_root,
        ]
        root_path = next((p for p in train_root_candidates if os.path.exists(p)), train_root_candidates[0])
        from orchestranet.data.kins_dataset import KINSAmodalDataset
        dataset = KINSAmodalDataset(
            root=root_path,
            ann_file=ann_path,
            transforms=train_transforms,
        )
    else:
        root_path = os.path.join(args.data_root, "train2017")
        ann_path = os.path.join(args.data_root, "annotations/instances_train2017.json")
        dataset = COCODetectionDataset(
            root=root_path,
            ann_file=ann_path,
            transforms=train_transforms,
            occlusion_aug=occ_aug,
        )
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(dataset, **train_loader_kwargs)

    # Validation DataLoader
    val_loader = None
    val_images_limit = None
    if getattr(args, "val_images", 0) and args.val_images > 0:
        val_images_limit = args.val_images
    elif getattr(args, "val_num_images", None) and args.val_num_images > 0:
        val_images_limit = args.val_num_images

    if is_kins:
        val_ann_candidates = [
            os.path.join(args.data_root, "update_test_2020.json"),
            os.path.join(args.data_root, "annotations", "update_test_2020.json"),
            os.path.join(args.data_root, "instances_val.json"),
        ]
        default_val_ann = next((p for p in val_ann_candidates if os.path.exists(p)), val_ann_candidates[0])
        val_root_candidates = [
            os.path.join(args.data_root, "testing/image_2"),
            os.path.join(args.data_root, "image_2"),
            args.data_root,
        ]
        default_val_root = next((p for p in val_root_candidates if os.path.exists(p)), val_root_candidates[0])
    else:
        default_val_root = os.path.join(args.data_root, "val2017")
        default_val_ann = os.path.join(args.data_root, "annotations/instances_val2017.json")

    val_root = args.val_root if args.val_root else default_val_root
    val_ann = args.val_ann_file if args.val_ann_file else default_val_ann
    val_transforms = get_val_transforms(img_size=640)

    if is_kitti:
        val_dataset = KITTIDepthDataset(
            root=args.data_root,
            split="val",
            img_size=640,
            transforms=val_transforms,
        )
        if len(val_dataset) == 0:
            logger.warning(
                f"⚠️  No KITTI validation pairs found in {args.data_root}. "
                "Validation will be skipped."
            )
        else:
            val_loader_kwargs = {
                "batch_size": args.batch_size,
                "shuffle": False,
                "num_workers": args.num_workers,
                "pin_memory": True,
            }
            if args.num_workers > 0:
                val_loader_kwargs["persistent_workers"] = True
                val_loader_kwargs["prefetch_factor"] = 2
            val_loader = DataLoader(val_dataset, **val_loader_kwargs)
            logger.info(f"   Validation: {len(val_dataset)} images from {args.data_root}")
            if val_images_limit:
                logger.info(f"   Validation image cap: {val_images_limit}")
    elif os.path.exists(val_root) and os.path.exists(val_ann):
        if is_kins:
            from orchestranet.data.kins_dataset import KINSAmodalDataset
            val_dataset = KINSAmodalDataset(
                root=val_root,
                ann_file=val_ann,
                transforms=val_transforms,
            )
        else:
            val_dataset = COCODetectionDataset(
                root=val_root,
                ann_file=val_ann,
                transforms=val_transforms,
            )
        val_loader_kwargs = {
            "batch_size": 1 if args.model == "m1" else args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": True,
        }
        if args.num_workers > 0:
            val_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 2

        val_loader = DataLoader(val_dataset, **val_loader_kwargs)
        logger.info(f"   Validation: {len(val_dataset)} images from {val_root}")
        if val_images_limit:
            logger.info(f"   Validation image cap: {val_images_limit}")
    else:
        logger.warning(
            f"⚠️  Validation dataset not found ({val_root} or {val_ann}). "
            "Validation will be skipped."
        )

    # Optimizer
    trainable_params = [p for p in trainer.all_params if p.requires_grad]
    named_trainable_params = []
    for prefix, module in [("backbone", trainer.backbone), ("fpn", trainer.fpn), ("model", trainer.model)]:
        for name, p in module.named_parameters():
            if p.requires_grad:
                named_trainable_params.append((f"{prefix}.{name}", p))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Scheduler with warmup
    warmup_steps = args.warmup_epochs * len(loader)
    cosine_steps = (args.epochs - args.warmup_epochs) * len(loader)

    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")

    if args.warmup_epochs >= args.epochs:
        raise ValueError(
            "warmup_epochs must be smaller than epochs "
            "because cosine annealing requires at least one epoch."
        )

    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_steps,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_steps),
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_steps),
        )

    # Mixed precision
    use_scaler = (args.device == "cuda" and args.amp_dtype == "fp16")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
        init_scale=1024.0,
    )

    if args.device == "cuda":
        if args.amp_dtype == "bf16":
            logger.info("   AMP Mode: BFloat16 (BF16) on CUDA (GradScaler disabled per standard design)")
        elif args.amp_dtype == "fp16":
            logger.info("   AMP Mode: Float16 (FP16) on CUDA (GradScaler enabled, init_scale=1024.0)")
        else:
            logger.info("   AMP Mode: Disabled (Full FP32)")
    else:
        logger.info("   Device: CPU (Full FP32)")

    # EMA
    ema = None
    if args.use_ema:
        ema = ModelEMA(trainer.model)

    # Resume
    start_epoch = 0
    best_loss = float("inf")
    best_val_loss = float("inf")
    best_map50 = 0.0
    global_step = 0
    if args.resume:
        # If Drive path was supplied for resume and drive_save_dir was not explicitly set, infer it
        if args.drive_save_dir is None and "drive" in str(args.resume).lower():
            args.drive_save_dir = str(Path(args.resume).parent)
            logger.info(f"   Inferred --drive-save-dir from --resume: {args.drive_save_dir}")

        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)

        trainer.model.load_state_dict(ckpt["model_state_dict"])

        if "backbone_state_dict" in ckpt:
            trainer.backbone.load_state_dict(ckpt["backbone_state_dict"])

        if "fpn_state_dict" in ckpt:
            trainer.fpn.load_state_dict(ckpt["fpn_state_dict"])

        # Evaluation-only does not need optimizer/scheduler state.
        if not getattr(args, "eval_only", False):
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            else:
                logger.warning("⚠️  'optimizer_state_dict' not found in checkpoint; optimizer initialized freshly.")

            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                logger.warning("⚠️  'scheduler_state_dict' not found in checkpoint; LR schedule will not match previous state.")

            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            else:
                logger.warning("⚠️  'scaler_state_dict' not found in checkpoint; GradScaler initialized freshly.")

            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("best_loss", float("inf"))
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_map50 = ckpt.get("best_map50", 0.0)
            global_step = ckpt.get("global_step", start_epoch * len(loader))

            if ema and "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info("   " + "=" * 50)
            logger.info(f"   Resume checkpoint: {args.resume}")
            logger.info(f"   Epoch: {ckpt.get('epoch', 0)} (continuing from epoch {start_epoch})")
            logger.info(f"   Best mAP50: {best_map50:.4f}")
            logger.info(f"   Best loss: {best_loss:.4f}")
            logger.info(f"   Global step: {global_step}")
            logger.info(f"   Current LR: {current_lr:.6e}")
            logger.info("   " + "=" * 50)
        else:
            if ema and "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])

            logger.info("   Loaded checkpoint weights for evaluation-only mode.")
    else:
        global_step = start_epoch * len(loader)

    # Evaluation only mode
    if getattr(args, "eval_only", False):
        if val_loader is None:
            logger.error("❌ Validation loader could not be created. Check --val-root and --val-ann-file / --data-root.")
            sys.exit(1)
        logger.info(f"\n🧪 Running evaluation only for {args.model.upper()}...")
        if ema:
            ema.apply_shadow(trainer.model)
        if args.model == "m1":
            val_metrics = validate_m1(
                trainer,
                val_loader,
                device=args.device,
                num_images=val_images_limit,
                amp_dtype=args.amp_dtype,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh,
            )
        elif args.model == "m6":
            val_metrics = validate_m6(
                trainer,
                val_loader,
                device=args.device,
                num_images=val_images_limit,
            )
        elif args.model == "m4":
            val_metrics = validate_m4(
                trainer,
                val_loader,
                device=args.device,
                num_images=val_images_limit,
                amp_dtype=args.amp_dtype,
            )
        else:
            val_metrics = validate_loss(
                trainer,
                val_loader,
                device=args.device,
                num_images=val_images_limit,
                amp_dtype=args.amp_dtype,
            )
        if ema:
            ema.restore(trainer.model)

        print(f"\n=== {args.model.upper()} Evaluation Results ===")
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
        return

    logger.info(f"\n🚀 Training {args.model} for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        trainer.backbone.train()
        trainer.fpn.train()
        trainer.model.train()

        loss_meter = AverageMeter("loss")
        component_meters = {}
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(loader):
            optimizer.zero_grad()

            autocast_enabled = (args.device == "cuda" and args.amp_dtype != "none")
            autocast_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
            with torch.amp.autocast("cuda", enabled=autocast_enabled, dtype=autocast_dtype):
                losses = trainer.train_step(images, targets)

            # Check total_loss with torch.isfinite() before backward
            total_loss = losses["total_loss"]
            if not torch.isfinite(total_loss):
                current_lr = optimizer.param_groups[0]["lr"]
                loss_comp_str = " | ".join(
                    f"{k}: {v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                    for k, v in losses.items()
                )
                logger.error(
                    f"\n{'!'*70}\n"
                    f"❌ NON-FINITE LOSS DETECTED at Epoch {epoch}, Batch {batch_idx}/{len(loader)}\n"
                    f"   LR: {current_lr:.6e}\n"
                    f"   Losses: {loss_comp_str}\n"
                    f"{'!'*70}\n"
                )
                raise RuntimeError(
                    f"Non-finite total_loss ({total_loss.item()}) at epoch {epoch}, batch {batch_idx}. "
                    f"Stopping training to prevent parameter corruption."
                )

            if use_scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
            else:
                total_loss.backward()

            # Check all gradients for finite values before clipping
            grads_finite = True
            first_bad_grad = None
            for name, p in named_trainable_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_finite = False
                    first_bad_grad = name
                    break

            if not grads_finite:
                current_lr = optimizer.param_groups[0]["lr"]
                loss_comp_str = " | ".join(
                    f"{k}: {v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                    for k, v in losses.items()
                )
                logger.error(
                    f"\n{'!'*70}\n"
                    f"⚠️  NON-FINITE GRADIENT DETECTED at Epoch {epoch}, Batch {batch_idx}/{len(loader)}\n"
                    f"   First offending parameter: {first_bad_grad}\n"
                    f"   LR: {current_lr:.6e}\n"
                    f"   Losses: {loss_comp_str}\n"
                    f"   Skipping optimizer.step() to protect model parameters.\n"
                    f"{'!'*70}\n"
                )
                optimizer.zero_grad()
                if use_scaler:
                    scaler.update()
                continue

            try:
                nn.utils.clip_grad_norm_(trainable_params, 10.0, error_if_nonfinite=True)
            except TypeError:
                nn.utils.clip_grad_norm_(trainable_params, 10.0)

            # Ensure optimizer.step() occurs before scheduler.step()
            if use_scaler:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                # Only advance scheduler if optimizer was actually stepped (not skipped by scaler)
                if not (scale_after < scale_before):
                    scheduler.step()
            else:
                optimizer.step()
                scheduler.step()

            # Lightweight finite-parameter check after optimizer.step()
            for name, p in named_trainable_params:
                if not torch.isfinite(p).all():
                    raise RuntimeError(
                        f"Non-finite parameter detected in '{name}' after optimizer.step() "
                        f"at epoch {epoch}, batch {batch_idx}. Halting training."
                    )

            # Update EMA
            if ema:
                ema.update(trainer.model)

            loss_val = losses["total_loss"].item()
            loss_meter.update(loss_val, images.shape[0])

            # Track individual loss components for epoch-level reporting.
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    if k not in component_meters:
                        component_meters[k] = AverageMeter(k)
                    component_meters[k].update(v.item(), images.shape[0])

            global_step += 1

            # Log to TensorBoard
            logger.log_scalar(f"train/{args.model}/total_loss", loss_val, global_step)
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and k != "total_loss":
                    logger.log_scalar(f"train/{args.model}/{k}", v.item(), global_step)
            logger.log_lr(optimizer.param_groups[0]["lr"], global_step)

            if batch_idx % args.print_freq == 0:
                loss_strs = " | ".join(
                    f"{k}: {v.item():.4f}" for k, v in losses.items()
                    if isinstance(v, torch.Tensor)
                )
                print(f"  [{batch_idx}/{len(loader)}] {loss_strs}")

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch}/{args.epochs} | Avg Loss: {loss_meter.avg:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        # ============================================================
        # Persistent epoch-level metrics
        # ============================================================
        # Keep a complete machine-readable record in training_log.jsonl.
        # Validation metrics are added below after validation completes.
        epoch_log = {
            "epoch": epoch,
            "avg_loss": float(loss_meter.avg),
            "lr": float(current_lr),
            "epoch_time_sec": float(epoch_time),
        }

        for k, meter in component_meters.items():
            epoch_log[k] = float(meter.avg)

        # Validation
        val_metrics = None
        is_best_map50 = False
        is_best_val_loss = False
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            if ema:
                ema.apply_shadow(trainer.model)

            if args.model == "m1":
                logger.info(f"  🔍 Validating M1 on COCO val2017...")
                val_metrics = validate_m1(
                    trainer,
                    val_loader,
                    device=args.device,
                    num_images=val_images_limit,
                    amp_dtype=args.amp_dtype,
                    conf_thresh=args.conf_thresh,
                    iou_thresh=args.iou_thresh,
                )
                diag = val_metrics.get("diagnostics", {})
                logger.info(
                    f"  📊 Val M1 | "
                    f"mAP@50: {val_metrics['mAP@50']:.4f} | "
                    f"mAP@50:95: {val_metrics['mAP@50:95']:.4f} | "
                    f"AP_s: {val_metrics['AP_small']:.4f} | "
                    f"AP_m: {val_metrics['AP_medium']:.4f} | "
                    f"AP_l: {val_metrics['AP_large']:.4f}"
                )
                if diag:
                    logger.info(
                        f"  🔬 Diagnostics: GT={diag.get('gt_count', 0)} | "
                        f"RawPreds={diag.get('raw_preds', 0):,} | "
                        f"AfterConf={diag.get('after_conf', 0):,} | "
                        f"AfterNMS={diag.get('after_nms', 0):,} | "
                        f"Score[min={diag.get('raw_score_min', 0.0):.4f}, mean={diag.get('raw_score_mean', 0.0):.4f}, max={diag.get('raw_score_max', 0.0):.4f}]"
                    )
                for m_k, m_v in val_metrics.items():
                    if isinstance(m_v, (int, float)):
                        logger.log_scalar(f"val/{args.model}/{m_k}", m_v, epoch)

                current_map50 = val_metrics["mAP@50"]
                if current_map50 > best_map50:
                    best_map50 = current_map50
                    is_best_map50 = True
            elif args.model == "m6":
                logger.info(f"  🔍 Validating M6 on validation set...")
                val_metrics = validate_m6(
                    trainer,
                    val_loader,
                    device=args.device,
                    num_images=val_images_limit,
                )
                loss_comp_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                )
                logger.info(f"  📊 Val M6 | {loss_comp_str}")
                for m_k, m_v in val_metrics.items():
                    logger.log_scalar(f"val/{args.model}/{m_k}", m_v, epoch)

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    is_best_val_loss = True
            elif args.model == "m4":
                logger.info(f"  🔍 Validating M4 on validation set...")
                val_metrics = validate_m4(
                    trainer,
                    val_loader,
                    device=args.device,
                    num_images=val_images_limit,
                    amp_dtype=args.amp_dtype,
                )
                loss_comp_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                )
                logger.info(f"  📊 Val M4 | {loss_comp_str}")
                for m_k, m_v in val_metrics.items():
                    logger.log_scalar(f"val/{args.model}/{m_k}", m_v, epoch)

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    is_best_val_loss = True
            else:
                logger.info(f"  🔍 Validating {args.model.upper()} on validation set...")
                val_metrics = validate_loss(
                    trainer,
                    val_loader,
                    device=args.device,
                    num_images=val_images_limit,
                    amp_dtype=args.amp_dtype,
                )
                loss_comp_str = " | ".join(
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                )
                logger.info(f"  📊 Val {args.model.upper()} | {loss_comp_str}")
                for m_k, m_v in val_metrics.items():
                    logger.log_scalar(f"val/{args.model}/{m_k}", m_v, epoch)

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    is_best_val_loss = True

            if ema:
                ema.restore(trainer.model)

        # ============================================================
        # Persist complete epoch record after validation
        # ============================================================
        if val_metrics is not None:
            for key, value in val_metrics.items():
                if key != "diagnostics":
                    if isinstance(value, (int, float)):
                        epoch_log[key] = float(value)
                    elif value is None:
                        epoch_log[key] = None

            # M1 diagnostics are nested inside val_metrics.
            if args.model == "m1":
                diagnostics = val_metrics.get("diagnostics", {})
                for diag_k in [
                    "gt_count", "raw_preds", "after_conf", "after_nms",
                    "raw_score_min", "raw_score_mean", "raw_score_max",
                    "conf_score_min", "conf_score_mean", "conf_score_max",
                    "raw_box_bounds", "kept_box_bounds",
                ]:
                    val = diagnostics.get(diag_k, None)
                    epoch_log[diag_k] = val
                    epoch_log[f"diagnostics_{diag_k}"] = val

                epoch_log["validation_images"] = int(val_images_limit or 0)
        else:
            if args.model == "m1":
                for diag_k in [
                    "gt_count", "raw_preds", "after_conf", "after_nms",
                    "raw_score_min", "raw_score_mean", "raw_score_max",
                    "conf_score_min", "conf_score_mean", "conf_score_max",
                    "raw_box_bounds", "kept_box_bounds",
                ]:
                    epoch_log[diag_k] = None
                    epoch_log[f"diagnostics_{diag_k}"] = None
                epoch_log["validation_images"] = None

        # Track lowest-loss checkpoint state
        is_best_loss = loss_meter.avg < best_loss
        best_loss = min(best_loss, loss_meter.avg)

        # Base checkpoint dictionary
        ckpt_data = {
            "epoch": epoch,
            "model_id": args.model,
            "model_state_dict": trainer.model.state_dict(),
            "backbone_state_dict": trainer.backbone.state_dict(),
            "fpn_state_dict": trainer.fpn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "avg_loss": float(loss_meter.avg),
            "best_loss": float(best_loss),
            "best_val_loss": float(best_val_loss),
            "best_map50": float(best_map50),
            "global_step": int(global_step),
            "args": vars(args),
        }
        if val_metrics is not None:
            ckpt_data["val_metrics"] = val_metrics
        if ema:
            ckpt_data["ema_state_dict"] = ema.state_dict()

        # Checkpoint protection: verify model parameters and optimizer state are finite
        def _check_dict_finite(d: dict, prefix: str):
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    if not torch.isfinite(v).all():
                        return f"{prefix}['{k}']"
                elif isinstance(v, dict):
                    res = _check_dict_finite(v, f"{prefix}['{k}']")
                    if res is not None:
                        return res
            return None

        bad_tensor_source = None
        for name, state in [
            ("model_state_dict", trainer.model.state_dict()),
            ("backbone_state_dict", trainer.backbone.state_dict()),
            ("fpn_state_dict", trainer.fpn.state_dict()),
            ("optimizer_state_dict", optimizer.state_dict()),
        ]:
            bad_tensor_source = _check_dict_finite(state, name)
            if bad_tensor_source is not None:
                break

        if bad_tensor_source is not None or not np.isfinite(loss_meter.avg):
            logger.error(
                f"\n{'!'*70}\n"
                f"❌ REFUSING TO SAVE CHECKPOINT at Epoch {epoch}!\n"
                f"   Non-finite values detected in: {bad_tensor_source or 'loss_meter.avg'}\n"
                f"   Preserving previous healthy checkpoints without overwriting.\n"
                f"{'!'*70}\n"
            )
            raise RuntimeError(
                f"Refusing to save checkpoint with non-finite values ({bad_tensor_source or 'loss_meter.avg'}) "
                f"at epoch {epoch}."
            )

        # 1. Save epoch checkpoint after EVERY completed epoch
        epoch_ckpt_name = f"{args.model}_epoch{epoch:03d}.pt"
        epoch_ckpt_path = Path(args.save_dir) / epoch_ckpt_name
        save_checkpoint_atomic(ckpt_data, epoch_ckpt_path)
        logger.info(f"  💾 Saved checkpoint: {epoch_ckpt_path}")

        drive_epoch_path = None
        if args.drive_save_dir:
            drive_epoch_path = sync_file_to_drive(epoch_ckpt_path, args.drive_save_dir, logger=logger)

        # 2. Update latest checkpoint atomically
        latest_ckpt_path = Path(args.save_dir) / f"{args.model}_latest.pt"
        save_checkpoint_atomic(ckpt_data, latest_ckpt_path)
        if args.drive_save_dir:
            sync_file_to_drive(latest_ckpt_path, args.drive_save_dir, logger=logger)

        # 3. Save best validation mAP@50 checkpoint if applicable
        if is_best_map50:
            best_map_path = Path(args.save_dir) / f"{args.model}_best_map50.pt"
            save_checkpoint_atomic(ckpt_data, best_map_path)
            logger.info(f"  🏆 New best validation mAP@50 ({best_map50:.4f}): {best_map_path}")
            if args.drive_save_dir:
                sync_file_to_drive(best_map_path, args.drive_save_dir, logger=logger)

        # 4. Save best validation loss checkpoint if applicable
        if is_best_val_loss:
            best_val_path = Path(args.save_dir) / f"{args.model}_best_val.pt"
            save_checkpoint_atomic(ckpt_data, best_val_path)
            logger.info(f"  🏆 New lowest validation loss ({best_val_loss:.4f}): {best_val_path}")
            if args.drive_save_dir:
                sync_file_to_drive(best_val_path, args.drive_save_dir, logger=logger)

        # 5. Save best training loss checkpoint if applicable
        if is_best_loss:
            best_loss_path = Path(args.save_dir) / f"{args.model}_best.pt"
            save_checkpoint_atomic(ckpt_data, best_loss_path)
            if args.model == "m1":
                logger.info(f"  📉 New lowest training loss checkpoint ({best_loss:.4f}): {best_loss_path}")
            else:
                logger.info(f"  🏆 New lowest training loss checkpoint ({best_loss:.4f}): {best_loss_path}")
            if args.drive_save_dir:
                sync_file_to_drive(best_loss_path, args.drive_save_dir, logger=logger)

        # 6. Record checkpoint paths and commit epoch record to JSONL
        epoch_log["checkpoint_path"] = str(epoch_ckpt_path)
        epoch_log["drive_checkpoint_path"] = drive_epoch_path
        epoch_log["best_map50"] = float(best_map50)
        epoch_log["best_loss"] = float(best_loss)
        logger.log_epoch(epoch, epoch_log)

    logger.flush()
    logger.close()
    logger.info(f"\n✅ {args.model} pre-training complete!")


if __name__ == "__main__":
    main()
