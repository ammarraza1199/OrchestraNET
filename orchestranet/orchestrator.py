"""
OrchestraNet Orchestrator — The Brain.

The main pipeline that coordinates backbone → router → micro-models → fusion → OA-NMS.
This is the entry point for both training and inference.

Architecture Flow:
  1. Input image → Shared Backbone → Multi-scale FPN features
  2. FPN features → Adaptive Router → Routing decision
  3. Routing decision → Activate selected micro-models (parallel)
  4. Model outputs → Cross-Attention Fusion → Merged detections
  5. Merged detections → OA-NMS → Final output with occlusion info
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import torch.nn as nn

from .backbone import MobileNetV4Backbone, LightweightFPN
from .models import (
    M1PrimaryDetector, M2OcclusionAnalyzer, M3SmallObjectEnhancer,
    M4DepthEstimator, M5SemanticContext, M6AmodalCompleter, M7ConfidenceCalibrator,
)
from .router import AdaptiveRouter
from .fusion import CrossAttentionFusion, occlusion_aware_nms


class OrchestraNet(nn.Module):
    """
    OrchestraNet: Multi-Model Orchestrated Detection Framework.

    Coordinates 7 specialized micro-models through an adaptive routing
    system for optimal speed-accuracy tradeoff on occluded and small objects.

    Args:
        num_classes: Number of object detection classes.
        backbone_name: timm model name for backbone.
        fpn_channels: Unified FPN channel dimension.
        pretrained_backbone: Whether to load pretrained backbone weights.
        router_thresholds: (low, high) thresholds for routing complexity.
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_name: str = "mobilenetv4_hybrid_medium",
        fpn_channels: int = 128,
        pretrained_backbone: bool = True,
        router_thresholds: tuple[float, float] = (0.3, 0.7),
    ):
        super().__init__()
        self.num_classes = num_classes

        # === Shared Backbone ===
        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            pretrained=pretrained_backbone,
        )
        backbone_channels = self.backbone.get_out_channels()

        # === Feature Pyramid Network ===
        self.fpn = LightweightFPN(
            in_channels=backbone_channels,
            out_channels=fpn_channels,
            use_depthwise=True,
        )

        # === Adaptive Router ===
        self.router = AdaptiveRouter(
            in_channels=fpn_channels,
            thresholds=router_thresholds,
        )

        # === Specialized Micro-Models ===
        self.models = nn.ModuleDict({
            "m1": M1PrimaryDetector(in_channels=fpn_channels, num_classes=num_classes),
            "m2": M2OcclusionAnalyzer(in_channels=fpn_channels),
            "m3": M3SmallObjectEnhancer(in_channels=fpn_channels),
            "m4": M4DepthEstimator(in_channels=fpn_channels),
            "m5": M5SemanticContext(in_channels=fpn_channels),
            "m6": M6AmodalCompleter(d_model=fpn_channels),
            "m7": M7ConfidenceCalibrator(),
        })

        # === Cross-Attention Fusion ===
        self.fusion = CrossAttentionFusion(d_model=fpn_channels)

    def forward(
        self,
        images: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
        conf_thresh: float = 0.05,
    ) -> dict[str, Any]:
        """
        Full OrchestraNet forward pass.

        Args:
            images: Input batch (B, 3, H, W)
            targets: Optional GT for training loss computation

        Returns:
            Dict with:
              - "detections": Final detection results after OA-NMS
              - "routing": Router decision details
              - "model_outputs": Raw outputs from each active model
              - "losses": Training losses (if targets provided)
        """
        # Fast-Path for standalone M1 monolithic baseline (bypasses backbone + FPN)
        if (getattr(self, "force_route", None) is not None and 
            str(self.force_route).lower() in ("monolithic", "m1_alone", "m1_only", "baseline") and 
            getattr(self.models["m1"], "pretrained_detector", None) is not None):
            m1_out = self.models["m1"](features=None, context={"m1_iou": 0.70}, images=images)
            obj = m1_out["objectness"]
            if obj.dim() == 3:
                obj = obj.squeeze(-1)
            scores_val = obj if (obj.numel() > 0 and obj.min() >= 0.0 and obj.max() <= 1.0) else torch.sigmoid(obj)
            detections = {
                "boxes": m1_out["decoded_boxes"],
                "scores": scores_val,
                "class_logits": m1_out["class_logits"],
            }
            final_detections = self._apply_oa_nms(detections, {"m1": m1_out}, score_threshold=conf_thresh)
            return {
                "detections": final_detections,
                "routing": {"routing_level": "monolithic", "active_models": ["m1"]},
                "model_outputs": {"m1": m1_out},
                "losses": {},
                "fpn_features": [],
            }

        # === Stage 1: Feature Extraction ===
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)

        # === Stage 2: Routing Decision ===
        routing = self.router(fpn_features)
        if getattr(self, "force_route", None) is not None:
            forced_level = str(self.force_route).lower()
            routing["routing_level"] = forced_level
            if forced_level in ("simple", "route1"):
                routing["active_models"] = ["m1", "m5"]
            elif forced_level in ("medium", "route2"):
                routing["active_models"] = ["m1", "m2", "m5", "m7"]
            elif forced_level in ("complex", "route3"):
                routing["active_models"] = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
            elif forced_level in ("monolithic", "m1_alone", "m1_only", "baseline"):
                routing["active_models"] = ["m1"]

        # Filter active models respecting ablation / manual deactivation
        active_models = [m for m in routing["active_models"] if getattr(self.models[m], "is_active", True)]

        # === Stage 3: Run Active Micro-Models ===
        model_outputs = {}
        context = {}  # Inter-model communication

        # M1 always runs first (other models may depend on it)
        if "m1" in active_models:
            model_outputs["m1"] = self.models["m1"](fpn_features, context=context, images=images)

        # M2 runs next (M6, M7 depend on occlusion info)
        if "m2" in active_models:
            model_outputs["m2"] = self.models["m2"](fpn_features)
            context["occlusion_features"] = model_outputs["m2"].get("occlusion_features")
            context["visibility_scores"] = model_outputs["m2"].get("visibility_scores")

        # Remaining models can run in parallel
        parallel_models = [m for m in active_models if m not in ("m1", "m2")]
        for model_id in parallel_models:
            model_outputs[model_id] = self.models[model_id](
                fpn_features, context=context
            )

        # === Stage 4: Fusion & Prediction Refinement ===
        if "m1" in model_outputs:
            m1_out = model_outputs["m1"]
            obj = m1_out["objectness"]
            if obj.dim() == 3:
                obj = obj.squeeze(-1)
            if obj.numel() > 0 and (obj.min() >= 0.0 and obj.max() <= 1.0):
                scores_val = obj
            else:
                scores_val = torch.sigmoid(obj)

            boxes = m1_out["decoded_boxes"].clone()
            scores = scores_val.clone()
            class_logits = m1_out["class_logits"].clone()
            B = boxes.shape[0]

            # 1. M5 (Scene Context) Refinement:
            # Re-scores class logits based on scene semantic object priors
            if "m5" in model_outputs:
                priors = model_outputs["m5"].get("object_priors")
                if priors is not None and priors.shape[-1] == class_logits.shape[-1]:
                    prior_mod = 0.20 * torch.log(priors.clamp(min=1e-4)).unsqueeze(1)
                    class_logits = class_logits + prior_mod

            # 2. M6 (Amodal Completer) Refinement:
            # Expands clipped visible boxes of occluded objects to their full amodal extent
            if "m6" in model_outputs and boxes.shape[1] > 0:
                amodal_offsets = model_outputs["m6"].get("amodal_bbox_offset")
                amodal_conf = model_outputs["m6"].get("completion_confidence")
                if amodal_offsets is not None and amodal_conf is not None:
                    M = min(boxes.shape[1], amodal_offsets.shape[1])
                    if M > 0:
                        w = (boxes[:, :M, 2] - boxes[:, :M, 0]).clamp(min=1)
                        h = (boxes[:, :M, 3] - boxes[:, :M, 1]).clamp(min=1)
                        offsets = torch.tanh(amodal_offsets[:, :M]) * amodal_conf[:, :M] * 0.08
                        boxes[:, :M, 0] = boxes[:, :M, 0] - offsets[:, :, 0] * w
                        boxes[:, :M, 1] = boxes[:, :M, 1] - offsets[:, :, 1] * h
                        boxes[:, :M, 2] = boxes[:, :M, 2] + offsets[:, :, 2] * w
                        boxes[:, :M, 3] = boxes[:, :M, 3] + offsets[:, :, 3] * h

            # 3. M3 (Small Object Enhancer) Refinement:
            # Selectively verifies and enhances small object detections (< 32x32)
            if "m3" in model_outputs and boxes.shape[1] > 0:
                conf_boost = model_outputs["m3"].get("confidence_boost")
                box_w = (boxes[:, :, 2] - boxes[:, :, 0]).clamp(min=0)
                box_h = (boxes[:, :, 3] - boxes[:, :, 1]).clamp(min=0)
                small_mask = (box_w * box_h) < 1024.0
                if conf_boost is not None and small_mask.any():
                    for b_i in range(B):
                        b_mask = small_mask[b_i]
                        if b_mask.any():
                            sampled_boost = self._sample_depth_at_boxes(conf_boost[b_i], boxes[b_i])
                            sr_factor = (sampled_boost - 0.5) * 0.12
                            scores[b_i, b_mask] = (scores[b_i, b_mask] * (1.0 + sr_factor[b_mask])).clamp(0.0, 1.0)

            # 4. M7 (Confidence Calibrator) Refinement:
            # Applies temperature calibration to align probabilities with true empirical precision
            if "m7" in model_outputs:
                temp = getattr(self.models["m7"], "temperature", torch.tensor(1.3, device=scores.device))
                temp_val = float(temp.mean().item()) if isinstance(temp, torch.Tensor) else float(temp)
                temp_val = max(0.8, min(2.0, temp_val))
                scores = torch.pow(scores.clamp(min=1e-5), 1.0 / temp_val)

            detections = {
                "boxes": boxes,
                "scores": scores,
                "class_logits": class_logits,
            }

        else:
            detections = {"boxes": torch.empty(0, 4), "scores": torch.empty(0),
                          "class_logits": torch.empty(0)}

        # === Stage 5: OA-NMS (per image in batch) ===
        # Only run post-processing NMS during inference (when targets are not provided)
        if not self.training and targets is None and "m1" in model_outputs:
            final_detections = self._apply_oa_nms(detections, model_outputs, score_threshold=conf_thresh)
        else:
            final_detections = detections

        # === Compute Losses ===
        losses = {}
        if targets is not None:
            for model_id, output in model_outputs.items():
                if hasattr(self.models[model_id], "get_loss"):
                    model_loss = self.models[model_id].get_loss(output, targets)
                    for k, v in model_loss.items():
                        losses[f"{model_id}_{k}"] = v

        return {
            "detections": final_detections,
            "routing": routing,
            "model_outputs": model_outputs,
            "losses": losses,
            "fpn_features": fpn_features,
        }

    def _apply_oa_nms(
        self,
        detections: dict[str, torch.Tensor],
        model_outputs: dict[str, dict],
        score_threshold: float = 0.05,
    ) -> list[dict[str, torch.Tensor]]:
        """Apply OA-NMS per image in the batch."""
        B = detections["boxes"].shape[0]
        results = []

        for b in range(B):
            boxes = detections["boxes"][b]
            scores = detections["scores"][b]
            class_probs = torch.sigmoid(detections["class_logits"][b])
            max_scores, labels = class_probs.max(dim=-1)
            combined_scores = scores * max_scores

            # Get occlusion and depth info if available
            occ_scores = None
            depth_vals = None

            if "m2" in model_outputs:
                if "occlusion_map" in model_outputs["m2"]:
                    occ_map = model_outputs["m2"]["occlusion_map"]
                    box_occ = self._sample_depth_at_boxes(occ_map[b], boxes)
                    occ_scores = (1.0 - box_occ).clamp(0.0, 1.0)
                elif "visibility_scores" in model_outputs["m2"]:
                    vis = model_outputs["m2"]["visibility_scores"]
                    if vis.dim() > 0:
                        occ_scores = vis[b].expand(combined_scores.shape[0])

            if "m4" in model_outputs:
                depth = model_outputs["m4"]["depth_map"]
                if depth.dim() >= 3:
                    # Sample depth at box centers
                    depth_vals = self._sample_depth_at_boxes(depth[b], boxes)

            result = occlusion_aware_nms(
                boxes=boxes,
                scores=combined_scores,
                labels=labels,
                occlusion_scores=occ_scores,
                depth_values=depth_vals,
                iou_threshold=0.65,
                score_threshold=score_threshold,
            )
            results.append(result)

        return results

    @staticmethod
    def _sample_depth_at_boxes(
        depth_map: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Sample depth values at the center of each detection box."""
        if boxes.shape[0] == 0:
            return torch.empty(0, device=depth_map.device)

        _, H, W = depth_map.shape
        cx = ((boxes[:, 0] + boxes[:, 2]) / 2).clamp(0, 639) / 640 * (W - 1)
        cy = ((boxes[:, 1] + boxes[:, 3]) / 2).clamp(0, 639) / 640 * (H - 1)
        cx = cx.long().clamp(0, W - 1)
        cy = cy.long().clamp(0, H - 1)
        return depth_map[0, cy, cx]

    def count_all_parameters(self) -> dict[str, dict[str, int]]:
        """Count parameters for each component."""
        result = {}
        result["backbone"] = {
            "total": sum(p.numel() for p in self.backbone.parameters()),
            "trainable": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
        }
        result["fpn"] = {
            "total": sum(p.numel() for p in self.fpn.parameters()),
            "trainable": sum(p.numel() for p in self.fpn.parameters() if p.requires_grad),
        }
        result["router"] = {
            "total": sum(p.numel() for p in self.router.parameters()),
            "trainable": sum(p.numel() for p in self.router.parameters() if p.requires_grad),
        }
        for model_id, model in self.models.items():
            result[model_id] = model.count_parameters()
        result["fusion"] = {
            "total": sum(p.numel() for p in self.fusion.parameters()),
            "trainable": sum(p.numel() for p in self.fusion.parameters() if p.requires_grad),
        }
        total = sum(v["total"] for v in result.values())
        trainable = sum(v["trainable"] for v in result.values())
        result["TOTAL"] = {"total": total, "trainable": trainable}
        return result
