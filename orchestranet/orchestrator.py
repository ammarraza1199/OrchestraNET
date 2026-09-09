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
        # === Stage 1: Feature Extraction ===
        backbone_features = self.backbone(images)
        fpn_features = self.fpn(backbone_features)

        # === Stage 2: Routing Decision ===
        routing = self.router(fpn_features)
        active_models = routing["active_models"]

        # === Stage 3: Run Active Micro-Models ===
        model_outputs = {}
        context = {}  # Inter-model communication

        # M1 always runs first (other models may depend on it)
        if "m1" in active_models:
            model_outputs["m1"] = self.models["m1"](fpn_features)

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

        # === Stage 4: Fusion ===
        if "m1" in model_outputs:
            m1_out = model_outputs["m1"]
            detections = {
                "boxes": m1_out["decoded_boxes"],
                "scores": torch.sigmoid(m1_out["objectness"]).squeeze(-1),
                "class_logits": m1_out["class_logits"],
            }

            # Apply confidence calibration if M7 was active
            if "m7" in model_outputs:
                cal = model_outputs["m7"]["calibrated_confidence"]
                # Modulate scores with calibrated confidence
                if cal.shape[-1] == 1 and detections["scores"].dim() == 2:
                    detections["scores"] = detections["scores"] * cal.squeeze(-1).unsqueeze(1)

        else:
            detections = {"boxes": torch.empty(0, 4), "scores": torch.empty(0),
                          "class_logits": torch.empty(0)}

        # === Stage 5: OA-NMS (per image in batch) ===
        if not self.training and "m1" in model_outputs:
            final_detections = self._apply_oa_nms(detections, model_outputs)
        else:
            final_detections = detections

        # === Compute Losses (training) ===
        losses = {}
        if targets is not None and self.training:
            for model_id, output in model_outputs.items():
                model_loss = self.models[model_id].get_loss(output, targets)
                for k, v in model_loss.items():
                    losses[f"{model_id}_{k}"] = v

        return {
            "detections": final_detections,
            "routing": routing,
            "model_outputs": model_outputs,
            "losses": losses,
        }

    def _apply_oa_nms(
        self,
        detections: dict[str, torch.Tensor],
        model_outputs: dict[str, dict],
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
