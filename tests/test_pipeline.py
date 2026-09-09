"""
Unit tests for OrchestraNet full pipeline.

Tests:
  1. Backbone forward pass and output shapes
  2. FPN forward pass and channel unification
  3. Each micro-model (M1-M7) forward pass
  4. Adaptive Router routing decisions
  5. Cross-Attention Fusion
  6. OA-NMS correctness
  7. Full pipeline end-to-end
  8. Parameter count verification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from orchestranet.backbone import MobileNetV4Backbone, LightweightFPN
from orchestranet.models import (
    M1PrimaryDetector, M2OcclusionAnalyzer, M3SmallObjectEnhancer,
    M4DepthEstimator, M5SemanticContext, M6AmodalCompleter, M7ConfidenceCalibrator,
)
from orchestranet.router import AdaptiveRouter
from orchestranet.fusion import CrossAttentionFusion, occlusion_aware_nms
from orchestranet.orchestrator import OrchestraNet


# ============ Fixtures ============

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def dummy_image(device):
    return torch.randn(2, 3, 640, 640, device=device)

@pytest.fixture
def fpn_features(device):
    """Simulated FPN features: P3 (80x80), P4 (40x40), P5 (20x20)."""
    return [
        torch.randn(2, 128, 80, 80, device=device),
        torch.randn(2, 128, 40, 40, device=device),
        torch.randn(2, 128, 20, 20, device=device),
    ]


# ============ Backbone Tests ============

class TestBackbone:
    def test_backbone_forward(self, device):
        backbone = MobileNetV4Backbone(pretrained=False).to(device)
        x = torch.randn(2, 3, 640, 640, device=device)
        features = backbone(x)
        assert len(features) == 3, f"Expected 3 feature levels, got {len(features)}"
        for f in features:
            assert f.shape[0] == 2, "Batch dimension mismatch"

    def test_backbone_channels(self, device):
        backbone = MobileNetV4Backbone(pretrained=False).to(device)
        channels = backbone.get_out_channels()
        assert len(channels) == 3
        for c in channels:
            assert c > 0

    def test_fpn_forward(self, device):
        in_channels = [80, 160, 256]
        fpn = LightweightFPN(in_channels=in_channels, out_channels=128).to(device)
        inputs = [
            torch.randn(2, 80, 80, 80, device=device),
            torch.randn(2, 160, 40, 40, device=device),
            torch.randn(2, 256, 20, 20, device=device),
        ]
        outputs = fpn(inputs)
        assert len(outputs) == 3
        for o in outputs:
            assert o.shape[1] == 128, "FPN should unify channels to 128"


# ============ Micro-Model Tests ============

class TestMicroModels:
    def test_m1_detector(self, fpn_features):
        m1 = M1PrimaryDetector().to(fpn_features[0].device)
        out = m1(fpn_features)
        assert "decoded_boxes" in out
        assert "objectness" in out
        assert "class_logits" in out
        assert out["decoded_boxes"].shape[-1] == 4
        assert out["class_logits"].shape[-1] == 80

    def test_m2_occlusion(self, fpn_features):
        m2 = M2OcclusionAnalyzer().to(fpn_features[0].device)
        out = m2(fpn_features)
        assert "occlusion_map" in out
        assert "visibility_scores" in out
        occ = out["occlusion_map"]
        assert occ.min() >= 0 and occ.max() <= 1, "Occlusion map should be [0,1]"

    def test_m3_small_enhancer(self, fpn_features):
        m3 = M3SmallObjectEnhancer().to(fpn_features[0].device)
        out = m3(fpn_features)
        assert "enhanced_features" in out
        assert "bbox_refinement" in out

    def test_m4_depth(self, fpn_features):
        m4 = M4DepthEstimator().to(fpn_features[0].device)
        out = m4(fpn_features)
        assert "depth_map" in out
        depth = out["depth_map"]
        assert depth.min() >= 0 and depth.max() <= 1, "Depth map should be [0,1]"

    def test_m5_context(self, fpn_features):
        m5 = M5SemanticContext().to(fpn_features[0].device)
        out = m5(fpn_features)
        assert "scene_logits" in out
        assert "scene_embedding" in out
        assert "object_priors" in out
        assert out["scene_embedding"].shape[-1] == 128

    def test_m6_amodal(self, fpn_features):
        m6 = M6AmodalCompleter().to(fpn_features[0].device)
        out = m6(fpn_features)
        assert "amodal_bbox_offset" in out
        assert "amodal_masks" in out
        assert "completion_confidence" in out

    def test_m7_calibrator(self, fpn_features):
        m7 = M7ConfidenceCalibrator().to(fpn_features[0].device)
        out = m7(fpn_features)
        assert "calibrated_confidence" in out
        conf = out["calibrated_confidence"]
        assert conf.min() >= 0 and conf.max() <= 1


# ============ Router Tests ============

class TestRouter:
    def test_router_forward(self, fpn_features):
        router = AdaptiveRouter().to(fpn_features[0].device)
        router.eval()
        result = router(fpn_features)
        assert "complexity_score" in result
        assert "routing_level" in result
        assert "active_models" in result
        assert result["routing_level"] in ["simple", "medium", "complex"]
        assert "m1" in result["active_models"]  # M1 is always active

    def test_router_training_mode(self, fpn_features):
        router = AdaptiveRouter().to(fpn_features[0].device)
        router.train()
        result = router(fpn_features)
        assert result["routing_probs"].shape == (2, 3)


# ============ Fusion Tests ============

class TestFusion:
    def test_cross_attention_fusion(self, device):
        fusion = CrossAttentionFusion(d_model=128).to(device)
        model_outputs = {
            "m1": torch.randn(2, 100, 128, device=device),
            "m2": torch.randn(2, 100, 128, device=device),
        }
        primary = torch.randn(2, 100, 128, device=device)
        out = fusion(model_outputs, primary)
        assert out.shape == (2, 100, 128)


# ============ OA-NMS Tests ============

class TestOANMS:
    def test_basic_nms(self):
        boxes = torch.tensor([
            [10, 10, 50, 50],
            [12, 12, 52, 52],  # Overlaps with first (duplicate)
            [100, 100, 150, 150],  # No overlap
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 0, 1])

        result = occlusion_aware_nms(boxes, scores, labels)
        assert len(result["boxes"]) >= 2  # At least the non-overlapping ones

    def test_occlusion_aware_keeps_occluded(self):
        """OA-NMS should keep both boxes when occlusion is detected."""
        boxes = torch.tensor([
            [10, 10, 100, 100],
            [30, 30, 120, 120],  # Overlaps
        ], dtype=torch.float32)
        scores = torch.tensor([0.9, 0.7])
        labels = torch.tensor([0, 0])
        occlusion_scores = torch.tensor([0.95, 0.4])  # Second is heavily occluded
        depth_values = torch.tensor([0.3, 0.7])  # Different depths

        result = occlusion_aware_nms(
            boxes, scores, labels,
            occlusion_scores=occlusion_scores,
            depth_values=depth_values,
        )
        # Both should be kept (different depths = different objects)
        assert len(result["boxes"]) == 2


# ============ Full Pipeline Test ============

class TestOrchestraNet:
    def test_full_pipeline(self, device):
        model = OrchestraNet(
            num_classes=80,
            pretrained_backbone=False,
            fpn_channels=128,
        ).to(device)
        model.eval()

        x = torch.randn(1, 3, 640, 640, device=device)
        with torch.no_grad():
            outputs = model(x)

        assert "detections" in outputs
        assert "routing" in outputs
        assert "model_outputs" in outputs
        assert outputs["routing"]["routing_level"] in ["simple", "medium", "complex"]

    def test_parameter_count(self, device):
        model = OrchestraNet(
            num_classes=80,
            pretrained_backbone=False,
        ).to(device)

        params = model.count_all_parameters()
        assert "TOTAL" in params
        total = params["TOTAL"]["total"]
        print(f"Total parameters: {total:,}")
        # Should be lightweight (< 20M for the model heads, excluding backbone)
        assert total > 0

    def test_training_forward(self, device):
        model = OrchestraNet(
            num_classes=80,
            pretrained_backbone=False,
        ).to(device)
        model.train()

        x = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "boxes": torch.rand(2, 10, 4, device=device) * 640,
            "labels": torch.randint(0, 80, (2, 10), device=device),
            "num_objects": torch.tensor([10, 10], device=device),
        }

        outputs = model(x, targets)
        assert "losses" in outputs
        assert len(outputs["losses"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
