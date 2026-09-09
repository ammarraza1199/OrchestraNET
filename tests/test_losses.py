"""
Tests for all OrchestraNet loss functions.

Verifies:
  1. Loss functions produce non-zero gradients (gradient flow)
  2. Output shapes are correct
  3. Loss values are in expected ranges
  4. Each model's get_loss() works with proper targets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn.functional as F

from orchestranet.models import (
    M1PrimaryDetector, M2OcclusionAnalyzer, M3SmallObjectEnhancer,
    M4DepthEstimator, M5SemanticContext, M6AmodalCompleter, M7ConfidenceCalibrator,
)
from orchestranet.losses.self_supervised_loss import (
    SelfSupervisedOcclusionLoss, OcclusionMaskGenerator,
)


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def fpn_features(device):
    return [
        torch.randn(2, 128, 80, 80, device=device),
        torch.randn(2, 128, 40, 40, device=device),
        torch.randn(2, 128, 20, 20, device=device),
    ]


# ============ M1 Detection Loss ============

class TestM1Loss:
    def test_loss_nonzero_with_targets(self, fpn_features, device):
        """M1 loss should be non-zero when there are GT boxes."""
        m1 = M1PrimaryDetector(in_channels=128).to(device)
        m1.train()
        predictions = m1(fpn_features)

        targets = {
            "boxes": torch.tensor([
                [[100, 100, 200, 200], [300, 300, 400, 400], [0, 0, 0, 0]],
                [[50, 50, 150, 150], [0, 0, 0, 0], [0, 0, 0, 0]],
            ], dtype=torch.float32, device=device),
            "labels": torch.tensor([
                [0, 5, 0],
                [1, 0, 0],
            ], dtype=torch.long, device=device),
            "num_objects": torch.tensor([2, 1], device=device),
        }

        losses = m1.get_loss(predictions, targets)

        assert "bbox_loss" in losses
        assert "cls_loss" in losses
        assert "obj_loss" in losses
        assert "total_loss" in losses
        # Total loss should be non-zero with GT
        assert losses["total_loss"].item() > 0

    def test_loss_gradient_flow(self, fpn_features, device):
        """Verify gradients flow back through all loss components."""
        m1 = M1PrimaryDetector(in_channels=128).to(device)
        m1.train()

        for p in m1.parameters():
            p.requires_grad_(True)

        predictions = m1(fpn_features)

        targets = {
            "boxes": torch.tensor([
                [[100, 100, 200, 200]],
                [[50, 50, 150, 150]],
            ], dtype=torch.float32, device=device),
            "labels": torch.tensor([[0], [1]], dtype=torch.long, device=device),
            "num_objects": torch.tensor([1, 1], device=device),
        }

        losses = m1.get_loss(predictions, targets)
        losses["total_loss"].backward()

        # Check that at least some parameters got gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in m1.parameters() if p.requires_grad)
        assert has_grad, "No gradients flowed through M1 loss"

    def test_loss_zero_with_no_gt(self, fpn_features, device):
        """M1 should still produce valid loss when no GT boxes exist."""
        m1 = M1PrimaryDetector(in_channels=128).to(device)
        predictions = m1(fpn_features)

        targets = {
            "boxes": torch.zeros(2, 0, 4, device=device),
            "labels": torch.zeros(2, 0, dtype=torch.long, device=device),
            "num_objects": torch.tensor([0, 0], device=device),
        }

        losses = m1.get_loss(predictions, targets)
        assert losses["total_loss"].item() >= 0
        assert not torch.isnan(losses["total_loss"])


# ============ M2 Occlusion Loss ============

class TestM2Loss:
    def test_occlusion_loss_with_mask(self, fpn_features, device):
        m2 = M2OcclusionAnalyzer(in_channels=128).to(device)
        predictions = m2(fpn_features)

        targets = {
            "occlusion_mask": torch.rand(2, 1, 80, 80, device=device),
        }

        losses = m2.get_loss(predictions, targets)
        assert "total_loss" in losses
        assert losses["total_loss"].item() > 0

    def test_occlusion_loss_without_mask(self, fpn_features, device):
        """Should return zero loss when no GT mask available."""
        m2 = M2OcclusionAnalyzer(in_channels=128).to(device)
        predictions = m2(fpn_features)
        losses = m2.get_loss(predictions, {})
        assert losses["total_loss"].item() == 0


# ============ M3 Small Enhancer Loss ============

class TestM3Loss:
    def test_sr_loss_nonzero(self, fpn_features, device):
        m3 = M3SmallObjectEnhancer(in_channels=128).to(device)
        predictions = m3(fpn_features)
        losses = m3.get_loss(predictions, {})

        assert "sr_loss" in losses
        assert "consistency_loss" in losses
        assert losses["total_loss"].item() > 0

    def test_sr_loss_gradient_flow(self, fpn_features, device):
        m3 = M3SmallObjectEnhancer(in_channels=128).to(device)
        m3.train()
        predictions = m3(fpn_features)
        losses = m3.get_loss(predictions, {})
        losses["total_loss"].backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in m3.parameters() if p.requires_grad)
        assert has_grad, "No gradients flowed through M3 loss"


# ============ M4 Depth Loss ============

class TestM4Loss:
    def test_depth_loss_with_gt(self, fpn_features, device):
        m4 = M4DepthEstimator(in_channels=128).to(device)
        predictions = m4(fpn_features)

        targets = {
            "depth_gt": torch.rand(2, 1, 80, 80, device=device).clamp(0.01, 1.0),
        }

        losses = m4.get_loss(predictions, targets)
        assert losses["total_loss"].item() > 0
        assert "depth_loss" in losses
        assert "smoothness_loss" in losses

    def test_depth_loss_without_gt(self, fpn_features, device):
        """Self-supervised smoothness loss should still be computed."""
        m4 = M4DepthEstimator(in_channels=128).to(device)
        predictions = m4(fpn_features)
        losses = m4.get_loss(predictions, {})

        assert "smoothness_loss" in losses
        assert losses["smoothness_loss"].item() > 0


# ============ M5 Context Loss ============

class TestM5Loss:
    def test_context_loss(self, fpn_features, device):
        m5 = M5SemanticContext(in_channels=128).to(device)
        predictions = m5(fpn_features)

        targets = {"scene_label": torch.randint(0, 365, (2,), device=device)}
        losses = m5.get_loss(predictions, targets)
        assert "total_loss" in losses
        assert losses["total_loss"].item() > 0


# ============ M6 Amodal Loss ============

class TestM6Loss:
    def test_amodal_bbox_loss(self, fpn_features, device):
        m6 = M6AmodalCompleter(d_model=128).to(device)
        predictions = m6(fpn_features)

        n_pred = min(predictions["amodal_bbox_offset"].shape[1], 10)
        targets = {
            "amodal_boxes": torch.rand(2, n_pred, 4, device=device),
        }

        losses = m6.get_loss(predictions, targets)
        assert losses["amodal_bbox_loss"].item() >= 0

    def test_amodal_mask_loss(self, fpn_features, device):
        m6 = M6AmodalCompleter(d_model=128, mask_resolution=28).to(device)
        predictions = m6(fpn_features)

        n_pred = min(predictions["amodal_masks"].shape[1], 10)
        mask_h, mask_w = predictions["amodal_masks"].shape[-2:]
        targets = {
            "amodal_masks": torch.rand(2, n_pred, mask_h, mask_w, device=device),
        }

        losses = m6.get_loss(predictions, targets)
        assert "amodal_mask_loss" in losses
        assert losses["amodal_mask_loss"].item() >= 0


# ============ Self-Supervised Loss ============

class TestSelfSupervisedLoss:
    def test_loss_components(self, device):
        loss_fn = SelfSupervisedOcclusionLoss().to(device)
        pred = torch.rand(2, 1, 80, 80, device=device, requires_grad=True)
        target = torch.rand(2, 1, 80, 80, device=device)
        losses = loss_fn(pred, target)

        assert "bce" in losses
        assert "dice" in losses
        assert "total" in losses
        assert losses["total"].requires_grad

    def test_mask_generator(self, device):
        gen = OcclusionMaskGenerator().to(device)
        features = torch.randn(2, 128, 80, 80, device=device)
        masked, mask = gen(features)
        assert masked.shape == features.shape
        assert mask.shape == (2, 1, 80, 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
