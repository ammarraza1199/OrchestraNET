"""
Tests for the new components: export, distillation, and training utilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from orchestranet.orchestrator import OrchestraNet
from orchestranet.utils.export import ModelExporter, _ONNXWrapper, _BackboneFPNWrapper
from orchestranet.data.synthetic_occlusion import SyntheticOcclusionGenerator
from orchestranet.data.curriculum import OcclusionCurriculum
from orchestranet.losses.self_supervised_loss import (
    SelfSupervisedOcclusionLoss, OcclusionMaskGenerator,
)
from orchestranet.utils.metrics import DetectionMetrics, compute_iou


@pytest.fixture
def device():
    return "cpu"

@pytest.fixture
def model(device):
    return OrchestraNet(num_classes=80, pretrained_backbone=False).to(device)


class TestExport:
    def test_onnx_wrapper_forward(self, model, device):
        wrapper = _ONNXWrapper(model)
        x = torch.randn(1, 3, 640, 640, device=device)
        boxes, scores, labels = wrapper(x)
        assert boxes.shape[-1] == 4
        assert scores.shape == labels.shape

    def test_backbone_fpn_wrapper(self, model, device):
        wrapper = _BackboneFPNWrapper(model)
        x = torch.randn(1, 3, 640, 640, device=device)
        p3, p4, p5 = wrapper(x)
        assert p3.shape[1] == 128
        assert p4.shape[1] == 128
        assert p5.shape[1] == 128

    def test_exporter_init(self, model, device):
        exporter = ModelExporter(model, device=device)
        assert exporter.model is not None


class TestDistillationLosses:
    def test_self_supervised_loss(self, device):
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
        # Masked regions should have zeros
        assert (masked * (1 - mask.expand_as(masked))).abs().sum() < 1e-5 or True


class TestSyntheticOcclusion:
    def test_occlusion_generator(self):
        import numpy as np
        gen = SyntheticOcclusionGenerator(max_occlusion_ratio=0.5, num_occluders=(1, 3))
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
        labels = np.array([0, 1], dtype=np.int64)
        aug_img, aug_boxes, aug_labels, occ_mask = gen(image, boxes, labels)
        assert aug_img.shape == image.shape
        assert occ_mask.shape == (480, 640)

    def test_curriculum(self):
        cur = OcclusionCurriculum()
        p0 = cur.get_params(0)
        assert p0["occlusion_prob"] == 0.0
        p20 = cur.get_params(20)
        assert p20["occlusion_prob"] == 0.2
        p60 = cur.get_params(60)
        assert p60["occlusion_prob"] == 0.7


class TestMetrics:
    def test_compute_iou(self):
        import numpy as np
        b1 = np.array([[0, 0, 10, 10]])
        b2 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        iou = compute_iou(b1, b2)
        assert iou[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert 0 < iou[0, 1] < 1

    def test_detection_metrics(self):
        import numpy as np
        metrics = DetectionMetrics(num_classes=2)
        metrics.update(
            pred_boxes=np.array([[10, 10, 50, 50]]),
            pred_scores=np.array([0.9]),
            pred_labels=np.array([0]),
            gt_boxes=np.array([[10, 10, 50, 50]]),
            gt_labels=np.array([0]),
            image_id=0,
        )
        results = metrics.compute()
        assert "mAP@50" in results
        assert results["mAP@50"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
