"""
Training Smoke Tests for OrchestraNet.

Verifies that training scripts can:
  1. Load models and datasets
  2. Run 1 batch of training without errors
  3. Produce non-zero loss with gradients
  4. Update model parameters
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from orchestranet.orchestrator import OrchestraNet
from orchestranet.backbone import MobileNetV4Backbone, LightweightFPN
from orchestranet.models import M1PrimaryDetector
from orchestranet.utils.config import Config
from orchestranet.utils.ema import ModelEMA
from orchestranet.utils.logger import TrainingLogger, AverageMeter


@pytest.fixture
def device():
    return "cpu"


# ============ Config Tests ============

class TestConfig:
    def test_config_from_yaml(self, tmp_path):
        # Create a temp config file
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
training:
  batch_size: 32
  learning_rate: 0.001
backbone:
  name: "mobilenetv4"
""")
        cfg = Config.from_yaml(str(config_path))
        assert cfg.training.batch_size == 32
        assert cfg.training.learning_rate == 0.001
        assert cfg.backbone.name == "mobilenetv4"

    def test_config_get_dot_notation(self, tmp_path):
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text("""
training:
  batch_size: 16
""")
        cfg = Config.from_yaml(str(config_path))
        assert cfg.get("training.batch_size") == 16
        assert cfg.get("nonexistent.key", 42) == 42

    def test_config_missing_file(self):
        cfg = Config.from_yaml("nonexistent_file.yaml")
        assert cfg is not None  # Should return empty config


# ============ EMA Tests ============

class TestEMA:
    def test_ema_update(self, device):
        model = nn.Linear(10, 5).to(device)
        ema = ModelEMA(model, decay=0.99)

        # Simulate parameter update
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)

        # EMA should have moved toward 1.0
        for name, shadow in ema.shadow.items():
            if "weight" in name:
                assert shadow.mean().item() != 0  # Not still at init

    def test_ema_apply_restore(self, device):
        model = nn.Linear(10, 5).to(device)
        original_weight = model.weight.data.clone()

        ema = ModelEMA(model, decay=0.99)

        # Update model weights
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)

        # Apply shadow (should change model weights)
        ema.apply_shadow(model)
        assert not torch.equal(model.weight.data, original_weight)

        # Restore should bring back the latest weights (1.0)
        ema.restore(model)
        assert torch.allclose(model.weight.data, torch.ones_like(model.weight.data))


# ============ Logger Tests ============

class TestLogger:
    def test_average_meter(self):
        meter = AverageMeter("test")
        meter.update(1.0, 1)
        meter.update(3.0, 1)
        assert meter.avg == 2.0
        assert meter.val == 3.0
        assert meter.count == 2

    def test_logger_init(self, tmp_path):
        logger = TrainingLogger(
            log_dir=str(tmp_path / "test_logs"),
            tb_enabled=False,
        )
        logger.info("Test message")
        logger.log_scalar("train/loss", 0.5, step=1)
        logger.close()


# ============ Training Smoke Test ============

class TestTrainingSmoke:
    def test_orchestranet_one_batch_forward(self, device):
        """Verify a single forward pass produces valid losses."""
        model = OrchestraNet(
            num_classes=80,
            pretrained_backbone=False,
        ).to(device)
        model.train()

        images = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "boxes": torch.tensor([
                [[100, 100, 200, 200], [300, 300, 400, 400]],
                [[50, 50, 150, 150], [250, 250, 350, 350]],
            ], dtype=torch.float32, device=device),
            "labels": torch.tensor([[0, 5], [1, 3]], dtype=torch.long, device=device),
            "num_objects": torch.tensor([2, 2], device=device),
        }

        outputs = model(images, targets)

        assert "losses" in outputs
        assert "routing" in outputs
        losses = outputs["losses"]

        # At least M1 loss should be present (M1 is always active)
        total = sum(v for k, v in losses.items()
                    if "total" in k and isinstance(v, torch.Tensor))
        assert total.item() > 0, "Total loss should be non-zero"

    def test_one_training_step(self, device):
        """Verify one complete training step (forward + backward + step)."""
        model = OrchestraNet(
            num_classes=80,
            pretrained_backbone=False,
        ).to(device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        images = torch.randn(1, 3, 320, 320, device=device)  # Smaller for speed
        targets = {
            "boxes": torch.tensor([[[50, 50, 100, 100]]], dtype=torch.float32, device=device),
            "labels": torch.tensor([[0]], dtype=torch.long, device=device),
            "num_objects": torch.tensor([1], device=device),
        }

        # Save initial params
        initial_params = {n: p.clone() for n, p in model.named_parameters()
                         if p.requires_grad}

        optimizer.zero_grad()
        outputs = model(images, targets)
        losses = outputs["losses"]
        total = sum(v for k, v in losses.items()
                    if "total" in k and isinstance(v, torch.Tensor))
        total.backward()
        optimizer.step()

        # Check some parameters changed
        changed = False
        for n, p in model.named_parameters():
            if n in initial_params and p.requires_grad:
                if not torch.equal(p, initial_params[n]):
                    changed = True
                    break
        assert changed, "No parameters changed after training step"

    def test_individual_model_training_step(self, device):
        """Test individual model pre-training flow."""
        backbone = MobileNetV4Backbone(pretrained=False).to(device)
        fpn = LightweightFPN(
            in_channels=backbone.get_out_channels(),
            out_channels=128,
        ).to(device)
        m1 = M1PrimaryDetector(in_channels=128).to(device)
        m1.train()

        images = torch.randn(2, 3, 320, 320, device=device)
        features = backbone(images)
        fpn_features = fpn(features)
        predictions = m1(fpn_features)

        targets = {
            "boxes": torch.tensor([
                [[50, 50, 100, 100]],
                [[30, 30, 80, 80]],
            ], dtype=torch.float32, device=device),
            "labels": torch.tensor([[0], [1]], dtype=torch.long, device=device),
            "num_objects": torch.tensor([1, 1], device=device),
        }

        losses = m1.get_loss(predictions, targets)
        assert losses["total_loss"].item() > 0
        losses["total_loss"].backward()

        # Verify gradients exist
        grad_count = sum(1 for p in m1.parameters()
                         if p.grad is not None and p.grad.abs().sum() > 0)
        assert grad_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
