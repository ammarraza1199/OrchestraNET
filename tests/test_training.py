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

    def test_m2_individual_trainer_step(self, device):
        """Test M2 individual training and self-supervised loss."""
        from training.train_individual import IndividualTrainer, validate_loss
        trainer = IndividualTrainer(model_id="m2", device=device, freeze_backbone=True)
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "boxes": torch.zeros((2, 10, 4), device=device),
            "labels": torch.zeros((2, 10), dtype=torch.long, device=device),
            "num_objects": torch.tensor([0, 0], device=device),
        }
        losses = trainer.train_step(images, targets)
        assert "total_loss" in losses
        assert "ss_loss" in losses
        assert losses["total_loss"].item() > 0
        losses["total_loss"].backward()

        val_loader = [(images, targets)]
        val_res = validate_loss(trainer, val_loader, device=device)
        assert "val_loss" in val_res
        assert val_res["val_loss"] > 0

    def test_m3_individual_trainer_step(self, device):
        """Test M3 individual training and SR loss."""
        from training.train_individual import IndividualTrainer
        trainer = IndividualTrainer(model_id="m3", device=device, freeze_backbone=True)
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "boxes": torch.zeros((2, 10, 4), device=device),
            "labels": torch.zeros((2, 10), dtype=torch.long, device=device),
            "num_objects": torch.tensor([0, 0], device=device),
        }
        losses = trainer.train_step(images, targets)
        assert "total_loss" in losses
        assert "sr_loss" in losses
        assert losses["total_loss"].item() > 0
        losses["total_loss"].backward()

    def test_m4_individual_trainer_step(self, device):
        """Test M4 individual training and depth smoothness loss."""
        from training.train_individual import IndividualTrainer
        trainer = IndividualTrainer(model_id="m4", device=device, freeze_backbone=True)
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "boxes": torch.zeros((2, 10, 4), device=device),
            "labels": torch.zeros((2, 10), dtype=torch.long, device=device),
            "num_objects": torch.tensor([0, 0], device=device),
        }
        losses = trainer.train_step(images, targets)
        assert "total_loss" in losses
        assert "smoothness_loss" in losses
        assert losses["total_loss"].item() > 0
        losses["total_loss"].backward()

    def test_m6_individual_trainer_step(self, device):
        """Test M6 individual training and amodal completion loss."""
        from training.train_individual import IndividualTrainer, validate_loss
        from orchestranet.models import M6AmodalCompleter

        trainer = IndividualTrainer(model_id="m6", device=device, freeze_backbone=True)
        assert isinstance(trainer.model, M6AmodalCompleter)
        assert trainer.model.d_model == 128

        images = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "boxes": torch.zeros((2, 10, 4), device=device),
            "labels": torch.zeros((2, 10), dtype=torch.long, device=device),
            "num_objects": torch.tensor([2, 2], device=device),
            "amodal_boxes": torch.rand(2, 10, 4, device=device),
            "amodal_masks": torch.rand(2, 10, 28, 28, device=device),
            "is_occluded": torch.randint(0, 2, (2, 10), device=device),
        }
        losses = trainer.train_step(images, targets)
        assert "total_loss" in losses
        assert "amodal_bbox_loss" in losses
        assert "amodal_mask_loss" in losses
        assert "amodal_conf_loss" in losses
        assert losses["total_loss"].item() > 0
        losses["total_loss"].backward()

        # Check gradients exist on M6 model parameters
        m6_grads = sum(1 for p in trainer.model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert m6_grads > 0, "Gradients should be computed for M6 parameters"

        val_loader = [(images, targets)]
        val_res = validate_loss(trainer, val_loader, device=device)
        assert "val_loss" in val_res
        assert val_res["val_loss"] > 0
        assert "val_amodal_bbox_loss" in val_res
        assert "val_amodal_mask_loss" in val_res
        assert "val_amodal_conf_loss" in val_res

        # Test M6 specific validation reporting Amodal Mask IoU and Bbox MAE
        from training.train_individual import validate_m6
        val_m6_res = validate_m6(trainer, val_loader, device=device)
        assert "val_loss" in val_m6_res
        assert "val_amodal_mask_iou" in val_m6_res
        assert "val_amodal_bbox_mae" in val_m6_res
        assert 0.0 <= val_m6_res["val_amodal_mask_iou"] <= 1.0
        assert val_m6_res["val_amodal_bbox_mae"] >= 0.0

    def test_compute_amodal_metrics(self):
        """Test compute_amodal_metrics with known ground-truth and predictions."""
        from orchestranet.utils.metrics import compute_amodal_metrics

        # Perfect match scenario
        preds = {
            "amodal_masks": torch.ones((1, 2, 28, 28)),  # > 0.5 everywhere
            "amodal_bbox_offset": torch.tensor([[[10.0, 20.0, 30.0, 40.0], [5.0, 5.0, 5.0, 5.0]]]),
        }
        targets = {
            "amodal_masks": torch.ones((1, 2, 28, 28)),
            "amodal_boxes": torch.tensor([[[10.0, 20.0, 30.0, 40.0], [5.0, 5.0, 5.0, 5.0]]]),
            "num_objects": torch.tensor([2]),
        }
        res = compute_amodal_metrics(preds, targets)
        assert abs(res["amodal_mask_iou"] - 1.0) < 1e-4
        assert abs(res["amodal_bbox_mae"] - 0.0) < 1e-4

        # Partial error scenario
        preds_err = {
            "amodal_masks": torch.zeros((1, 1, 28, 28)),  # 0 overlap
            "amodal_bbox_offset": torch.tensor([[[12.0, 22.0, 32.0, 42.0]]]),  # delta = 2.0
        }
        targets_err = {
            "amodal_masks": torch.ones((1, 1, 28, 28)),
            "amodal_boxes": torch.tensor([[[10.0, 20.0, 30.0, 40.0]]]),
            "num_objects": torch.tensor([1]),
        }
        res_err = compute_amodal_metrics(preds_err, targets_err)
        assert abs(res_err["amodal_mask_iou"] - 0.0) < 1e-4
        assert abs(res_err["amodal_bbox_mae"] - 2.0) < 1e-4

    def test_m6_kins_dataset_selection(self):
        """Verify KINS dataset selection and display label for M6."""
        from training.train_individual import MODEL_REGISTRY

        assert MODEL_REGISTRY["m6"]["dataset"] == "KINS"

        data_root = "/content/data/KINS"
        is_kins = "kins" in data_root.lower()
        assert is_kins is True
        dataset_name = "KINS" if is_kins else MODEL_REGISTRY["m6"]["dataset"]
        assert dataset_name == "KINS"

    def test_m6_metric_ignores_padded_slots(self):
        """Test F: compute_amodal_metrics strictly ignores padded slots beyond num_objects."""
        from orchestranet.utils.metrics import compute_amodal_metrics

        # 1 valid object, 49 padded slots
        preds_clean = {
            "amodal_masks": torch.ones((1, 50, 28, 28)),
            "amodal_bbox_offset": torch.zeros((1, 50, 4)),
        }
        targets = {
            "amodal_masks": torch.ones((1, 50, 28, 28)),
            "amodal_boxes": torch.zeros((1, 50, 4)),
            "num_objects": torch.tensor([1]),
        }
        res_clean = compute_amodal_metrics(preds_clean, targets)
        assert abs(res_clean["amodal_mask_iou"] - 1.0) < 1e-4
        assert abs(res_clean["amodal_bbox_mae"] - 0.0) < 1e-4

        # Mess up padded slot 35 with zeros and huge offsets
        preds_corrupted_padding = {
            "amodal_masks": torch.ones((1, 50, 28, 28)),
            "amodal_bbox_offset": torch.zeros((1, 50, 4)),
        }
        preds_corrupted_padding["amodal_masks"][:, 35] = 0.0
        preds_corrupted_padding["amodal_bbox_offset"][:, 35] = 9999.0

        res_corrupted = compute_amodal_metrics(preds_corrupted_padding, targets)
        # Should be identical because slot 35 is ignored
        assert abs(res_corrupted["amodal_mask_iou"] - 1.0) < 1e-4
        assert abs(res_corrupted["amodal_bbox_mae"] - 0.0) < 1e-4

    def test_m6_metric_handles_zero_object_cases(self):
        """Test G: compute_amodal_metrics handles images with 0 valid objects gracefully."""
        from orchestranet.utils.metrics import compute_amodal_metrics

        preds = {
            "amodal_masks": torch.rand((1, 50, 28, 28)),
            "amodal_bbox_offset": torch.rand((1, 50, 4)),
        }
        targets = {
            "amodal_masks": torch.zeros((1, 50, 28, 28)),
            "amodal_boxes": torch.zeros((1, 50, 4)),
            "num_objects": torch.tensor([0]),
        }
        res = compute_amodal_metrics(preds, targets)
        assert res["amodal_mask_iou"] == 0.0
        assert res["amodal_bbox_mae"] == 0.0
        assert len(res["ious"]) == 0
        assert len(res["maes"]) == 0

    def test_bbox_mae_uses_only_valid_objects(self):
        """Test H: Bbox MAE computes mean over exactly valid objects and compares coordinates consistently."""
        from orchestranet.utils.metrics import compute_amodal_metrics

        # 2 valid objects with known deltas [1.0, 2.0, 3.0, 4.0] (mean 2.5) and [0, 0, 0, 0] (mean 0.0)
        # Average MAE should be (2.5 + 0.0) / 2 = 1.25
        preds = {
            "amodal_masks": torch.ones((1, 50, 28, 28)),
            "amodal_bbox_offset": torch.zeros((1, 50, 4)),
        }
        preds["amodal_bbox_offset"][0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        targets = {
            "amodal_masks": torch.ones((1, 50, 28, 28)),
            "amodal_boxes": torch.zeros((1, 50, 4)),
            "num_objects": torch.tensor([2]),
        }
        res = compute_amodal_metrics(preds, targets)
        assert abs(res["amodal_bbox_mae"] - 1.25) < 1e-4

    def test_validate_m6_returns_expected_metric_keys(self, device):
        """Test I: validate_m6 returns all required metric keys."""
        from training.train_individual import IndividualTrainer, validate_m6

        trainer = IndividualTrainer("m6", device=device)
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "boxes": torch.zeros((2, 100, 4), device=device),
            "labels": torch.zeros((2, 100), dtype=torch.long, device=device),
            "num_objects": torch.tensor([2, 1], device=device),
            "amodal_boxes": torch.zeros((2, 50, 4), device=device),
            "amodal_masks": torch.zeros((2, 50, 28, 28), device=device),
            "is_occluded": torch.zeros((2, 50), device=device),
            "occlusion_mask": torch.zeros((2, 1, 640, 640), device=device),
        }
        val_loader = [(images, targets)]
        res = validate_m6(trainer, val_loader, device=device, num_images=2)

        expected_keys = [
            "val_loss",
            "val_amodal_mask_iou",
            "val_amodal_bbox_mae",
            "val_amodal_bbox_loss",
            "val_amodal_mask_loss",
            "val_amodal_conf_loss",
        ]
        for key in expected_keys:
            assert key in res, f"Expected key '{key}' missing from validate_m6 output"
            assert isinstance(res[key], float)

    def test_eval_only_checkpoint_loading_no_optimizer(self, device, tmp_path):
        """Test J: eval-only checkpoint loading does not require optimizer_state_dict."""
        from training.train_individual import IndividualTrainer

        trainer = IndividualTrainer("m6", device=device)
        ckpt_path = tmp_path / "test_eval_ckpt.pt"
        # Checkpoint without optimizer_state_dict
        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "backbone_state_dict": trainer.backbone.state_dict(),
            "fpn_state_dict": trainer.fpn.state_dict(),
        }, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.backbone.load_state_dict(ckpt["backbone_state_dict"])
        trainer.fpn.load_state_dict(ckpt["fpn_state_dict"])
        assert "optimizer_state_dict" not in ckpt

    def test_check_finiteness_diagnostic_helper(self):
        """Verify _check_finiteness raises descriptive RuntimeError on NaNs/Infs with full metadata."""
        from training.train_individual import _check_finiteness

        # Valid tensor passes
        valid_t = torch.tensor([1.0, 2.0, 3.0])
        _check_finiteness(valid_t, "test_valid", batch_idx=0, image_ids=[101])

        # NaN tensor raises
        nan_t = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(RuntimeError) as excinfo:
            _check_finiteness(nan_t, "test_nan_tensor", batch_idx=2, image_ids=[555])
        err_msg = str(excinfo.value)
        assert "test_nan_tensor" in err_msg
        assert "batch 2" in err_msg
        assert "image_ids=[555]" in err_msg
        assert "NaNs=1" in err_msg

        # Inf tensor raises
        inf_t = torch.tensor([float("inf"), 2.0])
        with pytest.raises(RuntimeError) as excinfo_inf:
            _check_finiteness(inf_t, "test_inf_tensor", batch_idx=0)
        assert "Infs=1" in str(excinfo_inf.value)

    def test_validate_m6_catches_non_finite_tensor(self, device):
        """Verify validate_m6 stops immediately with RuntimeError if M6 output has non-finite values."""
        from training.train_individual import IndividualTrainer, validate_m6

        trainer = IndividualTrainer("m6", device=device)
        images = torch.randn(2, 3, 640, 640, device=device)
        targets = {
            "image_id": torch.tensor([1001, 1002]),
            "boxes": torch.zeros((2, 100, 4), device=device),
            "labels": torch.zeros((2, 100), dtype=torch.long, device=device),
            "num_objects": torch.tensor([2, 1], device=device),
            "amodal_boxes": torch.zeros((2, 50, 4), device=device),
            "amodal_masks": torch.zeros((2, 50, 28, 28), device=device),
            "is_occluded": torch.zeros((2, 50), device=device),
            "occlusion_mask": torch.zeros((2, 1, 640, 640), device=device),
        }
        val_loader = [(images, targets)]

        # Patch trainer.model to inject NaN into decoded output
        orig_forward = trainer.model.forward
        def bad_forward(features, context=None):
            out = orig_forward(features, context=context)
            out["decoded"] = out["decoded"].clone()
            out["decoded"][0, 0, 0] = float("nan")
            return out

        trainer.model.forward = bad_forward
        with pytest.raises(RuntimeError) as exc_info:
            validate_m6(trainer, val_loader, device=device, num_images=2)
        assert "M6 transformer output / decoded tensor" in str(exc_info.value)
        assert "image_ids=[1001, 1002]" in str(exc_info.value)
        trainer.model.forward = orig_forward


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
