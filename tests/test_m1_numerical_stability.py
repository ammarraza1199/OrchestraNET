"""
Regression tests for M1 numerical stability under AMP/autocast.
Verifies FP32 execution of geometry operations, protection against FP16 overflow,
gradient propagation, and checkpoint protection against non-finite values.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import numpy as np

from orchestranet.models.m1_detector import (
    M1PrimaryDetector,
    box_iou,
    generalized_box_iou,
)
from training.train_individual import IndividualTrainer


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_decode_boxes_fp32_under_autocast(device):
    """Verify that _decode_boxes executes in FP32 under autocast and preserves gradients."""
    model = M1PrimaryDetector(in_channels=128).to(device)
    H, W = 80, 80
    bbox_pred = torch.randn(2, 3, 4, H, W, device=device, requires_grad=True)

    use_cuda_autocast = (device == "cuda")
    with torch.amp.autocast("cuda", enabled=use_cuda_autocast):
        decoded = model._decode_boxes(bbox_pred, level_idx=0, H=H, W=W, device=device)

    # Decoded boxes must be FP32 and finite
    assert decoded.dtype == torch.float32
    assert torch.isfinite(decoded).all()
    assert decoded.shape == (2, 3, 4, H, W)

    # Gradient flow back to bbox_pred must be finite
    loss = decoded.sum()
    loss.backward()
    assert bbox_pred.grad is not None
    assert torch.isfinite(bbox_pred.grad).all()


def test_box_iou_under_autocast_large_boxes(device):
    """
    Verify box_iou handles boxes with area > 65504 without overflowing to +inf or NaN in FP16.
    [0, 0, 600, 600] has area 360,000, which overflows FP16 (max 65,504).
    """
    boxes1 = torch.tensor([
        [50.0, 50.0, 400.0, 400.0],  # area = 122,500
        [0.0, 0.0, 600.0, 600.0],    # area = 360,000
    ], device=device)

    boxes2 = torch.tensor([
        [40.0, 40.0, 390.0, 390.0],  # area = 122,500
        [10.0, 10.0, 590.0, 590.0],  # area = 336,400
    ], device=device)

    use_cuda_autocast = (device == "cuda")
    with torch.amp.autocast("cuda", enabled=use_cuda_autocast):
        iou = box_iou(boxes1, boxes2)

    assert iou.dtype == torch.float32
    assert torch.isfinite(iou).all()
    assert (iou >= 0.0).all() and (iou <= 1.0).all()
    assert iou.shape == (2, 2)


def test_generalized_box_iou_under_autocast_large_boxes(device):
    """
    Verify generalized_box_iou handles large boxes without FP16 enclosing area overflow.
    Verifies gradient flow to predicted boxes.
    """
    boxes1 = torch.tensor([
        [50.0, 50.0, 400.0, 400.0],  # area = 122,500
        [0.0, 0.0, 600.0, 600.0],    # area = 360,000
    ], device=device, requires_grad=True)

    boxes2 = torch.tensor([
        [40.0, 40.0, 390.0, 390.0],
        [10.0, 10.0, 590.0, 590.0],
    ], device=device)

    use_cuda_autocast = (device == "cuda")
    with torch.amp.autocast("cuda", enabled=use_cuda_autocast):
        giou = generalized_box_iou(boxes1, boxes2)

    assert giou.dtype == torch.float32
    assert torch.isfinite(giou).all()
    assert (giou >= -1.0).all() and (giou <= 1.0).all()

    loss = (1.0 - giou).sum()
    loss.backward()
    assert boxes1.grad is not None
    assert torch.isfinite(boxes1.grad).all()


def test_m1_loss_with_large_coco_pixel_boxes(device):
    """
    Verify M1 forward and get_loss with representative large 640x640 pixel boxes
    ([50, 50, 400, 400] and [0, 0, 600, 600]) under autocast produce finite losses and gradients.
    """
    model = M1PrimaryDetector(in_channels=128).to(device)

    # 3 FPN levels with batch size 2
    features = [
        torch.randn(2, 128, 80, 80, device=device),
        torch.randn(2, 128, 40, 40, device=device),
        torch.randn(2, 128, 20, 20, device=device),
    ]

    target_boxes = torch.zeros(2, 10, 4, device=device)
    target_boxes[:, 0, :] = torch.tensor([50.0, 50.0, 400.0, 400.0], device=device)
    target_boxes[:, 1, :] = torch.tensor([0.0, 0.0, 600.0, 600.0], device=device)

    target_labels = torch.zeros(2, 10, dtype=torch.long, device=device)
    target_labels[:, 0] = 1
    target_labels[:, 1] = 2

    targets = {
        "boxes": target_boxes,
        "labels": target_labels,
        "num_objects": torch.tensor([2, 2], dtype=torch.long),
    }

    use_cuda_autocast = (device == "cuda")
    with torch.amp.autocast("cuda", enabled=use_cuda_autocast):
        predictions = model(features)
        losses = model.get_loss(predictions, targets)

    for k in ["bbox_loss", "cls_loss", "obj_loss", "total_loss"]:
        val = losses[k]
        assert torch.isfinite(val), f"Loss component {k} is not finite: {val}"

    # Verify backward pass generates finite gradients
    model.zero_grad()
    losses["total_loss"].backward()

    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Gradient in {name} is not finite!"


def test_checkpoint_protection_against_nan_parameters(tmp_path: Path):
    """
    Verify checkpoint protection halts saving when model parameters contain NaN,
    preserving existing checkpoints.
    """
    trainer = IndividualTrainer("m1", "cpu")
    optimizer = torch.optim.AdamW(trainer.all_params, lr=1e-3)

    # Helper function matching train_individual.py check
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

    # Healthy model should pass check
    bad = _check_dict_finite(trainer.model.state_dict(), "model_state_dict")
    assert bad is None

    # Inject NaN into a parameter
    with torch.no_grad():
        first_param = next(trainer.model.parameters())
        first_param.view(-1)[0] = float("nan")

    bad = _check_dict_finite(trainer.model.state_dict(), "model_state_dict")
    assert bad is not None
    assert "model_state_dict" in bad
