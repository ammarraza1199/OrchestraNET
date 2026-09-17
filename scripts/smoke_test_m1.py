import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from orchestranet.backbone import MobileNetV4Backbone, LightweightFPN
from orchestranet.models import M1PrimaryDetector

def run_smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running M1 Engineering Smoke Test on {device} ===")

    B = 4
    # Create models
    backbone = MobileNetV4Backbone(pretrained=False).to(device)
    fpn = LightweightFPN(in_channels=backbone.get_out_channels(), out_channels=128).to(device)
    m1 = M1PrimaryDetector(in_channels=128, num_classes=80).to(device)

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    trainable_params = list(fpn.parameters()) + list(m1.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    # Dummy images (640x640)
    images = torch.randn(B, 3, 640, 640, device=device)

    # Ground truth boxes in 640x640 canvas coordinates
    targets = {
        "boxes": torch.tensor([
            [[50.0, 60.0, 200.0, 220.0], [300.0, 310.0, 450.0, 480.0], [0.0, 0.0, 0.0, 0.0]],
            [[100.0, 120.0, 180.0, 250.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[20.0, 30.0, 80.0, 90.0], [200.0, 200.0, 400.0, 400.0], [500.0, 500.0, 600.0, 600.0]],
            [[150.0, 150.0, 350.0, 350.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ], dtype=torch.float32, device=device),
        "labels": torch.tensor([
            [0, 15, 0],
            [62, 0, 0],
            [1, 5, 79],
            [44, 0, 0],
        ], dtype=torch.long, device=device),
        "num_objects": torch.tensor([2, 1, 3, 1]),
    }

    # Forward
    optimizer.zero_grad()
    bb_feats = backbone(images)
    fpn_feats = fpn(bb_feats)
    predictions = m1(fpn_feats)

    # Shape checks
    assert predictions["decoded_boxes"].shape == (B, 25200, 4), f"Unexpected box shape: {predictions['decoded_boxes'].shape}"
    assert predictions["objectness"].shape == (B, 25200, 1), f"Unexpected obj shape: {predictions['objectness'].shape}"
    assert predictions["class_logits"].shape == (B, 25200, 80), f"Unexpected cls shape: {predictions['class_logits'].shape}"
    print(f"Forward pass successful. Output decoded_boxes shape: {list(predictions['decoded_boxes'].shape)}")

    # Loss
    losses = m1.get_loss(predictions, targets)
    total_loss = losses["total_loss"]
    bbox_loss = losses["bbox_loss"]
    cls_loss = losses["cls_loss"]
    obj_loss = losses["obj_loss"]
    num_pos = losses["num_pos"]

    print(f"Losses: total={total_loss.item():.4f}, bbox={bbox_loss.item():.4f}, cls={cls_loss.item():.4f}, obj={obj_loss.item():.4f}, num_pos={num_pos.item()}")

    assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss}"
    assert torch.isfinite(bbox_loss), f"Bbox loss is not finite: {bbox_loss}"
    assert torch.isfinite(cls_loss), f"Cls loss is not finite: {cls_loss}"
    assert torch.isfinite(obj_loss), f"Obj loss is not finite: {obj_loss}"
    assert num_pos.item() > 0, f"Expected num_pos > 0, got {num_pos.item()}"

    # Backward
    total_loss.backward()

    # Gradient check
    grad_norms = []
    for p in trainable_params:
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), "Non-finite gradient encountered!"
            grad_norms.append(p.grad.norm().item())

    assert len(grad_norms) > 0, "No gradients computed!"
    print(f"Backward pass successful. Total parameter tensors with finite gradients: {len(grad_norms)}")

    # Optimizer step
    nn.utils.clip_grad_norm_(trainable_params, 10.0)
    optimizer.step()
    print("Optimizer step successful.")
    print("=== M1 Engineering Smoke Test: PASSED! ===")

if __name__ == "__main__":
    run_smoke_test()
