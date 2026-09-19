#!/usr/bin/env python3
"""
KITTI M4 Smoke Test Script.

This script is designed for execution in Google Colab (or any environment with
the real KITTI dataset) to smoke test the KITTIDepthDataset pipeline and M4 depth estimator.

Default dataset path: /content/data/KITTI/extracted
"""

import argparse
import sys
import torch
from torch.utils.data import DataLoader

from orchestranet.data.kitti_depth_dataset import KITTIDepthDataset
from orchestranet.models.m4_depth import M4DepthEstimator
from orchestranet.evaluation.depth_metrics import DepthMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="KITTI M4 Depth Pipeline Smoke Test")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/data/KITTI/extracted",
        help="Path to KITTI dataset root (default: /content/data/KITTI/extracted)",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Optional path to KITTI raw sequences root",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for DataLoader test (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run smoke test on (default: cuda if available else cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("========================================")
    print("KITTI M4 SMOKE TEST")
    print("========================================")
    print(f"Dataset root:\n{args.data_root}\n")

    # 1. Inspect Train Dataset
    print("TRAIN DATASET")
    try:
        train_dataset = KITTIDepthDataset(
            root=args.data_root,
            split="train",
            img_size=640,
            raw_root=args.raw_root,
        )
        print(f"depth files: {train_dataset.total_depth_files}")
        print(f"paired samples: {train_dataset.paired_samples}")
        print(f"missing RGB: {train_dataset.missing_rgb}")
        print(f"missing GT: {train_dataset.missing_gt}")
    except Exception as e:
        print(f"Error loading train dataset: {e}")
        train_dataset = None

    if train_dataset is None or len(train_dataset) == 0:
        print("\nTRAINING RGB PAIRS: 0")
        print("Note: Train RGB images are absent or not yet extracted; proceeding with validation dataset.\n")

    # 2. Inspect Val Dataset
    print("VAL DATASET")
    try:
        val_dataset = KITTIDepthDataset(
            root=args.data_root,
            split="val",
            img_size=640,
        )
        print(f"samples: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading val dataset: {e}")
        val_dataset = None

    if val_dataset is None or len(val_dataset) == 0:
        raise RuntimeError("Missing validation pairs: No valid validation samples found in depth_selection/val_selection_cropped!")

    # Determine active dataset for sample verification & DataLoader
    active_dataset = train_dataset if (train_dataset is not None and len(train_dataset) > 0) else val_dataset
    dataset_name = "train" if active_dataset is train_dataset else "val"
    print(f"\nInspecting first samples from [{dataset_name}] dataset:")

    num_samples_to_print = min(5, len(active_dataset))
    for i in range(num_samples_to_print):
        sample = active_dataset[i]
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise TypeError(f"Sample {i}: expected tuple of (image, targets), got {type(sample)}")

        image, targets = sample
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Sample {i}: image must be a torch.Tensor, got {type(image)}")
        if not isinstance(targets, dict):
            raise TypeError(f"Sample {i}: targets must be a dict, got {type(targets)}")

        required_keys = {"depth_gt", "valid_mask", "depth_meters"}
        if not required_keys.issubset(targets.keys()):
            raise KeyError(f"Sample {i}: targets missing required keys: {required_keys - targets.keys()}")

        # Check shapes
        if image.shape != torch.Size([3, 640, 640]):
            raise ValueError(f"Sample {i}: unexpected image shape {image.shape}, expected [3, 640, 640]")
        if targets["depth_gt"].shape != torch.Size([1, 640, 640]):
            raise ValueError(f"Sample {i}: unexpected depth_gt shape {targets['depth_gt'].shape}, expected [1, 640, 640]")
        if targets["depth_meters"].shape != torch.Size([1, 640, 640]):
            raise ValueError(f"Sample {i}: unexpected depth_meters shape {targets['depth_meters'].shape}, expected [1, 640, 640]")

        # Check NaN / Inf
        if torch.isnan(image).any() or torch.isinf(image).any():
            raise ValueError(f"Sample {i}: image contains NaN or Inf values!")
        if torch.isnan(targets["depth_gt"]).any() or torch.isinf(targets["depth_gt"]).any():
            raise ValueError(f"Sample {i}: depth_gt contains NaN or Inf values!")
        if torch.isnan(targets["depth_meters"]).any() or torch.isinf(targets["depth_meters"]).any():
            raise ValueError(f"Sample {i}: depth_meters contains NaN or Inf values!")

        valid_count = int(targets["valid_mask"].sum().item())
        depth_min = float(targets["depth_gt"].min().item())
        depth_max = float(targets["depth_gt"].max().item())
        metric_min = float(targets["depth_meters"].min().item())
        metric_max = float(targets["depth_meters"].max().item())

        print(f"\n--- Sample {i+1} ---")
        print(f"RGB shape: {list(image.shape)}")
        print(f"depth shape: {list(targets['depth_gt'].shape)}")
        print(f"valid pixel count: {valid_count}")
        print(f"depth min: {depth_min:.4f}")
        print(f"depth max: {depth_max:.4f}")
        print(f"metric depth min: {metric_min:.4f}m")
        print(f"metric depth max: {metric_max:.4f}m")

    # 3. DataLoader creation
    batch_size = min(args.batch_size, len(active_dataset))
    loader = DataLoader(active_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch_images, batch_targets = next(iter(loader))

    batch_images = batch_images.to(device)
    batch_targets = {k: v.to(device) for k, v in batch_targets.items()}

    # Initialize M4 model
    model = M4DepthEstimator().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 4. Forward pass
    forward_pass = False
    try:
        outputs = model(batch_images)
        pred_depth = outputs["depth"]
        if torch.isnan(pred_depth).any() or torch.isinf(pred_depth).any():
            raise ValueError("Forward pass produced NaN or Inf predictions!")
        print("\nforward: PASS")
        forward_pass = True
    except Exception as e:
        print(f"\nforward: FAIL ({e})")
        sys.exit(1)

    # 5. Loss computation
    loss_pass = False
    try:
        loss_dict = model.compute_loss(outputs, batch_targets)
        total_loss = loss_dict["loss"]
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(f"Loss is NaN or Inf: {loss_dict}")
        print(
            f"loss: PASS (total_loss={total_loss.item():.4f}, "
            f"depth_loss={loss_dict['depth_loss'].item():.4f}, "
            f"smoothness_loss={loss_dict['smoothness_loss'].item():.4f})"
        )
        loss_pass = True
    except Exception as e:
        print(f"loss: FAIL ({e})")
        sys.exit(1)

    # 6. Backward pass
    backward_pass = False
    try:
        optimizer.zero_grad()
        total_loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise ValueError(f"Gradient for {name} contains NaN or Inf!")
        print("backward: PASS")
        backward_pass = True
    except Exception as e:
        print(f"backward: FAIL ({e})")
        sys.exit(1)

    # 7. Optimizer step
    optimizer_pass = False
    try:
        optimizer.step()
        print("optimizer: PASS")
        optimizer_pass = True
    except Exception as e:
        print(f"optimizer: FAIL ({e})")
        sys.exit(1)

    # 8. DepthMetrics
    metrics_pass = False
    try:
        model.eval()
        with torch.no_grad():
            eval_outputs = model(batch_images)
            eval_pred = eval_outputs["depth"] * 80.0
            eval_gt = batch_targets["depth_meters"]
            eval_mask = batch_targets["valid_mask"]

            evaluator = DepthMetrics(max_depth=80.0, min_depth=1e-3)
            evaluator.update(eval_pred, eval_gt, mask=eval_mask)
            results = evaluator.compute()

            for metric_name, val in results.items():
                if torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val)):
                    raise ValueError(f"Metric '{metric_name}' is NaN or Inf: {val}")

            print("metrics: PASS")
            print("\nExplicit Metrics:")
            for metric_name, val in results.items():
                print(f"  {metric_name}: {val:.4f}")
            metrics_pass = True
    except Exception as e:
        print(f"metrics: FAIL ({e})")
        sys.exit(1)

    print("\n========================================")
    print("KITTI M4 SMOKE TEST SUMMARY")
    print("========================================")
    print(f"forward: {'PASS' if forward_pass else 'FAIL'}")
    print(f"loss: {'PASS' if loss_pass else 'FAIL'}")
    print(f"backward: {'PASS' if backward_pass else 'FAIL'}")
    print(f"optimizer: {'PASS' if optimizer_pass else 'FAIL'}")
    print(f"metrics: {'PASS' if metrics_pass else 'FAIL'}")
    print("========================================")


if __name__ == "__main__":
    main()
