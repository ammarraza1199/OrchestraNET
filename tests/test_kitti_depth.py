"""
Unit tests for KITTI Depth Dataset pipeline, M4 model contract, loss masking, and depth metrics.
These tests use temporary synthetic directories and do NOT require the real Colab KITTI dataset.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from orchestranet.backbone import MobileNetV4Backbone, LightweightFPN
from orchestranet.data.kitti_depth_dataset import KITTIDepthDataset, DEPTH_SCALE, MAX_DEPTH
from orchestranet.models.m4_depth import M4DepthEstimator
from orchestranet.evaluation.depth_metrics import DepthMetrics


def create_synthetic_depth_png(path: Path, width: int = 120, height: int = 40, depth_value_m: float = 20.0):
    """Create a synthetic 16-bit depth PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_val = int(depth_value_m * DEPTH_SCALE)
    arr = np.zeros((height, width), dtype=np.uint16)
    # Put depth in the central region, leave border 0
    arr[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = raw_val
    img = Image.fromarray(arr)
    img.save(str(path))
    return arr


def create_synthetic_rgb_png(path: Path, width: int = 120, height: int = 40, color: tuple = (100, 150, 200)):
    """Create a synthetic RGB PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color)
    img.save(str(path))


# ============================================================
# Phase 9: Tests A - Q
# ============================================================

def test_train_pairing_image_02(tmp_path: Path):
    """Test A: Training image_02 pairing discovery."""
    seq_name = "2011_09_26_drive_0001_sync"
    gt_p = tmp_path / "train" / seq_name / "proj_depth" / "groundtruth" / "image_02" / "0000000001.png"
    rgb_p = tmp_path / "train" / seq_name / "image_02" / "data" / "0000000001.png"

    create_synthetic_depth_png(gt_p)
    create_synthetic_rgb_png(rgb_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="train", img_size=640)
    assert len(dataset) == 1
    assert dataset.image_02_count == 1
    assert dataset.paired_samples == 1
    assert dataset.missing_rgb == 0
    assert dataset.pairs[0]["camera"] == "image_02"


def test_train_pairing_image_03(tmp_path: Path):
    """Test B: Training image_03 pairing discovery."""
    seq_name = "2011_09_26_drive_0001_sync"
    gt_p = tmp_path / "train" / seq_name / "proj_depth" / "groundtruth" / "image_03" / "0000000002.png"
    rgb_p = tmp_path / "train" / seq_name / "image_03" / "data" / "0000000002.png"

    create_synthetic_depth_png(gt_p)
    create_synthetic_rgb_png(rgb_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="train", img_size=640)
    assert len(dataset) == 1
    assert dataset.image_03_count == 1
    assert dataset.pairs[0]["camera"] == "image_03"


def test_missing_rgb_handling(tmp_path: Path):
    """Test C: Missing RGB handling skips sample without crash and increments missing_rgb."""
    seq_name = "2011_09_26_drive_0001_sync"
    gt_p = tmp_path / "train" / seq_name / "proj_depth" / "groundtruth" / "image_02" / "0000000003.png"
    create_synthetic_depth_png(gt_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="train", img_size=640)
    assert len(dataset) == 0
    assert dataset.missing_rgb == 1
    assert dataset.paired_samples == 0


def test_missing_gt_handling(tmp_path: Path):
    """Test D: Missing GT depth in validation mode increments missing_gt."""
    val_root = tmp_path / "depth_selection" / "val_selection_cropped"
    rgb_p = val_root / "image" / "2011_09_26_drive_0002_sync_image_0000000005_image_02.png"
    create_synthetic_rgb_png(rgb_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="val", img_size=640)
    assert len(dataset) == 0
    assert dataset.missing_gt == 1


def test_validation_filename_parsing(tmp_path: Path):
    """Test E: Validation filename parsing matches _image_ to _groundtruth_depth_."""
    val_root = tmp_path / "depth_selection" / "val_selection_cropped"
    rgb_p = val_root / "image" / "2011_09_26_drive_0002_sync_image_0000000005_image_02.png"
    gt_p = val_root / "groundtruth_depth" / "2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png"

    create_synthetic_rgb_png(rgb_p)
    create_synthetic_depth_png(gt_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="val", img_size=640)
    assert len(dataset) == 1
    sample = dataset.pairs[0]
    assert sample["sequence"] == "2011_09_26_drive_0002_sync"
    assert sample["frame"] == "0000000005"
    assert sample["camera"] == "image_02"


def test_uint16_depth_preservation_and_conversion(tmp_path: Path):
    """Test F & G: uint16 precision is preserved and depth is converted to meters and normalized [0, 1]."""
    seq_name = "2011_09_26_drive_0001_sync"
    gt_p = tmp_path / "train" / seq_name / "proj_depth" / "groundtruth" / "image_02" / "0000000001.png"
    rgb_p = tmp_path / "train" / seq_name / "image_02" / "data" / "0000000001.png"

    target_depth_m = 25.6  # 25.6 * 256.0 = 6553.6 -> raw 6554 uint16
    create_synthetic_depth_png(gt_p, depth_value_m=target_depth_m)
    create_synthetic_rgb_png(rgb_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="train", img_size=640)
    img_tensor, targets = dataset[0]

    depth_meters = targets["depth_meters"]
    depth_gt = targets["depth_gt"]
    valid_mask = targets["valid_mask"]

    # In the central valid region:
    valid_px = valid_mask[0]
    assert valid_px.any()
    measured_m = depth_meters[0][valid_px].mean().item()
    assert pytest.approx(measured_m, abs=0.1) == target_depth_m

    expected_norm = target_depth_m / MAX_DEPTH
    measured_norm = depth_gt[0][valid_px].mean().item()
    assert pytest.approx(measured_norm, abs=0.01) == expected_norm


def test_zero_invalid_depth_mask(tmp_path: Path):
    """Test H: Zero/unmeasured depth pixels produce False in valid_mask and 0 in depth_gt."""
    seq_name = "2011_09_26_drive_0001_sync"
    gt_p = tmp_path / "train" / seq_name / "proj_depth" / "groundtruth" / "image_02" / "0000000001.png"
    rgb_p = tmp_path / "train" / seq_name / "image_02" / "data" / "0000000001.png"

    create_synthetic_depth_png(gt_p, depth_value_m=10.0)
    create_synthetic_rgb_png(rgb_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="train", img_size=640)
    _, targets = dataset[0]

    valid_mask = targets["valid_mask"][0]
    depth_gt = targets["depth_gt"][0]

    # Corner of the canvas (padding area) must be invalid
    assert not valid_mask[0, 0].item()
    assert depth_gt[0, 0].item() == 0.0


def test_letterbox_geometry_and_spatial_alignment(tmp_path: Path):
    """Test I & J: RGB and depth undergo identical aspect-ratio letterboxing and spatial alignment."""
    val_root = tmp_path / "depth_selection" / "val_selection_cropped"
    rgb_p = val_root / "image" / "seq1_image_0000000001_image_02.png"
    gt_p = val_root / "groundtruth_depth" / "seq1_groundtruth_depth_0000000001_image_02.png"

    # 120 x 40 aspect ratio (3:1)
    w, h = 120, 40
    create_synthetic_rgb_png(rgb_p, width=w, height=h)
    create_synthetic_depth_png(gt_p, width=w, height=h, depth_value_m=15.0)

    dataset = KITTIDepthDataset(root=tmp_path, split="val", img_size=640)
    img_tensor, targets = dataset[0]

    # Check canvas shape
    assert img_tensor.shape == (3, 640, 640)
    assert targets["depth_gt"].shape == (1, 640, 640)

    # Scale: 640 / 120 = 5.333 -> new_w = 640, new_h = round(40 * 640/120) = 213
    # pad_y = (640 - 213) / 2 = 213.5 -> int(213)
    # The top padding rows must have 0 valid depth
    assert not targets["valid_mask"][0, :50, :].any()
    # Center rows must have valid depth
    assert targets["valid_mask"][0, 320, 320].item()


def test_dataset_output_shape_and_types(tmp_path: Path):
    """Test K: Dataset return contract adheres to required tensor shapes and types."""
    val_root = tmp_path / "depth_selection" / "val_selection_cropped"
    rgb_p = val_root / "image" / "seq1_image_0000000001_image_02.png"
    gt_p = val_root / "groundtruth_depth" / "seq1_groundtruth_depth_0000000001_image_02.png"

    create_synthetic_rgb_png(rgb_p)
    create_synthetic_depth_png(gt_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="val", img_size=640)
    img_tensor, targets = dataset[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dtype == torch.float32
    assert img_tensor.shape == (3, 640, 640)

    assert "depth_gt" in targets
    assert targets["depth_gt"].dtype == torch.float32
    assert targets["depth_gt"].shape == (1, 640, 640)

    assert "valid_mask" in targets
    assert targets["valid_mask"].dtype == torch.bool
    assert targets["valid_mask"].shape == (1, 640, 640)

    assert "depth_meters" in targets
    assert targets["depth_meters"].dtype == torch.float32
    assert targets["depth_meters"].shape == (1, 640, 640)


def test_dataloader_batching(tmp_path: Path):
    """Test L: DataLoader batching works properly with multiple samples."""
    val_root = tmp_path / "depth_selection" / "val_selection_cropped"
    for i in range(3):
        rgb_p = val_root / "image" / f"seq1_image_{i:010d}_image_02.png"
        gt_p = val_root / "groundtruth_depth" / f"seq1_groundtruth_depth_{i:010d}_image_02.png"
        create_synthetic_rgb_png(rgb_p)
        create_synthetic_depth_png(gt_p)

    dataset = KITTIDepthDataset(root=tmp_path, split="val", img_size=640)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 640, 640)
    assert targets["depth_gt"].shape == (2, 1, 640, 640)
    assert targets["valid_mask"].shape == (2, 1, 640, 640)
    assert targets["depth_meters"].shape == (2, 1, 640, 640)


def test_m4_forward_pass():
    """Test M: M4DepthEstimator forward pass with FPN features."""
    model = M4DepthEstimator(in_channels=128)
    model.eval()

    # Features: [P3 (stride 8), P4 (stride 16), P5 (stride 32)]
    # For 640x640: P3=80x80, P4=40x40, P5=20x20
    features = [
        torch.randn(2, 128, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 128, 20, 20),
    ]

    out = model(features)
    assert "depth_map" in out
    assert "depth_features" in out
    assert out["depth_map"].shape == (2, 1, 40, 40)
    # Output of sigmoid must be strictly in (0, 1)
    assert (out["depth_map"] >= 0.0).all() and (out["depth_map"] <= 1.0).all()


def test_m4_loss_zero_depth_masking_regression():
    """Test N: Regression test proving zero GT depth does not corrupt scale-invariant loss."""
    model = M4DepthEstimator(in_channels=128)
    pred_depth = torch.full((1, 1, 40, 40), 0.5)
    predictions = {"depth_map": pred_depth}

    # Case 1: 50% valid depth (0.5), 50% zero depth (0.0)
    gt_sparse = torch.zeros((1, 1, 40, 40))
    gt_sparse[:, :, :20, :] = 0.5
    valid_mask = gt_sparse > 0.0

    targets1 = {"depth_gt": gt_sparse, "valid_mask": valid_mask}
    loss1 = model.get_loss(predictions, targets1)

    # In the valid region, pred == gt == 0.5, so diff == 0, si_loss should be 0.0
    assert pytest.approx(loss1["depth_loss"].item(), abs=1e-5) == 0.0

    # Case 2: Changing the invalid zero values to other arbitrary zeros/negatives
    # must NOT alter the depth loss
    gt_sparse2 = gt_sparse.clone()
    gt_sparse2[:, :, 20:, :] = -1.0  # Invalid negative
    targets2 = {"depth_gt": gt_sparse2, "valid_mask": valid_mask}
    loss2 = model.get_loss(predictions, targets2)

    assert pytest.approx(loss1["depth_loss"].item(), abs=1e-5) == loss2["depth_loss"].item()
    assert torch.isfinite(loss1["total_loss"])


def test_m4_backward_and_finite_gradients():
    """Test O & P: M4 loss backward pass succeeds and produces finite gradients."""
    model = M4DepthEstimator(in_channels=128)
    model.train()

    features = [
        torch.randn(2, 128, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 128, 20, 20),
    ]

    out = model(features)

    # Ground truth with sparse valid pixels
    gt = torch.zeros((2, 1, 640, 640))
    gt[:, :, 200:400, 200:400] = 0.3
    targets = {"depth_gt": gt, "valid_mask": (gt > 1e-3)}

    losses = model.get_loss(out, targets)
    total_loss = losses["total_loss"]
    assert torch.isfinite(total_loss)

    total_loss.backward()

    # Verify all trainable parameter gradients are finite
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_depth_metrics_evaluation():
    """Test Q: DepthMetrics calculates AbsRel, RMSE, and delta1 accurately over valid pixels."""
    metrics = DepthMetrics(min_depth=0.001, max_depth=80.0)

    # 1 image with known pred and GT
    gt = np.array([[10.0, 20.0], [0.0, 40.0]], dtype=np.float32)  # one 0 pixel
    pred = np.array([[10.0, 22.0], [5.0, 38.0]], dtype=np.float32)

    metrics.update(pred, gt)
    res = metrics.compute()

    assert res["n_valid_pixels"] == 3  # pixel with gt=0 is excluded
    assert pytest.approx(res["AbsRel"], abs=1e-3) == (0.0 + 2.0 / 20.0 + 2.0 / 40.0) / 3.0
    assert pytest.approx(res["d1"], abs=1e-3) == 1.0  # max(22/20, 20/22) = 1.1 < 1.25
    assert np.isfinite(res["RMSE"])
    assert np.isfinite(res["SILog"])


def test_m4_full_pipeline_batch_size_2():
    """Test full pipeline integration: batch of 2 images -> Backbone -> FPN -> M4."""
    backbone = MobileNetV4Backbone(pretrained=False)
    fpn = LightweightFPN(in_channels=backbone.get_out_channels(), out_channels=128)
    model = M4DepthEstimator(in_channels=128)

    images = torch.randn(2, 3, 640, 640)
    backbone_features = backbone(images)
    fpn_features = fpn(backbone_features)

    assert len(fpn_features) == 3
    assert fpn_features[0].shape == (2, 128, 80, 80)
    assert fpn_features[1].shape == (2, 128, 40, 40)
    assert fpn_features[2].shape == (2, 128, 20, 20)

    outputs = model(fpn_features)
    assert "depth_map" in outputs
    assert outputs["depth_map"].shape == (2, 1, 40, 40)


def test_m4_features_input_contract_regression():
    """
    Test reproducing the IndexError / TypeError when passing incorrect feature shapes.
    Passing a raw image Tensor of shape [2, 3, 640, 640] raises TypeError with helpful guidance.
    Passing [P4, P5] (2 levels) or [P3, P4, P5] (3 levels) executes forward successfully.
    """
    model = M4DepthEstimator(in_channels=128)

    # 1. Raw image tensor passed directly should fail informatively with TypeError
    raw_tensor = torch.randn(2, 3, 640, 640)
    with pytest.raises(TypeError, match="M4DepthEstimator expects FPN features"):
        model(raw_tensor)

    # 2. 2-level features [P4, P5] should work
    two_levels = [
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 128, 20, 20),
    ]
    out2 = model(two_levels)
    assert "depth_map" in out2
    assert out2["depth_map"].shape == (2, 1, 40, 40)

    # 3. 3-level features [P3, P4, P5] should work
    three_levels = [
        torch.randn(2, 128, 80, 80),
        torch.randn(2, 128, 40, 40),
        torch.randn(2, 128, 20, 20),
    ]
    out3 = model(three_levels)
    assert "depth_map" in out3
    assert out3["depth_map"].shape == (2, 1, 40, 40)

