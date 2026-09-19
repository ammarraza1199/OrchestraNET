"""
Tests for crash-safe, fully resumable M1 training and Drive synchronization.
"""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from orchestranet.utils.logger import TrainingLogger
from training.train_individual import (
    save_checkpoint_atomic,
    sync_file_to_drive,
    IndividualTrainer,
)


def test_save_checkpoint_atomic(tmp_path: Path):
    """Verify atomic checkpoint saving creates valid file and leaves no temp file."""
    ckpt_dir = tmp_path / "checkpoints"
    target_file = ckpt_dir / "m1_epoch000.pt"
    dummy_data = {"epoch": 0, "test_tensor": torch.tensor([1.0, 2.0, 3.0])}

    saved_path = save_checkpoint_atomic(dummy_data, target_file)
    assert saved_path == target_file
    assert target_file.is_file()
    assert not (ckpt_dir / "m1_epoch000.pt.tmp").exists()

    loaded = torch.load(target_file, weights_only=False)
    assert loaded["epoch"] == 0
    assert torch.equal(loaded["test_tensor"], dummy_data["test_tensor"])


def test_sync_file_to_drive_success(tmp_path: Path):
    """Verify sync_file_to_drive safely copies to destination atomically."""
    local_dir = tmp_path / "local"
    drive_dir = tmp_path / "drive" / "checkpoints"
    local_file = local_dir / "m1_latest.pt"

    dummy_data = {"epoch": 5, "model": "m1"}
    save_checkpoint_atomic(dummy_data, local_file)

    logger = TrainingLogger(log_dir=str(tmp_path / "logs"), tb_enabled=False)
    drive_dest = sync_file_to_drive(local_file, drive_dir, logger=logger)

    assert drive_dest is not None
    assert Path(drive_dest).is_file()
    assert Path(drive_dest).name == "m1_latest.pt"
    assert not (drive_dir / "m1_latest.pt.tmp").exists()

    loaded = torch.load(drive_dest, weights_only=False)
    assert loaded["epoch"] == 5
    logger.close()


def test_sync_file_to_drive_failure_non_blocking(tmp_path: Path):
    """Verify sync_file_to_drive logs error and returns None on failure without crashing or deleting local file."""
    local_file = tmp_path / "m1_epoch001.pt"
    save_checkpoint_atomic({"epoch": 1}, local_file)

    # Use a file as the drive directory to force a failure
    invalid_drive_dir = tmp_path / "not_a_dir.txt"
    invalid_drive_dir.write_text("blocking file")

    logger = TrainingLogger(log_dir=str(tmp_path / "logs"), tb_enabled=False)
    result = sync_file_to_drive(local_file, invalid_drive_dir, logger=logger)

    assert result is None
    # Source file must remain intact
    assert local_file.is_file()

    # Check that training.log recorded the failure
    log_content = (tmp_path / "logs" / "training.log").read_text()
    assert "Drive synchronisation failed" in log_content
    logger.close()


def test_training_logger_persistent_file_and_append(tmp_path: Path):
    """Verify TrainingLogger creates training.log with timestamps and appends across runs."""
    log_dir = tmp_path / "logs"
    logger1 = TrainingLogger(log_dir=str(log_dir), tb_enabled=False)
    logger1.info("Run 1 started")
    logger1.warning("Run 1 warning")
    logger1.error("Run 1 error")
    logger1.close()

    log_path = log_dir / "training.log"
    assert log_path.is_file()
    content1 = log_path.read_text(encoding="utf-8")
    assert " | INFO | Run 1 started" in content1
    assert " | WARN | Run 1 warning" in content1
    assert " | ERROR | Run 1 error" in content1

    # Open logger again (simulating resume)
    logger2 = TrainingLogger(log_dir=str(log_dir), tb_enabled=False)
    logger2.info("Run 2 resumed")
    logger2.close()

    content2 = log_path.read_text(encoding="utf-8")
    assert "Run 1 started" in content2
    assert "Run 2 resumed" in content2


def test_training_log_jsonl_completeness(tmp_path: Path):
    """Verify training_log.jsonl records complete epoch diagnostics including nulls when unavailable."""
    log_dir = tmp_path / "logs"
    logger = TrainingLogger(log_dir=str(log_dir), tb_enabled=False)

    epoch_record = {
        "epoch": 0,
        "avg_loss": 2.5,
        "lr": 0.001,
        "epoch_time_sec": 12.3,
        "checkpoint_path": "/path/to/m1_epoch000.pt",
        "drive_checkpoint_path": "/drive/m1_epoch000.pt",
        "best_map50": 0.35,
        "best_loss": 2.5,
        "gt_count": 100,
        "raw_preds": 50000,
        "after_conf": 1200,
        "after_nms": 150,
        "raw_score_min": 0.001,
        "raw_score_mean": 0.05,
        "raw_score_max": 0.98,
        "conf_score_min": 0.25,
        "conf_score_mean": 0.65,
        "conf_score_max": 0.98,
        "raw_box_bounds": [[0, 0, 10, 10], [100, 100, 200, 200]],
        "kept_box_bounds": [[5, 5, 20, 20], [90, 90, 180, 180]],
        "validation_images": 100,
    }
    logger.log_epoch(0, epoch_record)

    # Second epoch where validation was skipped
    epoch_record_no_val = {
        "epoch": 1,
        "avg_loss": 2.1,
        "lr": 0.0009,
        "epoch_time_sec": 11.8,
        "checkpoint_path": "/path/to/m1_epoch001.pt",
        "drive_checkpoint_path": None,
        "best_map50": 0.35,
        "best_loss": 2.1,
        "gt_count": None,
        "raw_preds": None,
        "after_conf": None,
        "after_nms": None,
        "raw_score_min": None,
        "raw_score_mean": None,
        "raw_score_max": None,
        "conf_score_min": None,
        "conf_score_mean": None,
        "conf_score_max": None,
        "raw_box_bounds": None,
        "kept_box_bounds": None,
        "validation_images": None,
    }
    logger.log_epoch(1, epoch_record_no_val)
    logger.close()

    jsonl_path = log_dir / "training_log.jsonl"
    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2

    # Check first record
    assert lines[0]["epoch"] == 0
    assert lines[0]["gt_count"] == 100
    assert lines[0]["drive_checkpoint_path"] == "/drive/m1_epoch000.pt"

    # Check second record uses null for unavailable metrics
    assert lines[1]["epoch"] == 1
    assert lines[1]["gt_count"] is None
    assert lines[1]["drive_checkpoint_path"] is None


def test_checkpoint_resume_state_restoration(tmp_path: Path):
    """Verify optimizer, scheduler, scaler, EMA, and global step states are saved and restored."""
    device = "cpu"
    trainer = IndividualTrainer(model_id="m1", device=device, fpn_channels=128)
    trainable_params = [p for p in trainer.all_params if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    # Step optimizer and scheduler a few times to create non-trivial state
    for _ in range(5):
        optimizer.zero_grad()
        # dummy loss
        loss = sum(p.sum() for p in trainable_params) * 0.0
        loss.backward()
        optimizer.step()
        scheduler.step()

    lr_after_5 = optimizer.param_groups[0]["lr"]

    ckpt_data = {
        "epoch": 4,
        "model_id": "m1",
        "model_state_dict": trainer.model.state_dict(),
        "backbone_state_dict": trainer.backbone.state_dict(),
        "fpn_state_dict": trainer.fpn.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "avg_loss": 1.234,
        "best_loss": 1.234,
        "best_val_loss": 2.345,
        "best_map50": 0.456,
        "global_step": 250,
        "args": {"model": "m1", "epochs": 50},
    }

    ckpt_path = tmp_path / "m1_epoch004.pt"
    save_checkpoint_atomic(ckpt_data, ckpt_path)

    # Now create fresh instances and restore
    trainer_restored = IndividualTrainer(model_id="m1", device=device, fpn_channels=128)
    optimizer_restored = torch.optim.AdamW(
        [p for p in trainer_restored.all_params if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
    scheduler_restored = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_restored, T_max=100)
    scaler_restored = torch.amp.GradScaler("cuda", enabled=False)

    ckpt_loaded = torch.load(ckpt_path, weights_only=False)

    trainer_restored.model.load_state_dict(ckpt_loaded["model_state_dict"])
    trainer_restored.backbone.load_state_dict(ckpt_loaded["backbone_state_dict"])
    trainer_restored.fpn.load_state_dict(ckpt_loaded["fpn_state_dict"])
    optimizer_restored.load_state_dict(ckpt_loaded["optimizer_state_dict"])
    scheduler_restored.load_state_dict(ckpt_loaded["scheduler_state_dict"])
    scaler_restored.load_state_dict(ckpt_loaded["scaler_state_dict"])

    resumed_epoch = ckpt_loaded["epoch"] + 1
    assert resumed_epoch == 5
    assert ckpt_loaded["global_step"] == 250
    assert ckpt_loaded["best_map50"] == 0.456
    assert ckpt_loaded["best_loss"] == 1.234

    # The learning rate must match the step at which it was saved
    resumed_lr = optimizer_restored.param_groups[0]["lr"]
    assert pytest.approx(resumed_lr, rel=1e-5) == lr_after_5


def test_m1_1epoch_training_and_resume_simulation(tmp_path: Path):
    """
    End-to-end simulation of 1-epoch training with Drive sync, followed by resuming
    from m1_latest.pt at epoch 1.
    """
    device = "cpu"
    save_dir = tmp_path / "checkpoints" / "individual"
    drive_dir = tmp_path / "drive" / "checkpoints" / "m1_50epoch"
    log_dir = tmp_path / "logs" / "m1"

    save_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Epoch 0 training simulation
    logger0 = TrainingLogger(log_dir=str(log_dir), tb_enabled=False)
    trainer0 = IndividualTrainer(model_id="m1", device=device, fpn_channels=128)
    trainable_params0 = [p for p in trainer0.all_params if p.requires_grad]
    optimizer0 = torch.optim.AdamW(trainable_params0, lr=1e-3)
    scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer0, T_max=50)
    scaler0 = torch.amp.GradScaler("cuda", enabled=False)
    from orchestranet.utils.ema import ModelEMA
    ema0 = ModelEMA(trainer0.model)

    # 1 batch step
    optimizer0.zero_grad()
    dummy_images = torch.randn(2, 3, 640, 640)
    bb = trainer0.backbone(dummy_images)
    fpn = trainer0.fpn(bb)
    out = trainer0.model(fpn)
    loss = out["decoded_boxes"].sum() * 0.0 + 1.5
    loss.backward()
    optimizer0.step()
    scheduler0.step()
    ema0.update(trainer0.model)

    epoch = 0
    best_loss = 1.5
    best_val_loss = float("inf")
    best_map50 = 0.42

    ckpt_data0 = {
        "epoch": epoch,
        "model_id": "m1",
        "model_state_dict": trainer0.model.state_dict(),
        "backbone_state_dict": trainer0.backbone.state_dict(),
        "fpn_state_dict": trainer0.fpn.state_dict(),
        "optimizer_state_dict": optimizer0.state_dict(),
        "scheduler_state_dict": scheduler0.state_dict(),
        "scaler_state_dict": scaler0.state_dict(),
        "avg_loss": 1.5,
        "best_loss": best_loss,
        "best_val_loss": best_val_loss,
        "best_map50": best_map50,
        "global_step": 1,
        "ema_state_dict": ema0.state_dict(),
        "args": {"model": "m1", "epochs": 50},
    }

    # Save epoch 0
    epoch0_ckpt = save_checkpoint_atomic(ckpt_data0, save_dir / f"m1_epoch{epoch:03d}.pt")
    sync_file_to_drive(epoch0_ckpt, drive_dir, logger=logger0)

    # Save latest
    latest_ckpt = save_checkpoint_atomic(ckpt_data0, save_dir / "m1_latest.pt")
    sync_file_to_drive(latest_ckpt, drive_dir, logger=logger0)

    # Save best
    best_ckpt = save_checkpoint_atomic(ckpt_data0, save_dir / "m1_best.pt")
    sync_file_to_drive(best_ckpt, drive_dir, logger=logger0)

    # Save best map50
    best_map_ckpt = save_checkpoint_atomic(ckpt_data0, save_dir / "m1_best_map50.pt")
    sync_file_to_drive(best_map_ckpt, drive_dir, logger=logger0)

    epoch_log0 = {
        "epoch": 0,
        "avg_loss": 1.5,
        "lr": optimizer0.param_groups[0]["lr"],
        "epoch_time_sec": 5.0,
        "checkpoint_path": str(epoch0_ckpt),
        "drive_checkpoint_path": str(drive_dir / epoch0_ckpt.name),
        "best_map50": best_map50,
        "best_loss": best_loss,
        "gt_count": 50,
        "raw_preds": 50000,
        "after_conf": 1000,
        "after_nms": 100,
        "raw_score_min": 0.001,
        "raw_score_mean": 0.05,
        "raw_score_max": 0.95,
        "conf_score_min": 0.25,
        "conf_score_mean": 0.6,
        "conf_score_max": 0.95,
        "raw_box_bounds": [[0, 0, 10, 10], [100, 100, 200, 200]],
        "kept_box_bounds": [[5, 5, 20, 20], [90, 90, 180, 180]],
        "validation_images": 100,
    }
    logger0.log_epoch(0, epoch_log0)
    logger0.close()

    # Verify files exist after epoch 0
    assert (save_dir / "m1_epoch000.pt").is_file()
    assert (drive_dir / "m1_epoch000.pt").is_file()
    assert (save_dir / "m1_latest.pt").is_file()
    assert (drive_dir / "m1_latest.pt").is_file()
    assert (save_dir / "m1_best.pt").is_file()
    assert (drive_dir / "m1_best.pt").is_file()
    assert (save_dir / "m1_best_map50.pt").is_file()
    assert (drive_dir / "m1_best_map50.pt").is_file()

    log_text = (log_dir / "training.log").read_text(encoding="utf-8")
    assert "Drive sync complete" in log_text

    jsonl_lines = [json.loads(l) for l in (log_dir / "training_log.jsonl").read_text(encoding="utf-8").strip().split("\n")]
    assert len(jsonl_lines) == 1
    assert jsonl_lines[0]["epoch"] == 0
    assert jsonl_lines[0]["drive_checkpoint_path"] == str(drive_dir / "m1_epoch000.pt")

    # Resume from Drive m1_latest.pt
    resume_path = drive_dir / "m1_latest.pt"
    ckpt_resumed = torch.load(resume_path, weights_only=False)

    start_epoch = ckpt_resumed["epoch"] + 1
    assert start_epoch == 1

    trainer1 = IndividualTrainer(model_id="m1", device=device, fpn_channels=128)
    trainer1.model.load_state_dict(ckpt_resumed["model_state_dict"])
    trainer1.backbone.load_state_dict(ckpt_resumed["backbone_state_dict"])
    trainer1.fpn.load_state_dict(ckpt_resumed["fpn_state_dict"])

    trainable_params1 = [p for p in trainer1.all_params if p.requires_grad]
    optimizer1 = torch.optim.AdamW(trainable_params1, lr=1e-3)
    optimizer1.load_state_dict(ckpt_resumed["optimizer_state_dict"])

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=50)
    scheduler1.load_state_dict(ckpt_resumed["scheduler_state_dict"])

    ema1 = ModelEMA(trainer1.model)
    ema1.load_state_dict(ckpt_resumed["ema_state_dict"])

    # Verify LR restored exactly
    assert pytest.approx(optimizer1.param_groups[0]["lr"], rel=1e-5) == optimizer0.param_groups[0]["lr"]
    assert ckpt_resumed["global_step"] == 1
    assert ckpt_resumed["best_map50"] == 0.42

