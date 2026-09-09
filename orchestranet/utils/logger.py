"""
Unified Logger for OrchestraNet Training.

Provides consistent logging to:
  - Console (with progress formatting)
  - TensorBoard (scalar plots, images, histograms)
  - JSON log files (for post-hoc analysis)

Usage:
    logger = TrainingLogger(log_dir="logs/run_001", tb_enabled=True)
    logger.log_scalar("train/loss", 0.5, step=100)
    logger.log_scalars("train", {"bbox": 0.3, "cls": 0.2}, step=100)
    logger.info("Epoch 5 complete")
"""

import json
import os
import sys
import time
from pathlib import Path


class TrainingLogger:
    """
    Multi-backend training logger.

    Args:
        log_dir: Directory for log files and TensorBoard.
        tb_enabled: Whether to log to TensorBoard.
        print_freq: Print to console every N steps.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        tb_enabled: bool = True,
        print_freq: int = 50,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.print_freq = print_freq
        self.tb_writer = None

        if tb_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
            except ImportError:
                print("⚠️  TensorBoard not available, logging to console only")

        # JSON log file
        self.json_path = self.log_dir / "training_log.jsonl"
        self._step_data = {}

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a single scalar value."""
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
        self._step_data[tag] = value

    def log_scalars(self, group: str, values: dict, step: int):
        """Log multiple scalars under a group."""
        for k, v in values.items():
            tag = f"{group}/{k}"
            if self.tb_writer:
                self.tb_writer.add_scalar(tag, v, step)
            self._step_data[tag] = v

    def log_lr(self, lr: float, step: int):
        """Log learning rate."""
        self.log_scalar("train/lr", lr, step)

    def log_epoch(self, epoch: int, metrics: dict):
        """Log end-of-epoch metrics to JSON."""
        record = {"epoch": epoch, "timestamp": time.time(), **metrics}
        with open(self.json_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def info(self, msg: str):
        """Print info message."""
        print(f"[INFO] {msg}")

    def warning(self, msg: str):
        """Print warning message."""
        print(f"[WARN] {msg}", file=sys.stderr)

    def log_model_graph(self, model, dummy_input):
        """Log model graph to TensorBoard."""
        if self.tb_writer:
            try:
                self.tb_writer.add_graph(model, dummy_input)
            except Exception:
                pass  # Graph logging is best-effort

    def flush(self):
        """Flush TensorBoard writer."""
        if self.tb_writer:
            self.tb_writer.flush()

    def close(self):
        """Close all loggers."""
        if self.tb_writer:
            self.tb_writer.close()


class AverageMeter:
    """
    Computes and stores the average and current value.

    Usage:
        meter = AverageMeter("loss")
        for batch in loader:
            meter.update(loss.item(), batch_size)
        print(meter)  # "loss: 0.3456 (avg: 0.3821)"
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"
