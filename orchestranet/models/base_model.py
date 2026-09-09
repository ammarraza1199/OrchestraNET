"""
Base Micro-Model Abstract Class for OrchestraNet.

All 7 specialized micro-models inherit from this base class, which enforces
a consistent interface for the orchestrator to manage them uniformly.

Key contracts:
  - Every model has a `forward()` that takes FPN features + optional context
  - Every model reports its parameter count and estimated latency
  - Every model can be individually enabled/disabled by the router
  - Every model defines its required input feature levels
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseMicroModel(ABC, nn.Module):
    """
    Abstract base class for all OrchestraNet micro-models.

    Subclasses must implement:
      - forward(): Run inference given FPN features and optional context
      - get_loss(): Compute task-specific training loss
      - required_levels(): Declare which FPN levels this model needs

    Attributes:
        model_id: Unique identifier (e.g., "m1", "m2", ...)
        model_name: Human-readable name
        is_active: Whether the router has activated this model for the current frame
    """

    def __init__(self, model_id: str, model_name: str):
        super().__init__()
        self.model_id = model_id
        self.model_name = model_name
        self.is_active = True  # Router can deactivate

    @abstractmethod
    def forward(
        self,
        features: list[torch.Tensor],
        context: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run model inference.

        Args:
            features: List of FPN feature tensors [FP3, FP4, FP5].
            context: Optional dictionary of outputs from other models
                     (e.g., M2 occlusion maps for M6 amodal completion).

        Returns:
            Dictionary of output tensors specific to this model's task.
        """
        ...

    @abstractmethod
    def get_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss for this model.

        Args:
            predictions: Output dict from forward().
            targets: Ground truth annotations.

        Returns:
            Dictionary of loss components (e.g., {"bbox_loss": ..., "cls_loss": ...}).
        """
        ...

    @abstractmethod
    def required_levels(self) -> list[str]:
        """
        Declare which FPN feature levels this model requires.
        Returns a list like ["P3", "P4"] or ["P5"].
        """
        ...

    def count_parameters(self) -> dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @torch.no_grad()
    def estimate_latency(
        self,
        features: list[torch.Tensor],
        num_runs: int = 100,
        warmup: int = 10,
    ) -> dict[str, float]:
        """
        Estimate inference latency in milliseconds.

        Args:
            features: Sample FPN features for benchmarking.
            num_runs: Number of timed runs.
            warmup: Number of warmup runs before timing.

        Returns:
            Dict with "mean_ms", "std_ms", "min_ms", "max_ms".
        """
        import time

        self.eval()
        device = next(self.parameters()).device

        # Warmup
        for _ in range(warmup):
            _ = self.forward(features)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = self.forward(features)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        import statistics
        return {
            "mean_ms": statistics.mean(times),
            "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_ms": min(times),
            "max_ms": max(times),
        }

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"id={self.model_id}, "
            f"name={self.model_name}, "
            f"params={params['total']:,}, "
            f"trainable={params['trainable']:,}, "
            f"active={self.is_active})"
        )
