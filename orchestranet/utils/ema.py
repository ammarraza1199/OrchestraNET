"""
Exponential Moving Average (EMA) Model Wrapper for OrchestraNet.

Maintains a running exponential moving average of model weights during
training. EMA model typically achieves better generalization than the
final training weights.

Used during training: update after each step.
Used during evaluation: swap in EMA weights for inference.
"""

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model with exponentially smoothed weights.
    This reduces training noise and typically improves final model quality.

    Args:
        model: The PyTorch model to track.
        decay: EMA decay rate (higher = more smoothing). Default 0.9999.
        warmup_steps: Number of steps before EMA starts (use smaller decay initially).

    Usage:
        ema = ModelEMA(model)
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            ema.update(model)

        # Evaluate with EMA weights
        ema.apply_shadow(model)
        evaluate(model)
        ema.restore(model)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

        # Shadow copy of model parameters
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def _get_decay(self) -> float:
        """Compute effective decay with warmup ramp."""
        if self.updates < self.warmup_steps:
            return min(self.decay, (1 + self.updates) / (10 + self.updates))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters after an optimizer step."""
        self.updates += 1
        d = self._get_decay()

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(param.data, alpha=1 - d)

    def apply_shadow(self, model: nn.Module):
        """
        Swap model weights with EMA weights for evaluation.
        Original weights are backed up and can be restored with restore().
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original (non-EMA) model weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
            "decay": self.decay,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict):
        """Load EMA state from checkpoint."""
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        self.decay = state["decay"]
        self.updates = state["updates"]
