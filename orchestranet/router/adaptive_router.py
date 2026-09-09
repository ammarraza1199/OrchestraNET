"""
Adaptive Compute Router for OrchestraNet.

Dynamically selects which micro-models to activate per frame based on
estimated scene complexity. This is a key novelty — achieving variable
latency inference where simple scenes run fast and complex scenes get
full analysis.

Training: Uses Gumbel-Softmax for differentiable routing during training,
switches to hard thresholds at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complexity_estimator import SceneComplexityEstimator


# Routing profiles: which models activate at each complexity level
ROUTING_PROFILES = {
    "simple": ["m1"],                                        # ~2ms
    "medium": ["m1", "m2", "m7"],                           # ~5ms
    "complex": ["m1", "m2", "m3", "m4", "m5", "m6", "m7"],  # ~12ms
}

ALL_MODELS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]


class AdaptiveRouter(nn.Module):
    """
    Adaptive Compute Router — The Brain's Decision Engine.

    Estimates scene complexity and returns a routing decision indicating
    which micro-models should be activated for the current frame.

    Args:
        in_channels: FPN channel dimension.
        thresholds: (low, high) thresholds for routing levels.
        temperature: Gumbel-Softmax temperature (training only).
    """

    def __init__(
        self,
        in_channels: int = 128,
        thresholds: tuple[float, float] = (0.3, 0.7),
        temperature: float = 1.0,
    ):
        super().__init__()
        self.complexity_estimator = SceneComplexityEstimator(in_channels)
        self.low_thresh, self.high_thresh = thresholds
        self.temperature = temperature

        # For Gumbel-Softmax routing during training
        self.route_classifier = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # 3 routing levels
        )

    def forward(
        self,
        features: list[torch.Tensor],
    ) -> dict:
        """
        Determine routing decision.

        Args:
            features: FPN feature list [FP3, FP4, FP5]

        Returns:
            Dict with:
              - "complexity_score": (B, 1) raw complexity estimate
              - "routing_level": str ("simple"/"medium"/"complex")
              - "active_models": list of model IDs to activate
              - "routing_probs": (B, 3) softmax routing probabilities
        """
        complexity = self.complexity_estimator(features)  # (B, 1)

        if self.training:
            # Gumbel-Softmax for differentiable routing
            logits = self.route_classifier(complexity)
            routing_probs = F.gumbel_softmax(
                logits, tau=self.temperature, hard=True
            )
            # Determine which level is selected (argmax of one-hot)
            level_idx = routing_probs.argmax(dim=-1)[0].item()
        else:
            # Hard threshold routing at inference
            score = complexity[0, 0].item()
            if score < self.low_thresh:
                level_idx = 0
            elif score < self.high_thresh:
                level_idx = 1
            else:
                level_idx = 2
            routing_probs = torch.zeros(1, 3, device=complexity.device)
            routing_probs[0, level_idx] = 1.0

        level_names = ["simple", "medium", "complex"]
        routing_level = level_names[level_idx]
        active_models = ROUTING_PROFILES[routing_level]

        return {
            "complexity_score": complexity,
            "routing_level": routing_level,
            "active_models": active_models,
            "routing_probs": routing_probs,
            "level_idx": level_idx,
        }

    def get_model_activation_mask(self, routing_result: dict) -> dict[str, bool]:
        """Return a dict mapping each model ID to whether it should run."""
        active = set(routing_result["active_models"])
        return {m: (m in active) for m in ALL_MODELS}

    def update_temperature(self, decay: float = 0.95):
        """Decay Gumbel temperature for sharper routing during training."""
        self.temperature = max(0.1, self.temperature * decay)
