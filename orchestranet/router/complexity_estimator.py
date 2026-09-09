"""
Scene Complexity Estimator for the Adaptive Compute Router.

A lightweight CNN head (~50K params) that estimates scene complexity from
FPN features to determine which micro-models need activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneComplexityEstimator(nn.Module):
    """
    Estimates scene complexity on a scale of [0, 1].

    Low complexity (< 0.3): Few objects, no occlusion → M1 only
    Medium complexity (0.3–0.7): Some occlusion → M1 + M2 + M7
    High complexity (≥ 0.7): Heavy occlusion, small objects → ALL models

    Architecture: Global Average Pool → FC → ReLU → FC → Sigmoid
    """

    def __init__(self, in_channels: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fuse all FPN levels before estimating
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of FPN tensors [FP3, FP4, FP5]

        Returns:
            Complexity score: (B, 1) in [0, 1]
        """
        # Resize all to smallest spatial size (P5)
        target_size = features[-1].shape[2:]
        resized = [F.interpolate(f, size=target_size, mode="nearest") for f in features]
        fused = self.fuse(torch.cat(resized, dim=1))
        pooled = self.pool(fused).flatten(1)
        return self.fc(pooled)
