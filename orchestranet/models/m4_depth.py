"""
M4: Depth Estimator — Monocular Pseudo-Depth for Occlusion Ordering.

Produces relative depth maps to disambiguate foreground vs background
in occlusion scenarios. Uses a lightweight DPT-inspired architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseMicroModel


class LightweightTransformerBlock(nn.Module):
    """Efficient self-attention block for depth feature refinement."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_flat = x_flat + self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.transpose(1, 2).view(B, C, H, W)


class M4DepthEstimator(BaseMicroModel):
    """
    M4: Monocular Depth Estimator (~1M params).

    Uses P4+P5 features with lightweight transformer attention.
    Output: relative depth map (0=near, 1=far).
    """
    def __init__(self, in_channels=128, hidden_dim=64, num_heads=4):
        super().__init__(model_id="m4", model_name="Depth Estimator")

        # Feature fusion from P4 + P5
        self.p4_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True))
        self.p5_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.SiLU(inplace=True))

        # Transformer refinement
        self.transformer = LightweightTransformerBlock(hidden_dim, num_heads)

        # Depth decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, features, context=None):
        p4, p5 = features[1], features[2]
        f4 = self.p4_proj(p4)
        f5 = self.p5_proj(p5)
        f5_up = F.interpolate(f5, size=f4.shape[2:], mode="bilinear", align_corners=False)
        fused = f4 + f5_up
        refined = self.transformer(fused)
        depth = self.decoder(refined)
        return {"depth_map": depth, "depth_features": refined}

    def get_loss(self, predictions, targets):
        """
        Compute depth estimation loss:
          1. Scale-invariant log loss (when GT depth available)
          2. Edge-aware gradient smoothness (always active as regularizer)
        """
        device = predictions["depth_map"].device
        pred = predictions["depth_map"]  # (B, 1, H, W)

        # === Edge-Aware Gradient Smoothness Loss ===
        # Encourages smooth depth while preserving edges
        grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        smoothness_loss = grad_x.mean() + grad_y.mean()

        if "depth_gt" in targets:
            gt = F.interpolate(
                targets["depth_gt"], size=pred.shape[2:],
                mode="bilinear", align_corners=False
            )
            # Scale-invariant log loss
            diff = torch.log(pred + 1e-6) - torch.log(gt + 1e-6)
            si_loss = (diff ** 2).mean() - 0.5 * (diff.mean() ** 2)

            # Edge-aware weighting: reduce smoothness penalty at GT edges
            gt_grad_x = torch.abs(gt[:, :, :, :-1] - gt[:, :, :, 1:])
            gt_grad_y = torch.abs(gt[:, :, :-1, :] - gt[:, :, 1:, :])
            edge_weight_x = torch.exp(-gt_grad_x)
            edge_weight_y = torch.exp(-gt_grad_y)
            smooth_loss_weighted = (
                (edge_weight_x * grad_x).mean()
                + (edge_weight_y * grad_y).mean()
            )

            total = si_loss + 0.1 * smooth_loss_weighted
            return {
                "depth_loss": si_loss,
                "smoothness_loss": smooth_loss_weighted,
                "total_loss": total,
            }

        # Self-supervised: smoothness only (no GT depth available)
        return {
            "depth_loss": torch.tensor(0.0, device=device),
            "smoothness_loss": smoothness_loss,
            "total_loss": 0.01 * smoothness_loss,
        }

    def required_levels(self):
        return ["P4", "P5"]
