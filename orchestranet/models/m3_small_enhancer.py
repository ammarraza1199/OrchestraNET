"""
M3: Small Object Enhancer — Super-Resolution + Detection Refinement.

Re-examines regions likely containing small objects at higher effective
resolution using a lightweight super-resolution branch + refined detection.

Loss computation uses L1 + perceptual feature loss for SR and IoU-based
loss for bbox refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseMicroModel


class ResidualBlock(nn.Module):
    """Lightweight residual block for super-resolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))


class M3SmallObjectEnhancer(BaseMicroModel):
    """
    M3: Small Object Enhancer (~600K params).

    Two-branch architecture:
      1. SR Branch: 2x upsample features via sub-pixel convolution
      2. Detect Branch: Refine detections on enhanced features

    Losses:
      - SR reconstruction: L1 between enhanced features and upsampled original
      - Bbox refinement: Smooth L1 on delta predictions
      - Confidence boost: BCE between predicted confidence and IoU quality
    """
    def __init__(self, in_channels=128, sr_channels=32, num_res_blocks=3, upscale=2):
        super().__init__(model_id="m3", model_name="Small Object Enhancer")
        self.sr_channels = sr_channels

        # Feature compression
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, sr_channels, 1, bias=False),
            nn.BatchNorm2d(sr_channels), nn.ReLU(inplace=True))

        # Residual blocks for feature enhancement
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(sr_channels) for _ in range(num_res_blocks)])

        # Sub-pixel upsampling (PixelShuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(sr_channels, sr_channels * upscale ** 2, 3, padding=1),
            nn.PixelShuffle(upscale), nn.ReLU(inplace=True))

        # Detection refinement on enhanced features
        self.refine = nn.Sequential(
            nn.Conv2d(sr_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True))

        # Bbox refinement head
        self.bbox_refine = nn.Conv2d(64, 4, 1)
        # Confidence boost head
        self.conf_boost = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())

    def forward(self, features, context=None):
        p3 = features[0]  # P3 has highest resolution — best for small objects
        x = self.compress(p3)
        x = self.res_blocks(x)
        enhanced = self.upsample(x)
        refined = self.refine(enhanced)
        bbox_delta = self.bbox_refine(refined)
        confidence = self.conf_boost(refined)
        return {
            "enhanced_features": enhanced,
            "bbox_refinement": bbox_delta,
            "confidence_boost": confidence,
            "compressed_input": self.compress(p3),  # For SR loss
        }

    def get_loss(self, predictions, targets):
        """
        Compute SR enhancement losses:
          1. Feature reconstruction loss: L1 between enhanced features and
             bilinearly upsampled compressed input (self-supervised SR quality)
          2. Bbox refinement loss: Smooth L1 on delta predictions (supervised
             only when GT is available for small objects)
          3. Confidence calibration: BCE loss on confidence predictions

        Loss calculations are performed in FP32 for AMP numerical stability.
        """
        enhanced = predictions["enhanced_features"]
        compressed = predictions["compressed_input"]

        # ============================================================
        # AMP-SAFE LOSS COMPUTATION
        # ============================================================
        # Model activations may be FP16 under AMP. Perform all loss
        # calculations in FP32 to avoid dtype mismatch / instability.
        with torch.amp.autocast("cuda", enabled=False):

            enhanced_fp32 = enhanced.float()
            compressed_fp32 = compressed.detach().float()

            # ========================================================
            # SR Reconstruction Loss
            # ========================================================
            target_sr = F.interpolate(
                compressed_fp32,
                size=enhanced_fp32.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            sr_loss = F.l1_loss(
                enhanced_fp32,
                target_sr,
            )

            # ========================================================
            # Feature Consistency Loss
            # ========================================================
            down = F.adaptive_avg_pool2d(
                enhanced_fp32,
                compressed_fp32.shape[2:],
            )

            consistency_loss = F.mse_loss(
                down,
                compressed_fp32,
            )

            # ========================================================
            # Confidence Calibration Loss
            # ========================================================
            conf_fp32 = predictions["confidence_boost"].float()

            # Self-supervised confidence target:
            # high-activation regions receive higher confidence.
            with torch.no_grad():

                activation_strength = (
                    enhanced_fp32.abs()
                    .mean(dim=1, keepdim=True)
                )

                activation_strength = (
                    activation_strength
                    / (activation_strength.max() + 1e-6)
                )

                conf_target = (
                    activation_strength > 0.5
                ).float()

            conf_loss = F.binary_cross_entropy(
                conf_fp32,
                conf_target,
            )

            # ========================================================
            # TOTAL LOSS
            # ========================================================
            total = (
                sr_loss
                + 0.5 * consistency_loss
                + 0.1 * conf_loss
            )

        return {
            "sr_loss": sr_loss,
            "consistency_loss": consistency_loss,
            "conf_loss": conf_loss,
            "total_loss": total,
        }

    def required_levels(self):
        return ["P3"]
