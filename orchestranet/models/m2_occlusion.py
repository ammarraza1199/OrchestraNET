"""
M2: Occlusion Analyzer — Core Novelty Module.

Predicts per-pixel occlusion maps, per-detection visibility ratios,
and occluder-occludee relationships using U-Net-Lite with attention gates.
Includes self-supervised occlusion prediction pretext task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseMicroModel


class AttentionGate(nn.Module):
    """Attention gate for skip connections — highlights relevant features."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.w_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        g = F.interpolate(g, size=s.shape[2:], mode="nearest")
        return skip * self.psi(self.relu(g + s))


class M2OcclusionAnalyzer(BaseMicroModel):
    """
    M2: Occlusion Map Predictor & Visibility Scorer (~800K params).

    Architecture: U-Net-Lite with attention gates
    Input: P3 features from FPN
    Outputs:
      - occlusion_map: Per-pixel occlusion probability (B, 1, H, W)
      - visibility_scores: Per-detection visibility ratio [0,1] (B, N, 1)
    """
    def __init__(self, in_channels=128, mid_channels=64, out_channels=32):
        super().__init__(model_id="m2", model_name="Occlusion Analyzer")

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.SiLU(inplace=True))

        # Attention gates
        self.ag1 = AttentionGate(16, out_channels, 16)
        self.ag2 = AttentionGate(out_channels, mid_channels, 32)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(16 + out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.Conv2d(out_channels + mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.SiLU(inplace=True))

        # Occlusion map head
        self.occ_head = nn.Conv2d(mid_channels, 1, 1)

        # Visibility scoring head (global pooling → MLP)
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(mid_channels, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, features, context=None):
        p3 = features[0]  # Use P3 (highest resolution FPN level)

        # Encode
        e1 = self.enc1(p3)
        e2 = self.enc2(e1)
        bn = self.bottleneck(e2)

        # Decode with attention
        a1 = self.ag1(bn, e2)
        d1 = F.interpolate(bn, size=e2.shape[2:], mode="nearest")
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        a2 = self.ag2(d1, e1)
        d2 = F.interpolate(d1, size=e1.shape[2:], mode="nearest")
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        # Outputs
        occ_map = torch.sigmoid(self.occ_head(d2))
        vis_score = self.vis_head(d2)

        return {
            "occlusion_map": occ_map,
            "visibility_scores": vis_score,
            "occlusion_features": d2,  # For downstream models (M6, M7)
        }

    def get_loss(self, predictions, targets):
        device = predictions["occlusion_map"].device
        if "occlusion_mask" in targets:
            gt = targets["occlusion_mask"]
            pred = predictions["occlusion_map"]
            gt_resized = F.interpolate(gt, size=pred.shape[2:], mode="nearest")
            bce = F.binary_cross_entropy(pred, gt_resized)
            # Dice loss for better boundary prediction
            inter = (pred * gt_resized).sum()
            dice = 1 - (2 * inter + 1) / (pred.sum() + gt_resized.sum() + 1)
            return {"occ_loss": bce + dice, "total_loss": bce + dice}
        return {"occ_loss": torch.tensor(0.0, device=device),
                "total_loss": torch.tensor(0.0, device=device)}

    def self_supervised_loss(self, features_masked, original_mask):
        """Self-supervised pretext: predict where artificial occlusion was applied."""
        pred = self.forward(features_masked)["occlusion_map"]
        mask_resized = F.interpolate(original_mask, size=pred.shape[2:], mode="nearest")
        bce = F.binary_cross_entropy(pred, mask_resized)
        inter = (pred * mask_resized).sum()
        dice = 1 - (2 * inter + 1) / (pred.sum() + mask_resized.sum() + 1)
        return bce + dice

    def required_levels(self):
        return ["P3"]
