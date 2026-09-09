"""
M7: Confidence Calibrator — Occlusion-Aware Score Adjustment.

Recalibrates detection confidence by combining raw M1 scores with
occlusion visibility, depth ordering, and scene context. Prevents
score suppression of correctly detected but heavily occluded objects.
"""

import torch
import torch.nn as nn
from .base_model import BaseMicroModel


class M7ConfidenceCalibrator(BaseMicroModel):
    """
    M7: Confidence Calibrator (~100K params).

    MLP that takes multi-model signals and produces calibrated scores.
    Input features: M1 confidence + M2 visibility + M4 depth + M5 context + bbox stats.
    """
    def __init__(self, input_dim=134, hidden_layers=None):
        super().__init__(model_id="m7", model_name="Confidence Calibrator")
        hidden_layers = hidden_layers or [128, 64, 32]
        self.input_dim = input_dim

        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(in_dim, h_dim), nn.GELU(), nn.Dropout(0.1)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        # Adaptive projection to handle variable input dimensions
        self.adaptive_proj = None  # Lazily initialized

    def _get_projection(self, feat_dim: int, device: torch.device) -> nn.Linear:
        """Lazily create a projection layer to map features to expected input_dim."""
        if self.adaptive_proj is None or self.adaptive_proj.in_features != feat_dim:
            self.adaptive_proj = nn.Linear(feat_dim, self.input_dim, bias=False).to(device)
            nn.init.kaiming_normal_(self.adaptive_proj.weight)
        return self.adaptive_proj

    def forward(self, features, context=None):
        """
        Expects context dict with concatenated feature vector per detection.
        Falls back to creating features from available FPN features.
        """
        if context and "calibration_input" in context:
            x = context["calibration_input"]
        else:
            # Build feature vector from available signals
            p5 = features[-1]
            x = p5.mean(dim=[2, 3])  # Global average pool: (B, C)
            # Project to expected input_dim (handles any channel count)
            if x.shape[-1] != self.input_dim:
                proj = self._get_projection(x.shape[-1], x.device)
                x = proj(x)

        calibrated = self.mlp(x / self.temperature)
        return {"calibrated_confidence": calibrated, "temperature": self.temperature}

    def build_calibration_input(self, m1_conf, m2_vis, m4_depth_val,
                                 m5_embed, bbox_area, aspect_ratio):
        """Assemble the input feature vector from multi-model outputs."""
        return torch.cat([
            m1_conf, m2_vis, m4_depth_val,
            m5_embed, bbox_area, aspect_ratio
        ], dim=-1)

    def get_loss(self, predictions, targets):
        device = predictions["calibrated_confidence"].device
        if "gt_confidence" in targets:
            loss = nn.functional.binary_cross_entropy(
                predictions["calibrated_confidence"], targets["gt_confidence"])
            return {"cal_loss": loss, "total_loss": loss}
        return {"cal_loss": torch.tensor(0.0, device=device),
                "total_loss": torch.tensor(0.0, device=device)}

    def required_levels(self):
        return ["P5"]
