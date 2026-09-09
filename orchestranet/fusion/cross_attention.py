"""
Cross-Attention Fusion Module for OrchestraNet.

Merges outputs from all active micro-models using multi-head cross-attention.
Allows each model's output to attend to and refine others' predictions,
producing a unified detection tensor that leverages all available signals.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Fuses multi-model outputs via cross-attention.

    Each model's output features are treated as queries that can attend
    to all other models' features, enabling rich inter-model communication.

    Args:
        d_model: Feature dimension for attention.
        num_heads: Number of attention heads.
        num_models: Maximum number of models whose outputs are fused.
    """

    def __init__(self, d_model: int = 128, num_heads: int = 4, num_models: int = 7):
        super().__init__()
        self.d_model = d_model
        self.num_models = num_models

        # Project each model's output to shared dimension
        self.input_projs = nn.ModuleDict({
            f"m{i}": nn.Linear(d_model, d_model)
            for i in range(1, num_models + 1)
        })

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward refinement
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Output projection to detection space
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        primary_detections: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse model outputs with cross-attention.

        Args:
            model_outputs: Dict mapping model_id to feature tensor (B, N, D).
            primary_detections: M1's detection features as the query basis (B, N, D).

        Returns:
            Fused detection features: (B, N, D)
        """
        B = primary_detections.shape[0]

        # Project all available model outputs to shared space
        projected = []
        for model_id, feat in model_outputs.items():
            if model_id in self.input_projs and feat is not None:
                proj = self.input_projs[model_id](feat)
                projected.append(proj)

        if not projected:
            return self.output_proj(primary_detections)

        # Concatenate all model outputs as keys/values
        kv = torch.cat(projected, dim=1)  # (B, sum_N, D)

        # Cross-attention: detections attend to all model outputs
        query = primary_detections
        attended, _ = self.cross_attn(query, kv, kv)
        query = self.norm1(query + attended)
        query = self.norm2(query + self.ffn(query))

        return self.output_proj(query)
