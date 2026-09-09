"""
M5: Semantic Context Engine — Scene-Level Priors.

Provides top-down contextual information: scene type embedding and
expected object priors to reduce false positives and boost occluded object recovery.
"""

import torch
import torch.nn as nn
from .base_model import BaseMicroModel


class M5SemanticContext(BaseMicroModel):
    """
    M5: Scene Context Engine (~400K params).

    Uses P5 global features to classify scene type and generate
    contextual priors for detection adjustment.
    """
    def __init__(self, in_channels=128, embed_dim=128, num_scene_classes=365):
        super().__init__(model_id="m5", model_name="Semantic Context")
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Scene classification
        self.scene_cls = nn.Sequential(
            nn.Linear(in_channels, embed_dim), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(embed_dim, num_scene_classes))

        # Scene embedding for downstream models
        self.scene_embed = nn.Sequential(
            nn.Linear(in_channels, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim))

        # Object prior head: which object classes are likely in this scene
        self.obj_prior = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 80), nn.Sigmoid())  # 80 COCO classes

    def forward(self, features, context=None):
        p5 = features[-1]  # Use deepest level for global semantics
        pooled = self.pool(p5).flatten(1)
        scene_logits = self.scene_cls(pooled)
        embedding = self.scene_embed(pooled)
        obj_priors = self.obj_prior(embedding)
        return {
            "scene_logits": scene_logits,
            "scene_embedding": embedding,
            "object_priors": obj_priors,
        }

    def get_loss(self, predictions, targets):
        device = predictions["scene_logits"].device
        if "scene_label" in targets:
            loss = nn.functional.cross_entropy(predictions["scene_logits"], targets["scene_label"])
            return {"scene_loss": loss, "total_loss": loss}
        return {"scene_loss": torch.tensor(0.0, device=device),
                "total_loss": torch.tensor(0.0, device=device)}

    def required_levels(self):
        return ["P5"]
