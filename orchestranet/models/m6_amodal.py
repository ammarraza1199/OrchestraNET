"""
M6: Amodal Completer — Complete Shape Prediction for Occluded Objects.

Uses detection features + occlusion maps to predict the full (amodal)
extent of partially visible objects, including complete bounding boxes
and shape masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseMicroModel


class M6AmodalCompleter(BaseMicroModel):
    """
    M6: Amodal Shape Completer (~900K params).

    Transformer decoder that takes M1 detection features + M2 occlusion
    info to predict complete object shapes.
    """
    def __init__(self, d_model=128, num_heads=4, num_layers=2,
                 dim_feedforward=256, mask_resolution=28, max_objects=50):
        super().__init__(model_id="m6", model_name="Amodal Completer")
        self.d_model = d_model
        self.max_objects = max_objects
        self.mask_resolution = mask_resolution

        # Input projection
        self.input_proj = nn.Linear(d_model + 1, d_model)  # +1 for visibility

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Object queries (learnable)
        self.queries = nn.Embedding(max_objects, d_model)

        # Amodal bbox head
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 4))  # dx, dy, dw, dh offset from visible bbox

        # Amodal mask head
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(inplace=True),
            nn.Linear(256, mask_resolution * mask_resolution))

        # Completion confidence
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, features, context=None):
        p3 = features[0]
        B = p3.shape[0]

        # Create memory from P3 features
        memory = p3.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Learnable object queries
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)

        # Add occlusion info if available from M2
        if context and "occlusion_features" in context:
            occ_feat = context["occlusion_features"].flatten(2).transpose(1, 2)
            occ_pooled = occ_feat.mean(dim=1, keepdim=True)
            # Broadcast visibility signal into memory
            vis_signal = torch.ones(B, memory.shape[1], 1, device=p3.device)
            if "visibility_scores" in context:
                vis_signal = vis_signal * context["visibility_scores"].unsqueeze(1)
            memory = self.input_proj(torch.cat([memory, vis_signal], dim=-1))
        else:
            vis_signal = torch.ones(B, memory.shape[1], 1, device=p3.device)
            memory = self.input_proj(torch.cat([memory, vis_signal], dim=-1))

        # Transformer decode
        decoded = self.transformer(queries, memory)

        # Predict amodal outputs
        amodal_bbox = self.bbox_head(decoded)
        amodal_mask = self.mask_head(decoded).view(
            B, self.max_objects, self.mask_resolution, self.mask_resolution)
        completion_conf = self.conf_head(decoded)

        return {
            "amodal_bbox_offset": amodal_bbox,
            "amodal_masks": torch.sigmoid(amodal_mask),
            "completion_confidence": completion_conf,
        }

    def get_loss(self, predictions, targets):
        """
        Compute amodal completion losses:
          1. Bbox offset L1 loss (amodal vs visible bbox delta)
          2. Mask BCE loss (predicted amodal mask vs GT)
          3. Completion confidence BCE (should be high for truly occluded objects)
        """
        device = predictions["amodal_bbox_offset"].device

        bbox_loss = torch.tensor(0.0, device=device)
        mask_loss = torch.tensor(0.0, device=device)
        conf_loss = torch.tensor(0.0, device=device)

        if "amodal_boxes" in targets:
            n_gt = min(
                predictions["amodal_bbox_offset"].shape[1],
                targets["amodal_boxes"].shape[1],
            )
            bbox_loss = F.l1_loss(
                predictions["amodal_bbox_offset"][:, :n_gt],
                targets["amodal_boxes"][:, :n_gt],
            )

        if "amodal_masks" in targets:
            n_gt = min(
                predictions["amodal_masks"].shape[1],
                targets["amodal_masks"].shape[1],
            )
            pred_masks = predictions["amodal_masks"][:, :n_gt]
            gt_masks = targets["amodal_masks"][:, :n_gt]
            # Resize GT masks to match prediction resolution
            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                gt_masks = F.interpolate(
                    gt_masks.flatten(0, 1).unsqueeze(1),
                    size=pred_masks.shape[-2:], mode="nearest"
                ).squeeze(1).unflatten(0, (pred_masks.shape[0], n_gt))
            with torch.amp.autocast("cuda", enabled=False):
                mask_loss = F.binary_cross_entropy(
                    pred_masks.float(),
                    gt_masks.float()
                )

        if "is_occluded" in targets:
            # Confidence should be high for occluded objects
            n_gt = min(
                predictions["completion_confidence"].shape[1],
                targets["is_occluded"].shape[1],
            )
            with torch.amp.autocast("cuda", enabled=False):
                conf_loss = F.binary_cross_entropy(
                    predictions["completion_confidence"][:, :n_gt].float(),
                    targets["is_occluded"][:, :n_gt].unsqueeze(-1).float(),
                )

        total = bbox_loss + mask_loss + 0.5 * conf_loss

        return {
            "amodal_bbox_loss": bbox_loss,
            "amodal_mask_loss": mask_loss,
            "amodal_conf_loss": conf_loss,
            "total_loss": total,
        }

    def required_levels(self):
        return ["P3"]
