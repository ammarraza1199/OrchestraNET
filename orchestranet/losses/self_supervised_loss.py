"""
Self-Supervised Occlusion Prediction Loss — Key Novelty.

Implements the self-supervised pretext task where the model learns to
predict artificially-added occlusion regions, enriching features for
real occlusion understanding without requiring expensive amodal annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSupervisedOcclusionLoss(nn.Module):
    """
    Loss for the self-supervised occlusion prediction pretext task.

    Process:
      1. Randomly mask regions of input features (simulating occlusion)
      2. M2 predicts where the masking occurred
      3. Loss = BCE + Dice between predicted and actual mask

    This teaches M2 to understand occlusion patterns without annotations.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        predicted_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute self-supervised occlusion loss.

        Args:
            predicted_mask: M2's predicted occlusion map (B, 1, H, W)
            target_mask: Ground truth artificial mask (B, 1, H, W)

        Returns:
            Dict with "bce", "dice", "total" losses
        """
        # Resize target to match prediction if needed
        if predicted_mask.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask, size=predicted_mask.shape[2:], mode="nearest"
            )

        # BCE loss
        bce = F.binary_cross_entropy(predicted_mask, target_mask)

        # Dice loss for better boundary prediction
        smooth = 1.0
        pred_flat = predicted_mask.flatten(1)
        target_flat = target_mask.flatten(1)
        intersection = (pred_flat * target_flat).sum(1)
        dice = 1 - (2 * intersection + smooth) / (
            pred_flat.sum(1) + target_flat.sum(1) + smooth
        )
        dice = dice.mean()

        total = self.bce_weight * bce + self.dice_weight * dice

        return {"bce": bce, "dice": dice, "total": total}


class OcclusionMaskGenerator(nn.Module):
    """
    Generates random occlusion masks for self-supervised training.
    Applied to FPN features to simulate occlusion at feature level.
    """

    def __init__(
        self,
        mask_ratio_range: tuple[float, float] = (0.1, 0.5),
        patch_size_range: tuple[int, int] = (2, 8),
    ):
        super().__init__()
        self.mask_ratio_range = mask_ratio_range
        self.patch_size_range = patch_size_range

    @torch.no_grad()
    def forward(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking to features.

        Args:
            features: (B, C, H, W) feature tensor

        Returns:
            Tuple of (masked_features, mask) where mask is (B, 1, H, W)
        """
        B, C, H, W = features.shape
        device = features.device

        mask = torch.ones(B, 1, H, W, device=device)

        for b in range(B):
            ratio = torch.empty(1).uniform_(*self.mask_ratio_range).item()
            num_patches = int(H * W * ratio)
            patch_h = torch.randint(*self.patch_size_range, (1,)).item()
            patch_w = torch.randint(*self.patch_size_range, (1,)).item()

            for _ in range(num_patches // (patch_h * patch_w)):
                top = torch.randint(0, max(1, H - patch_h), (1,)).item()
                left = torch.randint(0, max(1, W - patch_w), (1,)).item()
                mask[b, 0, top:top+patch_h, left:left+patch_w] = 0

        masked_features = features * mask
        occlusion_mask = 1 - mask  # 1 where occluded

        return masked_features, occlusion_mask
