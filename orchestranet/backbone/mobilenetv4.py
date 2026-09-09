"""
MobileNetV4-Hybrid Backbone for OrchestraNet.

Uses `timm` to instantiate a MobileNetV4-Hybrid backbone with multi-scale
feature extraction at P3 (stride 8), P4 (stride 16), and P5 (stride 32).
This provides the shared latent tensor used by all micro-models.

Design Decisions:
- MobileNetV4-Hybrid was chosen for its best-in-class latency/accuracy
  tradeoff among mobile-optimized architectures (2024).
- We extract features at 3 scales to balance small/medium/large object detection.
- Depthwise-separable convolutions keep the parameter count low (~3.5M).
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class MobileNetV4Backbone(nn.Module):
    """
    Multi-scale feature extractor based on MobileNetV4-Hybrid.

    Extracts feature maps at three scales:
      - P3: stride 8,  channels = out_channels[0]  (small objects)
      - P4: stride 16, channels = out_channels[1]  (medium objects)
      - P5: stride 32, channels = out_channels[2]  (large objects)

    Args:
        model_name: timm model identifier. Defaults to 'mobilenetv4_hybrid_medium'.
        pretrained: Whether to load ImageNet pretrained weights.
        out_indices: Indices of feature stages to extract (0-indexed).
        frozen_stages: Number of stages to freeze (0 = none frozen).
    """

    # Fallback channel dimensions for known model variants
    KNOWN_CHANNELS = {
        "mobilenetv4_hybrid_medium": [48, 80, 160, 256],
        "mobilenetv4_hybrid_large": [64, 128, 256, 512],
        "efficientnetv2_s": [48, 64, 160, 256],
    }

    def __init__(
        self,
        model_name: str = "mobilenetv4_hybrid_medium",
        pretrained: bool = True,
        out_indices: tuple = (1, 2, 3),
        frozen_stages: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.out_indices = out_indices

        if timm is not None:
            # Use timm for production-quality backbones
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
            )
            # Get actual channel dimensions from timm
            self.out_channels = self.backbone.feature_info.channels()
        else:
            # Fallback: build a lightweight custom backbone for dev/testing
            self.backbone = None
            channels = self.KNOWN_CHANNELS.get(
                model_name, [48, 80, 160, 256]
            )
            self.out_channels = [channels[i] for i in out_indices]
            self._build_fallback_backbone(channels, out_indices)

        # Freeze early stages if requested
        if frozen_stages > 0:
            self._freeze_stages(frozen_stages)

    def _build_fallback_backbone(self, channels, out_indices):
        """
        Build a simple fallback backbone for development/testing when timm
        is not available. Uses standard conv blocks at each stride level.
        """
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        all_channels = [32] + channels
        self.stages = nn.ModuleList()
        for i in range(len(channels)):
            stride = 2 if i > 0 else 2  # Each stage downsamples by 2x
            stage = nn.Sequential(
                # Depthwise separable conv block
                nn.Conv2d(
                    all_channels[i], all_channels[i], 3,
                    stride=stride, padding=1, groups=all_channels[i], bias=False
                ),
                nn.BatchNorm2d(all_channels[i]),
                nn.SiLU(inplace=True),
                nn.Conv2d(all_channels[i], channels[i], 1, bias=False),
                nn.BatchNorm2d(channels[i]),
                nn.SiLU(inplace=True),
                # Second block
                nn.Conv2d(
                    channels[i], channels[i], 3,
                    stride=1, padding=1, groups=channels[i], bias=False
                ),
                nn.BatchNorm2d(channels[i]),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels[i], channels[i], 1, bias=False),
                nn.BatchNorm2d(channels[i]),
                nn.SiLU(inplace=True),
            )
            self.stages.append(stage)

        self._out_indices = out_indices

    def _freeze_stages(self, num_stages: int):
        """Freeze parameters in the first `num_stages` stages."""
        if self.backbone is not None:
            # Freeze timm backbone stages
            for i, (name, param) in enumerate(self.backbone.named_parameters()):
                stage_idx = int(name.split(".")[0][-1]) if name[0].isdigit() else 0
                if stage_idx < num_stages:
                    param.requires_grad = False
        else:
            # Freeze fallback stages
            if num_stages >= 1:
                for param in self.stem.parameters():
                    param.requires_grad = False
            for i in range(min(num_stages - 1, len(self.stages))):
                for param in self.stages[i].parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features from input image.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of feature tensors at P3, P4, P5 scales:
              - P3: (B, C3, H/8, W/8)
              - P4: (B, C4, H/16, W/16)
              - P5: (B, C5, H/32, W/32)
        """
        if self.backbone is not None:
            return self.backbone(x)
        else:
            # Fallback backbone
            features = []
            x = self.stem(x)
            for i, stage in enumerate(self.stages):
                x = stage(x)
                if i in self._out_indices:
                    features.append(x)
            return features

    def get_out_channels(self) -> list[int]:
        """Return the channel dimensions of each output feature level."""
        return list(self.out_channels)

    @torch.no_grad()
    def get_feature_shapes(self, input_size: int = 640) -> dict:
        """
        Get output feature map shapes for a given input size.
        Useful for debugging and configuration.
        """
        dummy = torch.randn(1, 3, input_size, input_size)
        if torch.cuda.is_available():
            dummy = dummy.cuda()
            self.cuda()
        features = self.forward(dummy)
        shapes = {}
        for i, feat in enumerate(features):
            level = f"P{self.out_indices[i] + 2}"
            shapes[level] = {
                "shape": tuple(feat.shape),
                "channels": feat.shape[1],
                "spatial": (feat.shape[2], feat.shape[3]),
                "stride": input_size // feat.shape[2],
            }
        return shapes
