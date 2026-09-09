"""
Lightweight Feature Pyramid Network (FPN) for OrchestraNet.

Fuses multi-scale features from the backbone into a unified representation
with consistent channel dimensions. Uses depthwise-separable convolutions
for minimal parameter overhead.

This FPN produces the Shared Latent Tensor (SLT) that all micro-models consume.

Design Decisions:
- Depthwise-separable convolutions reduce params by ~8x vs standard convolutions.
- Top-down pathway with lateral connections preserves both semantics and spatial detail.
- Optional bottom-up augmentation path for enhanced small-object features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution: depthwise + pointwise."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class LightweightFPN(nn.Module):
    """
    Lightweight Feature Pyramid Network using depthwise-separable convolutions.

    Takes multi-scale features from the backbone and produces a unified
    multi-scale representation (Shared Latent Tensor) with consistent
    channel dimensions.

    Args:
        in_channels: List of input channel dimensions from backbone [P3, P4, P5].
        out_channels: Unified output channel dimension for all FPN levels.
        num_outs: Number of output feature levels.
        use_depthwise: Whether to use depthwise-separable convolutions.
        extra_convs_on_inputs: Whether to add extra downsampled outputs.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 128,
        num_outs: int = 3,
        use_depthwise: bool = True,
        extra_convs_on_inputs: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_ins = len(in_channels)

        conv_cls = DepthwiseSeparableConv if use_depthwise else self._standard_conv

        # Lateral connections: 1x1 conv to match channel dimensions
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )

        # Top-down pathway: smooth features after upsampling + addition
        self.td_convs = nn.ModuleList()
        for i in range(self.num_ins - 1):
            self.td_convs.append(conv_cls(out_channels, out_channels))

        # Bottom-up augmentation pathway (optional, for small object enhancement)
        self.bu_convs = nn.ModuleList()
        for i in range(self.num_ins - 1):
            self.bu_convs.append(
                conv_cls(out_channels, out_channels, stride=2, padding=1)
            )

        # Output convolutions
        self.out_convs = nn.ModuleList()
        for i in range(num_outs):
            self.out_convs.append(conv_cls(out_channels, out_channels))

        self._init_weights()

    @staticmethod
    def _standard_conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        """Standard convolution fallback."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass through the FPN.

        Args:
            inputs: List of feature tensors from backbone [P3, P4, P5]

        Returns:
            List of FPN feature tensors [FP3, FP4, FP5] with unified channels
        """
        assert len(inputs) == self.num_ins, (
            f"Expected {self.num_ins} inputs, got {len(inputs)}"
        )

        # Step 1: Lateral connections (channel alignment)
        laterals = [
            self.lateral_convs[i](inputs[i]) for i in range(self.num_ins)
        ]

        # Step 2: Top-down pathway (coarse → fine)
        for i in range(self.num_ins - 1, 0, -1):
            # Upsample higher level and add to lower level
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="nearest",
            )
            laterals[i - 1] = laterals[i - 1] + upsampled
            laterals[i - 1] = self.td_convs[i - 1](laterals[i - 1])

        # Step 3: Bottom-up augmentation pathway (fine → coarse)
        for i in range(self.num_ins - 1):
            laterals[i + 1] = laterals[i + 1] + self.bu_convs[i](laterals[i])

        # Step 4: Output convolutions
        outputs = [
            self.out_convs[i](laterals[i]) for i in range(self.num_outs)
        ]

        return outputs

    def get_out_channels(self) -> int:
        """Return the unified output channel dimension."""
        return self.out_channels
