"""
Shared Feature Backbone for OrchestraNet.

Provides multi-scale feature extraction using lightweight architectures
(MobileNetV4-Hybrid or EfficientNetV2-S) via the `timm` library.
Outputs multi-scale feature maps that feed into the FPN and all micro-models.
"""

from .mobilenetv4 import MobileNetV4Backbone
from .feature_pyramid import LightweightFPN

__all__ = ["MobileNetV4Backbone", "LightweightFPN"]
