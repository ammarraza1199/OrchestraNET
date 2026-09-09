"""
Specialized Micro-Models for OrchestraNet.

Seven task-specific lightweight models that operate on the Shared Latent Tensor:
  M1: Primary Detector - Always-active YOLO-Nano detection head
  M2: Occlusion Analyzer - Occlusion map prediction and visibility scoring
  M3: Small Object Enhancer - Super-resolution + detection refinement
  M4: Depth Estimator - Monocular pseudo-depth for occlusion ordering
  M5: Semantic Context Engine - Scene-level contextual priors
  M6: Amodal Completer - Complete shape prediction for occluded objects
  M7: Confidence Calibrator - Occlusion-aware confidence recalibration
"""

from .base_model import BaseMicroModel
from .m1_detector import M1PrimaryDetector
from .m2_occlusion import M2OcclusionAnalyzer
from .m3_small_enhancer import M3SmallObjectEnhancer
from .m4_depth import M4DepthEstimator
from .m5_context import M5SemanticContext
from .m6_amodal import M6AmodalCompleter
from .m7_calibrator import M7ConfidenceCalibrator

__all__ = [
    "BaseMicroModel",
    "M1PrimaryDetector",
    "M2OcclusionAnalyzer",
    "M3SmallObjectEnhancer",
    "M4DepthEstimator",
    "M5SemanticContext",
    "M6AmodalCompleter",
    "M7ConfidenceCalibrator",
]
