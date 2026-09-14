"""
OrchestraNet Evaluation Framework — Phase 3.

Provides a centralized, modular evaluation suite covering:
  - Detection metrics (COCO-style mAP)
  - Occlusion-aware metrics (AP_occ, Vis < 30%)
  - Depth estimation metrics (KITTI-standard)
  - System efficiency profiling (latency, memory, FLOPs)
  - Unified evaluate_system() interface

Usage:
    from orchestranet.evaluation import MetricRegistry, SystemEvaluator, EfficiencyProfiler
    from orchestranet.evaluation.evaluator import evaluate_system
"""

from .metric_registry import MetricRegistry
from .occlusion_metrics import OcclusionAwareMetrics
from .depth_metrics import DepthMetrics
from .efficiency_profiler import EfficiencyProfiler
from .evaluator import SystemEvaluator, evaluate_system

__all__ = [
    "MetricRegistry",
    "OcclusionAwareMetrics",
    "DepthMetrics",
    "EfficiencyProfiler",
    "SystemEvaluator",
    "evaluate_system",
]
