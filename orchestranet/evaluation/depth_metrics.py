"""
KITTI-Standard Depth Estimation Metrics — Phase 3.

Implements the 7 canonical KITTI depth evaluation metrics:
  AbsRel, SqRel, RMSE, RMSElog, δ<1.25, δ<1.25², δ<1.25³

Policy:
  - Metrics always computed in FP32 regardless of AMP mode.
  - When no GT depth is available, returns UNAVAILABLE sentinels (not 0.0).
  - Applies a valid-mask (gt > min_depth and gt < max_depth) before accumulation.
  - Designed for M4 (DepthEstimator) evaluation on KITTI or NYU Depth v2.

Reference:
  Eigen et al., "Depth Map Prediction from a Single Image using a Multi-Scale
  Deep Network", NeurIPS 2014. Evaluation protocol Section 5.

Usage:
    depth_metrics = DepthMetrics(min_depth=0.001, max_depth=80.0)
    for pred_depth, gt_depth in loader:
        depth_metrics.update(pred_depth, gt_depth)
    results = depth_metrics.compute()
"""

from __future__ import annotations

from typing import Any

import numpy as np

from orchestranet.evaluation.metric_registry import UNAVAILABLE, _Unavailable


_NO_DEPTH_REASON = (
    "No ground-truth depth maps were provided during evaluation. "
    "Depth metrics (AbsRel, SqRel, RMSE, δ thresholds) require 'depth_gt' "
    "targets, available from KITTI or NYU Depth v2 datasets. "
    "Standard COCO does not include depth annotations."
)


class DepthMetrics:
    """
    KITTI-standard monocular depth evaluation.

    Accumulates per-pixel errors over the valid depth range and reports
    the canonical 7-metric set at compute() time.

    Args:
        min_depth: Minimum valid depth value (metres). GT pixels below this
                   threshold are excluded from evaluation. Default: 0.001 m.
        max_depth: Maximum valid depth value. Default: 80.0 m (KITTI outdoor).
                   Use 10.0 for NYU Depth v2 (indoor).
        scale_aware: If True, apply median-scaling alignment before computing
                     errors (standard for self-supervised depth methods).
                     Default: False (absolute scale, for supervised methods).

    Phase 4 note:
        The compute() interface is identical to OcclusionAwareMetrics and
        DetectionMetrics: reset() → update() per image → compute().
        EfficiencyProfiler and SystemEvaluator both call this interface.
    """

    def __init__(
        self,
        min_depth: float = 0.001,
        max_depth: float = 80.0,
        scale_aware: bool = False,
    ):
        self.min_depth   = min_depth
        self.max_depth   = max_depth
        self.scale_aware = scale_aware
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated depth error statistics."""
        self._abs_rel_sum: float = 0.0
        self._sq_rel_sum:  float = 0.0
        self._rmse_sum:    float = 0.0
        self._rmselog_sum: float = 0.0
        self._d1_sum:      float = 0.0   # δ < 1.25
        self._d2_sum:      float = 0.0   # δ < 1.25²
        self._d3_sum:      float = 0.0   # δ < 1.25³
        self._n_images:    int   = 0
        self._n_valid_px:  int   = 0
        self._has_gt:      bool  = False

    def update(
        self,
        pred: "np.ndarray | torch.Tensor",
        gt: "np.ndarray | torch.Tensor",
    ) -> None:
        """
        Accumulate depth errors for one image.

        Args:
            pred: Predicted depth map, arbitrary shape (H, W) or (1, H, W) or (B, 1, H, W).
                  Values assumed to be in the same units as gt (metres).
            gt:   Ground-truth depth map, same shape as pred.

        Notes:
            - Converts to float32 numpy regardless of input dtype/device.
            - Applies valid-mask: gt > min_depth AND gt < max_depth.
            - If the valid mask is empty, this image is silently skipped.
        """
        # ---- to float32 numpy ----
        pred_np = self._to_numpy(pred)
        gt_np   = self._to_numpy(gt)

        # ---- flatten to 1D for vectorised ops ----
        pred_np = pred_np.flatten().astype(np.float32)
        gt_np   = gt_np.flatten().astype(np.float32)

        # ---- valid mask ----
        valid = (gt_np > self.min_depth) & (gt_np < self.max_depth) & np.isfinite(gt_np)
        if valid.sum() == 0:
            return

        pred_v = pred_np[valid]
        gt_v   = gt_np[valid]

        # ---- optional median-scale alignment (self-supervised methods) ----
        if self.scale_aware:
            scale = np.median(gt_v) / (np.median(pred_v) + 1e-9)
            pred_v = pred_v * scale

        # ---- clip predictions to valid range ----
        pred_v = np.clip(pred_v, self.min_depth, self.max_depth)

        # ---- per-pixel error terms ----
        abs_diff = np.abs(gt_v - pred_v)
        sq_diff  = abs_diff ** 2

        abs_rel  = (abs_diff / (gt_v + 1e-9)).mean()
        sq_rel   = (sq_diff  / (gt_v + 1e-9)).mean()
        rmse     = float(np.sqrt(sq_diff.mean()))
        rmselog  = float(np.sqrt(
            ((np.log(gt_v + 1e-9) - np.log(pred_v + 1e-9)) ** 2).mean()
        ))

        # ---- delta thresholds ----
        thresh = np.maximum(gt_v / (pred_v + 1e-9), pred_v / (gt_v + 1e-9))
        d1 = float((thresh < 1.25    ).mean())
        d2 = float((thresh < 1.25 ** 2).mean())
        d3 = float((thresh < 1.25 ** 3).mean())

        # ---- accumulate ----
        self._abs_rel_sum += float(abs_rel)
        self._sq_rel_sum  += float(sq_rel)
        self._rmse_sum    += rmse
        self._rmselog_sum += rmselog
        self._d1_sum      += d1
        self._d2_sum      += d2
        self._d3_sum      += d3
        self._n_images    += 1
        self._n_valid_px  += int(valid.sum())
        self._has_gt       = True

    def compute(self) -> dict[str, Any]:
        """
        Compute all 7 KITTI depth metrics averaged over accumulated images.

        Returns:
            Dict with keys:
              AbsRel, SqRel, RMSE, RMSElog, d1 (δ<1.25), d2 (δ<1.25²), d3 (δ<1.25³),
              n_images, n_valid_pixels.

            If no GT depth was provided, all metric values are UNAVAILABLE dicts.
        """
        if not self._has_gt or self._n_images == 0:
            _unav = _Unavailable(_NO_DEPTH_REASON).to_dict()
            return {
                "AbsRel":        _unav,
                "SqRel":         _unav,
                "RMSE":          _unav,
                "RMSElog":       _unav,
                "d1":            _unav,
                "d2":            _unav,
                "d3":            _unav,
                "n_images":      0,
                "n_valid_pixels": 0,
                "status":        "UNAVAILABLE",
                "reason":        _NO_DEPTH_REASON,
            }

        n = self._n_images
        return {
            "AbsRel":         float(self._abs_rel_sum  / n),
            "SqRel":          float(self._sq_rel_sum   / n),
            "RMSE":           float(self._rmse_sum     / n),
            "RMSElog":        float(self._rmselog_sum  / n),
            "d1":             float(self._d1_sum       / n),   # δ < 1.25
            "d2":             float(self._d2_sum       / n),   # δ < 1.25²
            "d3":             float(self._d3_sum       / n),   # δ < 1.25³
            "n_images":       n,
            "n_valid_pixels": self._n_valid_px,
            "depth_range":    f"[{self.min_depth}, {self.max_depth}] m",
            "scale_aware":    self.scale_aware,
        }

    # ------------------------------------------------------------------
    # Reference targets (for readiness report comparison)
    # ------------------------------------------------------------------

    @staticmethod
    def kitti_eigen_targets() -> dict[str, float]:
        """
        State-of-the-art reference values on KITTI Eigen split (0–80m range).
        Used for gap reporting in training_readiness.md.
        These are NOT OrchestraNet's current values — they are external reference
        points only.
        """
        return {
            "AbsRel": 0.060,   # DPT-BEiT-L (best known)
            "SqRel":  0.249,
            "RMSE":   2.551,
            "RMSElog": 0.090,
            "d1": 0.974,
            "d2": 0.997,
            "d3": 0.999,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """Convert tensor or array to float32 numpy, handling CUDA tensors."""
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float().numpy()
        except ImportError:
            pass
        return np.asarray(x, dtype=np.float32)
