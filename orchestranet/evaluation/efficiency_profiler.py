"""
System-Level Efficiency Profiler — Phase 3.

Profiles OrchestraNet end-to-end for:
  - Latency:  mean / median / p95 / p99 in ms, plus FPS
             (CUDA events for GPU, perf_counter for CPU)
  - Memory:   peak VRAM allocated during one forward pass (MB)
  - FLOPs:    optional — requires fvcore (soft dependency, never crashes)

fvcore policy (Q1 decision):
  If fvcore is NOT installed:
    - count_flops() returns a structured UNAVAILABLE result
    - does NOT return 0 FLOPs (which would be misleading)
    - includes reason "fvcore not installed"
    - does NOT crash or raise
  If fvcore IS installed:
    - count_flops() uses fvcore.nn.FlopCountAnalysis for accurate per-op counting

Usage:
    from orchestranet.evaluation.efficiency_profiler import EfficiencyProfiler

    profiler = EfficiencyProfiler(model, device="cuda")
    latency  = profiler.profile_latency(dummy_input, num_runs=200)
    memory   = profiler.profile_memory(dummy_input)
    flops    = profiler.count_flops(dummy_input)
    report   = profiler.report()

Phase 4 note:
    report() returns a JSON-serializable dict. Checkpoint evaluators
    in Phase 4 can call profiler.report() after each training phase
    to track efficiency regressions.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from orchestranet.evaluation.metric_registry import _Unavailable

_FVCORE_UNAVAILABLE_REASON = (
    "fvcore is not installed. Install with: pip install fvcore. "
    "FLOPs counting is an optional soft dependency and will not affect "
    "other metrics when unavailable."
)


def _try_import_fvcore():
    """Attempt to import fvcore; return (FlopCountAnalysis, parameter_count) or None."""
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
        return FlopCountAnalysis, parameter_count
    except ImportError:
        return None


class EfficiencyProfiler:
    """
    Profiles OrchestraNet inference efficiency.

    Args:
        model:  An nn.Module (typically OrchestraNet or a single micro-model).
        device: "cuda" or "cpu".

    All profiling methods are @torch.no_grad() and switch model to eval mode
    for the duration of the call, then restore the original training state.
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model  = model
        self.device = device
        self._latency_fp32:  dict[str, float] | None = None
        self._latency_fp16:  dict[str, float] | None = None
        self._memory_result: dict[str, float] | None = None
        self._flops_result:  dict[str, Any]   | None = None
        self._param_count:   dict[str, int]   | None = None

    # ------------------------------------------------------------------
    # Latency profiling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def profile_latency(
        self,
        dummy_input: torch.Tensor,
        num_runs: int = 200,
        warmup:   int = 10,
        use_amp:  bool = False,
    ) -> dict[str, float]:
        """
        Measure inference latency with CUDA events (GPU) or perf_counter (CPU).

        Args:
            dummy_input: Input tensor for benchmarking.
            num_runs:    Timed runs (default 200).
            warmup:      Warmup runs before timing starts (default 10).
            use_amp:     If True, profile under torch.amp.autocast (FP16).

        Returns:
            Dict with keys: mean_ms, median_ms, p95_ms, p99_ms, min_ms, max_ms, fps.
        """
        was_training = self.model.training
        self.model.eval()
        dummy_input = dummy_input.to(self.device)

        use_cuda = (str(self.device).startswith("cuda") and torch.cuda.is_available())
        amp_ctx  = torch.amp.autocast("cuda", enabled=use_amp) if use_cuda else _NullContext()

        # Warmup
        for _ in range(warmup):
            with amp_ctx:
                _ = self.model(dummy_input)
        if use_cuda:
            torch.cuda.synchronize()

        times_ms: list[float] = []

        if use_cuda:
            # Use CUDA events for accurate GPU timing
            for _ in range(num_runs):
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev   = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                with amp_ctx:
                    _ = self.model(dummy_input)
                end_ev.record()
                torch.cuda.synchronize()
                times_ms.append(start_ev.elapsed_time(end_ev))
        else:
            for _ in range(num_runs):
                t0 = time.perf_counter()
                with amp_ctx:
                    _ = self.model(dummy_input)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(times_ms, dtype=np.float64)
        result = {
            "mean_ms":   float(arr.mean()),
            "median_ms": float(np.median(arr)),
            "p95_ms":    float(np.percentile(arr, 95)),
            "p99_ms":    float(np.percentile(arr, 99)),
            "min_ms":    float(arr.min()),
            "max_ms":    float(arr.max()),
            "fps":       float(1000.0 / arr.mean()),
            "num_runs":  num_runs,
            "warmup":    warmup,
            "precision": "fp16_amp" if use_amp else "fp32",
            "device":    self.device,
        }

        if was_training:
            self.model.train()

        if use_amp:
            self._latency_fp16 = result
        else:
            self._latency_fp32 = result

        return result

    # ------------------------------------------------------------------
    # Memory profiling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def profile_memory(self, dummy_input: torch.Tensor) -> dict[str, float]:
        """
        Measure peak GPU VRAM allocated during a single forward pass.

        Falls back gracefully to {"status": "UNAVAILABLE"} on CPU.

        Returns:
            Dict with: peak_vram_mb, allocated_before_mb, allocated_after_mb.
            Or UNAVAILABLE dict if not on CUDA.
        """
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            result = _Unavailable(
                "Memory profiling requires a CUDA device. "
                f"Current device: {self.device}."
            ).to_dict()
            self._memory_result = result
            return result

        was_training = self.model.training
        self.model.eval()
        dummy_input = dummy_input.to(self.device)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device)
        before_mb = torch.cuda.memory_allocated(self.device) / 1024**2

        _ = self.model(dummy_input)
        torch.cuda.synchronize()

        peak_mb  = torch.cuda.max_memory_allocated(self.device) / 1024**2
        after_mb = torch.cuda.memory_allocated(self.device) / 1024**2

        result = {
            "peak_vram_mb":       float(peak_mb),
            "allocated_before_mb": float(before_mb),
            "allocated_after_mb":  float(after_mb),
            "delta_mb":            float(peak_mb - before_mb),
            "device":              self.device,
        }

        if was_training:
            self.model.train()

        self._memory_result = result
        return result

    # ------------------------------------------------------------------
    # FLOPs (optional soft dependency on fvcore — Q1 decision)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def count_flops(self, dummy_input: torch.Tensor) -> dict[str, Any]:
        """
        Count multiply-add operations using fvcore (soft dependency).

        fvcore availability:
          INSTALLED   → accurate per-op FLOPs via FlopCountAnalysis
          NOT INSTALLED → returns structured UNAVAILABLE result (not 0 FLOPs,
                          not a crash, not a warning-suppressed silent failure)

        Returns:
            If fvcore available:
                {"total_flops": int, "total_gflops": float, "status": "OK", ...}
            If fvcore NOT available:
                {"status": "UNAVAILABLE", "reason": "fvcore not installed", ...}
        """
        fvcore = _try_import_fvcore()

        if fvcore is None:
            # STRICT: never substitute 0 FLOPs for missing fvcore
            result = {
                **_Unavailable(_FVCORE_UNAVAILABLE_REASON).to_dict(),
                "total_flops": None,
                "total_gflops": None,
            }
            self._flops_result = result
            return result

        FlopCountAnalysis, parameter_count = fvcore

        was_training = self.model.training
        self.model.eval()
        dummy_input = dummy_input.to(self.device)

        try:
            flops = FlopCountAnalysis(self.model, dummy_input)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total = int(flops.total())
            result = {
                "total_flops":  total,
                "total_gflops": round(total / 1e9, 4),
                "status":       "OK",
                "device":       self.device,
                "input_shape":  list(dummy_input.shape),
            }
        except Exception as exc:
            result = {
                **_Unavailable(
                    f"fvcore FlopCountAnalysis failed: {exc}"
                ).to_dict(),
                "total_flops":  None,
                "total_gflops": None,
            }

        if was_training:
            self.model.train()

        self._flops_result = result
        return result

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """
        Count total and trainable parameters.

        If model has count_all_parameters() (OrchestraNet), uses that for
        per-component breakdown. Falls back to simple total otherwise.
        """
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        model_size_mb = round(total_bytes / (1024 * 1024), 2)

        if hasattr(self.model, "count_all_parameters"):
            result = self.model.count_all_parameters()
            if "TOTAL" in result:
                result["TOTAL"]["model_size_mb"] = model_size_mb
        else:
            total     = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            result = {"TOTAL": {"total": total, "trainable": trainable, "model_size_mb": model_size_mb}}

        self._param_count = result
        return result

    # ------------------------------------------------------------------
    # Unified report
    # ------------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """
        Assemble a single JSON-serializable efficiency report from all
        profiling results collected so far.

        Phase 4 note: This dict is the intended format for per-checkpoint
        efficiency tracking. Checkpoint evaluators should call report()
        and persist the result alongside metric results.
        """
        return {
            "latency_fp32":  self._latency_fp32,
            "latency_fp16":  self._latency_fp16,
            "memory":        self._memory_result,
            "flops":         self._flops_result,
            "parameters":    self._param_count,
        }


# ------------------------------------------------------------------
# Null context for CPU path
# ------------------------------------------------------------------

class _NullContext:
    """No-op context manager used when not on CUDA."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
