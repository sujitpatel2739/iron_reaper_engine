"""
MetricStore.py
--------------
Module-level metric store using Welford online accumulators.

WHY WELFORD:
  The old design appended raw floats to Python lists:
    _metrics[run][layer_id][name] -> [float, float, ...]
  At 1000 samples × 20 layers × 4 metrics that is 80,000 Python float
  objects (~2.2 MB of heap overhead, not counting list nodes). At 10,000
  samples: 22 MB, all live on the Python heap, never freed until clear().
  Numpy could hold the same in 320 KB as a contiguous float32 array, but
  we don't need the raw values at all — we only ever query mean/std/min/max.

  Welford's algorithm maintains {count, mean, M2, lo, hi} — exactly 5
  numbers — and gives numerically stable mean and variance in O(1) time
  and O(1) memory regardless of how many samples have been processed.
  Memory is now bounded: E × L × M × 5 floats × 8 bytes.
  For 10 epochs, 20 layers, 4 metrics: 32 KB total.

WHY EPOCH DIMENSION:
  Previously there was no concept of epochs — the store held a single
  flat list per run. To display cross-epoch trends (how does activation
  mean change over training?) we need each epoch's stats independently.

Structure:
  _stats[run_id][epoch][layer_id][metric_name] -> _Welford
"""

import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Welford accumulator
# ---------------------------------------------------------------------------

class _Welford:
    """
    Online mean/variance accumulator (Welford's algorithm).
    All arithmetic is done in Python floats — fast enough for scalar metrics.
    """
    __slots__ = ('n', 'mean', 'M2', 'lo', 'hi')

    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 0.0
        self.lo   = math.inf
        self.hi   = -math.inf

    def update(self, value: float) -> None:
        self.n   += 1
        delta     = value - self.mean
        self.mean += delta / self.n
        delta2    = value - self.mean
        self.M2  += delta * delta2
        if value < self.lo: self.lo = value
        if value > self.hi: self.hi = value

    def summary(self) -> Dict[str, float]:
        """Return {mean, std, min, max, count}. std is 0 if n < 2."""
        std = math.sqrt(self.M2 / self.n) if self.n >= 2 else 0.0
        return {
            "mean":  self.mean,
            "std":   std,
            "min":   self.lo   if self.n > 0 else 0.0,
            "max":   self.hi   if self.n > 0 else 0.0,
            "count": self.n,
        }

    @property
    def std(self) -> float:
        return math.sqrt(self.M2 / self.n) if self.n >= 2 else 0.0


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# _stats[run_id][epoch][layer_id][metric_name] -> _Welford
_stats: Dict[int, Dict[int, Dict[int, Dict[str, _Welford]]]] = {}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def update(run: int, epoch: int, layer_id: int, metric_name: str, value: float) -> None:
    """
    Feed one scalar observation into the accumulator for
    (run, epoch, layer_id, metric_name).  O(1), no allocation after first call.
    """
    run_d   = _stats.setdefault(run, {})
    epoch_d = run_d.setdefault(epoch, {})
    layer_d = epoch_d.setdefault(layer_id, {})
    acc     = layer_d.get(metric_name)
    if acc is None:
        acc = _Welford()
        layer_d[metric_name] = acc
    acc.update(value)


# ---------------------------------------------------------------------------
# Read — single cell
# ---------------------------------------------------------------------------

def get_summary(run: int, epoch: int, layer_id: int, metric_name: str) -> Optional[Dict[str, float]]:
    """
    Return {mean, std, min, max, count} for one (run, epoch, layer, metric).
    Returns None if the cell has not been written to.
    """
    acc = _stats.get(run, {}).get(epoch, {}).get(layer_id, {}).get(metric_name)
    return acc.summary() if acc is not None else None


def get_mean(run: int, epoch: int, layer_id: int, metric_name: str) -> Optional[float]:
    acc = _stats.get(run, {}).get(epoch, {}).get(layer_id, {}).get(metric_name)
    return acc.mean if acc is not None else None


# ---------------------------------------------------------------------------
# Read — epoch snapshot (used by streaming delivery)
# ---------------------------------------------------------------------------

def get_epoch_snapshot(run: int, epoch: int) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Return everything collected in one epoch:
      { layer_id: { metric_name: {mean, std, min, max, count} } }

    This is the payload sent in the epoch_done WebSocket event.
    """
    epoch_data = _stats.get(run, {}).get(epoch, {})
    result = {}
    for layer_id, metrics in epoch_data.items():
        result[layer_id] = {
            name: acc.summary()
            for name, acc in metrics.items()
        }
    return result


# ---------------------------------------------------------------------------
# Read — cross-epoch sequence (used by Profiles, backward compat)
# ---------------------------------------------------------------------------

def get_sequence(
    run:         int,
    metric_name: str,
    epoch:       int  = 0,
    agg:         str  = "mean",
) -> Dict[int, float]:
    """
    Return {layer_id: scalar} across all layers for one metric in one epoch.
    agg: 'mean' | 'std' | 'min' | 'max' | 'count'
    """
    epoch_data = _stats.get(run, {}).get(epoch, {})
    result = {}
    for layer_id, metrics in epoch_data.items():
        acc = metrics.get(metric_name)
        if acc is not None:
            s = acc.summary()
            result[layer_id] = s.get(agg, s["mean"])
    return result


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def runs() -> List[int]:
    return sorted(_stats.keys())

def epochs(run: int) -> List[int]:
    return sorted(_stats.get(run, {}).keys())

def layer_ids(run: int, epoch: int) -> List[int]:
    return sorted(_stats.get(run, {}).get(epoch, {}).keys())

def metric_names(run: int, epoch: int, layer_id: int) -> List[str]:
    return list(_stats.get(run, {}).get(epoch, {}).get(layer_id, {}).keys())


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def clear() -> None:
    """Drop all data across all runs."""
    _stats.clear()

def clear_run(run: int) -> None:
    """Drop all epochs for one run."""
    _stats.pop(run, None)

def clear_epoch(run: int, epoch: int) -> None:
    """Drop one epoch — useful to free memory after streaming epoch_done."""
    _stats.get(run, {}).pop(epoch, None)


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def debug_repr(run: int, epoch: int = 0) -> str:
    if run not in _stats:
        return f"MetricStore: run {run} not found"
    epoch_data = _stats[run].get(epoch, {})
    lines = [f"MetricStore (run={run}, epoch={epoch}):"]
    for lid in sorted(epoch_data.keys()):
        names = list(epoch_data[lid].keys())
        lines.append(f"  layer {lid}: {names}")
    return "\n".join(lines)