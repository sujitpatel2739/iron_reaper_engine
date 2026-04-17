"""
Profiles.py
-----------
Interpreter profiles. Read from MetricStore using the new epoch-aware API.

WHY THE OLD API BROKE:
  The old MetricStore stored raw lists and offered get_sequence(run, name, agg).
  The new MetricStore stores Welford accumulators and offers:
    get_sequence(run, metric_name, epoch, agg)  — same shape, now epoch-scoped
    get_summary(run, epoch, layer_id, metric)   — {mean, std, min, max, count}

  Profiles are still used by the HTTP /run/start endpoint (legacy path).
  For the streaming WS /run/full path, runner.py builds the snapshot directly
  from MetricStore without going through profiles.  Profiles are kept for
  backward compatibility and for the Profiles tab if it is re-added later.

WHY EPOCH=0 DEFAULT:
  Profiles are called after a single diagnostic pass (one epoch).
  They default to epoch=0, which is always correct for the single-epoch case.
  Multi-epoch profile aggregation (e.g. mean of epoch means) can be added later.
"""

import numpy as np
import diag.MetricStore as MetricStore


class InterpreterProfile:
    def __init__(self, name: str, run_id: int = 0, epoch: int = 0):
        self.name   = name
        self.run_id = run_id
        self.epoch  = epoch

    def __call__(self) -> dict:
        raise NotImplementedError


class SignalStatsProfile(InterpreterProfile):
    """
    Per-layer mean-aggregated activation and gradient stats for one epoch.
    Returns {metric_name: {layer_id: mean_value}} ready for plotting.
    """

    def __call__(self) -> dict:
        return {
            "activation_mean": MetricStore.get_sequence(self.run_id, "activation_mean", epoch=self.epoch, agg="mean"),
            "activation_std":  MetricStore.get_sequence(self.run_id, "activation_std",  epoch=self.epoch, agg="mean"),
            "grad_in_norm":    MetricStore.get_sequence(self.run_id, "grad_in_norm",     epoch=self.epoch, agg="mean"),
            "grad_in_std":     MetricStore.get_sequence(self.run_id, "grad_in_std",      epoch=self.epoch, agg="mean"),
        }


class PathDominanceProfile(InterpreterProfile):
    """
    Computes fractional energy split between residual and shortcut paths.
    Only meaningful for ResBlock-style layers that write residual/shortcut stats.
    """

    def __call__(self) -> dict:
        raw_residual = MetricStore.get_sequence(self.run_id, "residual_energy", epoch=self.epoch, agg="mean")
        raw_shortcut = MetricStore.get_sequence(self.run_id, "shortcut_energy", epoch=self.epoch, agg="mean")

        residual, shortcut = {}, {}
        for (l, r), (_, s) in zip(raw_residual.items(), raw_shortcut.items()):
            total        = r + s if (r + s) > 0 else 1.0
            residual[l]  = r / total
            shortcut[l]  = s / total

        return {"residual": residual, "shortcut": shortcut}