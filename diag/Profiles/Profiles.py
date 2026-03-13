"""
Profiles.py
-----------
Interpreter profiles. MetricStore is imported as a module — no instance,
no injection. Profiles read directly from MetricStore.get_sequence().

Profiles no longer accept a store argument — they call MetricStore directly.
The only argument needed is run_id to scope which run's data to read.
"""

import numpy as np
import diag.MetricStore as MetricStore


class InterpreterProfile:
    def __init__(self, name: str, run_id: int = 0):
        self.name   = name
        self.run_id = run_id

    def __call__(self) -> dict:
        raise NotImplementedError


class SignalStatsProfile(InterpreterProfile):
    """
    Reads activation and gradient statistics per layer from MetricStore.
    Returns per-layer mean-aggregated values ready for plotting.
    """

    def __call__(self) -> dict:
        return {
            "activation_mean": MetricStore.get_sequence(self.run_id, "activation_mean", agg="mean"),
            "activation_var":  MetricStore.get_sequence(self.run_id, "activation_var",  agg="mean"),
            "grad_norm":       MetricStore.get_sequence(self.run_id, "grad_norm",        agg="mean"),
            "grad_var":        MetricStore.get_sequence(self.run_id, "grad_var",         agg="mean"),
        }


class PathDominanceProfile(InterpreterProfile):
    """
    Computes fractional energy split between residual and shortcut paths.
    Only meaningful for layers that wrote residual_energy / shortcut_energy
    (i.e. ResBlock-style composite layers).
    """

    def __call__(self) -> dict:
        raw_residual = MetricStore.get_sequence(self.run_id, "residual_energy", agg="mean")
        raw_shortcut = MetricStore.get_sequence(self.run_id, "shortcut_energy", agg="mean")

        residual, shortcut = {}, {}
        for (l, r), (_, s) in zip(raw_residual.items(), raw_shortcut.items()):
            total       = r + s
            residual[l] = r / total
            shortcut[l] = s / total

        return {"residual": residual, "shortcut": shortcut}