"""
Observers.py
------------
Pull-based observers. Both CacheStore and MetricStore are imported
as modules — no instances, no injection into constructors.

Lifecycle flow
--------------
WrappedLayer fires on_forward_post(layer_id) etc.
Each observer reads exactly the slots it needs from CacheStore,
computes its metrics, and writes them straight into MetricStore.

MetricStore is written to here directly — no intermediate collection,
no hand-off to a runner. The data is there as soon as the hook fires.
"""

from collections import defaultdict
import numpy as np

import cache.CacheStore as CacheStore
import diag.MetricStore as MetricStore
from cache.CacheStore import (
    SLOT_INPUT, SLOT_OUTPUT,
    SLOT_GRAD_OUT, SLOT_GRAD_IN,
    SLOT_RESIDUAL, SLOT_SHORTCUT,
)


class LayerObserver:
    """
    Base observer. Subclasses override only the hooks they care about.
    run_id scopes all MetricStore writes to the current run.
    """

    def __init__(self, name: str, run_id: int = 0):
        self.name   = name
        self.run_id = run_id

    def on_forward_pre(self,   layer_id: int) -> None: pass
    def on_forward_post(self,  layer_id: int) -> None: pass
    def on_backward_pre(self,  layer_id: int) -> None: pass
    def on_backward_post(self, layer_id: int) -> None: pass

    def _data(self, layer_id: int, slot: str):
        """Read a slot from CacheStore, return numpy array or None."""
        t = CacheStore.read(layer_id, slot)
        return None if t is None else t.data

    def _record(self, layer_id: int, metric_name: str, value: float) -> None:
        """Write a scalar metric directly into MetricStore."""
        MetricStore.add(self.run_id, layer_id, metric_name, [value])


# ---------------------------------------------------------------------------
# SignalStatsObserver
# ---------------------------------------------------------------------------

class SignalStatsObserver(LayerObserver):
    """
    Reads : SLOT_OUTPUT   → activation_mean, activation_var
            SLOT_GRAD_OUT → grad_norm, grad_var
    Writes: MetricStore
    """

    def __init__(self, run_id: int = 0):
        super().__init__("SignalStatsObserver", run_id)

    def on_forward_post(self, layer_id: int) -> None:
        data = self._data(layer_id, SLOT_OUTPUT)
        if data is None:
            return
        self._record(layer_id, "activation_mean", float(data.mean()))
        self._record(layer_id, "activation_var",  float(data.var()))

    def on_backward_pre(self, layer_id: int) -> None:
        data = self._data(layer_id, SLOT_GRAD_OUT)
        if data is None:
            return
        self._record(layer_id, "grad_norm", float(np.linalg.norm(data)))
        self._record(layer_id, "grad_var",  float(data.var()))


# ---------------------------------------------------------------------------
# SignalShapeObserver
# ---------------------------------------------------------------------------

class SignalShapeObserver(LayerObserver):
    """
    Reads : SLOT_INPUT, SLOT_OUTPUT, SLOT_GRAD_OUT, SLOT_GRAD_IN
    Writes: self.logs only (shapes are not numeric — kept local, not in MetricStore)
    """

    def __init__(self, run_id: int = 0):
        super().__init__("SignalShapeObserver", run_id)
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_pre(self, layer_id: int) -> None:
        data = self._data(layer_id, SLOT_INPUT)
        if data is not None:
            self.logs[layer_id]["input_shape"].append(data.shape)

    def on_forward_post(self, layer_id: int) -> None:
        data = self._data(layer_id, SLOT_OUTPUT)
        if data is not None:
            self.logs[layer_id]["output_shape"].append(data.shape)

    def on_backward_pre(self, layer_id: int) -> None:
        data = self._data(layer_id, SLOT_GRAD_OUT)
        if data is not None:
            self.logs[layer_id]["grad_out_shape"].append(data.shape)

    def on_backward_post(self, layer_id: int) -> None:
        data = self._data(layer_id, SLOT_GRAD_IN)
        if data is not None:
            self.logs[layer_id]["grad_in_shape"].append(data.shape)


# ---------------------------------------------------------------------------
# ResidualEnergyObserver
# ---------------------------------------------------------------------------

class ResidualEnergyObserver(LayerObserver):
    """
    Reads : SLOT_RESIDUAL, SLOT_SHORTCUT
    Writes: MetricStore  (residual_energy, shortcut_energy)
    Silently skips layers that don't write these slots.
    """

    def __init__(self, run_id: int = 0):
        super().__init__("ResidualEnergyObserver", run_id)

    def on_forward_post(self, layer_id: int) -> None:
        f = self._data(layer_id, SLOT_RESIDUAL)
        s = self._data(layer_id, SLOT_SHORTCUT)
        if f is None or s is None:
            return
        self._record(layer_id, "residual_energy", float(np.mean(f ** 2)))
        self._record(layer_id, "shortcut_energy", float(np.mean(s ** 2)))