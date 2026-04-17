"""
Observers.py
------------
Pull-based observers. CacheStore, MetricStore, and AnomalyStore are all
imported as modules — no instances, no injection.

WHY OUTPUT-ONLY (no SLOT_INPUT):
  SLOT_INPUT is written by every layer before the forward computation.
  We decided to only observe SLOT_OUTPUT: the output is what flows
  downstream and what matters diagnostically. Storing both input and
  output would double memory usage in CacheStore for no diagnostic gain.

WHY GRAD_IN-ONLY (no SLOT_GRAD_OUT):
  SLOT_GRAD_OUT is the gradient arriving at a layer from downstream.
  SLOT_GRAD_IN is the gradient this layer passes upstream after its
  own backward. SLOT_GRAD_IN is the meaningful signal: it tells you
  how strongly this layer is propagating learning signal. Vanishing/
  exploding gradient is detected on SLOT_GRAD_IN.

WHY WELFORD CALLS HERE:
  Observers are the only place that reads raw tensors from CacheStore.
  Computing statistics here (at the point of reading) means we never
  store raw tensors in MetricStore at all. The stat update is O(1) and
  the tensor can be immediately released from CacheStore after.

WHY ANOMALIES INLINE:
  Anomaly detection uses the same tensor the observer just read for
  metrics — no extra pass, no extra CacheStore read. The checks are
  cheap (np.isnan, norm comparison) and do not affect the forward/
  backward pass since we only read from CacheStore, never write.

Lifecycle:
  WrappedLayer fires:
    on_forward_post(layer_id)  → observer reads SLOT_OUTPUT
    on_backward_post(layer_id) → observer reads SLOT_GRAD_IN

  After observers fire, WrappedLayer calls CacheStore.clear_layer()
  to release the tensor references immediately.
"""

import numpy as np

import cache.CacheStore as CacheStore
import diag.MetricStore as MetricStore
import diag.AnomalyStore as AnomalyStore
from cache.CacheStore import SLOT_OUTPUT, SLOT_GRAD_IN


# ---------------------------------------------------------------------------
# Thresholds for anomaly detection
# ---------------------------------------------------------------------------

VANISHING_GRAD_THRESHOLD  = 1e-7   # grad_in norm below this → vanishing
EXPLODING_GRAD_THRESHOLD  = 1e3    # grad_in norm above this → exploding
DEAD_OUTPUT_THRESHOLD     = 1e-8   # both mean and std below this → dead


class LayerObserver:
    """
    Base observer. run_id and epoch scope all writes.
    Subclasses override only the hooks they care about.
    """

    def __init__(self, name: str, run_id: int = 0):
        self.name   = name
        self.run_id = run_id
        self.epoch  = 0   # set by RunSession before each epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def on_forward_pre(self,   layer_id: int) -> None: pass
    def on_forward_post(self,  layer_id: int) -> None: pass
    def on_backward_pre(self,  layer_id: int) -> None: pass
    def on_backward_post(self, layer_id: int) -> None: pass

    def _output_data(self, layer_id: int):
        """Read SLOT_OUTPUT from CacheStore. Returns numpy array or None."""
        t = CacheStore.read(layer_id, SLOT_OUTPUT)
        return None if t is None else t.data

    def _grad_in_data(self, layer_id: int):
        """Read SLOT_GRAD_IN from CacheStore. Returns numpy array or None."""
        t = CacheStore.read(layer_id, SLOT_GRAD_IN)
        return None if t is None else t.data

    def _update(self, layer_id: int, metric_name: str, value: float) -> None:
        """Push one scalar into the Welford accumulator for this epoch."""
        MetricStore.update(self.run_id, self.epoch, layer_id, metric_name, value)

    def _anomaly(
        self,
        layer_id: int,
        kind:     str,
        fired:    bool,
        value=None,
    ) -> None:
        """Record one anomaly check for this sample in this epoch."""
        AnomalyStore.record(self.run_id, self.epoch, layer_id, kind, fired, value)


# ---------------------------------------------------------------------------
# SignalStatsObserver
# ---------------------------------------------------------------------------

class SignalStatsObserver(LayerObserver):
    """
    Forward: reads SLOT_OUTPUT → activation_mean, activation_std, output_shape.
             Checks for dead outputs and NaN/Inf.
    Backward: reads SLOT_GRAD_IN → grad_in_norm, grad_in_std.
              Checks for vanishing/exploding gradients and NaN.

    No SLOT_INPUT or SLOT_GRAD_OUT are read — see module docstring.
    """

    def __init__(self, run_id: int = 0):
        super().__init__("SignalStatsObserver", run_id)

    def on_forward_post(self, layer_id: int) -> None:
        data = self._output_data(layer_id)
        if data is None:
            return

        mean_val = float(data.mean())
        std_val  = float(data.std())

        # Metric updates
        self._update(layer_id, "activation_mean", mean_val)
        self._update(layer_id, "activation_std",  std_val)

        # Store output shape as a special non-scalar metric key.
        # Shape doesn't change per sample so we can safely overwrite it by
        # storing it in a side channel in MetricStore as a one-element "mean"
        # of the int values. We store each dim separately, prefixed "shape_".
        for i, dim in enumerate(data.shape):
            self._update(layer_id, f"shape_{i}", float(dim))

        # Anomaly: dead output (mean ≈ 0 and std ≈ 0 simultaneously)
        dead = abs(mean_val) < DEAD_OUTPUT_THRESHOLD and std_val < DEAD_OUTPUT_THRESHOLD
        self._anomaly(layer_id, "dead_output", dead,
                      sample_value=std_val if dead else None)

        # Anomaly: NaN in output
        has_nan = bool(np.any(np.isnan(data)))
        self._anomaly(layer_id, "nan_output", has_nan)

        # Anomaly: Inf in output
        has_inf = bool(np.any(np.isinf(data)))
        self._anomaly(layer_id, "inf_output", has_inf)

    def on_backward_post(self, layer_id: int) -> None:
        data = self._grad_in_data(layer_id)
        if data is None:
            return

        norm_val = float(np.linalg.norm(data))
        std_val  = float(data.std())

        # Metric updates
        self._update(layer_id, "grad_in_norm", norm_val)
        self._update(layer_id, "grad_in_std",  std_val)

        # Anomaly: vanishing gradient
        vanishing = norm_val < VANISHING_GRAD_THRESHOLD
        self._anomaly(layer_id, "vanishing_gradient", vanishing,
                      sample_value=norm_val if vanishing else None)

        # Anomaly: exploding gradient
        exploding = norm_val > EXPLODING_GRAD_THRESHOLD
        self._anomaly(layer_id, "exploding_gradient", exploding,
                      sample_value=norm_val if exploding else None)

        # Anomaly: NaN in gradient
        has_nan = bool(np.any(np.isnan(data)))
        self._anomaly(layer_id, "nan_gradient", has_nan)


# ---------------------------------------------------------------------------
# SignalShapeObserver
# ---------------------------------------------------------------------------

class SignalShapeObserver(LayerObserver):
    """
    Lightweight observer that records only the output shape of each layer.
    Shape is constant across samples for a given network configuration,
    so it only needs to be recorded once — but we check every sample for
    correctness (a shape change mid-run indicates a bug).

    Writes to MetricStore as shape_0, shape_1, … dimension integers.
    Observers that want the shape as a list should reconstruct it from
    get_summary(run, epoch, layer_id, "shape_N").
    """

    def __init__(self, run_id: int = 0):
        super().__init__("SignalShapeObserver", run_id)

    def on_forward_post(self, layer_id: int) -> None:
        data = self._output_data(layer_id)
        if data is None:
            return
        for i, dim in enumerate(data.shape):
            self._update(layer_id, f"shape_{i}", float(dim))