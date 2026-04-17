"""
AnomalyStore.py
---------------
Module-level store for discrete anomaly events.

WHY SEPARATE FROM MetricStore:
  Anomalies are discrete flagged events, not continuous statistics.
  They have a severity level, a triggering value, and a frequency.
  Storing them in MetricStore's Welford accumulators would lose the
  categorical structure (kind/severity) and make them unqueryable
  as a diagnostic list. A separate store keeps concerns clean.

WHY PER-EPOCH:
  An anomaly that fires on every sample in epoch 0 but disappears by
  epoch 5 tells a completely different story from a persistent anomaly.
  Storing per-epoch lets the frontend heatmap show this pattern.

Structure:
  _store[run_id][epoch][layer_id] -> { kind -> _AnomalyAccumulator }

Each _AnomalyAccumulator tracks how often an anomaly fired (count) vs
how many times it was checked (total), plus the most recent sample_value
that triggered it.

Anomaly kinds (written by Observers):
  'dead_output'           — output mean and std both near zero (dead neurons)
  'nan_output'            — NaN detected in output tensor
  'inf_output'            — Inf detected in output tensor
  'vanishing_gradient'    — grad_in norm below threshold
  'exploding_gradient'    — grad_in norm above threshold
  'nan_gradient'          — NaN detected in grad_in tensor

Severity levels:
  'warning'   — potentially concerning, review recommended
  'critical'  — numerically unstable or training-breaking
"""

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Severity map — fixed per kind
# ---------------------------------------------------------------------------

ANOMALY_SEVERITY: Dict[str, str] = {
    "dead_output":           "warning",
    "nan_output":            "critical",
    "inf_output":            "critical",
    "vanishing_gradient":    "warning",
    "exploding_gradient":    "critical",
    "nan_gradient":          "critical",
}


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class _AnomalyAccumulator:
    __slots__ = ('count', 'total', 'sample_value', 'kind', 'severity')

    def __init__(self, kind: str):
        self.kind         = kind
        self.severity     = ANOMALY_SEVERITY.get(kind, "warning")
        self.count        = 0
        self.total        = 0
        self.sample_value = None   # most recent triggering value

    def record(self, fired: bool, sample_value: Any = None) -> None:
        """
        Call once per sample.
        fired=True  → anomaly was detected this sample.
        fired=False → check was done, anomaly was not present.
        """
        self.total += 1
        if fired:
            self.count += 1
            self.sample_value = sample_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind":         self.kind,
            "severity":     self.severity,
            "frequency":    self.count / self.total if self.total > 0 else 0.0,
            "count":        self.count,
            "total":        self.total,
            "sample_value": float(self.sample_value) if self.sample_value is not None else None,
        }


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# _store[run_id][epoch][layer_id][kind] -> _AnomalyAccumulator
_store: Dict[int, Dict[int, Dict[int, Dict[str, _AnomalyAccumulator]]]] = {}


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record(
    run:          int,
    epoch:        int,
    layer_id:     int,
    kind:         str,
    fired:        bool,
    sample_value: Any = None,
) -> None:
    """
    Record one anomaly check for a single sample.

    Called by observers after every forward/backward pass per layer.
    fired=True  → the anomaly condition was met this sample.
    fired=False → checked but not triggered.
    """
    run_d   = _store.setdefault(run, {})
    epoch_d = run_d.setdefault(epoch, {})
    layer_d = epoch_d.setdefault(layer_id, {})
    acc     = layer_d.get(kind)
    if acc is None:
        acc = _AnomalyAccumulator(kind)
        layer_d[kind] = acc
    acc.record(fired, sample_value)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_layer_anomalies(
    run:      int,
    epoch:    int,
    layer_id: int,
) -> List[Dict[str, Any]]:
    """
    Return a list of anomaly dicts for one layer in one epoch.
    Only returns anomaly kinds that fired at least once.
    """
    layer_d = _store.get(run, {}).get(epoch, {}).get(layer_id, {})
    return [
        acc.to_dict()
        for acc in layer_d.values()
        if acc.count > 0
    ]


def get_epoch_anomalies(run: int, epoch: int) -> Dict[int, List[Dict[str, Any]]]:
    """
    Return anomalies for all layers in one epoch.
    Shape: { layer_id: [ { kind, severity, frequency, count, total, sample_value } ] }
    Only includes layers/kinds that fired at least once.
    """
    epoch_d = _store.get(run, {}).get(epoch, {})
    result  = {}
    for layer_id, kinds in epoch_d.items():
        fired = [acc.to_dict() for acc in kinds.values() if acc.count > 0]
        if fired:
            result[layer_id] = fired
    return result


def has_any(run: int, epoch: int, layer_id: int) -> bool:
    return bool(_store.get(run, {}).get(epoch, {}).get(layer_id))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def clear() -> None:
    _store.clear()

def clear_run(run: int) -> None:
    _store.pop(run, None)

def clear_epoch(run: int, epoch: int) -> None:
    _store.get(run, {}).pop(epoch, None)