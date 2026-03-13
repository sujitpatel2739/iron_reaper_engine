"""
MetricStore.py
--------------
Module-level metric store. No instantiation — import and use directly.

    import MetricStore
    MetricStore.add(run, layer_id, "activation_var", values)
    MetricStore.get_sequence(run, "activation_var", agg="mean")
    MetricStore.clear()

Python's import system guarantees this module is loaded exactly once,
so _metrics is naturally a singleton — no class needed.

Structure
---------
    _metrics[run][layer_id][metric_name] -> [values]
"""

import numpy as np
from typing import List, Dict, Any, Optional


# -- Module-level state ------------------------------------------------------

_metrics: dict[int, dict[int, dict[str, list]]] = {}


# -- Write / Read ------------------------------------------------------------

def add(run: int, layer_id: int, metric_name: str, values: List) -> None:
    """Append values to (run, layer_id, metric_name)."""
    _metrics.setdefault(run, {}).setdefault(layer_id, {}).setdefault(metric_name, []).extend(values)


def get(run: int, layer_id: int, metric_name: str) -> Optional[List]:
    """Return raw value list for (run, layer_id, metric_name), or None."""
    return _metrics.get(run, {}).get(layer_id, {}).get(metric_name)


def get_sequence(run: int, metric_name: str, agg: str = "mean") -> Dict[int, Any]:
    """
    Return {layer_id: aggregated_value} across all layers for one metric.

    agg options: "mean", "max", "min", "none" (returns raw list)
    """
    if run not in _metrics:
        return {}

    _agg = {
        "mean": np.mean,
        "max":  np.max,
        "min":  np.min,
        "none": lambda x: x,
    }
    if agg not in _agg:
        raise ValueError(f"Unsupported agg='{agg}'. Choose from: {list(_agg)}")

    fn = _agg[agg]
    result = {}
    for layer_id, layer_metrics in _metrics[run].items():
        if metric_name in layer_metrics:
            result[layer_id] = fn(layer_metrics[metric_name])
    return result


def get_all(run: int) -> Dict[int, Dict[str, List]]:
    """Return everything stored for a run: {layer_id: {metric: [values]}}"""
    return _metrics.get(run, {})


# -- Inspection --------------------------------------------------------------

def runs() -> List[int]:
    return sorted(_metrics.keys())


def layer_ids(run: int) -> List[int]:
    return sorted(_metrics.get(run, {}).keys())


def metric_names(run: int, layer_id: int) -> List[str]:
    return list(_metrics.get(run, {}).get(layer_id, {}).keys())


# -- Lifecycle ---------------------------------------------------------------

def clear() -> None:
    """Drop everything. Call between runs."""
    _metrics.clear()


def clear_run(run: int) -> None:
    """Drop all data for one run."""
    _metrics.pop(run, None)


# -- Debug -------------------------------------------------------------------

def debug_repr(run: int) -> str:
    if run not in _metrics:
        return f"MetricStore: run {run} not found"
    lines = [f"MetricStore (run={run}):"]
    for lid in layer_ids(run):
        names = metric_names(run, lid)
        lines.append(f"  layer {lid}: {names}")
    return "\n".join(lines)