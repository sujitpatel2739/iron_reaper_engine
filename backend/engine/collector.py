"""
collector.py
------------
Reads CacheStore after a layer step completes and returns a clean metrics
dict for immediate WebSocket delivery in step mode.

WHY THIS STILL EXISTS:
  Step mode delivers live per-layer metrics to the frontend after each
  individual NEXT click.  These metrics are immediate and transient —
  they do not go through MetricStore or AnomalyStore.  collector.py is
  the bridge between CacheStore and StepEvent.metrics for that use case.

WHY ONLY OUTPUT AND GRAD_IN:
  We removed SLOT_INPUT and SLOT_GRAD_OUT writes from Layer.py entirely.
  This module must match — reading a slot that is never written returns
  None, which was handled gracefully already, but we clean up the dead
  code paths for clarity.

  Step mode shows:
    - output_shape       (from SLOT_OUTPUT)
    - activation_mean    (from SLOT_OUTPUT)
    - activation_std     (from SLOT_OUTPUT)
    - grad_in_norm       (from SLOT_GRAD_IN)
    - grad_in_std        (from SLOT_GRAD_IN)

  These are the same metrics MetricStore accumulates across samples,
  so the step-mode panel and the full-run metrics panel show the same
  quantities — just immediate vs. aggregated.
"""

import numpy as np
from typing import Dict, Any

import cache.CacheStore as CacheStore
from cache.CacheStore import SLOT_OUTPUT, SLOT_GRAD_IN


def collect(layer_id_int: int) -> Dict[str, Any]:
    """
    Pull SLOT_OUTPUT and SLOT_GRAD_IN from CacheStore for one layer and
    return a plain dict of JSON-serialisable scalars and lists.

    Keys are absent (not None) when the slot has not been written yet —
    e.g. grad_in_norm is absent during the forward pass.
    """
    metrics: Dict[str, Any] = {}

    out_t   = CacheStore.read(layer_id_int, SLOT_OUTPUT)
    gin_t   = CacheStore.read(layer_id_int, SLOT_GRAD_IN)

    if out_t is not None:
        out = out_t.data
        metrics["output_shape"]    = list(out.shape)
        metrics["activation_mean"] = float(out.mean())
        metrics["activation_std"]  = float(out.std())

    if gin_t is not None:
        gin = gin_t.data
        metrics["grad_in_norm"] = float(np.linalg.norm(gin))
        metrics["grad_in_std"]  = float(gin.std())

    return metrics