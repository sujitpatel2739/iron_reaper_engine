"""
collector.py
------------
Reads CacheStore after a layer completes and returns a clean metrics dict.

Single responsibility: CacheStore → dict.

Used by step_engine.py after every forward and backward step to populate
StepEvent.metrics. Nothing else in the project should do this conversion.
"""

import numpy as np
from typing import Dict, Any

import cache.CacheStore as CacheStore
from cache.CacheStore import (
    SLOT_INPUT, SLOT_OUTPUT,
    SLOT_GRAD_OUT, SLOT_GRAD_IN,
    SLOT_RESIDUAL, SLOT_SHORTCUT,
)


def collect(layer_id_int: int) -> Dict[str, Any]:
    """
    Pull all available slots from CacheStore for one layer and return a
    plain dict of JSON-serialisable scalars and lists.

    Only includes keys for slots that were actually written — callers
    should treat missing keys as "not yet available".
    """
    metrics: Dict[str, Any] = {}

    def _np(slot):
        t = CacheStore.read(layer_id_int, slot)
        return t.data if t is not None else None

    inp   = _np(SLOT_INPUT)
    out   = _np(SLOT_OUTPUT)
    g_out = _np(SLOT_GRAD_OUT)
    g_in  = _np(SLOT_GRAD_IN)
    res   = _np(SLOT_RESIDUAL)
    short = _np(SLOT_SHORTCUT)

    if inp  is not None:
        metrics["input_shape"]      = list(inp.shape)

    if out  is not None:
        metrics["output_shape"]     = list(out.shape)
        metrics["activation_mean"]  = float(out.mean())
        metrics["activation_var"]   = float(out.var())

    if g_out is not None:
        metrics["grad_norm"]        = float(np.linalg.norm(g_out))
        metrics["grad_var"]         = float(g_out.var())

    if g_in  is not None:
        metrics["grad_in_norm"]     = float(np.linalg.norm(g_in))

    if res   is not None:
        metrics["residual_energy"]  = float(np.mean(res ** 2))

    if short is not None:
        metrics["shortcut_energy"]  = float(np.mean(short ** 2))

    return metrics