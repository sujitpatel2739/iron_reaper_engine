"""
registry.py
-----------
Single source of truth for every supported type in the engine.

Changes from previous version
------------------------------
1. SLOT_INPUT / SLOT_GRAD_OUT removed from OBSERVER_TYPES description
   to match the new Observers.py that only reads SLOT_OUTPUT / SLOT_GRAD_IN.

2. node_id = uuid.UUID bug fixed.
   The old code set node_id = uuid.UUID (the class itself, not an instance).
   This was passed as the node_id to every Node constructor, meaning all
   op nodes had the same id. Fixed to use layer_id (the integer already
   assigned by the planner for this node's position in the topo order).

3. build_observers now always includes SignalStatsObserver regardless of
   selection when the list is non-empty, because anomaly detection lives
   inside SignalStatsObserver and must always run during full runs.
   (This behaviour can be relaxed later if dedicated anomaly observers
   are split out.)
"""

from typing import Any, Dict, List
from Observers.Observers import (
    SignalStatsObserver,
    SignalShapeObserver,
)


# ---------------------------------------------------------------------------
# Type schemas  (served to frontend via GET /layers and GET /nodes)
# ---------------------------------------------------------------------------

LAYER_TYPES: Dict[str, dict] = {
    "Linear": {
        "fields": [
            {"name": "in_features",  "type": "int",   "min": 1, "default": 128},
            {"name": "out_features", "type": "int",   "min": 1, "default": 64},
        ],
    },
    "Relu": {
        "fields": [],
    },
    "LayerNorm": {
        "fields": [
            {"name": "in_features", "type": "int",   "min": 1,  "default": 64},
            {"name": "eps",         "type": "float",             "default": 1e-5},
        ],
    },
}

NODE_TYPES: Dict[str, dict] = {
    "AddNode":       {"fields": [{"name": "axis", "type": "int_or_null", "default": None}], "inputs": "N"},
    "SubNode":       {"fields": [],                                                          "inputs": 2},
    "MulNode":       {"fields": [],                                                          "inputs": "N"},
    "DivNode":       {"fields": [],                                                          "inputs": 2},
    "SqNode":        {"fields": [],                                                          "inputs": 1},
    "NegNode":       {"fields": [],                                                          "inputs": 1},
    "SqrtNode":      {"fields": [],                                                          "inputs": 1},
    "RangeclipNode": {"fields": [{"name": "min_val", "type": "float", "default": -1.0},
                                 {"name": "max_val", "type": "float", "default":  1.0}],    "inputs": 1},
    "ConcatNode":    {"fields": [{"name": "axis",     "type": "int",   "default": 1}],      "inputs": "N"},
    "SplitNode":     {"fields": [{"name": "n_splits", "type": "int",   "default": 2},
                                 {"name": "axis",     "type": "int",   "default": 1}],      "inputs": 1},
}

OBSERVER_TYPES: Dict[str, dict] = {
    "SignalStatsObserver": {
        "label": "Signal Stats",
        "desc":  "output activation stats + gradient-in stats + anomaly detection",
    },
    "SignalShapeObserver": {
        "label": "Signal Shape",
        "desc":  "output tensor shapes through the network",
    },
}

# ---------------------------------------------------------------------------
# Observer factory
# ---------------------------------------------------------------------------

_OBSERVER_CLASSES = {
    "SignalStatsObserver": SignalStatsObserver,
    "SignalShapeObserver": SignalShapeObserver,
}