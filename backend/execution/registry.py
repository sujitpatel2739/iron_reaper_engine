"""
registry.py
-----------
Single source of truth for every supported type in the engine.

Three responsibilities — all about mapping names to objects:

1. LAYER_TYPES / NODE_TYPES
   Config schemas served by GET /layers and GET /nodes.
   Adding a new layer = add one entry here + one in the factory below.

2. build_layer(node_data, layer_id)
   Instantiates the correct Layer or Node from a graph node's data dict.

3. build_observers(selected_names, run_id)
   Instantiates the requested observers for a run.
"""

from typing import Any, Dict, List

from neural_network.Layers.Layer import Linear, Relu, LayerNorm
from neural_network.Nodes.Nodes import (
    AddNode, SubNode, MulNode, DivNode,
    SqNode, NegNode, SqrtNode, ScaleNode,
    ClipNode, ConcatNode, SplitNode,
)
from Observers.Observers import (
    SignalStatsObserver,
    SignalShapeObserver,
    ResidualEnergyObserver,
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
            {"name": "in_features", "type": "int",   "min": 1,      "default": 64},
            {"name": "eps",         "type": "float",                 "default": 1e-5},
        ],
    },
}

NODE_TYPES: Dict[str, dict] = {
    "AddNode":    {"fields": [{"name": "axis", "type": "int_or_null", "default": None}], "inputs": "N"},
    "SubNode":    {"fields": [],                                                          "inputs": 2},
    "MulNode":    {"fields": [],                                                          "inputs": "N"},
    "DivNode":    {"fields": [],                                                          "inputs": 2},
    "SqNode":     {"fields": [],                                                          "inputs": 1},
    "NegNode":    {"fields": [],                                                          "inputs": 1},
    "SqrtNode":   {"fields": [],                                                          "inputs": 1},
    "ScaleNode":  {"fields": [{"name": "scalar",   "type": "float", "default": 1.0}],    "inputs": 1},
    "ClipNode":   {"fields": [{"name": "min_val",  "type": "float", "default": -1.0},
                               {"name": "max_val",  "type": "float", "default":  1.0}],  "inputs": 1},
    "ConcatNode": {"fields": [{"name": "axis",     "type": "int",   "default": 1}],      "inputs": "N"},
    "SplitNode":  {"fields": [{"name": "n_splits", "type": "int",   "default": 2},
                               {"name": "axis",     "type": "int",   "default": 1}],     "inputs": 1},
}

OBSERVER_TYPES: Dict[str, dict] = {
    "SignalStatsObserver":    {"label": "Signal Stats",    "desc": "activation mean/var, grad norm/var"},
    "SignalShapeObserver":    {"label": "Signal Shape",    "desc": "tensor shapes through the network"},
    "ResidualEnergyObserver": {"label": "Residual Energy", "desc": "residual vs shortcut path energy"},
}


# ---------------------------------------------------------------------------
# Layer / Node factory
# ---------------------------------------------------------------------------

def build_layer(node_data: dict, layer_id: int) -> Any:
    """
    Instantiate a Layer or Node from a React Flow node's data dict.

    node_data keys:
        kind      : "layer" | "node"
        layerType : e.g. "Linear"   (when kind == "layer")
        nodeType  : e.g. "AddNode"  (when kind == "node")
        config    : dict of field values
    """
    kind   = node_data.get("kind", "layer")
    config = node_data.get("config", {})

    if kind == "layer":
        ltype = node_data.get("layerType", "")

        if ltype == "Linear":
            return Linear(
                layer_id,
                int(config.get("in_features", 128)),
                int(config.get("out_features", 64)),
            )
        if ltype == "Relu":
            return Relu(layer_id)

        if ltype == "LayerNorm":
            return LayerNorm(
                layer_id,
                int(config.get("in_features", 64)),
                float(config.get("eps", 1e-5)),
            )

        raise ValueError(f"registry: unknown layer type '{ltype}'")

    if kind == "node":
        ntype = node_data.get("nodeType", "")

        _node_factories = {
            "AddNode":    lambda c: AddNode(axis=c.get("axis")),
            "SubNode":    lambda c: SubNode(),
            "MulNode":    lambda c: MulNode(),
            "DivNode":    lambda c: DivNode(),
            "SqNode":     lambda c: SqNode(),
            "NegNode":    lambda c: NegNode(),
            "SqrtNode":   lambda c: SqrtNode(),
            "ScaleNode":  lambda c: ScaleNode(scalar=float(c.get("scalar", 1.0))),
            "ClipNode":   lambda c: ClipNode(
                              min_val=float(c.get("min_val", -1.0)),
                              max_val=float(c.get("max_val",  1.0))),
            "ConcatNode": lambda c: ConcatNode(axis=int(c.get("axis", 1))),
            "SplitNode":  lambda c: SplitNode(
                              n_splits=int(c.get("n_splits", 2)),
                              axis=int(c.get("axis", 1))),
        }

        if ntype not in _node_factories:
            raise ValueError(f"registry: unknown node type '{ntype}'")
        return _node_factories[ntype](config)

    raise ValueError(f"registry: unknown node kind '{kind}'")


# ---------------------------------------------------------------------------
# Observer factory
# ---------------------------------------------------------------------------

_OBSERVER_CLASSES = {
    "SignalStatsObserver":    SignalStatsObserver,
    "SignalShapeObserver":    SignalShapeObserver,
    "ResidualEnergyObserver": ResidualEnergyObserver,
}


def build_observers(selected_names: List[str], run_id: int = 0) -> list:
    """
    Instantiate the requested observers.
    Unknown names are silently skipped (they may have been removed from the registry).
    """
    observers = []
    for name in selected_names:
        cls = _OBSERVER_CLASSES.get(name)
        if cls is not None:
            observers.append(cls(run_id))
    return observers