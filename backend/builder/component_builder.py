from typing import List
from neural_network.Layers.InputLayer import InputLayer
from neural_network.Layers.Layer import Linear, LayerNorm
from neural_network.Layers.activation_fns import Relu, Sigmoid, Tanh, LeakyRelu, Elu
from neural_network.Nodes.Nodes import (
    AddNode, SubNode, MulNode, DivNode,
    SqNode, NegNode, SqrtNode,
    RangeclipNode, ConcatNode, SplitNode,
)

# ---------------------------------------------------------------------------
# Layer & Node factory
# ---------------------------------------------------------------------------

_node_factories = {
    "add":           lambda c, node_id: AddNode(node_id=node_id, axis=c.get("axis")),
    "addnode":       lambda c, node_id: AddNode(node_id=node_id, axis=c.get("axis")),
    "sub":           lambda c, node_id: SubNode(node_id=node_id),
    "subnode":       lambda c, node_id: SubNode(node_id=node_id),
    "mul":           lambda c, node_id: MulNode(node_id=node_id),
    "mulnode":       lambda c, node_id: MulNode(node_id=node_id),
    "div":           lambda c, node_id: DivNode(node_id=node_id),
    "divnode":       lambda c, node_id: DivNode(node_id=node_id),
    "sq":            lambda c, node_id: SqNode(node_id=node_id),
    "sqnode":        lambda c, node_id: SqNode(node_id=node_id),
    "neg":           lambda c, node_id: NegNode(node_id=node_id),
    "negnode":       lambda c, node_id: NegNode(node_id=node_id),
    "sqrt":          lambda c, node_id: SqrtNode(node_id=node_id),
    "sqrtnode":      lambda c, node_id: SqrtNode(node_id=node_id),
    "clip":          lambda c, node_id: RangeclipNode(node_id=node_id, min_val=float(c.get("min_val", -1.0)), max_val=float(c.get("max_val", 1.0))),
    "rangeclip":     lambda c, node_id: RangeclipNode(node_id=node_id, min_val=float(c.get("min_val", -1.0)), max_val=float(c.get("max_val", 1.0))),
    "rangeclipnode": lambda c, node_id: RangeclipNode(node_id=node_id, min_val=float(c.get("min_val", -1.0)), max_val=float(c.get("max_val", 1.0))),
    "concat":        lambda c, node_id: ConcatNode(node_id=node_id, axis=int(c.get("axis", 1))),
    "concatnode":    lambda c, node_id: ConcatNode(node_id=node_id, axis=int(c.get("axis", 1))),
    "split":         lambda c, node_id: SplitNode(node_id=node_id, n_splits=int(c.get("n_splits", 2)), axis=int(c.get("axis", 1))),
    "splitnode":     lambda c, node_id: SplitNode(node_id=node_id, n_splits=int(c.get("n_splits", 2)), axis=int(c.get("axis", 1))),
}

def _build_layer(layer_id, component, config):
     # Support both old key names and new unified 'type'
    ltype = (
        component.get("type")
        or component.get("layerType")
        or ""
    ).strip().lower()
    if ltype == "linear":
        return Linear(
            layer_id,
            int(config.get("in_features", 128)),
            int(config.get("out_features", 64)),
        )
    if ltype == "relu":
        return Relu(layer_id)
    if ltype == "layernorm":
        return LayerNorm(
            layer_id,
            int(config.get("in_features", 64)),
            float(config.get("eps", 1e-5)),
        )
    if ltype in ("inputlayer", "input_layer", "input"):
        inputs = component.get("ports") or config.get("inputs") or []
        return InputLayer(layer_id, inputs)
    raise ValueError(f"registry: unknown layer type '{ltype}'")


def _build_node(node_id, component, config):
    # Support both old key names and new unified 'type'
    ntype = (
        component.get("type")
        or component.get("nodeType")
        or ""
    ).strip().lower()
    if ntype not in _node_factories:
        raise ValueError(f"registry: unknown node type '{ntype}'")
    return _node_factories[ntype](config, node_id=node_id)


def build_component(component: dict, component_id_int: int):
    """
    Instantiate a Layer or Node from a React Flow node's data dict.

    component keys:
        kind      : "layer" | "node"
        type      : lowercase type string, e.g. "linear", "add"
        config    : dict of field values
    """
    kind   = component.get("kind", "layer")
    config = component.get("config", {})

    if kind == "layer":
        return _build_layer(component_id_int, component, config)

    if kind == "node":
        return _build_node(component_id_int, component, config)
    
    raise ValueError(f"registry: unknown node kind '{kind}'")


def build_observers(selected_names: List[str], run_id: int = 0) -> list:
    """
    Instantiate the requested observers for a run.

    SignalStatsObserver is always included when any observer is requested
    because it is the observer that runs anomaly detection.  If the user
    deselects it from the UI, we still need it for AnomalyStore writes.
    SignalShapeObserver is additive and optional.
    """
    if not selected_names:
        return []

    # Always include SignalStatsObserver (carries anomaly detection).
    names = list(selected_names)
    if "SignalStatsObserver" not in names:
        names.insert(0, "SignalStatsObserver")

    observers = []
    for name in names:
        cls = _OBSERVER_CLASSES.get(name)
        if cls is not None:
            observers.append(cls(run_id))
    return observers