"""
api/bridge.py
-------------
Adapts a PyTorch nn.Module into the React Flow graph format
(nodes + edges JSON) used by the frontend.

For the full diagnostic run it also attaches forward/backward hooks
that feed activation and gradient data into the observer pipeline via
ironframe Tensors written to CacheStore.

Two public surfaces
-------------------
graph_from_model(model)
    Static inspection only — no data pass.
    Returns the graph JSON for rendering the architecture in the UI.

ModelBridge(model, observers)
    Attaches hooks, runs inference, feeds observers.
    Used by runner.py for full diagnostic runs.
"""

import io
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.ironframe.ironframe import Tensor
from backend.Observers.Observers import LayerObserver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _wrap_frozen(arr: np.ndarray) -> Tensor:
    """Wrap numpy array as a frozen ironframe Tensor for observer reads."""
    t = Tensor(arr, requires_grad=False)
    t.freezed = True
    return t


def _is_leaf(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def _param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Type mapping: torch layer names → our layer type names
# ---------------------------------------------------------------------------

_TORCH_TO_OURS = {
    "linear":         "Linear",
    "relu":           "Relu",
    "layernorm":      "LayerNorm",
    "batchnorm1d":    "LayerNorm",   # approximate — no direct equivalent yet
    "batchnorm2d":    "LayerNorm",
}


# ---------------------------------------------------------------------------
# Static graph extraction
# ---------------------------------------------------------------------------

def graph_from_model(model: nn.Module) -> dict:
    """
    Walk a PyTorch model and return a React Flow graph dict
    { "nodes": [...], "edges": [...] }.

    The graph is a linear chain of leaf modules in traversal order.
    Branch detection from TorchScript graphs is not supported yet.
    """
    graph_nodes: List[dict] = []
    graph_edges: List[dict] = []

    leaf_iter = [
        (name, mod)
        for name, mod in model.named_modules()
        if _is_leaf(mod)
    ]

    for i, (name, mod) in enumerate(leaf_iter):
        torch_type  = type(mod).__name__.lower()
        our_type    = _TORCH_TO_OURS.get(torch_type, torch_type.capitalize())
        node_id     = f"imported_{i}"

        config: dict = {}
        if hasattr(mod, "in_features"):
            config["in_features"]  = mod.in_features
        if hasattr(mod, "out_features"):
            config["out_features"] = mod.out_features
        if hasattr(mod, "normalized_shape"):
            config["in_features"]  = mod.normalized_shape[-1]
        if hasattr(mod, "eps"):
            config["eps"]          = mod.eps

        graph_nodes.append({
            "id":       node_id,
            "type":     "layerNode",
            "position": {"x": 200, "y": i * 130},
            "data": {
                "kind":      "layer",
                "layerType": our_type,
                "config":    config,
                "status":    "idle",
                "metrics":   None,
                "warnings":  [],
            },
        })

        if i > 0:
            prev_id = f"imported_{i - 1}"
            graph_edges.append({
                "id":     f"e_{prev_id}_{node_id}",
                "source": prev_id,
                "target": node_id,
            })

    return {"nodes": graph_nodes, "edges": graph_edges}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(file_bytes: bytes) -> tuple:
    """
    Load a PyTorch model from raw bytes (.pt / .pth).
 
    Supports full models saved with torch.save(model, path).
    Does NOT support bare state dicts — raises a clear ValueError if detected.
 
    PyTorch 2.6 compatibility
    -------------------------
    We try three strategies in order, stopping at the first that succeeds:
 
    1. weights_only=True  — safest, works for state dicts and simple models
    2. weights_only=False — required for full model objects containing custom classes
    3. Neither worked     — raise a descriptive error pointing the user to the fix
 
    Returns (model, warnings: List[str])
    """
    warnings = []
 
    # Strategy 1: weights_only=True (safe, PyTorch 2.6+ default)
    try:
        buf = io.BytesIO(file_bytes)
        obj = torch.load(buf, map_location="cpu", weights_only=True)
        return _validate_and_return(obj, warnings)
    except Exception:
        pass  # fall through to strategy 2
 
    # Strategy 2: weights_only=False (required for full model objects)
    try:
        buf = io.BytesIO(file_bytes)
        obj = torch.load(buf, map_location="cpu", weights_only=False)
        warnings.append(
            "Model loaded with weights_only=False because it contains a full "
            "model class (not just a state dict). Only load files from trusted sources."
        )
        return _validate_and_return(obj, warnings)
    except Exception as e:
        # Both strategies failed — give a clear, actionable error
        raise ValueError(
            f"Failed to load the uploaded file as a PyTorch model.\n\n"
            f"Most likely cause: the file was saved with torch.save(model, path) "
            f"using a custom model class that is not available in this environment.\n\n"
            f"To fix this:\n"
            f"  • Make sure you upload the full model, not just a state dict.\n"
            f"  • If your model uses a custom class, torch.save(model.state_dict(), path) "
            f"cannot be used — the full model object is required.\n\n"
            f"Original error: {e}"
        )
 
 
def _validate_and_return(obj, warnings):
    """Check the loaded object is a usable nn.Module and return it."""
    if isinstance(obj, nn.Module):
        obj.eval()
        return obj, warnings
 
    if isinstance(obj, dict):
        # Looks like a state dict
        raise ValueError(
            "The uploaded file appears to be a state dict "
            "(saved with torch.save(model.state_dict(), path)), not a full model.\n"
            "Please re-save your model with torch.save(model, path) and upload again."
        )
 
    raise ValueError(
        f"Unrecognised object type in file: {type(obj).__name__}. "
        "Expected a torch.nn.Module."
    )


def total_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Hook-based bridge for full diagnostic runs
# ---------------------------------------------------------------------------

class ModelBridge:
    """
    Attaches PyTorch forward/backward hooks to every leaf module and routes
    intercepted tensors into the observer pipeline.

    The model runs natively in PyTorch. Hooks wrap tensors as frozen
    ironframe Tensors and call observer lifecycle methods directly.
    """

    def __init__(self, model: nn.Module, observers: List[LayerObserver]):
        self.model     = model
        self.observers = observers
        self._hooks: List[Any] = []

        # Assign stable integer ids to leaf modules
        self._id_map: Dict[int, int] = {}
        for i, (_, mod) in enumerate(
            (x for x in model.named_modules() if _is_leaf(x[1]))
        ):
            self._id_map[id(mod)] = i

    def attach(self) -> None:
        for _, mod in self.model.named_modules():
            if not _is_leaf(mod):
                continue
            lid = self._id_map[id(mod)]
            self._hooks.append(
                mod.register_forward_pre_hook(self._fwd_pre(lid))
            )
            self._hooks.append(
                mod.register_forward_hook(self._fwd_post(lid))
            )
            self._hooks.append(
                mod.register_full_backward_hook(self._bwd(lid))
            )

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- Hook factories ------------------------------------------------------

    def _fwd_pre(self, layer_id: int):
        observers = self.observers
        def hook(module, args):
            if args and isinstance(args[0], torch.Tensor):
                x = _wrap_frozen(_to_numpy(args[0]))
                for obs in observers:
                    obs.on_forward_pre(layer_id)
        return hook

    def _fwd_post(self, layer_id: int):
        import cache.CacheStore as CacheStore
        from cache.CacheStore import SLOT_OUTPUT
        observers = self.observers
        def hook(module, args, output):
            if isinstance(output, torch.Tensor):
                t = Tensor(_to_numpy(output), requires_grad=False)
                CacheStore.write(layer_id, SLOT_OUTPUT, t)
                for obs in observers:
                    obs.on_forward_post(layer_id)
        return hook

    def _bwd(self, layer_id: int):
        import cache.CacheStore as CacheStore
        from cache.CacheStore import SLOT_GRAD_OUT, SLOT_GRAD_IN
        observers = self.observers
        def hook(module, grad_input, grad_output):
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                t = Tensor(_to_numpy(grad_output[0]), requires_grad=False)
                CacheStore.write(layer_id, SLOT_GRAD_OUT, t)
                for obs in observers:
                    obs.on_backward_pre(layer_id)
            if grad_input and isinstance(grad_input[0], torch.Tensor):
                t = Tensor(_to_numpy(grad_input[0]), requires_grad=False)
                CacheStore.write(layer_id, SLOT_GRAD_IN, t)
                for obs in observers:
                    obs.on_backward_post(layer_id)
        return hook