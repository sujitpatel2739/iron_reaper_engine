"""
cache/CacheStore.py
-------------------
Module-level cache store. No instantiation — import and use directly.

Stores whatever the backend produced — either an ironframe Tensor
(custom engine) or a torch.Tensor (PyTorch bridge). No wrapping,
no conversion. Consumers handle both types themselves.

Structure:  _store[layer_id][slot] -> Tensor | torch.Tensor
"""

from typing import Optional, List, Union, Any

SLOT_INPUT    = "input"
SLOT_OUTPUT   = "output"
SLOT_GRAD_OUT = "grad_out"
SLOT_GRAD_IN  = "grad_in"
SLOT_RESIDUAL = "residual"
SLOT_SHORTCUT = "shortcut"

_store: dict[int, dict[str, Any]] = {}


def write(layer_id: int, slot: str, tensor: Any) -> None:
    if layer_id not in _store:
        _store[layer_id] = {}
    _store[layer_id][slot] = tensor


def read(layer_id: int, slot: str) -> Optional[Any]:
    return _store.get(layer_id, {}).get(slot)


def read_required(layer_id: int, slot: str) -> Any:
    t = read(layer_id, slot)
    if t is None:
        raise KeyError(
            f"CacheStore: nothing at layer_id={layer_id}, slot='{slot}'. "
            f"Available: {slots(layer_id)}"
        )
    return t


def slots(layer_id: int) -> List[str]:
    return list(_store.get(layer_id, {}).keys())


def layer_ids() -> List[int]:
    return sorted(_store.keys())


def has(layer_id: int, slot: str) -> bool:
    return read(layer_id, slot) is not None


def clear() -> None:
    _store.clear()


def clear_layer(layer_id: int) -> None:
    _store.pop(layer_id, None)


def debug_repr() -> str:
    if not _store:
        return "CacheStore: (empty)"
    lines = ["CacheStore:"]
    for lid in layer_ids():
        entries = []
        for slot, val in _store[lid].items():
            t = type(val).__name__
            shape = getattr(val, 'shape', None) or getattr(getattr(val, 'data', None), 'shape', None)
            entries.append(f"{slot}:{t}{list(shape) if shape else ''}")
        lines.append(f"  layer {lid}: {entries}")
    return "\n".join(lines)
