# """
# CacheStore.py
# -------------
# Module-level cache store. No instantiation — import and use directly.

#     import CacheStore
#     CacheStore.write(layer_id, "output", tensor)
#     CacheStore.read(layer_id, "output")
#     CacheStore.clear()

# Python's import system guarantees this module is loaded exactly once,
# so self._store is naturally a singleton — no __new__ tricks needed.

# Structure
# ---------
#     self._store[layer_id][slot] -> Tensor   (live reference, autograd graph intact)

# Named slots
# -----------
# Use the constants below instead of raw strings to avoid typos.
# Layers may write any additional slot name they need beyond the standard set.
# """

# from typing import Optional, List
# from core.ironframe.ironframe import Tensor


# # -- Slot name constants ----------------------------------------------------

# SLOT_INPUT    = "input"
# SLOT_OUTPUT   = "output"
# SLOT_GRAD_OUT = "grad_out"
# SLOT_GRAD_IN  = "grad_in"
# SLOT_RESIDUAL = "residual"
# SLOT_SHORTCUT = "shortcut"


# class Cachestore:
#     # -- Module-level state ------------------------------------------------------
#     _instance = None
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(Cachestore, cls).__new__(cls)
#         return cls._instance

#     def __init__(self):
#         self._store: dict[int, dict[str, Tensor]] = {}
        
#     # -- Write / Read ------------------------------------------------------------

#     def write(self, layer_id: int, slot: str, tensor: Tensor) -> None:
#         """
#         Store a live Tensor reference at (layer_id, slot).
#         The autograd graph on the tensor is never touched — it stays fully alive.
#         """
#         if layer_id not in self._store:
#             self._store[layer_id] = {}
#         self._store[layer_id][slot] = tensor


#     def read(self, layer_id: int, slot: str) -> Optional[Tensor]:
#         """
#         Return the Tensor at (layer_id, slot), or None if absent.
#         The returned Tensor is the original — not a copy, not frozen.
#         """
#         return self._store.get(layer_id, {}).get(slot)


#     def read_required(self, layer_id: int, slot: str) -> Tensor:
#         """
#         Like read(), but raises KeyError if absent.
#         Use inside layers that cannot proceed without the value.
#         """
#         t = self.read(layer_id, slot)
#         if t is None:
#             raise KeyError(
#                 f"CacheStore: nothing at layer_id={layer_id}, slot='{slot}'. "
#                 f"Available: {self.slots(layer_id)}"
#             )
#         return t


#     # -- Inspection --------------------------------------------------------------

#     def slots(self, layer_id: int) -> List[str]:
#         """All slot names written for a given layer_id."""
#         return list(self._store.get(layer_id, {}).keys())


#     def layer_ids(self, ) -> List[int]:
#         """All layer_ids that have at least one written slot."""
#         return sorted(self._store.keys())


#     def has(self, layer_id: int, slot: str) -> bool:
#         return self.read(layer_id, slot) is not None


#     # -- Lifecycle ---------------------------------------------------------------

#     def clear(self, ) -> None:
#         """
#         Drop all Tensor references.
#         Call between forward/backward pass pairs to release the computation
#         graph and prevent stale reads across passes.
#         """
#         self._store.clear()


#     def clear_layer(self, layer_id: int) -> None:
#         """Drop all slots for one layer."""
#         self._store.pop(layer_id, None)


#     def clear_slot(self, layer_id: int, slot: str) -> None:
#         """Drop a single slot."""
#         if layer_id in self._store:
#             self._store[layer_id].pop(slot, None)


#     # -- Debug -------------------------------------------------------------------

#     def debug_repr(self, ) -> str:
#         if not self._store:
#             return "CacheStore: (empty)"
#         lines = ["CacheStore:"]
#         for lid in self.layer_ids():
#             lines.append(f"  layer {lid}: {self.slots(lid)}")
#         return "\n".join(lines)
    

# # Initialization of CacheStore
# cachestore = Cachestore()


"""
CacheStore.py
-------------
Module-level cache store. No instantiation — import and use directly.

    import CacheStore
    CacheStore.write(layer_id, "output", tensor)
    CacheStore.read(layer_id, "output")
    CacheStore.clear()

Python's import system guarantees this module is loaded exactly once,
so _store is naturally a singleton — no __new__ tricks needed.

Structure
---------
    _store[layer_id][slot] -> Tensor   (live reference, autograd graph intact)

Named slots
-----------
Use the constants below instead of raw strings to avoid typos.
Layers may write any additional slot name they need beyond the standard set.
"""

from typing import Optional, List
from core.ironframe.ironframe import Tensor


# -- Slot name constants ----------------------------------------------------

SLOT_INPUT    = "input"
SLOT_OUTPUT   = "output"
SLOT_GRAD_OUT = "grad_out"
SLOT_GRAD_IN  = "grad_in"
SLOT_RESIDUAL = "residual"
SLOT_SHORTCUT = "shortcut"


# -- Module-level state ------------------------------------------------------

_store: dict[int, dict[str, Tensor]] = {}


# -- Write / Read ------------------------------------------------------------

def write(layer_id: int, slot: str, tensor: Tensor) -> None:
    """
    Store a live Tensor reference at (layer_id, slot).
    The autograd graph on the tensor is never touched — it stays fully alive.
    """
    if layer_id not in _store:
        _store[layer_id] = {}
    _store[layer_id][slot] = tensor


def read(layer_id: int, slot: str) -> Optional[Tensor]:
    """
    Return the Tensor at (layer_id, slot), or None if absent.
    The returned Tensor is the original — not a copy, not frozen.
    """
    return _store.get(layer_id, {}).get(slot)


def read_required(layer_id: int, slot: str) -> Tensor:
    """
    Like read(), but raises KeyError if absent.
    Use inside layers that cannot proceed without the value.
    """
    t = read(layer_id, slot)
    if t is None:
        raise KeyError(
            f"CacheStore: nothing at layer_id={layer_id}, slot='{slot}'. "
            f"Available: {slots(layer_id)}"
        )
    return t


# -- Inspection --------------------------------------------------------------

def slots(layer_id: int) -> List[str]:
    """All slot names written for a given layer_id."""
    return list(_store.get(layer_id, {}).keys())


def layer_ids() -> List[int]:
    """All layer_ids that have at least one written slot."""
    return sorted(_store.keys())


def has(layer_id: int, slot: str) -> bool:
    return read(layer_id, slot) is not None


# -- Lifecycle ---------------------------------------------------------------

def clear() -> None:
    """
    Drop all Tensor references.
    Call between forward/backward pass pairs to release the computation
    graph and prevent stale reads across passes.
    """
    _store.clear()


def clear_layer(layer_id: int) -> None:
    """Drop all slots for one layer."""
    _store.pop(layer_id, None)


def clear_slot(layer_id: int, slot: str) -> None:
    """Drop a single slot."""
    if layer_id in _store:
        _store[layer_id].pop(slot, None)


# -- Debug -------------------------------------------------------------------

def debug_repr() -> str:
    if not _store:
        return "CacheStore: (empty)"
    lines = ["CacheStore:"]
    for lid in layer_ids():
        lines.append(f"  layer {lid}: {slots(lid)}")
    return "\n".join(lines)