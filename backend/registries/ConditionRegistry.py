"""
ConditionRegistry.py
--------------------
Module-level registry for named condition functions used by ConditionNode.

    import ConditionRegistry
    ConditionRegistry.register("my_condition", lambda x: x > 0.5)
    cond = ConditionRegistry.get("my_condition")

No instantiation needed — Python's import system guarantees this module
is loaded exactly once, so _registry is naturally a singleton.

Structure
---------
    _registry[name] -> Condition(name, fn)

A Condition wraps:
    name : str        — unique identifier, used for serialization
    fn   : callable   — takes a numpy array, returns a boolean numpy array
                        of the same shape (elementwise mask)

Built-in conditions are registered at module load time.
User conditions are registered at runtime before building/loading networks.

Serialization contract
----------------------
ConditionNode saves only the condition name to disk.
On load, the name is looked up here — so any custom condition must be
re-registered before loading a network that uses it.
If a name is not found, get() raises a KeyError with a helpful message.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional

# ---------------------------------------------------------------------------
# Condition dataclass
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """
    Wraps a condition name and its callable together.

    Attributes
    ----------
    name : str
        Unique string identifier. Used for serialization/deserialization.
        Must match a key in the registry.

    fn : callable
        Takes a numpy array, returns a boolean numpy array of the same shape.
        Example: lambda x: x > 0

    Example
    -------
        cond = Condition(name="greater_than_zero", fn=lambda x: x > 0)
        mask = cond.fn(some_array)   # -> boolean ndarray
    """
    name: str
    fn:   Callable[[np.ndarray], np.ndarray]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the condition function directly on a numpy array."""
        return self.fn(x)

    def __repr__(self):
        return f"Condition(name='{self.name}')"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_registry: dict[str, Condition] = {}

# ---------------------------------------------------------------------------
# Built-in conditions
# ---------------------------------------------------------------------------
# All built-ins operate on raw numpy arrays (x is np.ndarray).
# They return a boolean np.ndarray of the same shape — elementwise masks.
# ---------------------------------------------------------------------------

def _register_builtins() -> None:

    # -- Sign conditions -----------------------------------------------------
    register("greater_than_zero",    lambda x: x > 0)
    register("less_than_zero",       lambda x: x < 0)
    register("greater_equal_zero",   lambda x: x >= 0)
    register("less_equal_zero",      lambda x: x <= 0)
    register("equal_zero",           lambda x: x == 0)
    register("not_equal_zero",       lambda x: x != 0)

    # -- Threshold conditions ------------------------------------------------
    register("greater_than_one",     lambda x: x > 1)
    register("less_than_one",        lambda x: x < 1)
    register("greater_than_half",    lambda x: x > 0.5)
    register("less_than_half",       lambda x: x < 0.5)
    register("in_unit_range",        lambda x: (x >= 0) & (x <= 1))

    # -- Statistical conditions (whole-tensor, broadcast to shape) -----------
    register("above_mean",           lambda x: x > x.mean())
    register("below_mean",           lambda x: x < x.mean())
    register("above_median",         lambda x: x > np.median(x))
    register("below_median",         lambda x: x < np.median(x))

    # -- Magnitude conditions ------------------------------------------------
    register("large_magnitude",      lambda x: np.abs(x) > 1)
    register("small_magnitude",      lambda x: np.abs(x) <= 1)
    register("is_finite",            lambda x: np.isfinite(x))
    register("is_nan",               lambda x: np.isnan(x))
    register("is_inf",               lambda x: np.isinf(x))


# Run at module load — these names are locked in as built-ins
_register_builtins()

# Snapshot of built-in names — used by is_builtin() and unregister()
_BUILTIN_NAMES: frozenset = frozenset(_registry.keys())

# ---------------------------------------------------------------------------
# Write / Read
# ---------------------------------------------------------------------------

def register(name: str, fn: Callable[[np.ndarray], np.ndarray], overwrite: bool = False) -> None:
    """
    Register a condition function under a unique name.

    Parameters
    ----------
    name      : str       — unique identifier for this condition
    fn        : callable  — takes np.ndarray, returns boolean np.ndarray
    overwrite : bool      — if False (default), raises if name already exists

    Raises
    ------
    ValueError  if name already registered and overwrite=False
    TypeError   if fn is not callable

    Example
    -------
        ConditionRegistry.register("threshold_half", lambda x: x > 0.5)
    """
    if not callable(fn):
        raise TypeError(
            f"ConditionRegistry: fn must be callable, got {type(fn).__name__}."
        )
    if name in _registry and not overwrite:
        raise ValueError(
            f"ConditionRegistry: '{name}' is already registered. "
            f"Pass overwrite=True to replace it."
        )
    _registry[name] = Condition(name=name, fn=fn)


def get(name: str) -> Condition:
    """
    Retrieve a registered Condition by name.

    Raises
    ------
    KeyError  if name is not found, with a message listing available names

    Example
    -------
        cond = ConditionRegistry.get("greater_than_zero")
        mask = cond(x.data)
    """
    if name not in _registry:
        raise KeyError(
            f"ConditionRegistry: '{name}' not found.\n"
            f"Register custom conditions with ConditionRegistry.register() "
            f"before building or loading a network that uses them."
        )
    return _registry[name]


# def make(name: str, fn: Callable[[np.ndarray], np.ndarray]) -> Condition:
#     """
#     Register a condition and return it in one step.
#     Useful when building a network inline.

#     Example
#     -------
#         cond = ConditionRegistry.make("my_gate", lambda x: x > 0.5)
#         node = ConditionNode(condition=cond, ...)
#     """
#     register(name, fn)
#     return get(name)


def from_name(name: str) -> Condition:
    """
    Alias for get(). More readable at call sites that are purely loading.

    Example
    -------
        cond = ConditionRegistry.from_name(saved_condition_name)
    """
    return get(name)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def list_all() -> List[str]:
    """Return all registered condition names, sorted."""
    return sorted(_registry.keys())


def has(name: str) -> bool:
    """Return True if a condition with this name is registered."""
    return name in _registry


def is_builtin(name: str) -> bool:
    """Return True if name is a built-in condition (not user-registered)."""
    return name in _BUILTIN_NAMES


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def unregister(name: str) -> None:
    """
    Remove a condition from the registry.
    Built-in conditions cannot be unregistered.

    Raises
    ------
    ValueError  if name is a built-in
    KeyError    if name is not registered
    """
    if name in _BUILTIN_NAMES:
        raise ValueError(
            f"ConditionRegistry: '{name}' is a built-in condition and cannot "
            f"be unregistered."
        )
    if name not in _registry:
        raise KeyError(f"ConditionRegistry: '{name}' is not registered.")
    del _registry[name]


def clear_user_conditions() -> None:
    """
    Remove all user-registered conditions, keeping built-ins intact.
    Useful for test teardown or resetting between experiments.
    """
    for name in list(_registry.keys()):
        if name not in _BUILTIN_NAMES:
            del _registry[name]


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def debug_repr() -> str:
    if not _registry:
        return "ConditionRegistry: (empty)"
    lines = ["ConditionRegistry:"]
    for name in list_all():
        tag = " [built-in]" if is_builtin(name) else " [user]"
        lines.append(f"  {name}{tag}")
    return "\n".join(lines)
