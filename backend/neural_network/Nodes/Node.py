"""
Node.py
-------
Operation nodes for graph junction points in a network.

Nodes are fundamentally different from Layers:

  Layer  — single input → single output, has an integer id, participates in
            the observer pipeline, may have learnable parameters.

  Node   — N inputs → single output, no id, no parameters, not observable.
            A Node is purely a wiring/computation primitive used inside
            composite blocks (e.g. FlexResBlock) to express fork/merge topology.

All nodes implement:
    forward(*inputs)   → Tensor
    backward(grad)     → List[Tensor]   (one grad per input, in the same order)

The backward contract mirrors ironframe: grads are plain numpy-backed Tensors,
not autograd graph nodes — the Node owns its own backward math explicitly.

Available nodes
---------------
  AddNode     — element-wise sum of N inputs (with optional axis for broadcasting)
  SubNode     — t1 - t2
  MulNode     — element-wise product of N inputs
  DivNode     — t1 / t2
  SqNode      — t ** 2  (element-wise square, single input)
  NegNode     — -t      (negate, single input)
  SqrtNode    — sqrt(t) (single input)
  ConcatNode  — concatenate N inputs along a given axis
  SplitNode   — split one input into N chunks along a given axis
  ScaleNode   — multiply a single input by a scalar constant
  ClipNode    — clamp values into [min_val, max_val]
"""

import numpy as np
from typing import List, Optional, Any
from ironframe.ironframe import Tensor
from cache import CacheStore

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Node:
    """
    Abstract base for all operation nodes.

    Subclasses must implement:
        _forward(self, *inputs: Tensor) -> Tensor
        _backward(self, grad: Tensor)   -> List[Tensor]

    The base class handles:
        - Input validation helpers
        - Caching inputs for backward
        - A repr for debugging
    """

    def __init__(self, node_id: Any, name: str = ""):
        self.id = node_id
        self._inputs: Any   # cached during forward for backward use
        self._output: Any
        self._state = {}   # private working memory for this Node's backward

    def _forward(self, *inputs: Tensor) -> Optional[Any]|None:
        raise NotImplementedError

    def _backward(self, grad) -> Optional[Any]|None:
        raise NotImplementedError

    # -- Public interface ----------------------------------------------------

    def __call__(self, *args) -> Optional[Any]|None:
        self._inputs = list(*args)
        out = self._forward(*args)
        self._output = out
        return out

    def backward(self, grad) -> Optional[Any]|None:
        """
        Returns a list of gradient Tensors, one per input passed to forward(),
        in the same order.
        """
        return self._backward(grad)

    def _require_n_inputs(self, inputs, n: int):
        if len(inputs) != n or None in inputs:
            raise ValueError(
                f"{type(self).__name__} expects exactly {n} input(s), "
                f"got {len(inputs)}."
            )

    def _require_at_least(self, inputs, n: int):
        if len(inputs) < n or inputs.count(None) == len(inputs):
            raise ValueError(
                f"{type(self).__name__} expects at least {n} input(s), "
                f"got {len(inputs)}."
            )

    def _wrap(self, data: np.ndarray, requires_grad: bool = False) -> Tensor:
        return Tensor(data, requires_grad=requires_grad)

    def __repr__(self):
        n_in = len(self._inputs)
        return f"{type(self).__name__}(cached_inputs={n_in})"
    
    # -- CacheStore helpers --------------------------------------------------

    def _write(self, slot: str, tensor) -> None:
        CacheStore.write(self.id, slot, tensor)

    def _read(self, slot: str) -> Tensor:
        return CacheStore.read_required(self.id, slot)
