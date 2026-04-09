"""
Nodes.py
--------
Concrete Node implementations.

All arithmetic ops delegate to ironframe functions (add, sub, mul, div, sqrt)
so broadcasting and autograd are handled consistently in one place.

_state stores whatever the backward pass needs — inputs, masks, split indices.
Every node accepts node_id: Any and name: str and passes them to Node.__init__.
"""

import numpy as np
from typing import Callable, List, Optional, Any

from ironframe.ironframe import Tensor, sqrt, split, rangeclip, concate
from cache.CacheStore import SLOT_INPUT, SLOT_OUTPUT, SLOT_GRAD_OUT, SLOT_GRAD_IN
from registries.ConditionRegistry import Condition
from .Node import Node

# ---------------------------------------------------------------------------
# Arithmetic nodes
# ---------------------------------------------------------------------------

class AddNode(Node):
    """
    Element-wise sum of N input tensors.

    Backward
    --------
    dL/dx_i = grad for all i.
    If a broadcast happened (shapes differ), _revbroadcast collapses the
    gradient back to the original shape before returning it to that input.
    """

    def __init__(self, node_id: Any, name: str = "", axis: Optional[int] = None):
        super().__init__(node_id, name)
        self.axis = axis

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        # Delegate to ironframe add — handles broadcasting and autograd graph
        out = inputs[0]
        for t in inputs[1:]:
            out = out + t
        # Save inputs for backward
        self._state['inputs'] = list(inputs)
        self._state['out'] = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in
        

class SubNode(Node):
    """
    t1 - t2  (exactly two inputs).

    Backward
    --------
    dL/dt1 =  grad
    dL/dt2 = -grad
    Both are _revbroadcast'd back to their original shapes.
    """

    def __init__(self, node_id: Any, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 2)
        t1, t2 = inputs
        out = t1 - t2
        self._state['inputs'] = [t1, t2]
        self._state['out']    = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in

class MulNode(Node):
    """
    Element-wise product of N input tensors.

    Backward
    --------
    dL/dx_i = grad * product_of_all_others
    _revbroadcast collapses the result back to each input's original shape.
    """

    def __init__(self, node_id: Any, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        out = inputs[0]
        for t in inputs[1:]:
            out = out * t
        self._state['inputs'] = list(inputs)
        self._state['out']    = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in

class DivNode(Node):
    """
    t1 / t2  (exactly two inputs).

    Backward
    --------
    dL/dt1 =  grad / t2
    dL/dt2 = -grad * t1 / t2^2
    """

    def __init__(self, node_id: Any, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 2)
        t1, t2 = inputs
        out = t1 / t2
        self._state['inputs'] = [t1, t2]
        self._state['out']    = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in

class SqNode(Node):
    """
    Element-wise square:  out = t ** 2  (single input).

    Backward
    --------
    dL/dt = grad * 2 * t
    """

    def __init__(self, node_id: Any, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t   = inputs[0]
        out = t ** 2
        self._state['input'] = t
        self._state['out']   = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in


# ---------------------------------------------------------------------------
# Unary utility nodes
# ---------------------------------------------------------------------------

class NegNode(Node):
    """
    Negate:  out = -t  (single input).

    Backward
    --------
    dL/dt = -grad
    """

    def __init__(self, node_id: Any, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t   = inputs[0]
        out = -t
        self._state['input'] = t
        self._state['out']   = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in


class SqrtNode(Node):
    """
    Element-wise square root:  out = sqrt(t)  (single input).

    Backward
    --------
    dL/dt = grad / (2 * sqrt(t))
    Delegates to ironframe sqrt for consistency.
    """

    def __init__(self, node_id: Any, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t   = inputs[0]
        out = sqrt(t)
        self._state['input'] = t
        self._state['out']   = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in

# class ScaleNode(Node):
#     """
#     Multiply by a fixed scalar constant:  out = scalar * t  (single input).
#     The scalar is not a learnable parameter.

#     Backward
#     --------
#     dL/dt = grad * scalar
#     """

#     def __init__(self, node_id: Any, scalar: float, name: str = ""):
#         super().__init__(node_id, name)
#         self.scalar = scalar

#     def _forward(self, *inputs: Tensor) -> Tensor:
#         self._require_n_inputs(inputs, 1)
#         t   = inputs[0]
#         out = t * self.scalar
#         self._state['input'] = t
#         self._state['out']   = out
          # self._write(SLOT_OUTPUT, out)
#         return out

#     def _backward(self, grad: Tensor) -> List[Tensor]:
#         t = self._state['input']
#         return [self._wrap(grad.data * self.scalar, t.requires_grad)]


class RangeclipNode(Node):
    """
    Clamp values into [min_val, max_val].

    Backward
    --------
    Gradient passes through where input was in range, zero elsewhere.
    """

    def __init__(self, node_id: Any, min_val: float, max_val: float, name: str = ""):
        super().__init__(node_id, name)
        self.min_val = min_val
        self.max_val = max_val

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        out, mask = rangeclip(inputs[0], self.min_val, self.max_val)
        self._state['mask']  = mask
        self._state['out']   = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in

# ---------------------------------------------------------------------------
# Shape nodes
# ---------------------------------------------------------------------------

class ConcatNode(Node):
    """
    Concatenate N input tensors along a given axis.

    Backward
    --------
    Split the gradient along the same axis using the input sizes.
    """

    def __init__(self, node_id: Any, axis: int = 1, name: str = ""):
        super().__init__(node_id, name)
        self.axis = axis

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        out, split_indices = concate(inputs, self.axis)
        self._state['inputs'] = list(inputs)
        self._state['split_indices'] = split_indices
        self._state['out'] = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        grad_in = out.backward(grad)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in # List of grad_in(s)

class SplitNode(Node):
    """
    Split a single input tensor into N chunks along a given axis.

    Returns a list of Tensors (unlike other nodes that return one).
    backward() accepts a list of grads, one per chunk.

    Backward
    --------
    Concatenate incoming gradient chunks along the split axis.
    """

    def __init__(self, node_id: Any, n_splits: int, axis: int = 0, name: str = ""):
        super().__init__(node_id, name)
        self.n_splits = n_splits
        self.axis     = axis

    def _forward(self, *inputs: Tensor) -> List[Tensor]:
        self._require_n_inputs(inputs, 1)
        out = split(self.n_splits, self.axis)
        self._state['out'] = out
        self._write(SLOT_OUTPUT, out)
        return out # List of splited Tensors

    def _backward(self, grads: List[Tensor]) -> List[Tensor]:
        out = self._state['out']
        grad_in = out.backward(grads)
        self._write(SLOT_GRAD_IN, grad_in)
        return grad_in # One grad_in Tensor

        