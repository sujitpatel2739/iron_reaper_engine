"""
Nodes.py
--------
Concrete Node implementations.

All arithmetic ops delegate to ironframe functions (add, sub, mul, div, sqrt)
so broadcasting and autograd are handled consistently in one place.

_state stores whatever the backward pass needs — inputs, masks, split indices.
Every node accepts node_id: int and name: str and passes them to Node.__init__.
"""

import numpy as np
from typing import Callable, List, Optional, Any

from ironframe.ironframe import Tensor, add, sub, mul, div, sqrt, _revbroadcast
from cache.CacheStore import SLOT_INPUT, SLOT_OUTPUT, SLOT_GRAD_OUT, SLOT_GRAD_IN
from ConditionRegistry import Condition
from Node import Node


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

    def __init__(self, node_id: int, name: str = "", axis: Optional[int] = None):
        super().__init__(node_id, name)
        self.axis = axis

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        out = inputs[0]
        for t in inputs[1:]:
            out = out + t
        # Save inputs for backward
        self._state['inputs'] = list(inputs)
        self._state['out'] = out
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        return out.backward(grad)

class SubNode(Node):
    """
    t1 - t2  (exactly two inputs).

    Backward
    --------
    dL/dt1 =  grad
    dL/dt2 = -grad
    Both are _revbroadcast'd back to their original shapes.
    """

    def __init__(self, node_id: int, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 2)
        t1, t2 = inputs
        out = t1 - t2
        self._state['inputs'] = [t1, t2]
        self._state['out']    = out
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        return out.backward(grad)

class MulNode(Node):
    """
    Element-wise product of N input tensors.

    Backward
    --------
    dL/dx_i = grad * product_of_all_others
    _revbroadcast collapses the result back to each input's original shape.
    """

    def __init__(self, node_id: int, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        out = inputs[0]
        for t in inputs[1:]:
            out = out * t
        self._state['inputs'] = list(inputs)
        self._state['out']    = out
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        return out.backward(grad)

class DivNode(Node):
    """
    t1 / t2  (exactly two inputs).

    Backward
    --------
    dL/dt1 =  grad / t2
    dL/dt2 = -grad * t1 / t2^2
    """

    def __init__(self, node_id: int, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 2)
        t1, t2 = inputs
        out = t1 / t2
        self._state['inputs'] = [t1, t2]
        self._state['out']    = out
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        return out.backward(grad)

class SqNode(Node):
    """
    Element-wise square:  out = t ** 2  (single input).

    Backward
    --------
    dL/dt = grad * 2 * t
    """

    def __init__(self, node_id: int, name: str = ""):
        super().__init__(node_id, name)

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t   = inputs[0]
        out = t ** 2
        self._state['input'] = t
        self._state['out']   = out
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        out = self._state['out']
        return out.backward(grad)

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

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t = inputs[0]
        return self._wrap(-t.data, t.requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t = self._inputs[0]
        return [self._wrap(-grad.data, t.requires_grad)]


class SqrtNode(Node):
    """
    Element-wise square root:  out = sqrt(t)  (single input).

    Backward
    --------
    dL/dt = grad / (2 * sqrt(t))
    """

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t = inputs[0]
        return self._wrap(np.sqrt(t.data), t.requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t = self._inputs[0]
        return [self._wrap(grad.data / (2.0 * np.sqrt(t.data)), t.requires_grad)]


class ScaleNode(Node):
    """
    Multiply by a fixed scalar constant:  out = scalar * t  (single input).
    The scalar is not a learnable parameter — use this for things like
    the alpha scaling in a ResBlock.

    Backward
    --------
    dL/dt = grad * scalar
    """

    def __init__(self, scalar: float):
        super().__init__()
        self.scalar = scalar

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t = inputs[0]
        return self._wrap(t.data * self.scalar, t.requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t = self._inputs[0]
        return [self._wrap(grad.data * self.scalar, t.requires_grad)]


class ClipNode(Node):
    """
    Clamp values into [min_val, max_val]:  out = clip(t, min_val, max_val).
    Useful for gradient clipping or activation bounding.

    Backward
    --------
    Gradient passes through where input was inside [min_val, max_val],
    zero elsewhere (straight-through for the active region).
    """

    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t = inputs[0]
        return self._wrap(np.clip(t.data, self.min_val, self.max_val), t.requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t = self._inputs[0]
        mask = (t.data >= self.min_val) & (t.data <= self.max_val)
        return [self._wrap(grad.data * mask.astype(np.float32), t.requires_grad)]


# ---------------------------------------------------------------------------
# Shape nodes
# ---------------------------------------------------------------------------

class ConcatNode(Node):
    """
    Concatenate N input tensors along a given axis.

    Parameters
    ----------
    axis : int  (default 1, i.e. feature axis)

    Backward
    --------
    Split the gradient along the same axis using the input sizes,
    returning one gradient slice per input.
    """

    def __init__(self, axis: int = 1):
        super().__init__()
        self.axis = axis
        self._split_indices: List[int] = []

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        # Record where to split on backward
        sizes = [t.data.shape[self.axis] for t in inputs]
        self._split_indices = np.cumsum(sizes[:-1]).tolist()
        requires_grad = any(t.requires_grad for t in inputs)
        return self._wrap(
            np.concatenate([t.data for t in inputs], axis=self.axis),
            requires_grad
        )

    def _backward(self, grad: Tensor) -> List[Tensor]:
        slices = np.split(grad.data, self._split_indices, axis=self.axis)
        return [
            self._wrap(s, inp.requires_grad)
            for s, inp in zip(slices, self._inputs)
        ]


class SplitNode(Node):
    """
    Split a single input tensor into N chunks along a given axis.

    Unlike other nodes whose forward() returns one Tensor, SplitNode
    returns a list of Tensors. Its backward() accepts a list of grads
    (one per output chunk) and returns a list with a single gradient
    for the original input.

    Parameters
    ----------
    n_splits : int   — number of equal-sized chunks
    axis     : int   — axis to split along (default 1)

    Backward
    --------
    Concatenate the incoming gradient chunks along the split axis.
    """

    def __init__(self, n_splits: int, axis: int = 1):
        super().__init__()
        self.n_splits = n_splits
        self.axis = axis

    def forward(self, *inputs: Tensor):            # override: returns List[Tensor]
        self._require_n_inputs(inputs, 1)
        self._inputs = list(inputs)
        chunks = np.split(inputs[0].data, self.n_splits, axis=self.axis)
        requires_grad = inputs[0].requires_grad
        return [self._wrap(c, requires_grad) for c in chunks]

    def backward(self, grads: List[Tensor]) -> List[Tensor]:  # accepts list
        g = np.concatenate([g.data for g in grads], axis=self.axis)
        return [self._wrap(g, self._inputs[0].requires_grad)]

    def _forward(self, *inputs):   # satisfies ABC but not used directly
        pass

    def _backward(self, grad):     # satisfies ABC but not used directly
        pass
    
class ConditionNode(Node):
    """
    Executes paths based on condition.
    
    Parameters
    __________
    condition : Callable
    
    Backward
    ________
    """
    
    def __init__(self, node_id:int, condition: Callable[[Any], Any],
                path_true: Any|None, path_false: Any|None, name: str):
        super().__init__(node_id=node_id)
        self._require_n_inputs((path_true, path_false), 1)
        self.condition = Condition(name, condition)
        # Example condition: x > 0.5 and x.shape == (1, 2) ...
        self.path_true = path_true
        self.path_false = path_false

    def _forward(self, operand: Any, input:Any|None) -> Optional[Any]|None:
        mask = self.condition(operand)
        self._state['mask'] = mask
            
        if isinstance(mask, Tensor|np.ndarray):
            path_true = self.path_true if self.path_true else lambda x:np.empty_like(input)
            path_false = self.path_false if self.path_false else lambda x:np.empty_like(input)
            result = np.where(mask.data, path_true(input), path_false(input))
        else:
            path_true = self.path_true if self.path_true else lambda x:None
            path_false = self.path_false if self.path_false else lambda x:None
            if mask == True:
                result = path_true(input)
            else:
                result = path_false(input)

        return result

    def _backward(self, grad: Tensor) -> List[Tensor]:
        ...
    
