from neural_network.Nodes.Node import Node
import numpy as np
from typing import List, Optional
from ironframe.ironframe import Tensor

# ---------------------------------------------------------------------------
# Arithmetic nodes
# ---------------------------------------------------------------------------

class AddNode(Node):
    """
    Element-wise sum of N input tensors.

    Parameters
    ----------
    axis : int or None
        If set, inputs are broadcast-summed along this axis.
        For the common residual-connection case (just element-wise add),
        leave axis=None.

    Backward
    --------
    dL/dx_i = grad  for all i   (addition distributes the gradient equally)
    If a broadcast happened (shapes differ), the gradient is summed back over
    the broadcast dimension before being returned to that input.
    """

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.axis = axis

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        data = inputs[0].data.copy()
        for t in inputs[1:]:
            data = data + t.data
        requires_grad = any(t.requires_grad for t in inputs)
        return self._wrap(data, requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        grads = []
        for inp in self._inputs:
            g = grad.data
            # If shapes differ due to broadcasting, reduce back
            if g.shape != inp.data.shape:
                reduce_axes = tuple(
                    i for i, (gs, is_) in enumerate(
                        zip(g.shape[::-1], inp.data.shape[::-1])
                    ) if gs != is_
                )
                if reduce_axes:
                    g = np.sum(g, axis=reduce_axes, keepdims=True)
                g = g.reshape(inp.data.shape)
            grads.append(self._wrap(g, inp.requires_grad))
        return grads


class SubNode(Node):
    """
    t1 - t2  (exactly two inputs).

    Backward
    --------
    dL/dt1 =  grad
    dL/dt2 = -grad
    """

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 2)
        t1, t2 = inputs
        requires_grad = t1.requires_grad or t2.requires_grad
        return self._wrap(t1.data - t2.data, requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t1, t2 = self._inputs
        return [
            self._wrap( grad.data, t1.requires_grad),
            self._wrap(-grad.data, t2.requires_grad),
        ]


class MulNode(Node):
    """
    Element-wise product of N input tensors.

    Backward
    --------
    dL/dx_i = grad * product_of_all_others
    Computed efficiently: total_product / x_i  (with zero-safe fallback).
    """

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        data = inputs[0].data.copy()
        for t in inputs[1:]:
            data = data * t.data
        requires_grad = any(t.requires_grad for t in inputs)
        return self._wrap(data, requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        grads = []
        n = len(self._inputs)
        for i, inp in enumerate(self._inputs):
            # product of all inputs except i
            others = np.ones_like(inp.data)
            for j, other in enumerate(self._inputs):
                if j != i:
                    others = others * other.data
            grads.append(self._wrap(grad.data * others, inp.requires_grad))
        return grads


class DivNode(Node):
    """
    t1 / t2  (exactly two inputs).

    Backward
    --------
    dL/dt1 =  grad / t2
    dL/dt2 = -grad * t1 / t2^2
    """

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 2)
        t1, t2 = inputs
        requires_grad = t1.requires_grad or t2.requires_grad
        return self._wrap(t1.data / t2.data, requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t1, t2 = self._inputs
        return [
            self._wrap( grad.data / t2.data,                    t1.requires_grad),
            self._wrap(-grad.data * t1.data / (t2.data ** 2),   t2.requires_grad),
        ]


class SqNode(Node):
    """
    Element-wise square:  out = t ** 2  (single input).

    Backward
    --------
    dL/dt = grad * 2 * t
    """

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t = inputs[0]
        return self._wrap(t.data ** 2, t.requires_grad)

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t = self._inputs[0]
        return [self._wrap(grad.data * 2.0 * t.data, t.requires_grad)]


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
    condition : str — Condition(s)
    
    Backward
    ________
    """
    
    def __init__(self, condition: str):
        super().__init__()

    def _forward(self, *inputs: Tensor) -> Tensor:
        ...

    def _backward(self, grad: Tensor) -> List[Tensor]:
        ...
    
