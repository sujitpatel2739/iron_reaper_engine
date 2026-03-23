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

from ironframe.ironframe import Tensor, sqrt, split
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
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
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
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
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
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
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
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Any:
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

    def __init__(self, node_id: int, name: str = ""):
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
        return out.backward(grad)


class SqrtNode(Node):
    """
    Element-wise square root:  out = sqrt(t)  (single input).

    Backward
    --------
    dL/dt = grad / (2 * sqrt(t))
    Delegates to ironframe sqrt for consistency.
    """

    def __init__(self, node_id: int, name: str = ""):
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
        return out.backward(grad)

# class ScaleNode(Node):
#     """
#     Multiply by a fixed scalar constant:  out = scalar * t  (single input).
#     The scalar is not a learnable parameter.

#     Backward
#     --------
#     dL/dt = grad * scalar
#     """

#     def __init__(self, node_id: int, scalar: float, name: str = ""):
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


class ClipNode(Node):
    """
    Clamp values into [min_val, max_val].

    Backward
    --------
    Gradient passes through where input was in range, zero elsewhere.
    """

    def __init__(self, node_id: int, min_val: float, max_val: float, name: str = ""):
        super().__init__(node_id, name)
        self.min_val = min_val
        self.max_val = max_val

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_n_inputs(inputs, 1)
        t    = inputs[0]
        out  = self._wrap(np.clip(t.data, self.min_val, self.max_val), t.requires_grad)
        mask = (t.data >= self.min_val) & (t.data <= self.max_val)
        self._state['input'] = t
        self._state['mask']  = mask      # save mask — backward needs it
        self._state['out']   = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        t    = self._state['input']
        mask = self._state['mask']
        return [self._wrap(grad.data * mask.astype(np.float32), t.requires_grad)]


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

    def __init__(self, node_id: int, axis: int = 1, name: str = ""):
        super().__init__(node_id, name)
        self.axis = axis

    def _forward(self, *inputs: Tensor) -> Tensor:
        self._require_at_least(inputs, 2)
        sizes         = [t.shape[self.axis] for t in inputs]
        split_indices = np.cumsum(sizes[:-1]).tolist()
        requires_grad = any(t.requires_grad for t in inputs)
        out           = self._wrap(
            np.concatenate([t.data for t in inputs], axis=self.axis),
            requires_grad
        )
        self._state['inputs']        = list(inputs)
        self._state['split_indices'] = split_indices
        self._state['out']           = out
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> List[Tensor]:
        inputs        = self._state['inputs']
        split_indices = self._state['split_indices']
        slices        = np.split(grad.data, split_indices, axis=self.axis)
        return [
            self._wrap(s, inp.requires_grad)
            for s, inp in zip(slices, inputs)
        ]


class SplitNode(Node):
    """
    Split a single input tensor into N chunks along a given axis.

    Returns a list of Tensors (unlike other nodes that return one).
    backward() accepts a list of grads, one per chunk.

    Backward
    --------
    Concatenate incoming gradient chunks along the split axis.
    """

    def __init__(self, node_id: int, n_splits: int, axis: int = 0, name: str = ""):
        super().__init__(node_id, name)
        self.n_splits = n_splits
        self.axis     = axis

    def _forward(self, *inputs: Tensor) -> List[Tensor]:
        self._require_n_inputs(inputs, 1)
        self._inputs  = list(inputs)
        t             = inputs[0]
        chunks        = split(t, self.n_splits, axis=self.axis)
        requires_grad = t.requires_grad
        out_chunks    = [self._wrap(c, requires_grad) for c in chunks]
        self._state['input']  = t
        self._state['chunks'] = out_chunks
        # self._write(SLOT_OUTPUT, out_chunks)
        return out_chunks

    def _backward(self, grads: List[Tensor]) -> List[Tensor]:
        t = self._state['input']
        out = self._state['out']
        return out.backward(grads)

# --------------------------------------------------------------------------
# Condition node
# ---------------------------------------------------------------------------

class ConditionNode(Node):
    """
    Routes input through two paths based on an elementwise or scalar condition.

    Parameters
    ----------
    node_id    : int
    condition  : Condition  — a Condition from ConditionRegistry wrapping name + fn.
                             fn receives `operand` and returns a boolean mask or bool.
    path_true  : callable | None  — called with `input` when condition is True.
                                    None = identity (input passes through unchanged).
    path_false : callable | None  — called with `input` when condition is False.
                                    None = identity (input passes through unchanged).
    name       : str

    Forward
    -------
    operand and input are kept separate intentionally:
        operand — what the condition is evaluated on (can be anything)
        input   — what gets routed through the paths (optional, can be None
                  if the paths don't require tensor input)

    Two cases:
        Elementwise mask (Tensor or ndarray) — both paths run, outputs stitched
                                               with np.where using the mask.
        Scalar bool                          — only the matching path runs,
                                               the other is never called.

    Backward
    --------
    Elementwise: gradient is gated by the mask — each path receives only the
                 gradient for the positions it was responsible for.
    Scalar:      gradient flows only through the path that ran.
                 If the path has no .backward(), gradient is returned as-is.
    """

    def __init__(
        self,
        node_id:    int,
        condition:  Condition,
        path_true:  Optional[Any] = None,
        path_false: Optional[Any] = None,
        name:       str = "",
    ):
        super().__init__(node_id, name)

        if not isinstance(condition, Condition):
            raise TypeError(
                f"ConditionNode: condition must be a Condition instance from "
                f"ConditionRegistry, got {type(condition).__name__}. "
                f"Use ConditionRegistry.get('name') or ConditionRegistry.make('name', fn)."
            )

        self.condition  = condition
        self.path_true  = path_true
        self.path_false = path_false

    # -- helpers -------------------------------------------------------------

    def _run_path(self, path: Optional[Any], input: Any) -> Any:
        """Call a path if it exists, otherwise return input unchanged."""
        if path is None:
            return input
        return path(input)

    def _is_elementwise(self, mask: Any) -> bool:
        """True if mask is an array-like (elementwise), False if scalar bool."""
        return isinstance(mask, (np.ndarray, Tensor))

    def _mask_data(self, mask: Any) -> np.ndarray:
        """Always return a plain numpy boolean array from the mask."""
        if isinstance(mask, Tensor):
            return mask.data.astype(bool)
        if isinstance(mask, np.ndarray):
            return mask.astype(bool)
        # scalar bool — wrap into a 0-d array for uniform handling
        return np.array(mask, dtype=bool)

    # -- forward -------------------------------------------------------------

    def _forward(self, operand: Any, input: Any = None) -> Any:
        """
        Parameters
        ----------
        operand : Any   — value the condition is evaluated on
        input   : Any   — value passed to the paths (can be None)
        """
        mask      = self.condition(operand)
        mask_data = self._mask_data(mask)

        self._state['mask']    = mask_data
        self._state['input']   = input
        self._state['operand'] = operand

        if self._is_elementwise(mask):
            # Both paths run — outputs stitched elementwise
            out_true  = self._run_path(self.path_true,  input)
            out_false = self._run_path(self.path_false, input)

            self._state['out_true']  = out_true
            self._state['out_false'] = out_false

            # Extract raw data for np.where — handle Tensor or plain array
            data_true  = out_true.data  if isinstance(out_true,  Tensor) else np.asarray(out_true)
            data_false = out_false.data if isinstance(out_false, Tensor) else np.asarray(out_false)

            result_data = np.where(mask_data, data_true, data_false)
            requires_grad = (
                (isinstance(out_true,  Tensor) and out_true.requires_grad)  or
                (isinstance(out_false, Tensor) and out_false.requires_grad)
            )
            result = self._wrap(result_data, requires_grad)

        else:
            # Scalar condition — only one path runs
            if mask_data.item():
                result = self._run_path(self.path_true,  input)
                self._state['bool'] = 'true'
            else:
                result = self._run_path(self.path_false, input)
                self._state['bool'] = 'false'

        self._state['out'] = result
        self._write(SLOT_OUTPUT, result)
        return result

    # -- backward ------------------------------------------------------------

    def _backward(self, grad: Tensor) -> List[Tensor]:
        mask_data = self._state['mask']

        if self._is_elementwise(mask_data) or mask_data.ndim > 0:
            # Elementwise case — gate gradient by mask, route to each path
            grad_true_data  = np.where(mask_data,  grad.data, 0.0)
            grad_false_data = np.where(~mask_data, grad.data, 0.0)

            grad_true  = self._wrap(grad_true_data,  grad.requires_grad)
            grad_false = self._wrap(grad_false_data, grad.requires_grad)

            # Propagate through each path if it supports backward
            if self.path_true is not None and hasattr(self.path_true, 'backward'):
                self.path_true.backward(grad_true)

            if self.path_false is not None and hasattr(self.path_false, 'backward'):
                self.path_false.backward(grad_false)

            # Return gated grads — one for operand (no grad), one for input
            return [self._wrap(np.zeros_like(mask_data, dtype=np.float32), False), grad]

        else:
            # Scalar case — gradient flows only through the path that ran
            taken = self._state.get('taken', 'true')
            path  = self.path_true if taken == 'true' else self.path_false

            if path is not None and hasattr(path, 'backward'):
                path.backward(grad)

            return [self._wrap(np.array(0.0), False), grad]
