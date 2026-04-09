import numpy as np
from typing import Callable, List, Optional, Any

from ironframe.ironframe import Tensor
from cache.CacheStore import SLOT_INPUT, SLOT_OUTPUT, SLOT_GRAD_OUT, SLOT_GRAD_IN
from registries.ConditionRegistry import Condition
from .Node import Node

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