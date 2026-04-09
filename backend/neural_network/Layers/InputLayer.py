"""
InputLayer.py
---------------
Each entry in `inputs` is a label string identifying one input tensor.
Labels default to positional names ('input_0', 'input_1', ...) if not
provided. They are editable in the frontend.

Each input can feed any number of downstream edges — the InputLayer does
not control fan-out. The graph topology (edges) determines that.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from Layer import Layer
from ironframe.ironframe import Tensor
import cache.CacheStore as CacheStore
from cache.CacheStore import SLOT_INPUT, SLOT_OUTPUT, SLOT_GRAD_OUT, SLOT_GRAD_IN


# ---------------------------------------------------------------------------
# InputLayer
# ---------------------------------------------------------------------------

class InputLayer(Layer):
    """
    Entry point of a neural network graph.

    Parameters
    ----------
    layer_id : int
        Unique integer ID. Assigned by the user, must not clash with
        other layer/node IDs in the same network.

    inputs : List[str] | None
        Labels for each input port. If None, defaults to
        ['input_0', 'input_1', ...] — but since the default case is
        a single input, the default list is just ['input_0'].
        Labels are editable in the frontend.

    name : str
        Display name. Default 'inputlayer'. Editable in the frontend.
    """

    def __init__(
        self,
        layer_id: Any,
        inputs: list[str],
        name:     str = 'inputlayer',
    ):
        self.id         = layer_id
        self.name       = name
        self.type       = 'inputlayer'
        self._state = {}
        self.parameters = {}

        # Input labels — one per input port
        # Default to ['input_0'] for the single-input case
        self.inputs = inputs if inputs is not None else ['input_0']

        if len(self.inputs) == 0:
            raise ValueError(
                f"InputLayer '{self.name}': must have at least one input label. "
                f"Pass inputs=['name'] or leave as default."
            )
        
        

    def _forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Store inputs in _state and CacheStore, return unchanged.
        Missing keys are silently ignored — no validation by design.
        """

        for label, tensor in inputs.items():
            CacheStore.write(self.id, f"{SLOT_OUTPUT}_{label}", tensor)
            self._state[label] = tensor
        return inputs

    def _backward(self, grad: Any) -> None:
        """
        Gradients arrive here and stop.
        No parameters to update. No upstream to propagate to.
        """
        return None

    # -- Dynamic input management --------------------------------------------

    def add_input(self, label: str) -> None:
        """
        Add a new input port with the given label.
        Mirrors the frontend + button.

        Raises
        ------
        ValueError if label already exists — labels must be unique.
        """
        if label in self.inputs:
            raise ValueError(
                f"InputLayer '{self.name}': label '{label}' already exists. "
                f"Labels must be unique. Current labels: {self.inputs}"
            )
        self.inputs.append(label)

    def remove_input(self, label: str) -> None:
        """
        Remove an input port by label.
        Mirrors the frontend - button.

        Raises
        ------
        ValueError if label not found.
        ValueError if removing would leave zero inputs.
        """
        if label not in self.inputs:
            raise ValueError(
                f"InputLayer '{self.name}': label '{label}' not found. "
                f"Current labels: {self.inputs}"
            )
        if len(self.inputs) == 1:
            raise ValueError(
                f"InputLayer '{self.name}': cannot remove the last input. "
                f"InputLayer must have at least one input port."
            )
        self.inputs.remove(label)

    def rename_input(self, old_label: str, new_label: str) -> None:
        """
        Rename an input port label.
        Mirrors frontend label editing.

        Raises
        ------
        ValueError if old_label not found.
        ValueError if new_label already exists.
        """
        if old_label not in self.inputs:
            raise ValueError(
                f"InputLayer '{self.name}': label '{old_label}' not found. "
                f"Current labels: {self.inputs}"
            )
        if new_label in self.inputs:
            raise ValueError(
                f"InputLayer '{self.name}': label '{new_label}' already exists. "
                f"Labels must be unique."
            )
        idx = self.inputs.index(old_label)
        self.inputs[idx] = new_label

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InputLayer("
            f"id={self.id}, "
            f"name='{self.name}', "
            f"inputs={self.inputs})"
        )