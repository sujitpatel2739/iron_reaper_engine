"""
InputLayer.py
-------------
Entry point of a neural network graph.

Each port label identifies one named input tensor. Labels are editable
in the frontend. The InputLayer stores each input in CacheStore keyed
by label so downstream layers can read them by name.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from neural_network.Layers.Layer import Layer
from ironframe.ironframe import Tensor
import cache.CacheStore as CacheStore
from cache.CacheStore import SLOT_OUTPUT


class InputLayer(Layer):
    """
    Entry point of a neural network graph.

    Parameters
    ----------
    layer_id : int | str
        Unique ID. Must not clash with other layer/node IDs.

    inputs : List[str] | None
        Labels for each input port.
        Defaults to ['input_0'] for the single-input case.
        Labels are editable in the frontend.

    name : str
        Display name. Default 'inputlayer'. Editable in the frontend.
    """

    def __init__(
        self,
        layer_id: Any,
        inputs:   Optional[List[str]] = None,
        name:     str = 'inputlayer',
    ):
        # InputLayer does not call Layer.__init__ because it does not use
        # the detach machinery or _state dict in the same way.
        self.id         = layer_id
        self.name       = name
        self.type       = 'inputlayer'
        self._state     = {}
        self.parameters = {}

        # Port labels — at least one required
        self.inputs = inputs if inputs is not None else ['input_0']

        if len(self.inputs) == 0:
            raise ValueError(
                f"InputLayer '{self.name}': must have at least one input label."
            )

    # -- Forward / backward --------------------------------------------------

    def _forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Store each named input in CacheStore and return unchanged.
        Missing port keys are silently ignored.
        """
        for label, tensor in inputs.items():
            CacheStore.write(self.id, f"{SLOT_OUTPUT}_{label}", tensor)
            self._state[label] = tensor
        return inputs

    def _backward(self, grad: Any) -> None:
        return None

    # -- Port management -----------------------------------------------------

    def add_input(self, label: str) -> None:
        if label in self.inputs:
            raise ValueError(
                f"InputLayer '{self.name}': label '{label}' already exists."
            )
        self.inputs.append(label)

    def remove_input(self, label: str) -> None:
        if label not in self.inputs:
            raise ValueError(
                f"InputLayer '{self.name}': label '{label}' not found."
            )
        if len(self.inputs) == 1:
            raise ValueError(
                f"InputLayer '{self.name}': cannot remove the last input port."
            )
        self.inputs.remove(label)

    def rename_input(self, old_label: str, new_label: str) -> None:
        if old_label not in self.inputs:
            raise ValueError(f"InputLayer '{self.name}': '{old_label}' not found.")
        if new_label in self.inputs:
            raise ValueError(f"InputLayer '{self.name}': '{new_label}' already exists.")
        self.inputs[self.inputs.index(old_label)] = new_label

    def __repr__(self) -> str:
        return f"InputLayer(id={self.id}, name='{self.name}', inputs={self.inputs})"