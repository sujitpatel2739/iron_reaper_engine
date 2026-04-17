import numpy as np
from typing import Any
from backend.ironframe.ironframe import Tensor
from backend.neural_network.Layers.Layer import  Layer

# ---------------------------------------------------------------------------
# Activation layers
# ---------------------------------------------------------------------------

class Relu(Layer):
    def __init__(self, layer_id: Any, name: str = ""):
        super().__init__(layer_id, name)

    def _forward(self, x: Tensor) -> Tensor:
        self._state['mask'] = x.data > 0
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        grad_X = Tensor(grad.data * self._state['mask'], requires_grad=grad.requires_grad)
        return grad_X


class Tanh(Layer):
    def __init__(self, layer_id: Any, name: str = ""):
        super().__init__(layer_id, name)

    def _forward(self, x: Tensor) -> Tensor:
        raw = (np.exp(x.data) - np.exp(-x.data)) / (np.exp(x.data) + np.exp(-x.data))
        out = Tensor(raw, requires_grad=x.requires_grad)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        out    = self._state['out']
        grad_X = grad * (1 - (out.data ** 2))
        return grad_X


class Sigmoid(Layer):
    def __init__(self, layer_id: Any, name: str = ""):
        super().__init__(layer_id, name)

    def _forward(self, x: Tensor) -> Tensor:
        out = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        out    = self._state['out']
        grad_X = grad * out * (1 - out)
        return grad_X


class LeakyRelu(Layer):
    def __init__(self, layer_id: Any, alpha: float = 0.01, name: str = ""):
        super().__init__(layer_id, name)
        self.alpha = alpha

    def _forward(self, x: Tensor) -> Tensor:
        self._state['mask'] = x.data > 0
        out = Tensor(
            np.where(self._state['mask'], x.data, x.data * self.alpha),
            requires_grad=x.requires_grad,
        )
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        grad_X = Tensor(
            np.where(self._state['mask'], grad.data, grad.data * self.alpha),
            requires_grad=grad.requires_grad,
        )
        return grad_X


class Elu(Layer):
    def __init__(self, layer_id: Any, alpha: float = 0.01, name: str = ""):
        super().__init__(layer_id, name)
        self.alpha = alpha

    def _forward(self, x: Tensor) -> Tensor:
        self._state['input'] = x.data
        self._state['mask']  = x.data > 0
        out = Tensor(
            np.where(self._state['mask'], x.data, self.alpha * (np.exp(x.data) - 1)),
            requires_grad=x.requires_grad,
        )
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        grad_X = Tensor(
            np.where(
                self._state['mask'],
                grad.data,
                grad.data * self.alpha * np.exp(self._state['input']),
            ),
            requires_grad=grad.requires_grad,
        )
        return grad_X