"""
Layer.py
--------
"""

import numpy as np
from types import MappingProxyType
from typing import Any

import cache.CacheStore as CacheStore
from cache.CacheStore import SLOT_OUTPUT, SLOT_GRAD_IN
from ironframe.ironframe import Tensor, add, mul, matmul, mean, sqrt

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Layer:
    def __init__(self, layer_id: Any, name: str = ""):
        self.id         = layer_id
        self.name       = f"layer_{layer_id}" if not name else name
        self.type       = type(self).__name__.lower()
        self.parameters = {}
        self._state     = {}
        self._detached  = False
        object.__setattr__(self, '_detached', False)

    def __call__(self, x: Tensor) -> Tensor:
        if self._detached:
            return x
        return self._forward(x)

    def backward(self, grad: Tensor) -> Tensor:
        if self._detached:
            return grad
        return self._backward(grad)

    def _forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def __setattr__(self, name, value):
        if getattr(self, '_detached', False):
            raise AttributeError(
                f"Layer '{self.name}' is detached (immutable). "
                f"Cannot set '{name}'."
            )
        super().__setattr__(name, value)

    def detach(self):
        if hasattr(self, 'parameters'):
            object.__setattr__(self, 'parameters', MappingProxyType(self.parameters))
        object.__setattr__(self, '_detached', True)
        return self

    def _write_output(self, tensor: Tensor) -> None:
        CacheStore.write(self.id, SLOT_OUTPUT, tensor)

    def _write_grad_in(self, tensor: Tensor) -> None:
        CacheStore.write(self.id, SLOT_GRAD_IN, tensor)

# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear(Layer):
    def __init__(self, layer_id: Any, in_features: int, out_features: int, name: str = ""):
        super().__init__(layer_id, name)
        std             = np.sqrt(2.0 / in_features)
        self.W          = Tensor(np.random.normal(0, std, (in_features, out_features)), requires_grad=True)
        self.b          = Tensor(np.zeros((1, out_features)), requires_grad=True)
        self.parameters = {'W': self.W, 'b': self.b}

    def _forward(self, X: Tensor) -> Tensor:
        self._state['X'] = X          # needed by backward for dW
        out = (X @ self.W) + self.b
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        grad_X = grad @ self.W.transpose
        return grad_X


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(Layer):
    def __init__(self, layer_id: Any, in_features: int, eps: float = 1e-5, name: str = ""):
        super().__init__(layer_id, name)
        self.eps        = eps
        self.gamma      = Tensor(np.ones((1, in_features)),  requires_grad=True)
        self.beta       = Tensor(np.zeros((1, in_features)), requires_grad=True)
        self.parameters = {'gamma': self.gamma, 'beta': self.beta}

    def _forward(self, X: Tensor) -> Tensor:
        mu    = mean(X, axis=-1, keepdims=True)
        X_mu  = X - mu
        var   = mean(X_mu * X_mu, axis=-1, keepdims=True)
        eps   = Tensor(np.ones_like(var.data) * self.eps, requires_grad=False)
        std   = sqrt(var + eps)
        X_hat = X_mu / std
        out   = (self.gamma * X_hat) + self.beta

        self._state['X_hat'] = X_hat
        self._state['std']   = std
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        X_hat = self._state['X_hat']
        std   = self._state['std']

        dg = mean(grad * X_hat, axis=0)
        db = mean(grad, axis=0)
        self.gamma.grad = self.gamma.grad + dg if self.gamma.grad is not None else dg
        self.beta.grad  = self.beta.grad  + db if self.beta.grad  is not None else db

        dX_hat = grad * self.gamma
        term1  = dX_hat
        term2  = mean(dX_hat, axis=-1, keepdims=True)
        term3  = X_hat * mean(dX_hat * X_hat, axis=-1, keepdims=True)
        grad_X = ((term1 - term2) - term3) / std

        return grad_X


# ---------------------------------------------------------------------------
# Conv2d (forward only; backward stub preserved)
# ---------------------------------------------------------------------------

class Conv2d(Layer):
    def __init__(self, layer_id: Any, in_channels: int, out_channels: int,
                 kernel_size=3, stride=1, padding='same', name: str = ""):
        super().__init__(layer_id, name)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride       = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding      = padding
        std               = np.sqrt(2.0 / in_channels)
        self.W            = Tensor(np.random.normal(0, std, (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])), requires_grad=True)
        self.b            = Tensor(np.zeros((out_channels, 1, 1)), requires_grad=True)
        self.parameters   = {'W': self.W, 'b': self.b}

    def _forward(self, X: Tensor) -> Tensor:
        if isinstance(self.padding, tuple):
            padding_H, padding_W = self.padding
        elif self.padding == 'same':
            padding_H = (self.kernel_size[0] - 1) // 2
            padding_W = (self.kernel_size[1] - 1) // 2
        else:
            raise ValueError(f'Unknown padding: {self.padding}')

        in_C, in_H, in_W = X.shape
        out_H = (in_H - self.kernel_size[0] + 2 * padding_H) // self.stride[0] + 1
        out_W = (in_W - self.kernel_size[1] + 2 * padding_W) // self.stride[1] + 1

        X_padded = np.pad(X.data, ((0,0),(padding_H,padding_H),(padding_W,padding_W)), mode='constant')
        X_tf = []
        for i in range(out_H):
            for j in range(out_W):
                patch = X_padded[:, i*self.stride[0]:i*self.stride[0]+self.kernel_size[0],
                                     j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]]
                X_tf.append(patch.reshape(1, -1))
        X_tf = np.vstack(X_tf)
        W_tf = self.W.reshape(self.out_channels, -1)
        out_tf = X_tf @ W_tf.transpose
        out = out_tf.transpose.reshape(self.out_channels, out_H, out_W) + self.b

        self._state.update({'X_tf': X_tf, 'X_padded': X_padded, 'W_tf': W_tf,
                            'out_H': out_H, 'out_W': out_W,
                            'padding': (padding_H, padding_W)})
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        # Backward not yet implemented — returns zeros as placeholder
        grad_X = Tensor(np.zeros_like(grad.data), requires_grad=False)
        return grad_X