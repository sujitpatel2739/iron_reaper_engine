"""
Layer.py
--------
Base Layer and primitive layer implementations.

CacheStore is imported as a module — no instance, no injection.
Layers call CacheStore.write() / CacheStore.read_required() directly.

_state remains a private instance dict on each layer — it holds
internal working memory (mask, X_hat, std) that only that layer's
own backward ever reads. It is never accessed from outside.
"""

import numpy as np
from types import MappingProxyType
from typing import Any

import cache.CacheStore as CacheStore
from cache.CacheStore import SLOT_INPUT, SLOT_OUTPUT, SLOT_GRAD_OUT, SLOT_GRAD_IN
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
        self._state     = {}   # private working memory for this layer's backward
        self._detached = False
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

    # -- CacheStore helpers --------------------------------------------------

    def _write(self, slot: str, tensor: Tensor) -> None:
        CacheStore.write(self.id, slot, tensor)

    def _read(self, slot: str) -> Tensor:
        return CacheStore.read_required(self.id, slot)

# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear(Layer):
    def __init__(self, layer_id: Any, in_features: int, out_features: int, name: str = ""):
        super().__init__(layer_id, name)
        std      = np.sqrt(2.0 / in_features)
        self.W   = Tensor(np.random.normal(0, std, (in_features, out_features)), requires_grad=True)
        self.b   = Tensor(np.zeros((1, out_features)), requires_grad=True)
        self.parameters = {'W': self.W, 'b': self.b}

    def _forward(self, X: Tensor) -> Tensor:
        self._write(SLOT_INPUT, X)
        out = (X @ self.W) + self.b
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)
        grad_X = grad @ self.W.transpose
        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X

class InputLayer(Layer):
    def __init__(self, layer_id: Any, inputs: list, name: str = ""):
        super().__init__(layer_id, name)
        self.inputs = list(inputs)
        self.parameters = {}

    def _forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return inputs

    def _backward(self, grad: Tensor) -> Tensor:
        return grad


# ---------------------------------------------------------------------------
# Activation Layers
# ---------------------------------------------------------------------------

class Relu(Layer):
    def __init__(self, layer_id: Any, name: str = ""):
        super().__init__(layer_id, name)

    def _forward(self, x: Tensor) -> Tensor:
        self._write(SLOT_INPUT, x)
        self._state['mask'] = x.data > 0
        out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)
        grad_X = Tensor(grad.data * self._state['mask'], requires_grad=grad.requires_grad)
        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X

class Tanh(Layer):
    def __init__(self, layer_id: Any, name: str = ""):
        super().__init__(layer_id, name)

    def _forward(self, x: Tensor) -> Tensor:
        self._write(SLOT_INPUT, x)
        out = (np.exp(x.data) - np.exp(-x.data)) / (np.exp(x.data) + np.exp(-x.data))
        out = Tensor(out, requires_grad=x.requires_grad)
        self._write(SLOT_OUTPUT, out)
        self._state['out'] = out
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)
        out = self._state['out']
        grad_X = grad * (1 - (out.data ** 2))
        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X

class Sigmoid(Layer):
    def __init__(self, layer_id: Any, name: str = ""):
        super().__init__(layer_id, name)
        
    def _forward(self, x: Tensor) -> Tensor:
        self._write(SLOT_INPUT, x)
        out = Tensor(1 / (1 + np.exp(-x.data)) , requires_grad=x.requires_grad)
        self._write(SLOT_OUTPUT, out)
        self._state['out'] = out
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)
        out = self._state['out']
        grad_X = grad * out * (1 - out)
        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X

class LeakyRelu(Layer):
    def __init__(self, layer_id: Any, alpha: float = 0.01, name: str = ""):
        super().__init__(layer_id, name)
        self.alpha = alpha

    def _forward(self, x: Tensor) -> Tensor:
        self._write(SLOT_INPUT, x)
        self._state['mask'] = x.data > 0
        out = Tensor(np.where(self._state['mask'], x.data, x.data * self.alpha), requires_grad=x.requires_grad)
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)
        mask = self._state['mask']
        grad_X = Tensor(
            np.where(mask, grad.data, grad.data * self.alpha),
            requires_grad=grad.requires_grad
        )
        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X
    
class Elu(Layer):
    def __init__(self, layer_id: Any, alpha: float = 0.01, name: str = ""):
        super().__init__(layer_id, name)
        self.alpha = alpha

    def _forward(self, x: Tensor) -> Tensor:
        self._write(SLOT_INPUT, x)
        self._state['input'] = x.data
        self._state['mask'] = x.data > 0
        out = Tensor(np.where(self._state['mask'], x.data, self.alpha * (np.exp(x.data) - 1)), requires_grad=x.requires_grad)
        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)
        x = self._state['input']
        mask = self._state['mask']
        grad_X = Tensor(
            np.where(mask, grad.data, grad.data * self.alpha * np.exp(x)),
            requires_grad=grad.requires_grad
        )
        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X

# class Gelu(Layer):
#     def __init__(self, layer_id: Any, name: str = ""):
#         super().__init__(layer_id, name)
        
#     def _forward(self, x: Tensor) -> Tensor:
#         self._write(SLOT_INPUT, x)
#         self._state['mask'] = x.data > 0
#         out = Tensor(1 / 1 + np.exp(x.data) , requires_grad=x.requires_grad)
#         self._write(SLOT_OUTPUT, out)
#         return out

#     def _backward(self, grad: Tensor) -> Tensor:
#         self._write(SLOT_GRAD_OUT, grad)
#         out = self._read(SLOT_OUTPUT, 'out')
#         grad_X = grad * out * (1 - out)
#         self._write(SLOT_GRAD_IN, grad_X)
#         return grad_X
    
# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm(Layer):
    def __init__(self, layer_id: Any, in_features: int, eps: float = 1e-5, name: str = ""):
        super().__init__(layer_id, name)
        self.eps   = eps
        self.gamma = Tensor(np.ones((1, in_features)),  requires_grad=True)
        self.beta  = Tensor(np.zeros((1, in_features)), requires_grad=True)
        self.parameters = {'gamma': self.gamma, 'beta': self.beta}

    def _forward(self, X: Tensor) -> Tensor:
        self._write(SLOT_INPUT, X)

        mu    = mean(X, axis=-1, keepdims=True)
        X_mu  = X - mu
        var   = mean(X_mu * X_mu, axis=-1, keepdims=True)
        eps   = Tensor(np.ones_like(var.data) * self.eps, requires_grad=False)
        std   = sqrt(var + eps)
        X_hat = X_mu / std
        out   = (self.gamma * X_hat) + self.beta

        self._state['X_hat'] = X_hat   # private — only _backward reads this
        self._state['std']   = std

        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)

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

        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X
    
# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

class Conv2d(Layer):
    def __init__(self, layer_id: Any, in_channels: int, out_channels: int,
                 kernel_size: tuple|int = (3,3), stride: tuple|int = (1,1),
                 padding:tuple|str = 'same', name: str = ""):
        super().__init__(layer_id, name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        std = np.sqrt(2.0/in_channels)
        self.W = Tensor(np.random.normal(0, std, (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])), requires_grad=True)
        self.b = Tensor(np.zeros((out_channels, 1, 1)), requires_grad=True)
        self.parameters = {'W': self.W, 'b': self.b}
        

    def _forward(self, X: Tensor) -> Tensor:
        self._write(SLOT_INPUT, X)

        if isinstance(self.padding, tuple):
            padding_H, padding_W = self.padding
        elif self.padding == 'same':
            padding_H = (self.kernel_size[0] - 1) // 2
            padding_W = (self.kernel_size[1] - 1) // 2
        else:
            raise Exception('Error: Unknown padding type!')

        # X shape: (in_channels, H, W) — no batch dim
        in_C, in_H, in_W = X.shape

        out_H = (in_H - self.kernel_size[0] + 2 * padding_H) // self.stride[0] + 1
        out_W = (in_W - self.kernel_size[1] + 2 * padding_W) // self.stride[1] + 1

        # pad the raw numpy array
        X_padded = np.pad(
            X.data,
            ((0, 0), (padding_H, padding_H), (padding_W, padding_W)),
            mode='constant',
            constant_values=0
        )

        # im2col — each patch becomes one row
        X_tf = []
        for i in range(out_H):
            for j in range(out_W):
                row_start = i * self.stride[0]
                col_start = j * self.stride[1]

                patch = X_padded[
                    :,
                    row_start : row_start + self.kernel_size[0],
                    col_start : col_start + self.kernel_size[1]
                ]
                X_tf.append(patch.reshape(1, -1))  # (1, in_C * kH * kW)

        # shape: (out_H * out_W, in_C * kH * kW)
        X_tf = np.vstack(X_tf)

        # reshape kernels: (out_channels, in_C * kH * kW)
        W_tf = self.W.reshape(self.out_channels, -1)

        # matmul: (out_H*out_W, in_C*kH*kW) @ (in_C*kH*kW, out_channels)
        out_tf = X_tf @ W_tf.transpose  # shape: (out_H*out_W, out_channels)

        # reshape to (out_channels, out_H, out_W)
        out = out_tf.transpose.reshape(self.out_channels, out_H, out_W)
        out = out + self.b

        self._state['X_tf']    = X_tf
        self._state['X_padded'] = X_padded
        self._state['W_tf']    = W_tf
        self._state['out_H']    = out_H
        self._state['out_W']    = out_W
        self._state['padding']  = (padding_H, padding_W)

        self._write(SLOT_OUTPUT, out)
        return out

    def _backward(self, grad: Tensor) -> Tensor:
        self._write(SLOT_GRAD_OUT, grad)
        ...
        self._write(SLOT_GRAD_IN, grad_X)
        return grad_X