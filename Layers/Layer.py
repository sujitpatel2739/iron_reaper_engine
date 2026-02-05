
import numpy as np
from core.ironframe.ironframe import Tensor, add, mul, matmul, mean, sub, div, sqrt
from types import MappingProxyType

class Layer:
    def __init__(self, layer_id: int, name: str = ""):
        object.__setattr__(self, '_detached', False)
        self.id = layer_id
        self.name = "layer_" + str(layer_id) if name == "" else name
        self.type = str(self.__class__.__name__).lower()
        self.parameters = {}
        # _cache structure: inputs, outputs, grads, paths, running
        self._cache = tuple([{}, {}, {}, {}, {}])
        
    def __call__(self, x):
        if self._detached:
            print("Layer detached! Passing input unchanged!")
            return x
        return self._forward(x)
    
    def _forward(self, x):
        pass

    def backward(self, grad):
        if self._detached:
            return grad
        return self._backward(grad)         

    def _backward(self, grad):
        pass
    
    def __setattr__(self, name, value):
        if getattr(self, '_detached', True):
            raise AttributeError(f"Layer is detached (immutable). Cannot modify '{name}'.")
        super().__setattr__(name, value)
    
    def detach(self):
        if hasattr(self, 'parameters'):
            read_only_parameters = MappingProxyType(self.parameters)
            object.__setattr__(self, 'parameters', read_only_parameters)

        self._detached = True
        print("Layer detached: ", self)
        return self


class Linear(Layer):
    def __init__(self, layer_id, in_features, out_features, name=""):
        super().__init__(layer_id, name)
        std = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.normal(0, std**2, (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)
        self.parameters = {'W': self.W, 'b': self.b}
        
    def _forward(self, X):
        self._cache[0]['in'] = X
        out = add(matmul(X, self.W), self.b)
        self._cache[1]['out'] = out
        # out.shape: (batch, out_features)
        return out
    
    def _backward(self, grad):
        # grad.shape: (batch, out_features)
        self._cache[2]['grad_out'] = grad
        grad_X = matmul(grad, self.W.transpose())
        self._cache[2]['grad_in'] = grad_X
        # grad_input.shape: (batch, in_features)
        return grad_X
        
class Relu(Layer):
    def __init__(self, layer_id, name=""):
        super().__init__(layer_id, name)
        self.mask = None
    
    def _forward(self, input):
        out = Tensor(np.maximum(0, input.data), requires_grad=input.requires_grad)
        self.mask = input.data > 0
        return out
    
    def _backward(self, grad):
        return Tensor(grad.data * self.mask, requires_grad=grad.requires_grad)
    
class LayerNorm(Layer):
    def __init__(self, layer_id, in_features, eps, name=""):
        super().__init__(layer_id, name)

        self.eps = eps

        self.gamma = Tensor(np.ones((1, in_features)), requires_grad=True)
        self.beta  = Tensor(np.zeros((1, in_features)), requires_grad=True)

        self.parameters = {'gamma': self.gamma, 'beta': self.beta}
    
    def _forward(self, X):
        # mean over features (per sample)
        self._cache[0]['in'] = X
        mu = mean(X, axis=-1, keepdims=True)

        # variance
        X_mu = X - mu
        var = mean(X_mu * X_mu, axis=-1, keepdims=True)

        # normalize
        eps = Tensor(np.ones_like(var.data) * self.eps, requires_grad=False)
        std = sqrt(var * eps)
        X_hat = X_mu / std

        # affine
        out = (self.gamma * X_hat) + self.beta

        # cache everything needed for backward
        self._cache[4].update({
            'X_hat': X_hat,
            'std': std,
            'X_mu': X_mu})
        
        self._cache[1]['out'] = out
        return out
    
    def _backward(self, grad):
        """
        grad.shape = (batch_size, in_features)
        Using the cached values from forward pass to compute gradients.
        Cached values:
        - X_hat
        - std
        - X_mu
        """
        self._cache[2]['grad_out'] = grad
        X_hat = self._cache[4]['X_hat']
        std   = self._cache[4]['std']
        X_mu  = self._cache[4]['X_mu']
        N = X_hat.data.shape[-1]

        # gradients for gamma and beta
        if self.gamma.grad:
            self.gamma.grad += mean(grad * X_hat, axis=0)
        else:
            self.gamma.grad = mean(grad * X_hat, axis=0)
        if self.beta.grad:
            self.beta.grad += mean(grad, axis=0)
        else:
            self.beta.grad = mean(grad, axis=0)
            

        # grad wrt normalized input
        dX_hat = grad * self.gamma

        # LayerNorm backward (derived, exact)
        term1 = dX_hat
        term2 = mean(dX_hat, axis=-1, keepdims=True)
        term3 = X_hat * mean(dX_hat * X_hat, axis=-1, keepdims=True)

        grad_X = ((term1 - term2) - term3) / std
        self._cache[2]['grad_in'] = grad_X
        return grad_X