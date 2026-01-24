
import numpy as np
from ironframe import Tensor, add, mul, matmul, mean, sub, div, sqrt

class Layer:
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.parameters = []
        self._cache = {}

    def forward(self, x):
        pass

    def backward(self, grad):
        pass

    # --- lifecycle hooks (do NOT implement logic here) ---
    def forward_pre(self, x): pass
    def forward_post(self, out): pass
    def backward_pre(self, grad): pass
    def backward_post(self, grad_out): pass


class Linear(Layer):
    def __init__(self, layer_id, in_features, out_features):
        super().__init__(layer_id)
        std = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.normal(0, std**2, (in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)
        self.parameters = [self.W, self.b]
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, X):
        self._cache['X'] = X
        out = add(matmul(X, self.W), self.b)
        self._cache['out'] = out
        # out.shape: (batch, out_features)
        return out
    
    def backward(self, grad):
        # grad.shape: (batch, out_features)
        grad_X = matmul(grad, self.W.transpose())
        return grad_X
        # grad_input.shape: (batch, in_features)
        
class Relu(Layer):
    def __init__(self, layer_id):
        super().__init__(layer_id)
        self.mask = None
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        out = Tensor(np.maximum(0, input.data), requires_grad=input.requires_grad)
        self.mask = input.data > 0
        return out
    
    def backward(self, grad):
        return Tensor(grad.data * self.mask, requires_grad=grad.requires_grad)
    
class LayerNorm(Layer):
    def __init__(self, layer_id, in_features, eps):
        super().__init__(layer_id)

        self.eps = eps

        self.gamma = Tensor(np.ones((1, in_features)), requires_grad=True)
        self.beta  = Tensor(np.zeros((1, in_features)), requires_grad=True)

        self.parameters = [self.gamma, self.beta]
        self._cache = {}
        
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        # mean over features (per sample)
        mu = mean(X, axis=-1, keepdims=True)

        # varianceo
        X_mu = sub(X, mu)
        var = mean(mul(X_mu, X_mu), axis=-1, keepdims=True)

        # normalize
        eps = Tensor(np.ones_like(var.data) * self.eps, requires_grad=False)
        std = sqrt(add(var, eps))
        X_hat = div(X_mu, std)

        # affine
        out = add(mul(self.gamma, X_hat), self.beta)

        # cache everything needed for backward
        self._cache = {
            'X_hat': X_hat,
            'std': std,
            'X_mu': X_mu
        }
        return out
    
    def backward(self, grad):
        """
        grad.shape = (batch_size, in_features)
        Using the cached values from forward pass to compute gradients.
        Cached values:
        - X_hat
        - std
        - X_mu
        """
        X_hat = self._cache['X_hat']
        std   = self._cache['std']
        X_mu  = self._cache['X_mu']
        N = X_hat.data.shape[-1]

        # gradients for gamma and beta
        if self.gamma.grad:
            self.gamma.grad += mean(mul(grad, X_hat), axis=0)
        else:
            self.gamma.grad = mean(mul(grad, X_hat), axis=0)
        if self.beta.grad:
            self.beta.grad += mean(grad, axis=0)
        else:
            self.beta.grad = mean(grad, axis=0)
            

        # grad wrt normalized input
        dX_hat = mul(grad, self.gamma)

        # LayerNorm backward (derived, exact)
        term1 = dX_hat
        term2 = mean(dX_hat, axis=-1, keepdims=True)
        term3 = mul(X_hat, mean(mul(dX_hat, X_hat), axis=-1, keepdims=True))

        grad_X = div(sub(sub(term1, term2), term3), std)

        return grad_X