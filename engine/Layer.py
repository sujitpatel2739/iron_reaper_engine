
import numpy as np
from ironframe import Tensor, add, mul, matmul

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
        self.W = Tensor(np.random.randn(in_features, out_features), requires_grad=True)
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