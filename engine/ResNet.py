from Layer import Layer, Linear, Relu
from ironframe import add

class ResBlock(Layer):
    def __init__(self, layer_id, in_features, out_features):
        super().__init__(layer_id)

        self.linear = Linear(layer_id, in_features, out_features)
        self.relu = Relu(layer_id + 1)
        self.parameters = self.linear.parameters
        self._cache = {}

    def forward(self, X):
        self._cache['X'] = X
        out = self.linear.forward(X)
        out = self.relu.forward(out)
        # We'll fix shape mismatch and edge cases later.
        out = add(X, out)
        self._cache['out'] = out
        return out

    def backward(self, grad):
        """
        grad = dL/d(out)
        """
        # Identity path
        grad_identity = grad

        # Residual path
        grad_f = grad
        grad_f = self.relu.backward(grad_f)
        grad_f = self.linear.backward(grad_f)
        return add(grad_identity, grad_f)
