from Layer import Layer, Linear, Relu, LayerNorm
from ironframe import add

class ResBlock(Layer):
    def __init__(self, layer_id, in_features, out_features):
        super().__init__(layer_id)

        # Residual branch F(x)
        self.linear = Linear(layer_id, in_features, out_features)
        self.relu = Relu(layer_id + 1)
        self.lnorm = LayerNorm(layer_id + 2, out_features, 1e-5)

        # Shortcut branch S(x)
        if in_features != out_features:
            self.shortcut = Linear(layer_id + 3, in_features, out_features)
        else:
            self.shortcut = None

        # Collect parameters
        self.parameters = []
        self.parameters += self.linear.parameters
        if self.shortcut:
            self.parameters += self.shortcut.parameters

        self._cache = {}

    def forward(self, X):
        self._cache['X'] = X
        # Residual path
        f = self.linear.forward(X)
        f = self.relu.forward(f)
        f = self.lnorm(f)

        # Shortcut path
        if self.shortcut:
            s = self.shortcut.forward(X)
        else:
            s = X

        out = add(s, f)
        self._cache['residual'] = f
        self._cache['shortcut'] = s
        self._cache['out'] = out
        return out

    def backward(self, grad):
        """
        grad = dL/d(out)
        """

        # ----- Residual path -----
        grad_f = grad
        grad_f = self.lnorm.backward(grad_f)
        grad_f = self.relu.backward(grad_f)
        grad_f = self.linear.backward(grad_f)

        # ----- Shortcut path -----
        if self.shortcut:
            grad_s = self.shortcut.backward(grad)
        else:
            grad_s = grad

        # Combine gradients
        grad_x = add(grad_s, grad_f)
        return grad_x
