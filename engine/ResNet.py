from Layer import Layer, Linear, Relu, LayerNorm
from ironframe import add

class ResBlock(Layer):
    def __init__(self, layer_id, in_features, out_features, alpha, lnorm_mode='post'):
        super().__init__(layer_id)

        # Residual branch F(x)
        self.linear = Linear(layer_id, in_features, out_features)
        self.relu = Relu(layer_id + 1)
        self.lnorm = LayerNorm(layer_id + 2, out_features, 1e-5)
        self.alpha = alpha
        self.lnorm_mode = lnorm_mode  # 'pre' or 'post' or 'none'

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
        if self.lnorm_mode == 'pre':
            # Pre-activation normalization
            f = self.lnorm.forward(f)
        f = f * self.alpha

        # Shortcut path
        if self.shortcut:
            s = self.shortcut.forward(X)
        else:
            s = X

        out = s + f
        if self.lnorm_mode == 'post':
            # Post-activation normalization
            out = self.lnorm.forward(out)
            
        self._cache['residual'] = f
        self._cache['shortcut'] = s
        self._cache['out'] = out
        return out

    def backward(self, grad):
        # 1. Handle Post-Activation LayerNorm
        if self.lnorm_mode == 'post':
            grad = self.lnorm.backward(grad)

        # 2. The gradient splits at the addition (Z = S + F)
        # Both branches receive the same incoming gradient.
        grad_s = grad
        grad_f = grad

        # -------------------------------------------
        # 3. Backpropagate through Residual Path (F)
        # -------------------------------------------
    
        grad_f = grad_f * self.alpha

        # 3b. Handle Pre-Activation LayerNorm
        if self.lnorm_mode == 'pre':
            grad_f = self.lnorm.backward(grad_f)

        grad_f = self.relu.backward(grad_f)
        grad_f = self.linear.backward(grad_f)

        # -------------------------------------------
        # 4. Backpropagate through Shortcut Path (S)
        # -------------------------------------------
        if self.shortcut:
            grad_s = self.shortcut.backward(grad_s)
        # Else: If shortcut is None (identity connection), grad_s flows through unchanged.

        # 5. Sum gradients at the split point
        return grad_f + grad_s