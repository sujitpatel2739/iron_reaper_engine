"""
engine.py
---------
Main sequential execution engine. Wraps layers in WrappedLayer and
runs full forward and backward passes.
"""

from neural_network.Layers.WrappedLayer import WrappedLayer


class Engine:
    def __init__(self, layers, observers):
        self.observers     = observers
        self.wrapped_layers = [WrappedLayer(layer, observers) for layer in layers]

    def forward(self, x):
        out = x
        for wl in self.wrapped_layers:
            out = wl.forward(out)
        return out

    def backward(self, grad):
        grad_out = grad
        for wl in reversed(self.wrapped_layers):
            grad_out = wl.backward(grad_out)
        return grad_out