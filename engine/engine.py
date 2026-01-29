from ironframe import Tensor
from WrappedLayer import WrappedLayer
from ResNet import ResBlock
from Layer import Linear
from LayerObserver import SignalShapeObserver, SignalStatsObserver, ResidualEnergyObserver
import numpy as np

class Engine:
    def __init__(self, layers, observers):
        self.observers = observers

        self.wrapped_layers = [
            WrappedLayer(layer, observers)
            for layer in layers
        ]

    def forward(self, x):
        out = x
        for wrapped_layer in self.wrapped_layers:
            out = wrapped_layer.forward(out)
        return out

    def backward(self, grad):
        grad_out = grad
        for wrapped_layer in reversed(self.wrapped_layers):
            grad_out = wrapped_layer.backward(grad_out)
        return grad_out

