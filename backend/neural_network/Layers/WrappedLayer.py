"""
WrappedLayer.py
---------------
Wraps a Layer and fires lifecycle hooks on observers.
Signals only — no data is passed, no stores are touched here.
"""

from ironframe.ironframe import Tensor


class WrappedLayer:
    def __init__(self, layer, observers: list):
        self.layer     = layer
        self.observers = observers

    def forward(self, x: Tensor) -> Tensor:
        layer_id = self.layer.id

        for obs in self.observers:
            obs.on_forward_pre(layer_id)

        out = self.layer(x)

        for obs in self.observers:
            obs.on_forward_post(layer_id)

        return out

    def backward(self, grad: Tensor) -> Tensor:
        layer_id = self.layer.id

        for obs in self.observers:
            obs.on_backward_pre(layer_id)

        grad_in = self.layer.backward(grad)

        for obs in self.observers:
            obs.on_backward_post(layer_id)

        return grad_in