"""
WrappedLayer.py
---------------
"""

import cache.CacheStore as CacheStore
from ironframe.ironframe import Tensor


class WrappedLayer:
    def __init__(self, layer, observers: list, release_after_observe: bool = False):
        self.layer                = layer
        self.observers            = observers
        self.release_after_observe = release_after_observe

    def forward(self, x: Tensor) -> Tensor:
        layer_id = self.layer.id

        for obs in self.observers:
            obs.on_forward_pre(layer_id)

        out = self.layer(x)

        for obs in self.observers:
            obs.on_forward_post(layer_id)

        # Release CacheStore entry for this layer to free the tensor reference
        # and allow the autograd graph to be garbage-collected.
        # Only in full-run mode — step mode still needs CacheStore for collector.
        if self.release_after_observe:
            CacheStore.clear_layer(layer_id)

        return out

    def backward(self, grad: Tensor) -> Tensor:
        layer_id = self.layer.id

        for obs in self.observers:
            obs.on_backward_pre(layer_id)

        grad_in = self.layer.backward(grad)

        for obs in self.observers:
            obs.on_backward_post(layer_id)

        if self.release_after_observe:
            CacheStore.clear_layer(layer_id)

        return grad_in

    def set_release_mode(self, release: bool) -> None:
        """Toggle between full-run mode (release=True) and step mode (release=False)."""
        self.release_after_observe = release