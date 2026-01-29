from copy import copy as copy_as_view

class WrappedLayer:
    def __init__(self, layer, observers):
        self.layer = layer
        self.observers = observers

    def forward(self, x):
        x_view = copy_as_view(x)
        for observer in self.observers:
            observer.on_forward_pre(self.layer.id, x_view, self.layer._cache)

        out = self.layer(x)

        cache = copy_as_view(self.layer).detach()._cache
        for observer in self.observers:
            observer.on_forward_post(self.layer.id, cache)
        return out

    def backward(self, grad):
        cache = copy_as_view(self.layer).detach()._cache
        for observer in self.observers:
            observer.on_backward_pre(self.layer.id, cache)

        grad_in = self.layer.backward(grad)

        cache = copy_as_view(self.layer).detach()._cache
        for observer in self.observers:
            observer.on_backward_post(self.layer.id, cache)

        return grad_in
    