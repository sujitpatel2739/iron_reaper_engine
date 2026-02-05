from copy import copy as copy_as_view

class WrappedLayer:
    def __init__(self, layer, observers):
        self.layer = layer
        self.observers = observers

    def forward(self, x):
        x_view = copy_as_view(x)
        for observer in self.observers:
            observer.on_forward_pre(self.layer.id, x_view)

        out = self.layer(x)

        detached_layer = copy_as_view(self.layer).detach()
        for observer in self.observers:
            observer.on_forward_post(self.layer.id, detached_layer._cache)
        return out

    def backward(self, grad_out):
        grad_out_view = copy_as_view(grad_out)
        for observer in self.observers:
            observer.on_backward_pre(self.layer.id, grad_out_view)

        grad_in = self.layer.backward(grad_out)

        detached_layer = copy_as_view(self.layer).detach()
        for observer in self.observers:
            observer.on_backward_post(self.layer.id, detached_layer._cache)

        return grad_in
    