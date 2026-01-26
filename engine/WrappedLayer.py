from copy import copy as view_by_copy

class WrappedLayer:
    def __init__(self, layer, observers):
        self.layer = layer
        self.observers = observers

    def forward(self, x):
        x_view = view_by_copy(x).detach()
        for observer in self.observers:
            observer.on_forward_pre(self.layer, x_view)

        out = self.layer(x)

        out_view = view_by_copy(out).detach()
        for observer in self.observers:
            observer.on_forward_post(self.layer, out_view)
        return out

    def backward(self, grad):
        grad_view = view_by_copy(grad).detach()
        for observer in self.observers:
            observer.on_backward_pre(self.layer, grad_view)

        grad_in = self.layer.backward(grad)

        grad_in_view = view_by_copy(grad_in).detach()
        for observer in self.observers:
            observer.on_backward_post(self.layer, grad_in_view)

        return grad_in
