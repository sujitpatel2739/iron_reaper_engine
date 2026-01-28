from copy import copy as copy_as_view

class WrappedLayer:
    def __init__(self, layer, observers):
        self.layer = layer
        self.observers = observers

    def forward(self, x):
        x_view = copy_as_view(x).freeze()
        for observer in self.observers:
            observer.on_forward_pre(self.layer.id, x_view, self.layer._cache)

        out = self.layer(x)

        out_view = copy_as_view(out).freeze()
        for observer in self.observers:
            observer.on_forward_post(self.layer.id, self.layer._cache)
        return out

    def backward(self, grad):
        grad_view = copy_as_view(grad).freeze()
        for observer in self.observers:
            observer.on_backward_pre(self.layer.id, self.layer._cache)

        grad_in = self.layer.backward(grad)

        grad_in_view = copy_as_view(grad_in).freeze()
        for observer in self.observers:
            observer.on_backward_post(self.layer.id, self.layer._cache)

        return grad_in
    
    
