class WrappedLayer:
    def __init__(self, layer, observers):
        self.layer = layer
        self.observers = observers

    def forward(self, x):
        for observer in self.observers:
            observer.on_forward_pre(self.layer)

        out = self.layer.forward(x)

        for observer in self.observers:
            observer.on_forward_post(self.layer)

        return out

    def backward(self, grad):
        for observer in self.observers:
            observer.on_backward_pre(self.layer)

        grad_out = self.layer.backward(grad)

        for observer in self.observers:
            observer.on_backward_post(self.layer)

        return grad_out
