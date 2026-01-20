class WrappedLayer:
    def __init__(self, layer, observers):
        self.layer = layer
        self.observers = observers

    def forward(self, x):
        for observer in self.observers:
            observer.on_forward_pre(self.layer, x)

        out = self.layer.forward(x)

        for observer in self.observers:
            observer.on_forward_post(self.layer, out)

        return out

    def backward(self, grad):
        for observer in self.observers:
            observer.on_backward_pre(self.layer, grad)

        grad_out = self.layer.backward(grad)

        for observer in self.observers:
            observer.on_backward_post(self.layer, grad_out)

        return grad_out
