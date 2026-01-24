class WrappedLayer:
    def __init__(self, layer, observers):
        self.layer = layer
        self.observers = observers

    def forward(self, x):
        for observer in self.observers:
            observer.on_forward_pre(self.layer, x.detach())

        out = self.layer.forward(x)

        for observer in self.observers:
            observer.on_forward_post(self.layer, out.detach())

        return out

    def backward(self, grad):
        for observer in self.observers:
            observer.on_backward_pre(self.layer, grad.detach())

        grad_out = self.layer.backward(grad)

        for observer in self.observers:
            observer.on_backward_post(self.layer, grad_out.detach())

        return grad_out
