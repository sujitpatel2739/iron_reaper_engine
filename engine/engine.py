from ironframe import Tensor
from WrappedLayer import WrappedLayer
from ResNet import ResBlock
from LayerObserver import SignalStatsObserver
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



# Drive code ----------------------------------------------------------
batch_size = 100
n_layers = 50
in_features = 0
out_features = 10
layer_id = 0
layers = []

for i in range(n_layers):
    if i < 25:
        in_features += 10
        out_features = in_features + 10
    else:
        in_features = out_features - 10
        out_features -= 10
    
    layers.append(ResBlock(layer_id, in_features, out_features))
    layer_id += 2    # +2 because ResBlock contains 2 layers: Linear + Relu
    
observers = [SignalStatsObserver()]
        
E1 = Engine(
    layers,
    observers
)

X = Tensor(
    np.random.randn(batch_size, in_features),  # batch_size=32, in_features=10
    requires_grad=True
)

out = E1.forward(X)
grad_out = Tensor(np.random.randn(*out.data.shape))  # same shape as out
grad_in = E1.backward(grad_out)

for observer in observers:
    for layer_id, metrics in observer.logs.items():
        print(f"Layer {layer_id} stats:")
        for metric, values in metrics.items():
            mean_value = np.mean(values)
            print(f"  {metric}: {mean_value:.4f}")