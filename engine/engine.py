from ironframe import Tensor
from WrappedLayer import WrappedLayer
from ResNet import ResBlock
from Layer import Linear
from LayerObserver import SignalShapeObserver, SignalStatsObserver, ResidualEnergyObserver
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
n_layers = 20
in_features = 10
out_features = 10
layer_id = 0
layers = []

for i in range(n_layers):
    if i < 10:
        out_features = in_features + 1
    else:
        out_features = in_features - 1

    # layers.append(Linear(layer_id, in_features, out_features))
    layers.append(ResBlock(layer_id, in_features, out_features, alpha=1, lnorm_mode='post'))
    in_features = out_features
    layer_id += 4  # linear + relu + lnorm + optional shortcut

    
observers = [
    SignalStatsObserver(),
    SignalShapeObserver(),
    ResidualEnergyObserver(),
]
        
E1 = Engine(
    layers,
    observers
)

batch_size = 100
in_features = 10
X = Tensor(
    np.random.randn(batch_size, in_features),  # batch_size=32, in_features=10
    requires_grad=True
)

out = E1.forward(X)
grad_out = Tensor(np.random.randn(*out.data.shape))  # same shape as out
grad_in = E1.backward(grad_out)

for layer_no, layer_id in enumerate(range(0, n_layers, 4)):
    print(f"Layer {layer_no} (ID: {layer_id}):")
    for observer in observers:
        observer_name = observer.__class__.__name__
        print(f" Observer: {observer_name}")
        for metric, values in observer.logs[layer_id].items():
            if(observer_name == "SignalStatsObserver"):
                print(f"    {metric}: {np.mean(values)}")
            else:
                print(f"    {metric}: {values}")