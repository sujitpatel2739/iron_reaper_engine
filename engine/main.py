from engine import Engine
from ResNet import ResBlock
from diag.engine import MIEngine
from Layer import Linear
from LayerObserver import SignalShapeObserver, SignalStatsObserver, ResidualEnergyObserver
import numpy as np
from ironframe import Tensor

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
    layers.append(ResBlock(layer_id, in_features, out_features, alpha=0.05, lnorm_mode='pre'))
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
                
                
# ------------------------------------------------------------------------------------
# Derieved metrics interpertation (DME):

# We need to creat different interpreter profiles, each interpreteing one or more specific metric(s).
class InterpreterProfile:
    def __init__(self, name):
        self.name = name
        
    def __call__(self, observers):
        return self._execute(observers)
    
    def _execute(self, observers):
        # Example interpretation logic for this profile
        print(f"Executing interpreter profile: {self.name}")
        for observer in observers:
            observer_name = observer.__class__.__name__
            print(f"  Observer: {observer_name}")

forwardVarianceProfile = InterpreterProfile("Forward_Variance")

mi_engine = MIEngine("Interpreter-1")