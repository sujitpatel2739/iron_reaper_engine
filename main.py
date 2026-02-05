from engine import Engine
from Layers.ResNet import ResBlock
from diag.engine import MIEngine
from Layers.Layer import Linear
from Observers.Observers import SignalShapeObserver, SignalStatsObserver, ResidualEnergyObserver
from diag.Profiles.Profiles import PathDominanceProfile, SignalStatsProfile
from diag.MetricStore import MetricStore
import numpy as np
from core.ironframe.ironframe import Tensor
import matplotlib.pyplot as plt
from collections import defaultdict

# Drive code ----------------------------------------------------------
RUN = 0

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
    np.random.randn(batch_size, in_features),
    requires_grad=True
)

out = E1.forward(X)
grad_out = Tensor(np.random.randn(*out.data.shape))  # same shape as out
grad_in = E1.backward(grad_out)

for layer_no, layer_id in enumerate(range(0, n_layers, 4)):
    print(f"Layer {layer_no} (ID: {layer_id}):")
    for observer in observers:
        obs_name = type(observer).__name__
        if layer_id in observer.logs:
            print(f"  Observer: {obs_name}")
            for metric, values in observer.logs[layer_id].items():
                if obs_name == "SignalShapeObserver":
                    print(f"    {metric}: {values[-1]}")
                else:
                    print(f"    {metric}: {np.mean(values)}")
                
                
# ------------------------------------------------------------------------------------
# Derieved metrics interpertation (DME):


# # Collect metrics into MetricStore
metric_store = MetricStore()

for observer in observers:
    if observer.name in ["SignalShapeObserver"]:
        continue  # skip shape observers
    for layer_id, metrics in observer.logs.items():
        for metric_name, values in metrics.items():
            metric_store.add_metric(RUN, layer_id, metric_name, values)

# -------------------------------------------------------------------------------------
            

def analyze_pattern(data):
    x = np.arange(len(data))
    y = np.array(data)

    threshold = 0.001
    slope, intercept = np.polyfit(x, y, 1)

    if slope > threshold:
        return "increasing"
    elif slope < -threshold:
        return "decreasing"
    else:
        return "stable"
    

# We need to create different interpreter profiles, each interpreteing one or more specific metric(s).

signal_stats_profile = SignalStatsProfile("signal_state", RUN)
path_dominance_profile = PathDominanceProfile("path_dominance", RUN)

signal_stats = signal_stats_profile(metric_store)
path_dominance = path_dominance_profile(metric_store)
print(path_dominance)

# ------------------------------------------------------------------------------------
# Noww we plot the derived metrics for visualization:
# Plot Forward Activation Variance per Layer
pattern = analyze_pattern([v for _, v in signal_stats['activation_var'].items()])
layers, variances = zip(*signal_stats['activation_var'].items())
# variances = np.mean(variances)
plt.figure(figsize=(10, 5))
plt.plot(layers, variances, marker='o')
plt.xlabel("Layer ID")
plt.ylabel(f"Forward Activation Variance ({pattern})")
plt.title("Forward Activation Variance per Layer")
plt.grid(True)
plt.show()

# Plot Backward Gradient Norm per Layer
pattern = analyze_pattern([v for _, v in signal_stats['grad_norm'].items()])
layers, grad_norms = zip(*signal_stats['grad_norm'].items())
plt.figure(figsize=(10, 5))
plt.plot(layers, grad_norms, marker='o')
plt.xlabel("Layer ID")
plt.ylabel(f"Backward Gradient Norm ({pattern})")
plt.title("Backward Gradient Norm per Layer")
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------------

# Plot path energies from ResNet
# Plot the resisual path energy
residual_pattern = analyze_pattern([v for _, v in path_dominance['residual'].items()])
shortcut_pattern = analyze_pattern([v for _, v in path_dominance['shortcut'].items()])
layers, residual_energies = zip(*path_dominance['residual'].items())
_, shortcut_energies = zip(*path_dominance['shortcut'].items())
plt.figure(figsize=(10, 5))
plt.plot(layers, residual_energies, marker='o')
plt.plot(layers, shortcut_energies, marker='x')
plt.xlabel("Layer ID")
plt.ylabel(f"Backward Gradient Norm ({pattern})")
plt.title("Backward Gradient Norm per Layer")
plt.grid(True)
plt.show()