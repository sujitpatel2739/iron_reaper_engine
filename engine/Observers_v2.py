from collections import defaultdict
import numpy as np

class LayerObserver:
    def on_forward_pre(self, layer): 
        pass

    def on_forward_post(self, layer): 
        pass

    def on_backward_pre(self, layer): 
        pass

    def on_backward_post(self, layer): 
        pass

class SignalStatsObserver(LayerObserver):
    def __init__(self):
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_post(self, layer):
        data = layer._cache['outputs']['out'].data
        self.logs[layer.layer_id]["activation_mean"].append(data.mean())
        self.logs[layer.layer_id]["activation_var"].append(data.var())

    def on_backward_pre(self, layer):
        grad = layer._cache['grads']['grad_in'].data
        self.logs[layer.layer_id]["grad_norm"].append(np.linalg.norm(grad))
        self.logs[layer.layer_id]["grad_var"].append(grad.var())

class SignalShapeObserver(LayerObserver):
    def __init__(self):
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_pre(self, layer, x):
        shape = x.data.shape
        self.logs[layer.layer_id]["input_shape"].append(shape)
    
    def on_forward_post(self, layer, out):
        shape = out.data.shape
        self.logs[layer.layer_id]["activation_shape"].append(shape)

    def on_backward_pre(self, layer, grad):
        shape = grad.data.shape
        self.logs[layer.layer_id]["grad_shape"].append(shape)
        
    def on_backward_post(self, layer, grad_out):
        shape = grad_out.data.shape
        self.logs[layer.layer_id]["grad_out_shape"].append(shape)
        
class ResidualEnergyObserver(LayerObserver):
    def __init__(self):
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))
    
    def on_forward_post(self, layer, out):
        f = layer._cache['residual'].detach()
        s = layer._cache['shortcut'].detach()
        self.logs[layer.layer_id]["residual"].append(np.mean((f**2).data))
        self.logs[layer.layer_id]["shortcut"].append(np.mean((s**2).data))
        