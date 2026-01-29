from collections import defaultdict
import numpy as np

class LayerObserver:
    def __init__(self): pass
        
    def on_forward_pre(self, *args, **kwargs):
        pass

    def on_forward_post(self, *args, **kwargs):
        pass

    def on_backward_pre(self, *args, **kwargs):
        pass

    def on_backward_post(self, *args, **kwargs):
        pass

class SignalStatsObserver(LayerObserver):
    def __init__(self):
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))      

    def on_forward_post(self, layer_id, layer_cache):
        data = layer_cache['outputs']['out'].data
        self.logs[layer_id]["activation_mean"].append(data.mean())
        self.logs[layer_id]["activation_var"].append(data.var())

    def on_backward_pre(self, layer_id, layer_cache):
        grad = layer_cache['grads']['grad_in'].data
        self.logs[layer_id]["grad_norm"].append(np.linalg.norm(grad))
        self.logs[layer_id]["grad_var"].append(grad.var())

class SignalShapeObserver(LayerObserver):
    def __init__(self):
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_pre(self, layer_id, x):
        shape = x.data.shape
        self.logs[layer_id]["input_shape"].append(shape)

    def on_forward_post(self, layer_id, layer_cache):
        shape = layer_cache['outputs']['out'].data.shape
        self.logs[layer_id]["activation_shape"].append(shape)

    def on_backward_pre(self, layer_id, layer_cache):
        grad = layer_cache['grads']['grad_in'].data
        self.logs[layer_id]["grad_shape"].append(grad.shape)

    def on_backward_post(self, layer_id, layer_cache):
        grad_in = layer_cache['grads']['grad_in']
        shape = grad_in.data.shape
        self.logs[layer_id]["grad_in_shape"].append(shape)

class ResidualEnergyObserver(LayerObserver):
    def __init__(self):
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_post(self, layer_id, layer_cache):
        f = layer_cache['paths']['residual']
        s = layer_cache['paths']['shortcut']
        self.logs[layer_id]["residual"].append(np.mean((f**2).data))
        self.logs[layer_id]["shortcut"].append(np.mean((s**2).data))