from collections import defaultdict
import numpy as np

class LayerObserver:
    def on_forward_pre(self, layer, x): 
        pass

    def on_forward_post(self, layer, out): 
        pass

    def on_backward_pre(self, layer, grad): 
        pass

    def on_backward_post(self, layer, grad_out): 
        pass

class SignalStatsObserver(LayerObserver):
    def __init__(self):
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_post(self, layer, out):
        data = out.data
        self.logs[layer.layer_id]["activation_mean"].append(data.mean())
        self.logs[layer.layer_id]["activation_var"].append(data.var())

    def on_backward_pre(self, layer, grad):
        data = grad.data
        self.logs[layer.layer_id]["grad_norm"].append(np.linalg.norm(data))
        self.logs[layer.layer_id]["grad_var"].append(data.var())
