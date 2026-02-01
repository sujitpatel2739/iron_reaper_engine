from collections import defaultdict
import numpy as np

class LayerObserver:
    def __init__(self, name): 
        self.name = name
        
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
        self.name = "SignalStatsObserver"
        super().__init__(self.name)
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))      

    def on_forward_post(self, layer_id, layer_cache):
        out = layer_cache[1]['out'].freeze().data
        if not out.freezed:
            print("[WARNING]: layer output is not freezed in SignalStatsObserver")
            return
        self.logs[layer_id]["activation_mean"].append(out.data.mean())
        self.logs[layer_id]["activation_var"].append(out.data.var())

    def on_backward_pre(self, layer_id, grad_out):
        grad_out = grad_out.freeze()
        data = grad_out.data
        if not grad_out.freezed:
            print("[WARNING]: grad_out is not freezed in SignalStatsObserver")
            return
        self.logs[layer_id]["grad_norm"].append(np.linalg.norm(data))
        self.logs[layer_id]["grad_var"].append(data.var())

class SignalShapeObserver(LayerObserver):
    def __init__(self):
        self.name = "SignalShapeObserver"
        super().__init__(self.name)
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_pre(self, layer_id, x):
        shape = x.freeze().data.shape
        self.logs[layer_id]["input_shape"].append(shape)

    def on_forward_post(self, layer_id, layer_cache):
        out = layer_cache[1]['out'].freeze()
        if not out.freezed:
            print("[WARNING]: layer output is not freezed in SignalShapeObserver")
            return
        shape = out.data.shape
        self.logs[layer_id]["activation_shape"].append(shape)

    def on_backward_pre(self, layer_id, grad_out):
        grad_out = grad_out.freeze()
        if not grad_out.freezed:
            print("[WARNING]: grad_out is not freezed in SignalShapeObserver")
            return
        data = grad_out.data
        self.logs[layer_id]["grad_shape"].append(data.shape)

    def on_backward_post(self, layer_id, layer_cache):
        grad_in = layer_cache[2]['grad_in'].freeze()
        if not grad_in.freezed:
            print("[WARNING]: grad_in is not freezed in SignalShapeObserver")
            return
        shape = grad_in.data.shape
        self.logs[layer_id]["grad_in_shape"].append(shape)

class ResidualEnergyObserver(LayerObserver):
    def __init__(self):
        self.name = "ResidualEnergyObserver"
        super().__init__(self.name)
        # logs[layer_id][metric] -> list of values
        self.logs = defaultdict(lambda: defaultdict(list))

    def on_forward_post(self, layer_id, layer_cache):
        f = layer_cache[3]['residual'].freeze()
        if not f.freezed:
            print("[WARNING]: residual is not freezed in ResidualEnergyObserver")
            return
        self.logs[layer_id]["residual"].append(np.mean((f**2).data))
        
        s = layer_cache[3]['shortcut'].freeze()
        if not s.freezed:
            print("[WARNING]: shortcut is not freezed in ResidualEnergyObserver")
            return
        self.logs[layer_id]["shortcut"].append(np.mean((s**2).data))