import numpy as np

from diag.MetricStore import MetricStore

class InterpreterProfile:
    def __init__(self, name, run):
        self.name = name
        self.run = run
        
    def __call__(self, store):
        raise NotImplementedError("InterpreterProfile __call__ method must be implemented in subclasses.")


class SignalStatsProfile(InterpreterProfile):
    name = "signal_state"

    def __call__(self, store):
        activation_mean = store.get_layer_sequence(self.run, "activation_mean")
        activation_var = store.get_layer_sequence(self.run, "activation_var")
        grad_norm = store.get_layer_sequence(self.run, "grad_norm")
        grad_var = store.get_layer_sequence(self.run, "grad_var")
        
        return {
            'activation_mean': activation_mean,
            'activation_var': activation_var,
            'grad_norm': grad_norm,
            'grad_var': grad_var
        }

                    
class PathDominanceProfile(InterpreterProfile):
    name = "path_dominance"

    def __call__(self, store):
        residual = store.get_layer_sequence(self.run, "residual_energy")
        shortcut = store.get_layer_sequence(self.run, "shortcut_energy")

        residual = {}
        shortcut = {}
        for (l, r), (_, s) in zip(residual.items(), shortcut.items()):
            residual[l]= r / (r + s)
            shortcut[l]= s / (r + s)
            
        return {
            'residual': residual,
            'shortcut': shortcut
        }
