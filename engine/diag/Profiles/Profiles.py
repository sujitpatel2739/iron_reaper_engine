import numpy as np

class InterpreterProfile:
    def __init__(self, name):
        self.name = name
        self.observer_name = None
        
    def __call__(self, observers):
        return self._execute(observers)
    
    def _execute(self, observers):
        # Example interpretation logic for this profile
        print(f"Executing interpreter profile: {self.name}")
        for observer in observers:
            observer_name = observer.__class__.__name__
            print(f"  Observer: {observer_name}")


class SignalStateProfile(InterpreterProfile):
    def __init__(self, name):
        super().__init__(name)
        self.observer_name = "SignalStatsObserver"
        self.layers_variance = []
        self.layers_grad_norm = []
        
    def forwardVariance(self, observer):
        print(f"Interpreting forward variance metrics:")
        if hasattr(observer, 'logs'):
            for layer_id, metrics in observer.logs.items():
                if 'activation_var' in metrics:
                    mean_variance = np.mean(metrics['activation_var'])
                    self.layers_variance.append((layer_id, mean_variance))
                    print(f"  Layer ID {layer_id}: Mean Variance = {mean_variance}")
                    
    def backwardGradientNorm(self, observer):
        print(f"Interpreting backward gradient norm metrics:")
        if hasattr(observer, 'logs'):
            for layer_id, metrics in observer.logs.items():
                if 'grad_norm' in metrics:
                    mean_grad_norm = np.mean(metrics['grad_norm'])
                    self.layers_grad_norm.append((layer_id, mean_grad_norm))
                    print(f"  Layer ID {layer_id}: Mean Gradient Norm = {mean_grad_norm}")