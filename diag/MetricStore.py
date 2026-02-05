import numpy as np

class MetricStore:
    def __init__(self):
        self.metrics = {
            # run: {
                # layer_id: {
                    #     metric_name: [values]
                # }
            # }
        }
        
    
    def add_metric(self, run, layer_id, metric_name, values):
        if run not in self.metrics:
            self.metrics[run] = {}
        if layer_id not in self.metrics[run]:
            self.metrics[run][layer_id] = {}
        if metric_name not in self.metrics[run][layer_id]:
            self.metrics[run][layer_id][metric_name] = []
        
        self.metrics[run][layer_id][metric_name].extend(values)
        
    def get_metric(self, run, layer_id, metric_name):
        """
        Returns:
        {layer_id: [values]}
        """
        if run in self.metrics:
            if layer_id in self.metrics[run] and metric_name in self.metrics[run][layer_id]:
                return self.metrics[run][layer_id][metric_name]
        return None
    
    def get_layer_sequence(self, run, name, agg="none"):
        """
        Returns:
        {layer_id: values
        ...}
        """
        if run not in self.metrics:
            return {}
        result = {}
        for l_id, layer_metrics in self.metrics[run].items():
            if name in layer_metrics:
                values = layer_metrics[name]
                if agg == "mean":
                    agg_value = np.mean(values)
                elif agg == "max": 
                    agg_value = np.max(values)
                elif agg == "min":
                    agg_value = np.min(values)
                elif agg == "none":
                    agg_value = values
                else:
                    raise ValueError(f"Unsupported aggregation method: {agg}")
                result[l_id] = agg_value
        return result
    
    def get_all_metrics(self, run):
        """
        Returns:
        {layer_id: {metric_name: [values]}}
        """
        return self.metrics.get(run, {})