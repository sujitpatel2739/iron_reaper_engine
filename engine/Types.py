class ExecutionContext:
    def __init__(
        self,
        *,
        layer_id: int,
        layer_type: str,
        phase: str,              # "forward" | "backward"
        inputs: dict,            # named tensors
        outputs: dict,           # named tensors
        grads: dict,             # named gradients
        paths: dict,             # named signal paths
        metadata: dict           # free-form, engine-provided
    ):
        pass
