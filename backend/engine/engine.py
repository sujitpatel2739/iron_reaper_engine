"""
step_engine.py
--------------
"""

import numpy as np
from typing import Dict, Generator, List, Optional

from backend.registries.network_registry import NetworkBuild
from ironframe.ironframe import Tensor
from engine import collector
from engine.events import StepEvent, EventKind
from builder.builder import StepKind, ExecutionStep


class Engine:
    """
    Drives an Executionnetwork_build.network forward and backward, yielding one StepEvent
    per layer at each step.

    Usage
    -----
        network_build.network    = network_build.networkner.build_network_build.network(graph_nodes, graph_edges, observers)
        engine  = StepEngine(network_build.network)
        engine.set_input(x_tensor)

        for event in engine.step_forward():
            send_to_client(event.to_dict())
            await client_next_signal()

        for event in engine.step_backward():
            send_to_client(event.to_dict())
            await client_prev_signal()
    """

    def __init__(self, network_build: NetworkBuild, dataset: Dict[str, Tensor]):
        
        self._dataset = dataset
        self.network_build = network_build

    def _pick_input(self) -> Tensor:
        """Select the input tensor from the dataset (currently: first entry)."""
        if not self._dataset:
            raise ValueError("No dataset provided.")
        t = next(iter(self._dataset.values()))
        t.requires_grad = True
        return t

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self):
        """
        Yield one StepEvent per ExecutionStep in forward order.
        Caller advances the generator by calling next().
        """
        input = self._pick_input()
        if input is None:
            raise ValueError('Input not found.')

        current = self._pick_input()
        
        steps = self.network_build.network.steps
        
        for idx, step in enumerate(steps):
            next_id = steps[idx + 1].component_id if idx + 1 < len(steps) else None

            if step.kind == StepKind.LAYER:
                current, event = self._run_layer_forward(step, current, next_id)
                yield event
                if event.kind == EventKind.ERROR:
                    return

            elif step.kind == StepKind.BRANCH_POINT:
                yield StepEvent(
                    kind=EventKind.BRANCH_POINT,
                    layer_id=step.component_id,
                    branches=step.branch_ids,
                )
 
                branch_outputs: List[Tensor] = []
                for b_idx, branch in enumerate(step.branches):
                    b_input = current
                    for b_step in branch:
                        b_input, b_event = self._run_layer_forward(b_step, b_input)
                        yield b_event
                        if b_event.kind == EventKind.ERROR:
                            return

                    branch_outputs.append(b_input)
                    yield StepEvent(kind=EventKind.BRANCH_DONE, branch=step.branch_ids[b_idx])

                # Merge branch outputs via element-wise addition
                # current = branch_outputs[0]
                # for extra in branch_outputs[1:]:
                #     current = current + extra

                yield StepEvent(kind=EventKind.BRANCHES_COMPLETE, layer_id=step.component_id)

            elif step.kind == StepKind.MERGE:
                current, event = self._run_layer_forward(step, current, next_id)
                yield event
                if event.kind == EventKind.ERROR:
                    return

        self._output = current
        yield StepEvent(kind=EventKind.FORWARD_COMPLETE)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self) -> Generator[StepEvent, None, None]:
        """
        Yield one StepEvent per ExecutionStep in reverse order.
        Caller advances the generator by calling next().
        """
        if self._output is None:
            yield StepEvent(kind=EventKind.ERROR, error="Forward pass not completed.")
            return

        grad    = Tensor(np.ones_like(self._output.data))
        steps   = self.network_build.network.steps

        for step in reversed(steps):
            if step.kind == StepKind.LAYER:
                grad, event = self._run_layer_backward(step, grad)
                yield event
                if event.kind == EventKind.ERROR:
                    return

            elif step.kind == StepKind.BRANCH_POINT:
                yield StepEvent(
                    kind=EventKind.BRANCH_POINT,
                    layer_id=step.component_id,
                    branches=step.branch_ids,
                )

                for b_idx, branch in enumerate(reversed(step.branches)):
                    b_grad = grad
                    for b_step in reversed(branch):
                        b_grad, b_event = self._run_layer_backward(b_step, b_grad)
                        yield b_event
                        if b_event.kind == EventKind.ERROR:
                            return

                    b_id = step.branch_ids[len(step.branches) - 1 - b_idx]
                    yield StepEvent(kind=EventKind.BRANCH_DONE, branch=b_id)

                yield StepEvent(kind=EventKind.BRANCHES_COMPLETE, layer_id=step.component_id)

            elif step.kind == StepKind.MERGE:
                grad, event = self._run_layer_backward(step, grad)
                yield event
                if event.kind == EventKind.ERROR:
                    return

        yield StepEvent(kind=EventKind.BACKWARD_COMPLETE)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_layer_forward(
        self,
        step:    ExecutionStep,
        x:       Tensor,
        next_id: Optional[str] = None,
    ):
        try:
            if isinstance(step.comp, WrappedLayer):
                out = step.comp.forward(x)
            else:
                out = step.comp.forward(x)   # raw Node
        except Exception as e:
            return x, StepEvent(kind=EventKind.ERROR, layer_id=step.component_id, error=str(e))

        layer_id_int = self.network_build.network.id_to_int.get(step.component_id, 0)
        metrics      = collector.collect(layer_id_int)

        return out, StepEvent(
            kind=EventKind.STEP_DONE,
            layer_id=step.component_id,
            metrics=metrics,
            next_layer_id=next_id,
        )

    def _run_layer_backward(
        self,
        step: ExecutionStep,
        grad: Tensor,
    ):
        if not isinstance(step.comp, WrappedLayer):
            # Raw nodes don't participate in the backward observer pipeline
            return grad, StepEvent(kind=EventKind.STEP_DONE, layer_id=step.component_id, metrics={})

        try:
            grad_in = step.comp.backward(grad)
        except Exception as e:
            return grad, StepEvent(kind=EventKind.ERROR, layer_id=step.component_id, error=str(e))

        layer_id_int = self.network_build.network.id_to_int.get(step.component_id, 0)
        metrics      = collector.collect(layer_id_int)

        return grad_in, StepEvent(
            kind=EventKind.STEP_DONE,
            layer_id=step.component_id,
            metrics=metrics,
        )