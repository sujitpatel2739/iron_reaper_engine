"""
LoopNode.py
-----------
LoopNode and its two probe sentinels — LoopStartProbe and LoopEndProbe.

Design
------
LoopNode is pure backend — state, counters, jump logic, and finalization.
It never touches body layers directly. The engine owns all layer traversal.

LoopStartProbe and LoopEndProbe are frontend sentinels placed by the user
in the network. They hold edges to the first and last body elements
respectively, and delegate all loop logic to the shared LoopNode.

Everything between the probes is completely unaware it is in a loop.
Body layers and nodes work exactly as normal — they just receive an input
and produce an output. The probes handle the jump.

Usage
-----
    loop = LoopNode(node_id=0, n_iterations=5)

    # Build network
    network = [
        layer1,
        layer2,
        loop.start_probe,   # node_id + 1
        layer3,
        layer4,
        loop.end_probe,     # node_id + 2
        layer5,
    ]

    # Wire probe edges to body boundaries — done after building network
    loop.start_probe.first_element = layer3   # first body element
    loop.end_probe.last_element    = layer4   # last body element

ID allocation
-------------
LoopNode takes node_id N and allocates:
    N+1  →  LoopStartProbe
    N+2  →  LoopEndProbe
The user must leave N, N+1, N+2 free in their ID space.

Forward flow
------------
1. Engine calls start_probe(x)
       → on_start() fires once — initializes forward state
       → x passes through to engine unchanged

2. Engine calls layer3(x), layer4(x) — body runs, engine drives traversal

3. Engine calls end_probe(x)
       → saves x to intermediates
       → if should_continue():
             calls loop_node.forward(x)
                 → increments iteration_count
                 → pushes x into first_element (layer3)
                 → engine picks up and runs body again naturally
                 → eventually end_probe receives the new output and repeats
       → if not should_continue():
             calls on_end('forward', x) → returns final output to engine

Backward flow
-------------
1. Engine calls end_probe.backward(grad)
       → on_start_backward() fires once — initializes backward state
       → grad passes through to engine unchanged

2. Engine calls layer4.backward, layer3.backward — engine drives traversal

3. Engine calls start_probe.backward(grad)
       → if should_continue_backward():
             calls loop_node.backward(grad)
                 → increments backward_iteration_count
                 → pushes grad into last_element (layer4)
                 → engine picks up and walks body backward again naturally
                 → eventually start_probe receives new grad and repeats
       → if not should_continue_backward():
             calls on_end('backward', grad) → returns final grad to engine

return_all
----------
If return_all=True, end_probe._forward returns a list of all post-body
outputs (one per completed iteration) instead of just the final one.
The caller is responsible for handling a list output in this case.
"""

import numpy as np
from typing import Any, List, Optional

from ironframe.ironframe import Tensor
from Nodes.Node import Node
from cache.CacheStore import SLOT_INPUT, SLOT_OUTPUT, SLOT_GRAD_OUT, SLOT_GRAD_IN
from Logger import Logger


# ---------------------------------------------------------------------------
# LoopStartProbe
# ---------------------------------------------------------------------------

class LoopStartProbe(Node):
    """
    Sentinel placed before the first element of the loop body.

    Forward  — activates loop once via on_start(), then passes x through
               unchanged. Engine handles body traversal from here.

    Backward — receives grad after engine has walked backward through the
               body. Decides: jump back into body for another backward
               iteration, or finalize via on_end('backward').

    Attributes
    ----------
    first_element : any callable
        Set by the user after network construction.
        Must be the first layer or node in the loop body.
        LoopNode.forward() pushes input into this to start each iteration.
    """

    def __init__(self, node_id: int, loop_node: "LoopNode", name: str = ""):
        super().__init__(node_id, name or f"loop_start_{node_id}")
        self.loop_node     = loop_node
        # self.first_element = None   # user sets: loop.start_probe.first_element = layer3

    def _forward(self, *inputs) -> Any:
        self._require_n_inputs(inputs, 1)
        x = inputs[0]

        # Activate loop once — guard prevents re-activation on jump-backs
        if not self.loop_node._state['active']:
            self.loop_node.on_start()

        # Pure pass-through — engine handles body traversal from here
        return x

    def _backward(self, grad: Any) -> List:
        """
        Receives grad after the engine has walked backward through the body
        for one full iteration. Decides: another backward iteration or exit.
        """
        if self.loop_node.should_continue_backward():
            # Jump — push grad back into last body element
            # Engine picks up and walks body backward again naturally
            result = self.loop_node.backward(grad)
            return [result]
        else:
            # All backward iterations done — finalize
            final_grad = self.loop_node.on_end('backward', grad)
            return [final_grad]


# ---------------------------------------------------------------------------
# LoopEndProbe
# ---------------------------------------------------------------------------

class LoopEndProbe(Node):
    """
    Sentinel placed after the last element of the loop body.

    Forward  — saves post-body output, then either jumps back into body
               via loop_node.forward() or finalizes via on_end('forward').

    Backward — activates backward loop once via on_start_backward(), then
               passes grad through unchanged. The actual backward jump
               decision lives in LoopStartProbe._backward.

    Attributes
    ----------
    last_element : any callable with .backward()
        Set by the user after network construction.
        Must be the last layer or node in the loop body.
        LoopNode.backward() pushes grad into this to start each
        backward iteration.
    """

    def __init__(self, node_id: int, loop_node: "LoopNode", name: str = ""):
        super().__init__(node_id, name or f"loop_end_{node_id}")
        self.loop_node    = loop_node
        self.last_element = None   # user sets: loop.end_probe.last_element = layer4

    def _forward(self, *inputs) -> Any:
        """
        Receives post-body output from engine.
        Saves it, then decides: continue looping or exit.
        """
        self._require_n_inputs(inputs, 1)
        x = inputs[0]

        # Save post-body output for diagnostics and return_all
        # self.loop_node._state['intermediates'].append(x)

        if self.loop_node.should_continue():
            # Jump — push x back into first body element
            # Returns the output after all remaining iterations complete
            return self.loop_node.forward(x)
        else:
            # All iterations done — finalize and return to engine
            return self.loop_node.on_end('forward', x)

    def _backward(self, grad: Any) -> List:
        """
        First backward call activates the backward loop.
        After that, pure pass-through — engine handles body backward traversal.
        LoopStartProbe._backward owns the jump decision.
        """
        # Activate backward loop exactly once
        if not self.loop_node._state['backward_active']:
            self.loop_node.on_start_backward()

        # Pure pass-through — engine walks body backward from here
        return [grad]


# ---------------------------------------------------------------------------
# LoopNode
# ---------------------------------------------------------------------------

class LoopNode(Node):
    """
    Backend brain of the loop. Owns all state and logic.
    Never touches body layers directly — the engine does that.

    Parameters
    ----------
    node_id      : int   — base ID. start_probe gets N+1, end_probe gets N+2.
    n_iterations : int   — how many times the body runs. Must be >= 1.
    max_iter_cap : int   — hard safety cap. Fires a Logger warning if hit.
                           Default 500.
    return_all   : bool  — if True, end_probe._forward returns list of all
                           post-body outputs instead of just the final one.
    name         : str
    """

    def __init__(
        self,
        node_id:      int,
        n_iterations: int,
        max_iter_cap: int  = 500,
        return_all:   bool = False,
        name:         str  = "",
    ):
        super().__init__(node_id, name or f"loop_{node_id}")

        if n_iterations < 1:
            raise ValueError(
                f"LoopNode '{self.name}': n_iterations must be >= 1, "
                f"got {n_iterations}."
            )

        if n_iterations > max_iter_cap:
            Logger.warning(
                f"LoopNode '{self.name}': n_iterations ({n_iterations}) exceeds "
                f"max_iter_cap ({max_iter_cap}). Clamping to {max_iter_cap}."
            )
            n_iterations = max_iter_cap

        self.n_iterations = n_iterations
        self.max_iter_cap = max_iter_cap
        self.return_all   = return_all

        # Probes — created by LoopNode, placed in network by user
        self.start_probe = LoopStartProbe(node_id + 1, loop_node=self)
        self.end_probe   = LoopEndProbe(node_id + 2,   loop_node=self)

        # Initialize clean state
        self._init_state()

    # -- State ---------------------------------------------------------------

    def _init_state(self) -> None:
        """Full state reset. Called in __init__ and at the start of each pass."""
        self._state.update({
            # forward
            'active':          False,
            'iteration_count': 0,
            'intermediates':   [],     # post-body outputs appended by end_probe
            'out':             None,

            # backward
            'backward_active':           False,
            'backward_iteration_count':  0,
            'grad_out':                  None,
        })

    # -- Forward lifecycle ---------------------------------------------------

    def on_start(self) -> None:
        """
        Called once by LoopStartProbe._forward when the loop first activates.
        Resets state for a fresh forward pass.
        Not called again for subsequent iterations — the active guard prevents it.
        """
        self._init_state()
        self._state['active'] = True
        Logger.debug(
            f"LoopNode '{self.name}': forward loop activated "
            f"({self.n_iterations} iterations)."
        )

    def forward(self, x: Any) -> Any:
        """
        Called by LoopEndProbe._forward when should_continue() is True.
        Increments the iteration counter and pushes x into the first body element.
        The engine picks up naturally from there and runs the full body.
        Eventually end_probe receives the body's output and calls this again
        or calls on_end() — the loop drives itself through this cycle.
        """
        self._state['iteration_count'] += 1

        Logger.debug(
            f"LoopNode '{self.name}': forward iteration "
            f"{self._state['iteration_count']} / {self.n_iterations}."
        )

        if self.start_probe.first_element is None:
            raise RuntimeError(
                f"LoopNode '{self.name}': start_probe.first_element is not set. "
                f"Wire it after building the network:\n"
                f"    loop.start_probe.first_element = <first body layer/node>"
            )

        # Push x into the first body element — engine handles the rest
        return self.start_probe.first_element(x)

    def should_continue(self) -> bool:
        """
        True if the forward loop should run another iteration.
        Hard cap logs a warning and forces exit if reached.
        """
        count = self._state['iteration_count']

        if count >= self.max_iter_cap:
            Logger.warning(
                f"LoopNode '{self.name}': max_iter_cap ({self.max_iter_cap}) "
                f"reached during forward. Forcing loop exit."
            )
            return False

        return count < self.n_iterations

    def on_end(self, direction: str, value: Any) -> Any:
        """
        Called when the loop finishes — either forward or backward direction.
        Cleans up active flags, saves final value, returns output.

        Parameters
        ----------
        direction : 'forward' or 'backward'
        value     : final output tensor (forward) or final grad tensor (backward)
        """
        if direction == 'forward':
            self._state['active'] = False
            self._state['out']    = value
            Logger.debug(
                f"LoopNode '{self.name}': forward loop complete — "
                f"{self._state['iteration_count']} iteration(s) ran."
            )
            # return_all: return every post-body output, not just the last
            if self.return_all:
                return self._state['intermediates']
            return value

        else:  # backward
            self._state['backward_active'] = False
            self._state['grad_out']        = value
            Logger.debug(
                f"LoopNode '{self.name}': backward loop complete — "
                f"{self._state['backward_iteration_count']} iteration(s) ran."
            )
            return value

    # -- Backward lifecycle --------------------------------------------------

    def on_start_backward(self) -> None:
        """
        Called once by LoopEndProbe._backward when backward first activates.
        Resets backward counters.
        Not called again for subsequent backward iterations.
        """
        self._state['backward_active']          = True
        self._state['backward_iteration_count'] = 0
        Logger.debug(
            f"LoopNode '{self.name}': backward loop activated "
            f"({self.n_iterations} iterations)."
        )

    def backward(self, grad: Any) -> Any:
        """
        Called by LoopStartProbe._backward when should_continue_backward() is True.
        Increments backward counter and pushes grad into the last body element.
        The engine picks up naturally from there and walks the body backward.
        Eventually start_probe receives the result and calls this again
        or calls on_end('backward') — the backward loop drives itself.
        """
        self._state['backward_iteration_count'] += 1

        Logger.debug(
            f"LoopNode '{self.name}': backward iteration "
            f"{self._state['backward_iteration_count']} / {self.n_iterations}."
        )

        if self.end_probe.last_element is None:
            raise RuntimeError(
                f"LoopNode '{self.name}': end_probe.last_element is not set. "
                f"Wire it after building the network:\n"
                f"    loop.end_probe.last_element = <last body layer/node>"
            )

        # Push grad into the last body element — engine walks backward from there
        return self.end_probe.last_element.backward(grad)

    def should_continue_backward(self) -> bool:
        """
        True if the backward loop should run another iteration.
        Hard cap logs a warning and forces exit if reached.
        """
        count = self._state['backward_iteration_count']

        if count >= self.max_iter_cap:
            Logger.warning(
                f"LoopNode '{self.name}': max_iter_cap ({self.max_iter_cap}) "
                f"reached during backward. Forcing loop exit."
            )
            return False

        return count < self.n_iterations

    # -- Node interface -- not used directly ---------------------------------

    def _forward(self, *inputs) -> Any:
        raise NotImplementedError(
            f"LoopNode '{self.name}': do not call _forward directly. "
            f"Place loop.start_probe and loop.end_probe in your network, "
            f"then set loop.start_probe.first_element and "
            f"loop.end_probe.last_element."
        )

    def _backward(self, grad: Any) -> Any:
        raise NotImplementedError(
            f"LoopNode '{self.name}': do not call _backward directly. "
            f"The probes handle backward automatically."
        )

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        fwd    = self._state.get('iteration_count', 0)
        bwd    = self._state.get('backward_iteration_count', 0)
        active = self._state.get('active', False)
        return (
            f"LoopNode("
            f"id={self.id}, "
            f"name='{self.name}', "
            f"n_iterations={self.n_iterations}, "
            f"fwd_count={fwd}, "
            f"bwd_count={bwd}, "
            f"active={active})"
        )
