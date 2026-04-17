"""
planner.py
----------
Converts a React Flow graph (nodes + edges) into an ordered
ExecutionPlan — a list of ExecutionStep objects.

Responsibilities
----------------
- Topological sort of the graph
- Detection of fork nodes  (out-degree > 1)
- Detection of merge nodes (in-degree  > 1)
- Collection of per-branch step sequences
- Wrapping each Layer in a WrappedLayer with the supplied observers

This module knows nothing about how steps are executed.
It produces a plan and stops there.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from neural_network.Layers.Layer import Layer as BaseLayer
from neural_network.Layers.WrappedLayer import WrappedLayer
from backend.builder.component_builder import build_component


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------

class StepKind(Enum):
    LAYER        = auto()   # single layer / node, one input → one output
    BRANCH_POINT = auto()   # fork: one input → multiple parallel sub-sequences
    MERGE        = auto()   # merge node: multiple inputs → one output


@dataclass
class ExecutionStep:
    kind:     StepKind
    component_id: Optional[str] = None
    obj: Any = None
    branches: List[List['ExecutionStep']] = field(default_factory=list)
    # branch_ids[i] is the human-readable id sent to the frontend for branches[i]
    branch_ids: List[str]                = field(default_factory=list)


@dataclass
class ExecutionPlan:
    steps:       List[ExecutionStep]
    component_order:  List[str]
    id_to_int:   Dict[str, int]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_network(
    graph_nodes: List[dict],
    graph_edges: List[dict]
) -> ExecutionPlan:
    """
    Build and return an ExecutionPlan from the React Flow graph.

    Parameters
    ----------
    graph_nodes : list of React Flow node dicts
    graph_edges : list of React Flow edge dicts
    observers   : already-instantiated observer list (from registry.build_observers)
    """
    component_by_id  = {n["id"]: n for n in graph_nodes}
    out_edges, in_edges = _build_adjacency(graph_edges)
    order        = _topo_sort(graph_nodes, graph_edges)
    id_to_int    = {cid: i for i, cid in enumerate(order)}

    steps = _build_steps(order, component_by_id, out_edges, in_edges, id_to_int)
    return ExecutionPlan(steps=steps, component_order=order, id_to_int=id_to_int)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_adjacency(edges: List[dict]):
    out_edges: Dict[str, List[str]] = {}
    in_edges:  Dict[str, List[str]] = {}
    for e in edges:
        out_edges.setdefault(e["source"], []).append(e["target"])
        in_edges.setdefault(e["target"],  []).append(e["source"])
    return out_edges, in_edges


def _topo_sort(nodes: List[dict], edges: List[dict]) -> List[str]:
    """Kahn's algorithm — stable, respects insertion order."""
    in_degree = {n["id"]: 0 for n in nodes}
    adj: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        adj[e["source"]].append(e["target"])
        in_degree[e["target"]] += 1

    queue = [cid for cid, d in in_degree.items() if d == 0]
    order: List[str] = []
    while queue:
        cid = queue.pop(0)
        order.append(cid)
        for tgt in adj.get(cid, []):
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)
    return order


def _collect_branch(
    start_id:    str,
    out_edges:   Dict[str, List[str]],
    in_edges:    Dict[str, List[str]],
    component_by_id: Dict[str, dict],
    id_to_int:   Dict[str, int],
) -> List[ExecutionStep]:
    """
    Walk a single branch from start_id until we hit a merge node
    (in-degree > 1) or a dead end.
    """
    branch: List[ExecutionStep] = []
    curr = start_id
    while curr:
        node = component_by_id[curr]
        obj  = build_component(node, id_to_int[curr])
        branch.append(ExecutionStep(kind=StepKind.LAYER, component_id=curr, obj=obj))

        # Continue only through nodes that are NOT merge points
        nexts = [t for t in out_edges.get(curr, [])
                 if len(in_edges.get(t, [])) == 1]
        curr = nexts[0] if len(nexts) == 1 else None
    return branch


def _build_steps(
    order:       List[str],
    component_by_id: Dict[str, dict],
    out_edges:   Dict[str, List[str]],
    in_edges:    Dict[str, List[str]],
    id_to_int:   Dict[str, int],
) -> List[ExecutionStep]:
    """
    Walk the topological order and produce a flat list of ExecutionSteps,
    with BRANCH_POINT steps replacing the fork node and all its branches.
    """
    steps:   List[ExecutionStep] = []
    visited: set = set()

    i = 0
    while i < len(order):
        cid = order[i]
        if cid in visited:
            i += 1
            continue
        visited.add(cid)

        n_out = len(out_edges.get(cid, []))
        n_in  = len(in_edges.get(cid,  []))

        if n_out > 1:
            # Fork node — build one branch sequence per outgoing edge
            branch_seqs  = []
            branch_ids   = []
            for b_idx, target_id in enumerate(out_edges[cid]):
                b_seq = _collect_branch(
                    target_id, out_edges, in_edges,
                    component_by_id, id_to_int
                )
                branch_seqs.append(b_seq)
                branch_ids.append(f"branch_{b_idx}")
                # Mark all nodes in this branch as visited
                for bs in b_seq:
                    visited.add(bs.component_id)

            steps.append(ExecutionStep(
                kind=StepKind.BRANCH_POINT,
                component_id=cid,
                branches=branch_seqs,
                branch_ids=branch_ids,
            ))

        elif n_in > 1:
            # Merge node
            obj = build_component(component_by_id[cid]["data"], id_to_int[cid])
            steps.append(ExecutionStep(kind=StepKind.MERGE, component_id=cid, obj=obj))

        else:
            # Ordinary sequential node
            obj = build_component(component_by_id[cid]["data"], id_to_int[cid])
            steps.append(ExecutionStep(kind=StepKind.LAYER, component_id=cid, obj=obj))

        i += 1

    return steps