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
from execution.registry import build_layer


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
    node_id:  Optional[str]              = None
    obj:      Any                        = None   # WrappedLayer or raw Node
    branches: List[List['ExecutionStep']] = field(default_factory=list)
    # branch_ids[i] is the human-readable id sent to the frontend for branches[i]
    branch_ids: List[str]                = field(default_factory=list)


@dataclass
class ExecutionPlan:
    steps:       List[ExecutionStep]
    node_order:  List[str]               # all node ids in topo order
    id_to_int:   Dict[str, int]          # node_id → integer layer id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_plan(
    graph_nodes: List[dict],
    graph_edges: List[dict],
    observers:   list,
) -> ExecutionPlan:
    """
    Build and return an ExecutionPlan from the React Flow graph.

    Parameters
    ----------
    graph_nodes : list of React Flow node dicts
    graph_edges : list of React Flow edge dicts
    observers   : already-instantiated observer list (from registry.build_observers)
    """
    nodes_by_id  = {n["id"]: n for n in graph_nodes}
    out_edges, in_edges = _build_adjacency(graph_edges)
    order        = _topo_sort(graph_nodes, graph_edges)
    id_to_int    = {nid: i for i, nid in enumerate(order)}

    steps = _build_steps(order, nodes_by_id, out_edges, in_edges, id_to_int, observers)
    return ExecutionPlan(steps=steps, node_order=order, id_to_int=id_to_int)


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

    queue = [nid for nid, d in in_degree.items() if d == 0]
    order: List[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for tgt in adj.get(nid, []):
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)
    return order


def _make_step_obj(node: dict, layer_id: int, observers: list) -> Any:
    """Build a WrappedLayer (for Layers) or raw Node, from a graph node."""
    obj = build_layer(node["data"], layer_id)
    if isinstance(obj, BaseLayer):
        return WrappedLayer(obj, observers)
    return obj   # raw Node — not wrapped, not observable


def _collect_branch(
    start_id:    str,
    out_edges:   Dict[str, List[str]],
    in_edges:    Dict[str, List[str]],
    nodes_by_id: Dict[str, dict],
    id_to_int:   Dict[str, int],
    observers:   list,
) -> List[ExecutionStep]:
    """
    Walk a single branch from start_id until we hit a merge node
    (in-degree > 1) or a dead end.
    """
    branch: List[ExecutionStep] = []
    curr = start_id
    while curr:
        node = nodes_by_id[curr]
        obj  = _make_step_obj(node, id_to_int[curr], observers)
        branch.append(ExecutionStep(kind=StepKind.LAYER, node_id=curr, obj=obj))

        # Continue only through nodes that are NOT merge points
        nexts = [t for t in out_edges.get(curr, [])
                 if len(in_edges.get(t, [])) == 1]
        curr = nexts[0] if len(nexts) == 1 else None
    return branch


def _build_steps(
    order:       List[str],
    nodes_by_id: Dict[str, dict],
    out_edges:   Dict[str, List[str]],
    in_edges:    Dict[str, List[str]],
    id_to_int:   Dict[str, int],
    observers:   list,
) -> List[ExecutionStep]:
    """
    Walk the topological order and produce a flat list of ExecutionSteps,
    with BRANCH_POINT steps replacing the fork node and all its branches.
    """
    steps:   List[ExecutionStep] = []
    visited: set = set()

    i = 0
    while i < len(order):
        nid = order[i]
        if nid in visited:
            i += 1
            continue
        visited.add(nid)

        n_out = len(out_edges.get(nid, []))
        n_in  = len(in_edges.get(nid,  []))

        if n_out > 1:
            # Fork node — build one branch sequence per outgoing edge
            branch_seqs  = []
            branch_ids   = []
            for b_idx, target_id in enumerate(out_edges[nid]):
                b_seq = _collect_branch(
                    target_id, out_edges, in_edges,
                    nodes_by_id, id_to_int, observers,
                )
                branch_seqs.append(b_seq)
                branch_ids.append(f"branch_{b_idx}")
                # Mark all nodes in this branch as visited
                for bs in b_seq:
                    visited.add(bs.node_id)

            steps.append(ExecutionStep(
                kind=StepKind.BRANCH_POINT,
                node_id=nid,
                branches=branch_seqs,
                branch_ids=branch_ids,
            ))

        elif n_in > 1:
            # Merge node
            obj = _make_step_obj(nodes_by_id[nid], id_to_int[nid], observers)
            steps.append(ExecutionStep(kind=StepKind.MERGE, node_id=nid, obj=obj))

        else:
            # Ordinary sequential node
            obj = _make_step_obj(nodes_by_id[nid], id_to_int[nid], observers)
            steps.append(ExecutionStep(kind=StepKind.LAYER, node_id=nid, obj=obj))

        i += 1

    return steps