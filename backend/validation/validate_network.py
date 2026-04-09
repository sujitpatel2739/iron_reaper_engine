"""
validation/validate_network.py
--------------------------------
Shape compatibility validation for a React Flow graph.

Walk the graph in topological order, infer the output shape at each node,
and emit a warning whenever a node's expected input does not match
what its predecessor actually outputs.

Bugs fixed from previous version
----------------------------------
1. No return statement — always returned None.
2. Root nodes seeded with None so shape propagation was always blind.
3. _compat_warning received in_shapes[src] (src INPUT) not src OUTPUT.
4. out_shape was computed but never stored for downstream propagation.
"""

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_network(body: dict, input_shape: Optional[List[int]] = None) -> List[dict]:
    """
    Walk the graph, infer shapes, return a list of warning dicts.

    Parameters
    ----------
    body         : { "nodes": [...], "edges": [...] }
    input_shape  : seed shape for root nodes, e.g. [32, 128].
                   When None, roots start with an unknown shape and only
                   mismatches that can be inferred structurally are emitted.

    Returns
    -------
    List of dicts: [{ "node_id": str, "message": str }, ...]
    Empty list means the graph passed all checks.
    """
    nodes_by_id: Dict[str, dict] = {n["id"]: n for n in body.get("nodes", [])}
    edges:       List[dict]      = body.get("edges", [])
    warnings:    List[dict]      = []

    # out_shapes[node_id] = output shape that node produced after forward
    out_shapes:  Dict[str, Optional[List[int]]] = {}
    # in_shapes[node_id]  = input shape arriving into that node
    in_shapes:   Dict[str, Optional[List[int]]] = {} 

    # ------------------------------------------------------------------
    # Seed root nodes (no incoming edges) with the provided input_shape
    # ------------------------------------------------------------------
    targets = {e["target"] for e in edges}
    for n in nodes_by_id.values():
        if n["id"] not in targets:
            in_shapes[n["id"]] = input_shape

    # ------------------------------------------------------------------
    # Topological walk via in-degree tracking (Kahn's algorithm)
    # ------------------------------------------------------------------
    in_degree: Dict[str, int] = {nid: 0 for nid in nodes_by_id}
    for e in edges:
        in_degree[e["target"]] += 1

    queue   = [n for n in nodes_by_id.values() if in_degree[n["id"]] == 0]
    visited: set = set()

    while queue:
        node = queue.pop(0)
        nid  = node["id"]
        if nid in visited: 
            continue
        visited.add(nid)

        data     = node.get("data", {})
        ntype    = data.get("layerType") or data.get("nodeType", "")
        config   = data.get("config", {})
        in_shape = in_shapes.get(nid)

        # Compute and store this node's output shape
        out_shape        = _infer_out_shape(ntype, config, in_shape)
        out_shapes[nid]  = out_shape

        # ------------------------------------------------------------------
        # For each edge arriving at this node, check predecessor compatibility
        # using the predecessor's stored OUTPUT shape
        # ------------------------------------------------------------------
        for e in edges:
            if e["target"] != nid:
                continue
            src_id = e["source"]
            src    = nodes_by_id.get(src_id)
            if src is None:
                continue

            src_data   = src.get("data", {})
            src_type   = src_data.get("layerType") or src_data.get("nodeType", "")
            src_config = src_data.get("config", {})

            # Key fix: use out_shapes[src_id], not in_shapes[src_id]
            src_out = out_shapes.get(src_id)

            warn = _compat_warning(src_type, src_config, ntype, config, src_out)
            if warn:
                warnings.append({"node_id": nid, "message": warn})

        # ------------------------------------------------------------------
        # Propagate this node's output shape to all successors
        # ------------------------------------------------------------------
        for e in edges:
            if e["source"] != nid:
                continue
            tgt_id = e["target"]
            tgt    = nodes_by_id.get(tgt_id)
            if tgt is None:
                continue

            # For merge nodes receiving multiple inputs, accept first non-None
            if in_shapes.get(tgt_id) is None:
                in_shapes[tgt_id] = out_shape

            in_degree[tgt_id] -= 1
            if in_degree[tgt_id] == 0:
                queue.append(tgt)

    return warnings   # Bug 1 fix: was missing entirely


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def _compat_warning(
    src_type:  str,
    src_cfg:   dict,
    tgt_type:  str,
    tgt_cfg:   dict,
    src_out:   Optional[List[int]],   # src node's OUTPUT shape
) -> Optional[str]:
    """
    Return a warning string if src's output is incompatible with tgt's
    expected input, or None if compatible / shape is unknown.
    """
    if src_out is None:
        return None

    last_dim = src_out[-1]

    if tgt_type == "Linear":
        exp = tgt_cfg.get("in_features")
        if exp is not None and last_dim != int(exp):
            return (
                f"Shape mismatch: {src_type} outputs {src_out} "
                f"but Linear expects in_features={exp}"
            )

    if tgt_type == "LayerNorm":
        exp = tgt_cfg.get("in_features")
        if exp is not None and last_dim != int(exp):
            return (
                f"Shape mismatch: {src_type} outputs {src_out} "
                f"but LayerNorm expects in_features={exp}"
            )

    return None


# ---------------------------------------------------------------------------
# Output shape inference
# ---------------------------------------------------------------------------

def _infer_out_shape(
    node_type: str,
    config:    dict,
    in_shape:  Optional[List[int]],
) -> Optional[List[int]]:
    """
    Best-effort output shape for a single node given its input shape.
    Returns None if undeterminable.
    """
    if in_shape is None:
        return None

    try:
        if node_type == "Linear":
            return [*in_shape[:-1], int(config.get("out_features", 64))]

        if node_type in (
            "Relu", "LayerNorm",
            "AddNode", "MulNode", "SubNode", "DivNode",
            "SqNode", "NegNode", "SqrtNode",
            "ScaleNode", "ClipNode",
        ):
            return list(in_shape)

        if node_type == "SplitNode":
            n    = int(config.get("n_splits", 2))
            axis = int(config.get("axis", 1))
            if axis >= len(in_shape) or in_shape[axis] % n != 0:
                return None
            out       = list(in_shape)
            out[axis] = in_shape[axis] // n
            return out

        # ConcatNode needs all inputs to compute output — unknowable from one
        if node_type == "ConcatNode":
            return None

    except Exception:
        return None

    return None