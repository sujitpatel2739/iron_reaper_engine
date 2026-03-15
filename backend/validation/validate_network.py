from typing import Optional
from api.schemas import ValidationWarning

def validate_network(body: dict) -> Optional[list]:
    """Walk the graph, infer shapes, return per-node warnings."""
    nodes_by_id = {n["id"]: n for n in body.get("nodes", [])}
    edges       = body.get("edges", [])
    warnings    = []
    in_shapes   = {}
    print('body: ', body)
    
    targets = {e["target"] for e in edges}
    queue   = [n for n in nodes_by_id.values() if n["id"] not in targets]
    for r in queue:
        in_shapes[r["id"]] = None

    visited = set()
    while queue:
        node = queue.pop(0)
        nid  = node["id"]
        if nid in visited:
            continue
        visited.add(nid)

        data      = node.get("data", {})
        ntype     = data.get("layerType") or data.get("nodeType", "")
        config    = data.get("config", {})
        in_shape  = in_shapes.get(nid)
        out_shape = _infer_out_shape(ntype, config, in_shape)

        for e in edges:
            if e["target"] != nid:
                continue
            src = nodes_by_id.get(e["source"])
            if src is None:
                continue
            sd   = src.get("data", {})
            st   = sd.get("layerType") or sd.get("nodeType", "")
            sc   = sd.get("config", {})
            warn = _compat_warning(st, sc, ntype, config, in_shapes.get(src["id"]))
            if warn:
                warnings.append(ValidationWarning(node_id=nid, message=warn))

        for e in edges:
            if e["source"] != nid:
                continue
            tgt = nodes_by_id.get(e["target"])
            if tgt:
                in_shapes[tgt["id"]] = out_shape
                queue.append(tgt)

def _compat_warning(
    src_type: str, src_cfg: dict,
    tgt_type: str, tgt_cfg: dict,
    src_in_shape: Optional[list],
) -> Optional[str]:
    out = _infer_out_shape(src_type, src_cfg, src_in_shape)
    if out is None:
        return None
    if tgt_type == "Linear":
        exp = tgt_cfg.get("in_features")
        if exp and out[-1] != int(exp):
            return (f"Shape mismatch: {src_type} outputs {out} "
                    f"but {tgt_type} expects in_features={exp}")
    if tgt_type == "LayerNorm":
        exp = tgt_cfg.get("in_features")
        if exp and out[-1] != int(exp):
            return (f"Shape mismatch: {src_type} outputs {out} "
                    f"but LayerNorm expects in_features={exp}")
    return None

def _infer_out_shape(node_type: str, config: dict, in_shape: Optional[list]) -> Optional[list]:
    """Best-effort single-layer output shape inference."""
    if in_shape is None:
        return None
    try:
        if node_type in ("Linear",):
            return [in_shape[0], int(config.get("out_features", 64))]
        if node_type in ("Relu", "LayerNorm", "AddNode", "MulNode",
                          "SubNode", "SqNode", "NegNode", "SqrtNode",
                          "ScaleNode", "ClipNode"):
            return in_shape
        if node_type == "SplitNode":
            n    = int(config.get("n_splits", 2))
            axis = int(config.get("axis", 1))
            if in_shape[axis] % n != 0:
                return None
            out = list(in_shape)
            out[axis] = out[axis] // n
            return out
    except Exception:
        return None
    return None