"""
api/runner.py
-------------
Orchestrates a full (non-step) diagnostic run end to end:

  1. Load PyTorch model from uploaded bytes
  2. Attach ModelBridge hooks → observer pipeline
  3. Build a synthetic or user-supplied input tensor
  4. Forward + backward pass
  5. Collect MetricStore data
  6. Run profiles
  7. Return a structured DiagnosticReport

The runner is stateless — each call is an independent run.
"""

import io
import numpy as np
import torch
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import cache.CacheStore as CacheStore
import diag.MetricStore as MetricStore
from diag.Profiles.Profiles import SignalStatsProfile, PathDominanceProfile
from execution.registry import build_observers

from api.bridge import load_model, graph_from_model, ModelBridge, total_param_count
from api.schemas import (
    RunConfig, DiagnosticReport,
    LayerInfo, LayerMetrics,
    SignalStatsResult, PathDominanceResult,
)


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_PROFILE_REGISTRY = {
    "SignalStatsProfile":    SignalStatsProfile,
    "PathDominanceProfile":  PathDominanceProfile,
}


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

def _make_torch_input(config: RunConfig) -> torch.Tensor:
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    return torch.randn(*config.input_shape, requires_grad=True)


def _load_input_file(file_bytes: bytes, config: RunConfig) -> torch.Tensor:
    """Load a .npy or .csv file as a torch Tensor."""
    buf = io.BytesIO(file_bytes)
    try:
        arr = np.load(buf)
    except Exception:
        buf.seek(0)
        arr = np.loadtxt(buf, delimiter=",")
    return torch.from_numpy(arr.astype(np.float32)).requires_grad_(True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics(
    file_bytes:  bytes,
    config:      RunConfig,
    input_bytes: Optional[bytes] = None,
) -> DiagnosticReport:
    """
    Full pipeline: load → hook → forward → backward → metrics → profiles → report.
    """
    warnings: List[str] = []

    # 1. Load model
    model, load_warnings = load_model(file_bytes)
    warnings.extend(load_warnings)
    model_name  = type(model).__name__
    n_params    = total_param_count(model)

    # 2. Extract static layer info
    leaf_mods = [
        (name, mod)
        for name, mod in model.named_modules()
        if len(list(mod.children())) == 0
    ]
    layer_infos = [
        LayerInfo(
            layer_id=i,
            name=name or "root",
            type=type(mod).__name__.lower(),
            in_features=getattr(mod, "in_features", None),
            out_features=getattr(mod, "out_features", None),
            param_count=sum(p.numel() for p in mod.parameters()),
        )
        for i, (name, mod) in enumerate(leaf_mods)
    ]

    # 3. Build observers and attach bridge
    CacheStore.clear()
    MetricStore.clear_run(config.run_id)

    observers = build_observers(config.observers, config.run_id)
    bridge    = ModelBridge(model, observers)
    bridge.attach()

    # 4. Forward + backward
    x = (
        _load_input_file(input_bytes, config)
        if input_bytes
        else _make_torch_input(config)
    )

    try:
        out = model(x)
        if isinstance(out, torch.Tensor):
            out.sum().backward()
        else:
            warnings.append(
                f"Model output type {type(out).__name__} is not a Tensor — "
                "backward pass skipped."
            )
    except Exception as e:
        bridge.detach()
        raise RuntimeError(f"Forward/backward failed: {e}")

    bridge.detach()

    # 5. Build raw metrics response from MetricStore
    raw_metrics: List[LayerMetrics] = []
    id_to_name  = {info.layer_id: info.name for info in layer_infos}

    all_data = MetricStore.get_all(config.run_id)
    for layer_id, metrics_dict in sorted(all_data.items()):
        serialisable = {
            k: [float(v_) for v_ in v] if isinstance(v, list) else float(v)
            for k, v in metrics_dict.items()
        }
        raw_metrics.append(LayerMetrics(
            layer_id=layer_id,
            layer_name=id_to_name.get(layer_id, str(layer_id)),
            metrics=serialisable,
        ))

    # 6. Run profiles
    signal_stats_result:   Optional[SignalStatsResult]   = None
    path_dominance_result: Optional[PathDominanceResult] = None

    for profile_name in config.profiles:
        cls = _PROFILE_REGISTRY.get(profile_name)
        if cls is None:
            warnings.append(f"Unknown profile '{profile_name}', skipping.")
            continue

        profile = cls(profile_name, config.run_id)
        result  = profile()

        if profile_name == "SignalStatsProfile":
            def _agg(d):
                return {
                    int(k): float(np.mean(v)) if isinstance(v, list) else float(v)
                    for k, v in d.items()
                }
            signal_stats_result = SignalStatsResult(
                activation_mean=_agg(result.get("activation_mean", {})),
                activation_var= _agg(result.get("activation_var",  {})),
                grad_norm=      _agg(result.get("grad_norm",        {})),
                grad_var=       _agg(result.get("grad_var",         {})),
            )

        elif profile_name == "PathDominanceProfile":
            def _fk(d):
                return {int(k): float(v) for k, v in d.items()}
            path_dominance_result = PathDominanceResult(
                residual=_fk(result.get("residual", {})),
                shortcut=_fk(result.get("shortcut", {})),
            )

    return DiagnosticReport(
        run_id=config.run_id,
        model_name=model_name,
        total_params=n_params,
        layers=layer_infos,
        raw_metrics=raw_metrics,
        signal_stats=signal_stats_result,
        path_dominance=path_dominance_result,
        warnings=warnings,
    )