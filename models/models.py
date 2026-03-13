from pydantic import BaseModel
from typing import Dict, List, Any, Optional


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    """
    Optional config sent alongside the model upload.
    All fields have sensible defaults so the client can omit them.
    """
    run_id: int = 0
    batch_size: int = 32
    input_shape: List[int] = [32, 128]   # [batch, features] or [batch, C, H, W]
    observers: List[str] = [             # which observers to enable
        "SignalStatsObserver",
        "SignalShapeObserver",
        "ResidualEnergyObserver",
    ]
    profiles: List[str] = [             # which profiles to compute
        "SignalStatsProfile",
        "PathDominanceProfile",
    ]
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Layer-level response schemas
# ---------------------------------------------------------------------------

class LayerInfo(BaseModel):
    layer_id: int
    name: str
    type: str                           # "linear", "relu", "layernorm", etc.
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    param_count: int = 0


class LayerMetrics(BaseModel):
    layer_id: int
    layer_name: str
    metrics: Dict[str, Any]             # metric_name -> scalar or list


# ---------------------------------------------------------------------------
# Profile-level response schemas
# ---------------------------------------------------------------------------

class SignalStatsResult(BaseModel):
    activation_mean: Dict[int, float]
    activation_var: Dict[int, float]
    grad_norm: Dict[int, float]
    grad_var: Dict[int, float]


class PathDominanceResult(BaseModel):
    residual: Dict[int, float]
    shortcut: Dict[int, float]


# ---------------------------------------------------------------------------
# Top-level diagnostic response
# ---------------------------------------------------------------------------

class DiagnosticReport(BaseModel):
    run_id: int
    model_name: str
    total_params: int
    layers: List[LayerInfo]
    raw_metrics: List[LayerMetrics]
    signal_stats: Optional[SignalStatsResult] = None
    path_dominance: Optional[PathDominanceResult] = None
    warnings: List[str] = []


# ---------------------------------------------------------------------------
# Health / info response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"