"""
api/schemas.py
--------------
Pydantic request / response models for the FastAPI layer.
These are the wire-format contracts between frontend and backend.
Nothing in backend/ imports from here — only api/ files do.
"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    run_id:      int         = 0
    input_shape: List[int]   = [32, 128]
    observers:   List[str]   = [
        "SignalStatsObserver",
        "SignalShapeObserver",
        "ResidualEnergyObserver",
    ]
    profiles:    List[str]   = [
        "SignalStatsProfile",
        "PathDominanceProfile",
    ]
    seed:        Optional[int] = 42


# ---------------------------------------------------------------------------
# Layer / node info
# ---------------------------------------------------------------------------

class LayerInfo(BaseModel):
    layer_id:     int
    name:         str
    type:         str
    in_features:  Optional[int] = None
    out_features: Optional[int] = None
    param_count:  int = 0


class LayerMetrics(BaseModel):
    layer_id:   int
    layer_name: str
    metrics:    Dict[str, Any]


# ---------------------------------------------------------------------------
# Profile results
# ---------------------------------------------------------------------------

class SignalStatsResult(BaseModel):
    activation_mean: Dict[int, float]
    activation_var:  Dict[int, float]
    grad_norm:       Dict[int, float]
    grad_var:        Dict[int, float]


class PathDominanceResult(BaseModel):
    residual: Dict[int, float]
    shortcut: Dict[int, float]


# ---------------------------------------------------------------------------
# Top-level diagnostic report
# ---------------------------------------------------------------------------

class DiagnosticReport(BaseModel):
    run_id:         int
    model_name:     str
    total_params:   int
    layers:         List[LayerInfo]
    raw_metrics:    List[LayerMetrics]
    signal_stats:   Optional[SignalStatsResult]   = None
    path_dominance: Optional[PathDominanceResult] = None
    warnings:       List[str] = []


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status:  str
    version: str = "0.1.0"


class ValidationWarning(BaseModel):
    node_id: str
    message: str


class ValidationResponse(BaseModel):
    valid:    bool
    warnings: List[ValidationWarning] = []