"""
api/schemas.py
--------------
Pydantic request / response models for the FastAPI layer.
"""

from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Dataset input specs
# ---------------------------------------------------------------------------

class UploadedInput(BaseModel):
    """
    One named input slot backed by an uploaded file.
    The actual file bytes arrive as a multipart upload alongside this spec.
    max 100 samples enforced by the backend.
    """
    name:       str                   # matches an InputLayer port label
    file_key:   str                   # key used in the multipart form


class SyntheticInput(BaseModel):
    """
    One named input slot generated synthetically by the backend.
    """
    name:             str
    n_samples:        int   = 32      # max 100
    sample_shape:     List[int]       # shape of a single sample, e.g. [128]
    batch_size:       int   = 32
    distribution:     Literal["normal", "uniform", "zeros", "ones"] = "normal"
    seed:             Optional[int]   = 42

    @field_validator("n_samples")
    @classmethod
    def cap_samples(cls, v):
        if v > 100:
            raise ValueError("n_samples must be ≤ 100")
        return v


class DatasetSpec(BaseModel):
    """
    Full dataset spec attached to a run.
    At least one of uploaded_inputs or synthetic_inputs must be non-empty.
    """
    uploaded_inputs:  List[UploadedInput]  = []
    synthetic_inputs: List[SyntheticInput] = []


# ---------------------------------------------------------------------------
# Build request / response
# ---------------------------------------------------------------------------

class BuildRequest(BaseModel):
    graph:       dict          # { nodes, edges }
    run_config:  "RunConfig"


class BuildResponse(BaseModel):
    build_id:   str
    valid:      bool
    warnings:   List["ValidationWarning"] = []
    node_count: int
    edge_count: int


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    run_id:     int       = 0
    observers:  List[str] = [
        "SignalStatsObserver",
        "SignalShapeObserver",
        "ResidualEnergyObserver",
    ]
    profiles:   List[str] = [
        "SignalStatsProfile",
        "PathDominanceProfile",
    ]
    # dataset is sent separately as multipart — only spec refs here
    dataset:    Optional[DatasetSpec] = None


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
# Diagnostic report
# ---------------------------------------------------------------------------

class DiagnosticReport(BaseModel):
    run_id:         int
    build_id:       str
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


# Resolve forward refs
BuildResponse.model_rebuild()
BuildRequest.model_rebuild()