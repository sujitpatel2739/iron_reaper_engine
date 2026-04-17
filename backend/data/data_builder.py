"""
dataset_builder.py
------------------
Builds named input tensors from dataset specs.

Two sources:
    upload    — numpy .npy or .csv file, max 100 samples
    synthetic — generated from distribution spec, max 100 samples

Returns a dict:  { input_name: Tensor }
which is what the engine feeds to InputLayer ports.
"""

import io
import numpy as np
from typing import Dict, List, Optional

from ironframe.ironframe import Tensor
from api.schemas import SyntheticInput


MAX_SAMPLES = 100


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

def build_synthetic(spec: SyntheticInput) -> Tensor:
    """
    Generate a Tensor of shape (n_samples, *sample_shape) from spec.
    """
    n = min(spec.n_samples, MAX_SAMPLES)
    shape = (n, *spec.sample_shape)

    if spec.seed is not None:
        np.random.seed(spec.seed)

    dist = spec.distribution
    if dist == "normal":
        data = np.random.randn(*shape).astype(np.float32)
    elif dist == "uniform":
        data = np.random.uniform(0.0, 1.0, shape).astype(np.float32)
    elif dist == "zeros":
        data = np.zeros(shape, dtype=np.float32)
    elif dist == "ones":
        data = np.ones(shape, dtype=np.float32)
    else:
        raise ValueError(f"Unknown distribution '{dist}'")

    return Tensor(data, requires_grad=False)


# ---------------------------------------------------------------------------
# Upload parsing
# ---------------------------------------------------------------------------

def parse_upload(file_bytes: bytes, filename: str) -> Tensor:
    """
    Parse an uploaded file into a Tensor.
    Supports .npy and .csv.
    Max 100 samples — excess rows are silently truncated.
    """
    buf = io.BytesIO(file_bytes)

    if filename.endswith(".npy"):
        arr = np.load(buf)
    elif filename.endswith(".csv"):
        arr = np.loadtxt(buf, delimiter=",", dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported file format '{filename}'. "
            f"Please upload .npy or .csv files."
        )

    # Enforce max samples on the first axis
    if arr.ndim == 0:
        raise ValueError("Uploaded file produced a scalar — expected at least 1D array.")

    if arr.shape[0] > MAX_SAMPLES:
        arr = arr[:MAX_SAMPLES]

    return Tensor(arr.astype(np.float32), requires_grad=False)


# ---------------------------------------------------------------------------
# Build full dataset dict
# ---------------------------------------------------------------------------

def build_dataset(
    synthetic_specs: List[SyntheticInput],
    uploaded_files:  Dict[str, tuple],    # name → (bytes, filename)
) -> Dict[str, Tensor]:
    """
    Build the full { input_name: Tensor } dict from all specs.

    Parameters
    ----------
    synthetic_specs  : list of SyntheticInput specs
    uploaded_files   : dict mapping input name → (file_bytes, filename)

    Returns
    -------
    dict of { name: Tensor } ready to feed to the engine
    """
    dataset: Dict[str, Tensor] = {}

    for spec in synthetic_specs:
        dataset[spec.name] = build_synthetic(spec)

    for name, (file_bytes, filename) in uploaded_files.items():
        dataset[name] = parse_upload(file_bytes, filename)

    return dataset