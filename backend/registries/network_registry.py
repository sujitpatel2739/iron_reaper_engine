"""
network_registry.py
--------------------
Module-level registry of compiled (built) networks.

When the user clicks Build, the backend validates the graph, compiles
the ExecutionPlan, and stores it here under a unique build_id.
Subsequent run / step calls reference the build_id to retrieve the
pre-compiled plan — no re-parsing the graph on every run.

Any graph edit on the frontend invalidates the current build.
The frontend tracks build_id and sends it with run requests.

Structure
---------
    _registry[build_id] -> NetworkBuild

No instantiation — import and use as a module.
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NetworkBuild:
    build_id:    str
    graph:       dict          # original { nodes, edges }
    network:     Any           # ExecutionPlan from planner
    run_config:  dict
    built_at:    float = field(default_factory=time.time)
    node_count:  int   = 0
    edge_count:  int   = 0


# -- Module-level state ------------------------------------------------------

_registry: Dict[str, NetworkBuild] = {}


# -- Write / Read ------------------------------------------------------------

def register(graph: dict, network: Any, run_config: dict) -> str:
    """
    Store a compiled NetworkBuild and return its unique build_id.
    Only one build is kept at a time — previous build is replaced.
    (Multi-user support would require keying by session; not needed for dev.)
    """
    _registry.clear()   # one active build at a time

    build_id = str(uuid.uuid4())[:8]   # short readable id
    _registry[build_id] = NetworkBuild(
        build_id=build_id,
        graph=graph,
        network=network,
        run_config=run_config,
        node_count=len(graph.get("nodes", [])),
        edge_count=len(graph.get("edges", [])),
    )
    return build_id


def get(build_id: str) -> Optional[NetworkBuild]:
    """Return the build for this id, or None if not found / stale."""
    return _registry.get(build_id)


def get_required(build_id: str) -> NetworkBuild:
    build = get(build_id)
    if build is None:
        raise KeyError(
            f"No compiled network found for build_id='{build_id}'. "
            f"Please click Build before running."
        )
    return build


def current_build_id() -> Optional[str]:
    """Return the id of the currently active build, or None."""
    if _registry:
        return next(iter(_registry))
    return None


def clear() -> None:
    """Drop all builds. Called when graph is cleared."""
    _registry.clear()


def has_build() -> bool:
    return bool(_registry)