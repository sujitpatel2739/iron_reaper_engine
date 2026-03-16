"""
api/routes.py
-------------
All FastAPI routes.

HTTP
----
GET  /health
GET  /info
GET  /layers                  available layer types + config schemas
GET  /nodes                   available node types + config schemas
GET  /observers               available observer types
POST /network/validate        shape compatibility check (no instantiation)
POST /network/import          upload model → JSON graph
POST /network/save            JSON graph → downloadable file
POST /run/start               full diagnostic run → DiagnosticReport

WebSocket
---------
WS   /run/step                manual step-mode session

Run with:
    cd <project-root>
    PYTHONPATH=backend uvicorn api.main:app --reload --port 8000
"""

import json
import traceback
from typing import Optional

import numpy as np
from fastapi import (
    APIRouter, File, Form, HTTPException,
    Response, UploadFile, WebSocket, WebSocketDisconnect, status,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from execution.registry import LAYER_TYPES, NODE_TYPES, OBSERVER_TYPES, build_observers
from execution.planner import build_plan
from execution.step_engine import StepEngine
from validation.validate_network import validate_network
from ironframe.ironframe import Tensor
import cache.CacheStore as CacheStore
import diag.MetricStore as MetricStore

from api.bridge import load_model, graph_from_model
from api.runner import run_diagnostics
from api.schemas import (
    RunConfig, DiagnosticReport, HealthResponse,
    ValidationResponse,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    return HealthResponse(status="ok")


@router.get("/info", tags=["Meta"])
def info():
    import torch
    return {"version": "0.1.0", "pytorch_version": torch.__version__}


# ---------------------------------------------------------------------------
# Type registries
# ---------------------------------------------------------------------------

@router.get("/layers", tags=["Registry"])
def get_layer_types():
    return LAYER_TYPES


@router.get("/nodes", tags=["Registry"])
def get_node_types():
    return NODE_TYPES


@router.get("/observers", tags=["Registry"])
def get_observer_types():
    return OBSERVER_TYPES


# ---------------------------------------------------------------------------
# Network validation
# ---------------------------------------------------------------------------


@router.post("/network/validate", response_model=ValidationResponse, tags=["Network"])
def validate_network_route(body: dict):
    warnings = validate_network(body)
    return ValidationResponse(valid=len(warnings) == 0, warnings=warnings)


# ---------------------------------------------------------------------------
# Network import / save
# ---------------------------------------------------------------------------

@router.post("/network/import", tags=["Network"])
async def import_model(model_file: UploadFile = File(...)):
    """Upload .pt / .pth → React Flow graph JSON."""
    file_bytes = await model_file.read()
    try:
        model, _ = load_model(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return graph_from_model(model)


@router.post("/network/save", tags=["Network"])
async def save_network(body: dict):
    """Return graph JSON as a downloadable file."""
    return Response(
        content=json.dumps(body, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="network.json"'},
    )


# ---------------------------------------------------------------------------
# Full diagnostic run
# ---------------------------------------------------------------------------

@router.post("/run/start", response_model=DiagnosticReport, tags=["Run"])
async def run_start(
    model_file: UploadFile = File(...),
    config:     str        = Form(default="{}"),
    input_file: Optional[UploadFile] = File(default=None),
):
    """Full forward + backward pass. Returns DiagnosticReport."""
    try:
        run_config = RunConfig(**json.loads(config))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid config: {e}")

    model_bytes = await model_file.read()
    input_bytes = await input_file.read() if input_file else None

    try:
        return run_diagnostics(model_bytes, run_config, input_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ---------------------------------------------------------------------------
# WebSocket — manual step mode
# ---------------------------------------------------------------------------

@router.websocket("/run/step")
async def run_step(websocket: WebSocket):
    """
    Manual step-mode session.

    Client → Server actions:
        start   { graph, run_config }
        next    {}                        advance one forward step
        prev    {}                        advance one backward step
        follow  { branch: "branch_0" }   step into a branch
        stop    {}

    Server → Client events  (StepEvent.to_dict()):
        ready, step_done, branch_point, branch_done,
        branches_complete, forward_complete, backward_complete, error
    """
    await websocket.accept()

    engine:  Optional[StepEngine] = None
    fwd_gen                       = None
    bwd_gen                       = None
    phase                         = "idle"   # "forward" | "backward"

    def _make_input(cfg: dict) -> Tensor:
        seed = cfg.get("seed")
        if seed is not None:
            np.random.seed(seed)
        shape = cfg.get("input_shape", [32, 128])
        return Tensor(
            np.random.randn(*shape).astype(np.float32),
            requires_grad=True,
        )

    try:
        while True:
            raw    = await websocket.receive_text()
            msg    = json.loads(raw)
            action = msg.get("action")

            # ── START ──────────────────────────────────────────────────────
            if action == "start":
                graph      = msg.get("graph", {})
                run_config = msg.get("run_config", {})

                if not graph.get("nodes"):
                    await websocket.send_json(
                        {"event": "error", "message": "Graph has no nodes."}
                    )
                    continue

                CacheStore.clear()
                MetricStore.clear_run(run_config.get("run_id", 0))

                observers = build_observers(
                    run_config.get("observers", ["SignalStatsObserver"]),
                    run_config.get("run_id", 0),
                )
                plan   = build_plan(graph["nodes"], graph["edges"], observers)
                engine = StepEngine(plan)
                engine.set_input(_make_input(run_config))

                fwd_gen = engine.step_forward()
                phase   = "forward"

                await websocket.send_json({
                    "event":    "ready",
                    "layer_id": graph["nodes"][0]["id"],
                })

            # ── NEXT (forward) ─────────────────────────────────────────────
            elif action == "next" and phase == "forward" and fwd_gen is not None:
                try:
                    event = next(fwd_gen)
                    await websocket.send_json(event.to_dict())
                    if event.kind.name == "FORWARD_COMPLETE":
                        phase   = "backward"
                        bwd_gen = engine.step_backward()
                except StopIteration:
                    phase   = "backward"
                    bwd_gen = engine.step_backward()
                    await websocket.send_json({"event": "forward_complete"})

            # ── PREV (backward) ────────────────────────────────────────────
            elif action == "prev" and phase == "backward" and bwd_gen is not None:
                try:
                    event = next(bwd_gen)
                    await websocket.send_json(event.to_dict())
                except StopIteration:
                    await websocket.send_json({"event": "backward_complete"})

            # ── FOLLOW branch ──────────────────────────────────────────────
            elif action == "follow" and fwd_gen is not None:
                # Branch steps are embedded in the forward generator —
                # advancing next() steps through the active branch.
                try:
                    event = next(fwd_gen)
                    await websocket.send_json(event.to_dict())
                    if event.kind.name == "FORWARD_COMPLETE":
                        phase   = "backward"
                        bwd_gen = engine.step_backward()
                except StopIteration:
                    await websocket.send_json({"event": "forward_complete"})

            # ── STOP ───────────────────────────────────────────────────────
            elif action == "stop":
                await websocket.send_json({"event": "stopped"})
                break

            else:
                await websocket.send_json({
                    "event":   "error",
                    "message": f"Unknown action '{action}' in phase '{phase}'.",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
