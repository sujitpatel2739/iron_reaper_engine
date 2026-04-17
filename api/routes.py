"""
api/routes.py
-------------
All FastAPI routes.

Changes from previous version
------------------------------
1. WS /run/full added.
   New WebSocket endpoint that drives RunSession for multi-epoch streaming.
   Replaces the HTTP POST /run/start for full diagnostic runs.
   POST /run/start is kept for backward compatibility but is no longer
   the primary entry point.

2. /run/reset now also clears AnomalyStore.
   AnomalyStore is a new module that must be cleared alongside MetricStore
   and CacheStore when the user resets between graph edits.

3. WS /run/step is unchanged in protocol.
   The step-mode WebSocket still works exactly as before.

HTTP routes
-----------
GET  /health
GET  /info
GET  /layers
GET  /nodes
GET  /observers
POST /network/build
POST /network/import
POST /network/save
POST /dataset/upload
POST /dataset/validate-synthetic
POST /dataset/validate-upload
DELETE /dataset/validation/{id}
POST /run/reset
GET  /info/validation-store

WebSocket routes
----------------
WS   /run/full     ← NEW: multi-epoch streaming run
WS   /run/step     (unchanged)
"""

import json
import traceback
import uuid
from typing import Dict, List, Optional

import numpy as np
from fastapi import (
    APIRouter, File, Form, HTTPException,
    Response, UploadFile, WebSocket, WebSocketDisconnect, status,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.registries.registry import LAYER_TYPES, NODE_TYPES, OBSERVER_TYPES, build_observers
from backend.builder.builder import build_network
from backend.engine.engine import engine
from backend.step_engine.step_engine import StepEngine
from backend.validation.validate_network import validate_network
from backend.ironframe.ironframe import Tensor
import backend.cache.CacheStore as CacheStore
import backend.cache.ValidationStore as ValidationStore
import backend.diag.MetricStore as MetricStore
import backend.diag.AnomalyStore as AnomalyStore          # ← NEW

from api.bridge import load_model, graph_from_model
from api.runner import RunSession                           # ← NEW
from backend.registries import network_registry
from backend.data.data_builder import build_dataset, parse_upload
from api.schemas import (
    BuildResponse, RunConfig,
    HealthResponse, ValidationResponse, ValidationWarning,
    SyntheticInput,
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
    return {"version": "0.1.0"}


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
# Build
# ---------------------------------------------------------------------------

@router.post("/network/build", response_model=BuildResponse, tags=["Network"])
async def build_network(body: dict):
    graph      = body.get("graph", {})
    run_config = body.get("run_config", {})
    nodes      = graph.get("nodes", [])
    edges      = graph.get("edges", [])

    if not nodes:
        raise HTTPException(status_code=422, detail="Graph has no nodes.")

    input_shape  = run_config.get("input_shape")
    raw_warnings = validate_network({"nodes": nodes, "edges": edges}, input_shape)
    warnings     = [ValidationWarning(node_id=w["node_id"], message=w["message"])
                    for w in raw_warnings]

    try:
        network = build_network(nodes, edges)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"[Build] Failed to build network: {e}")

    CacheStore.clear()
    MetricStore.clear()
    AnomalyStore.clear()

    build_id = network_registry.register(
        graph      = {"nodes": nodes, "edges": edges},
        network       = network,
        run_config = run_config,
    )

    return BuildResponse(
        build_id   = build_id,
        valid      = len(warnings) == 0,
        warnings   = warnings,
        node_count = len(nodes),
        edge_count = len(edges),
    )


# ---------------------------------------------------------------------------
# Network import / save
# ---------------------------------------------------------------------------

@router.post("/network/import", tags=["Network"])
async def import_model(model_file: UploadFile = File(...)):
    file_bytes = await model_file.read()
    try:
        model, _ = load_model(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return graph_from_model(model)


@router.post("/network/save", tags=["Network"])
async def save_network(body: dict):
    return Response(
        content=json.dumps(body, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="network.json"'},
    )


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

@router.post("/dataset/validate-synthetic", tags=["Dataset"])
async def validate_synthetic_spec(body: dict):
    try:
        spec = SyntheticInput(**body)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid synthetic spec: {e}")

    validation_id = ValidationStore.store_validation('synthetic', {
        'name':         spec.name,
        'n_samples':    spec.n_samples,
        'sample_shape': spec.sample_shape,
        'batch_size':   spec.batch_size,
        'distribution': spec.distribution,
        'seed':         spec.seed,
    })
    return {'valid': True, 'validation_id': validation_id}


@router.post("/dataset/validate-upload", tags=["Dataset"])
async def validate_upload_spec(
    name:      str        = Form(...),
    data_file: UploadFile = File(...),
):
    file_bytes = await data_file.read()
    filename   = data_file.filename or "upload"
    try:
        tensor = parse_upload(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    validation_id = ValidationStore.store_validation('upload', {
        'name':      name,
        'data':      file_bytes,
        'filename':  filename,
        'shape':     list(tensor.data.shape),
        'dtype':     str(tensor.data.dtype),
        'n_samples': tensor.data.shape[0],
    })
    return {
        'valid':         True,
        'validation_id': validation_id,
        'shape':         list(tensor.data.shape),
        'dtype':         str(tensor.data.dtype),
        'n_samples':     tensor.data.shape[0],
    }


@router.delete("/dataset/validation/{validation_id}", tags=["Dataset"])
async def delete_dataset_validation(validation_id: str):
    deleted = ValidationStore.delete_validation(validation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Validation ID '{validation_id}' not found")
    return {"status": "deleted", "validation_id": validation_id}


# ---------------------------------------------------------------------------
# Run reset
# ---------------------------------------------------------------------------

@router.post("/run/reset", tags=["Run"])
def run_reset():
    """
    Clear all state. Must be called whenever the user edits the graph.
    Now also clears AnomalyStore.
    """
    network_registry.clear()
    CacheStore.clear()
    ValidationStore.clear()
    MetricStore.clear()
    AnomalyStore.clear()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Diagnostic endpoint
# ---------------------------------------------------------------------------

@router.get("/info/validation-store", tags=["Meta"])
def get_validation_store_info():
    ids = ValidationStore.validation_ids()
    return {"validation_count": len(ids), "validation_ids": ids}


# ---------------------------------------------------------------------------
# WebSocket — full multi-epoch streaming run  ← NEW
# ---------------------------------------------------------------------------

@router.websocket("/run/full")
async def run_full(websocket: WebSocket):
    """
    Full multi-epoch streaming run.

    Client → Server:
        start { build_id, run_config, validation_ids }
        stop  {}

    Server → Client:
        run_started  { run_id, n_epochs, n_batches, total_samples }
        epoch_start  { epoch }
        batch_done   { epoch, batch, n_batches }   (conditional)
        epoch_done   { epoch, snapshot, anomalies }
        run_done     { n_epochs }
        error        { message }
    """
    await websocket.accept()

    try:
        while True:
            raw    = await websocket.receive_text()
            msg    = json.loads(raw)
            action = msg.get("action")

            if action == "start":
                build_id   = msg.get("build_id")
                run_config = msg.get("run_config", {})
                vids       = msg.get("validation_ids", [])

                if not build_id:
                    await websocket.send_json({"event": "error", "message": "No build_id."})
                    continue

                build = network_registry.get(build_id)
                if build is None:
                    await websocket.send_json({"event": "error", "message": f"Build '{build_id}' not found. Rebuild."})
                    continue

                if not vids:
                    await websocket.send_json({"event": "error", "message": "No validation_ids provided."})
                    continue

                # Reconstruct dataset from ValidationStore
                synthetic_specs = []
                uploaded        = {}
                for vid in vids:
                    try:
                        kind, spec = ValidationStore.get_validation(vid)
                    except KeyError:
                        await websocket.send_json({"event": "error", "message": f"Validation ID '{vid}' not found."})
                        break
                    if kind == "synthetic":
                        synthetic_specs.append(SyntheticInput(**spec))
                    elif kind == "upload":
                        uploaded[spec["name"]] = (spec["data"], spec["filename"])
                else:
                    # Only reached if the for loop completed without break
                    try:
                        dataset = build_dataset(synthetic_specs, uploaded)
                    except ValueError as e:
                        await websocket.send_json({"event": "error", "message": str(e)})
                        continue

                    session = RunSession(build, run_config, dataset)
                    await session.run(websocket)
                    continue

            elif action == "stop":
                await websocket.send_json({"event": "stopped"})
                break

            else:
                await websocket.send_json({"event": "error", "message": f"Unknown action '{action}'."})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket — manual step mode  (unchanged)
# ---------------------------------------------------------------------------

@router.websocket("/run/step")
async def run_step(websocket: WebSocket):
    """
    Manual step-mode session.

    Client → Server: start | next | prev | follow | stop
    Server → Client: ready | step_done | branch_point | branch_done |
                     branches_complete | forward_complete | backward_complete | error
    """
    await websocket.accept()

    engine:  Optional[StepEngine] = None
    fwd_gen                       = None
    bwd_gen                       = None
    phase                         = "idle"

    def _make_input_from_dataset(dataset: dict) -> Tensor:
        if dataset:
            first = next(iter(dataset.values()))
            return first
        raise ValueError("No dataset provided for step mode.")

    try:
        while True:
            raw    = await websocket.receive_text()
            msg    = json.loads(raw)
            action = msg.get("action")

            if action == "start":
                build_id     = msg.get("build_id")
                run_config   = msg.get("run_config", {})
                dataset_spec = msg.get("dataset_spec", {})

                if not build_id:
                    await websocket.send_json({"event": "error", "message": "No build_id."})
                    continue

                build = network_registry.get(build_id)
                if build is None:
                    await websocket.send_json({"event": "error", "message": f"Build '{build_id}' not found."})
                    continue

                CacheStore.clear()
                MetricStore.clear_run(run_config.get("run_id", 0))

                synthetic_specs = [
                    SyntheticInput(**s)
                    for s in dataset_spec.get("synthetic_inputs", [])
                ]
                try:
                    dataset = build_dataset(synthetic_specs, {})
                except ValueError as e:
                    await websocket.send_json({"event": "error", "message": str(e)})
                    continue

                engine  = StepEngine(build.plan)
                x_input = _make_input_from_dataset(dataset)
                x_input.requires_grad = True
                engine.set_input(x_input)

                fwd_gen = engine.step_forward()
                phase   = "forward"

                first_node_id = build.graph["nodes"][0]["id"]
                await websocket.send_json({"event": "ready", "layer_id": first_node_id})

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

            elif action == "prev" and phase == "backward" and bwd_gen is not None:
                try:
                    event = next(bwd_gen)
                    await websocket.send_json(event.to_dict())
                except StopIteration:
                    await websocket.send_json({"event": "backward_complete"})

            elif action == "follow" and fwd_gen is not None:
                try:
                    event = next(fwd_gen)
                    await websocket.send_json(event.to_dict())
                    if event.kind.name == "FORWARD_COMPLETE":
                        phase   = "backward"
                        bwd_gen = engine.step_backward()
                except StopIteration:
                    await websocket.send_json({"event": "forward_complete"})

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