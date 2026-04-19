"""
api/runner.py
-------------
RunSession — drives a full multi-epoch diagnostic run.

WHY RunSession INSTEAD OF run_diagnostics():
  The old run_diagnostics() function ran forward+backward once, collected
  metrics, and returned a DiagnosticReport synchronously.  This is
  incompatible with multi-epoch streaming — you cannot hold an HTTP
  response open for minutes across many epochs.

  RunSession is a stateful object that:
    1. Loops over epochs and batches.
    2. Sets the current epoch on all observers before each epoch starts
       (so MetricStore/AnomalyStore writes go to the right epoch slot).
    3. Enables CacheStore release mode on all WrappedLayers before running
       (so tensors are freed immediately after observers read them).
    4. Sends WebSocket events after each epoch (and optionally after each
       batch if the batch was slow).
    5. Sends a run_done event when all epochs complete.

WHY RELEASE MODE:
  In full-run mode we iterate many samples.  Without releasing CacheStore
  after each layer's observers fire, every layer's output tensor stays
  referenced until CacheStore.clear() is called at the end.  For a
  100-layer network this is 100× more live tensor memory than needed.
  WrappedLayer.set_release_mode(True) enables per-layer clearing.
  Step mode does NOT set release mode because collector.collect() reads
  CacheStore after WrappedLayer.forward() returns.

WHY SET_EPOCH ON OBSERVERS:
  Observers write to MetricStore.update(run_id, epoch, layer_id, metric).
  If epoch is not updated between epochs, all samples across all epochs
  go into epoch 0.  Each observer carries a self.epoch int that must be
  updated before the epoch starts.

WHY ADAPTIVE BATCH STREAMING:
  We measure wall-clock time per batch.  If it completes in < 100ms we
  skip the batch_done event — sending one JSON message per batch for fast
  networks would flood the WebSocket.  If it takes >= 100ms we send it
  so the user can see progress on slow networks.

SNAPSHOT FORMAT (epoch_done payload):
  snapshot[node_id] = {
    "output_shape":    [N, D],         ← reconstructed from shape_0, shape_1
    "activation_mean": {mean, std, min, max, count},
    "activation_std":  {mean, std, min, max, count},
    "grad_in_norm":    {mean, std, min, max, count},
    "grad_in_std":     {mean, std, min, max, count},
  }
  anomalies[node_id] = [
    { kind, severity, frequency, count, total, sample_value }
  ]
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any

import backend.cache.CacheStore as CacheStore
import backend.diag.MetricStore as MetricStore
import backend.diag.AnomalyStore as AnomalyStore
from backend.engine.engine import Engine
from backend.engine.planner import StepKind
from backend.neural_network.Layers.WrappedLayer import WrappedLayer
from backend.ironframe.ironframe import Tensor
from backend.registries.network_registry import NetworkBuild
from backend.registries.registry import build_observers


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

def _build_snapshot(
    run_id:   int,
    epoch:    int,
    id_to_node: Dict[int, str],  # layer_id_int → graph node_id string
) -> Dict[str, Any]:
    """
    Read MetricStore for this epoch and build the snapshot dict keyed by
    graph node_id (string) for the frontend.
    """
    snapshot = {}

    for layer_int, node_id in id_to_node.items():
        layer_metrics: Dict[str, Any] = {}

        for metric in ("activation_mean", "activation_std", "grad_in_norm", "grad_in_std"):
            s = MetricStore.get_summary(run_id, epoch, layer_int, metric)
            if s is not None:
                layer_metrics[metric] = s

        # Reconstruct output_shape from shape_0, shape_1, … dimensions.
        # We collect shape_N until a gap is found.
        dims = []
        i = 0
        while True:
            s = MetricStore.get_summary(run_id, epoch, layer_int, f"shape_{i}")
            if s is None:
                break
            dims.append(int(round(s["mean"])))
            i += 1
        if dims:
            layer_metrics["output_shape"] = dims

        if layer_metrics:
            snapshot[node_id] = layer_metrics

    return snapshot


def _build_anomaly_snapshot(
    run_id:     int,
    epoch:      int,
    id_to_node: Dict[int, str],
) -> Dict[str, List[Any]]:
    """
    Read AnomalyStore for this epoch, keyed by graph node_id.
    """
    result = {}
    for layer_int, node_id in id_to_node.items():
        events = AnomalyStore.get_layer_anomalies(run_id, epoch, layer_int)
        if events:
            result[node_id] = events
    return result


# ---------------------------------------------------------------------------
# Helper: set epoch on all observers in a plan
# ---------------------------------------------------------------------------

def _set_epoch_on_observers(plan, epoch: int) -> None:
    """Walk all WrappedLayers in the plan and set their observers' epoch."""
    def _visit(steps):
        for step in steps:
            if step.kind == StepKind.BRANCH_POINT:
                for branch in step.branches:
                    _visit(branch)
            elif isinstance(step.obj, WrappedLayer):
                for obs in step.obj.observers:
                    obs.set_epoch(epoch)
    _visit(plan.steps)


def _set_release_mode(plan, release: bool) -> None:
    """Enable or disable CacheStore release on all WrappedLayers."""
    def _visit(steps):
        for step in steps:
            if step.kind == StepKind.BRANCH_POINT:
                for branch in step.branches:
                    _visit(branch)
            elif isinstance(step.obj, WrappedLayer):
                step.obj.set_release_mode(release)
    _visit(plan.steps)


def _count_params(plan) -> int:
    total = 0
    def _visit(steps):
        nonlocal total
        for step in steps:
            if step.kind == StepKind.BRANCH_POINT:
                for branch in step.branches:
                    _visit(branch)
            elif isinstance(step.obj, WrappedLayer):
                layer = step.obj.layer
                if hasattr(layer, "parameters"):
                    for p in layer.parameters.values():
                        if hasattr(p, "data"):
                            total += p.data.size
    _visit(plan.steps)
    return total


# ---------------------------------------------------------------------------
# RunSession
# ---------------------------------------------------------------------------

class RunSession:
    """
    Drives a full multi-epoch run and streams results via WebSocket.

    Usage (inside the WS /run/full handler):
        session = RunSession(build, run_config, dataset)
        await session.run(websocket)
    """

    BATCH_STREAM_THRESHOLD_MS = 100   # send batch_done if batch took longer than this

    def __init__(
        self,
        build:      NetworkBuild,
        run_config: dict,
        dataset:    Dict[str, Tensor],
    ):
        self.build      = build
        self.run_config = run_config
        self.dataset    = dataset

        self.run_id     = run_config.get("run_id", 0)
        self.n_epochs   = max(1, int(run_config.get("n_epochs", 1)))
        self.observers  = build_observers(
            run_config.get("observers", ["SignalStatsObserver"]),
            self.run_id,
        )

        # Build int→string id map for snapshot keying
        self.id_to_node: Dict[int, str] = {
            v: k for k, v in build.network.id_to_int.items()
        }

        # Compute n_batches from dataset — currently one tensor per input.
        # Each tensor's first dimension is n_samples; batch_size comes from
        # run_config. We treat the full dataset as one batch for now.
        # Multi-batch slicing is a future extension.
        self.n_batches = 1

    async def run(self, websocket) -> None:
        """
        Run all epochs, stream events, handle stop signal.
        websocket must support send_json() and receive_json() (FastAPI WebSocket).
        """
        import asyncio

        # Clear stores for this run
        CacheStore.clear()
        MetricStore.clear_run(self.run_id)
        AnomalyStore.clear_run(self.run_id)

        # Enable CacheStore release mode — tensors freed immediately after
        # observers read them, instead of accumulating for the whole run.
        _set_release_mode(self.build.plan, release=True)

        total_samples = sum(
            v.data.shape[0] if hasattr(v, 'data') else 1
            for v in self.dataset.values()
        )

        # Announce run start
        await websocket.send_json({
            "event":         "run_started",
            "run_id":        self.run_id,
            "n_epochs":      self.n_epochs,
            "n_batches":     self.n_batches,
            "total_samples": total_samples,
        })

        try:
            await self._run_engine(websocket, self.n_epochs)

        except Exception as e:
            await websocket.send_json({"event": "error", "message": str(e)})
            return
        finally:
            # Restore step mode (release=False) so next step session works correctly
            _set_release_mode(self.build.plan, release=False)
            CacheStore.clear()

        await websocket.send_json({
            "event":    "run_done",
            "n_epochs": self.n_epochs,
        })

    async def _run_engine(self, websocket, n_epochs: int) -> None:
        """Run one epoch: forward + backward, then stream epoch_done."""

        engine = Engine(self.dataset)
        engine.set_input()
        
        for epoch in range(n_epochs):
            await websocket.send_json({"event": "epoch_start", "epoch": epoch})

            # Point all observers at this epoch
            _set_epoch_on_observers(self.build.plan, epoch)
            MetricStore.clear_epoch(self.run_id, epoch)
            AnomalyStore.clear_epoch(self.run_id, epoch)

            # Run — currently one batch per epoch (the full dataset tensor)
            t0 = time.monotonic()

            # Full forward pass
            for event in engine.forward():
                if event.kind.name == "ERROR":
                    raise RuntimeError(event.error)
                if event.kind.name == "FORWARD_COMPLETE":
                    break

            # Full backward pass
            for event in engine.backward():
                if event.kind.name == "ERROR":
                    raise RuntimeError(event.error)
                if event.kind.name == "BACKWARD_COMPLETE":
                    break 

            elapsed_ms = (time.monotonic() - t0) * 1000

            # Conditionally send batch_done (adaptive streaming)
            if elapsed_ms >= self.BATCH_STREAM_THRESHOLD_MS:
                await websocket.send_json({
                    "event":    "batch_done",
                    "epoch":    epoch,
                    "batch":    0,
                    "n_batches": self.n_batches,
                })

            # Build and send epoch snapshot
            snapshot  = _build_snapshot(self.run_id, epoch, self.id_to_node)
            anomalies = _build_anomaly_snapshot(self.run_id, epoch, self.id_to_node)

            await websocket.send_json({
                "event":     "epoch_done",
                "epoch":     epoch,
                "snapshot":  snapshot,
                "anomalies": anomalies,
            })