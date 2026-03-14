"""
api/main.py
-----------
FastAPI application entry point.

Run from the project root:
    PYTHONPATH=backend uvicorn api.main:app --reload --port 8000

PYTHONPATH=backend makes all backend packages (cache, ironframe,
neural_network, execution, diag, Observers) importable directly.
api/ itself is importable because it sits at the project root.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.schemas import HealthResponse

app = FastAPI(
    title="Neural Systems Observatory",
    description=(
        "Diagnostic engine for neural network introspection. "
        "Build, import, observe, and step through neural networks."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
