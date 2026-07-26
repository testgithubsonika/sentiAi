"""
main.py
=======
FastAPI entrypoint for the Behavioral Anomaly Detection backend.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

On startup (see `lifespan`) this:
  1. Creates any missing tables (idempotent -- safe alongside
     `generate_and_load.py --create-tables`).
  2. Trains the IForest -> Bi-LSTM -> XGBoost stack on fresh synthetic
     data and persists the entity population (`ml_worker.start()`).
  3. Launches the background streaming task that feeds simulated live
     events through the trained pipeline, persists flagged alerts, and
     broadcasts everything over `/ws/stream`.

Tune or disable the training/streaming behavior via env vars -- see
`config.py` (e.g. `AUTO_TRAIN_ON_STARTUP=false` for a REST-only
instance pointed at an already-populated database).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import ml_worker
from config import settings
from database import engine
from models import Base
from routers import alerts, entities, metrics
from websocket_manager import manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Ensuring database schema exists...")
    Base.metadata.create_all(bind=engine)

    logger.info("Starting ML worker (training + streaming)...")
    await ml_worker.start()

    yield

    logger.info("Shutting down ML worker...")
    await ml_worker.stop()


app = FastAPI(
    title="Behavioral Anomaly Detection API",
    description=(
        "REST + WebSocket backend for the AI-powered behavioral anomaly "
        "detection dashboard: entity risk, alert triage, live metrics, "
        "and a real-time event/alert stream over WebSocket."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(entities.router)
app.include_router(alerts.router)
app.include_router(metrics.router)


@app.get("/api/health", tags=["ops"])
def health():
    """Lightweight liveness/readiness probe + worker status snapshot."""
    return {
        "status": "ok",
        "model_trained": ml_worker.state.is_trained,
        "streaming": ml_worker.state.is_streaming,
        "events_processed": ml_worker.state.events_processed,
        "alerts_raised": ml_worker.state.alerts_raised,
        "connected_dashboard_clients": manager.connection_count,
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Live feed of every streamed event (`type: "event"`) and every
    raised alert (`type: "alert"`) as JSON text frames. The server
    doesn't expect any particular client message -- `receive_text()` is
    just how FastAPI/Starlette detects disconnects; ping/keepalive
    frames sent by the client are read and discarded.
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
