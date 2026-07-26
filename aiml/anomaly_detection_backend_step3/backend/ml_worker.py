"""
ml_worker.py
============
The background ML worker. On app startup it:

  1. Trains the full detection stack (IForest baseline -> Bi-LSTM ->
     XGBoost) on a batch of fresh synthetic data, via
     `attack_pipeline.AttackClassificationPipeline` (Step 2), and wraps
     the trained models in a warm-started `AnomalyDetectionPipeline`
     (`anomaly_pipeline.py`) -- the same orchestrator that runs
     IForest -> Bi-LSTM -> XGBoost -> SHAP per event.
  2. Persists the synthetic entity population into `entity_profiles`.
  3. Launches an infinite async streaming loop that simulates a live
     event feed (fresh `SyntheticDataGenerator` batches, same
     population, natural 0.5-3% attack rate), runs each event through
     `AnomalyDetectionPipeline.process_event()`, writes
     `raw_access_logs` (+ `processed_streaming_logs` / `alert_queue` for
     flagged events), and broadcasts every event/alert over the
     WebSocket `ConnectionManager` for the live dashboard.

All CPU-bound / blocking ML and DB work runs via `asyncio.to_thread` so
the event loop stays free to serve REST + WebSocket traffic while
training and streaming happen in the background.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from anomaly_pipeline import Alert, AnomalyDetectionPipeline
from attack_pipeline import AttackClassificationPipeline
from config import settings
from data_generator import ATTACK_RATE_RANGE, SyntheticDataGenerator
from database import SessionLocal
from models import (
    AlertQueue,
    EntityProfile,
    EntityType,
    LabelType,
    ProcessedStreamingLog,
    RawAccessLog,
)
from websocket_manager import manager

logger = logging.getLogger("ml_worker")


class MLWorkerState:
    """Process-wide singleton holding the live pipeline + generator, so
    REST routers (e.g. entity history's adaptive-baseline field) can
    read from the same warm ML state the streaming loop is updating."""

    def __init__(self) -> None:
        self.pipeline: Optional[AnomalyDetectionPipeline] = None
        self.generator: Optional[SyntheticDataGenerator] = None
        self.is_trained: bool = False
        self.is_streaming: bool = False
        self.events_processed: int = 0
        self.alerts_raised: int = 0
        self.started_at: Optional[datetime] = None
        self._task: Optional[asyncio.Task] = None


state = MLWorkerState()


def get_pipeline() -> Optional[AnomalyDetectionPipeline]:
    return state.pipeline


# ---------------------------------------------------------------------------
# Training (blocking / CPU-bound)
# ---------------------------------------------------------------------------
def _train_stack() -> AnomalyDetectionPipeline:
    logger.info("Loading pretrained detection stack...")

    training_pipeline = AttackClassificationPipeline.load_pretrained(
        model_dir=settings.MODEL_DIR,
    )
    logger.info("Pretrained models loaded successfully.")

    gen = SyntheticDataGenerator(
        num_users=settings.TRAIN_NUM_USERS,
        num_service_accounts=settings.TRAIN_NUM_SERVICE_ACCOUNTS,
        num_devices=settings.TRAIN_NUM_DEVICES,
        seed=settings.TRAIN_SEED,
    )
    state.generator = gen

    return AnomalyDetectionPipeline(
        entity_profiles_df=training_pipeline.entity_profiles.reset_index(drop=True),
        xgb_classifier=training_pipeline.xgb_classifier,
        bilstm_trainer=training_pipeline.bilstm_trainer,
        resource_vocab=training_pipeline.resource_vocab,
        command_vocab=training_pipeline.command_vocab,
        window_size=training_pipeline.window_size,
        behavioral_profiler=training_pipeline.behavioral_profiler,
        severity_thresholds=settings.SEVERITY_THRESHOLDS,
        random_state=settings.TRAIN_SEED,
    )


def _persist_profiles(profiles_df: pd.DataFrame) -> None:
    """Upsert (insert-if-missing) entity profiles so `raw_access_logs`'
    FK to `entity_profiles` is always satisfiable."""
    with SessionLocal() as db:
        existing_ids = {row[0] for row in db.query(EntityProfile.entity_id).all()}
        objects = [
            EntityProfile(
                entity_id=row["entity_id"],
                entity_type=EntityType(row["entity_type"]),
                display_name=row["display_name"],
                habitual_hour_start=int(row["habitual_hour_start"]),
                habitual_hour_end=int(row["habitual_hour_end"]),
                home_city=row["home_city"],
                home_country=row["home_country"],
                home_lat=float(row["home_lat"]),
                home_lon=float(row["home_lon"]),
                home_subnet=row["home_subnet"],
                typical_resources=list(row["typical_resources"]),
                typical_auth_method=row["typical_auth_method"],
                typical_os=row["typical_os"],
                typical_mac=row["typical_mac"],
                typical_protocol=row["typical_protocol"],
                mean_session_seconds=float(row["mean_session_seconds"]),
                std_session_seconds=float(row["std_session_seconds"]),
            )
            for _, row in profiles_df.iterrows()
            if row["entity_id"] not in existing_ids
        ]
        if objects:
            db.bulk_save_objects(objects)
            db.commit()
        logger.info("Persisted %d new entity profiles (%d already existed).", len(objects), len(existing_ids))


# ---------------------------------------------------------------------------
# Per-event persistence
# ---------------------------------------------------------------------------
def _geo_lat_lon(event: dict) -> tuple[float, float]:
    geo = event.get("geo_location")
    if isinstance(geo, dict) and "lat" in geo:
        return float(geo["lat"]), float(geo["lon"])
    return float(event.get("geo_lat", 0.0) or 0.0), float(event.get("geo_lon", 0.0) or 0.0)


def _persist_event_and_alert(event: dict, alert: Optional[Alert]) -> Optional[dict]:
    """Writes `raw_access_logs` (+ `processed_streaming_logs` /
    `alert_queue` if flagged) in one transaction. Returns a JSON-ready
    broadcast payload for the WebSocket when an alert was raised."""
    lat, lon = _geo_lat_lon(event)
    with SessionLocal() as db:
        raw_log = RawAccessLog(
            log_id=uuid.uuid4(),
            entity_id=event["entity_id"],
            entity_type=EntityType(event["entity_type"]),
            timestamp=event["timestamp"],
            source_ip=event.get("source_ip", "0.0.0.0"),
            geo_location=event.get("geo_location") or {"lat": lat, "lon": lon},
            geo_lat=lat,
            geo_lon=lon,
            resource_accessed=event.get("resource_accessed", ""),
            auth_method=event.get("auth_method", "password"),
            auth_result=event.get("auth_result", "success"),
            session_duration=float(event.get("session_duration", 0.0) or 0.0),
            command_sequence=list(event.get("command_sequence") or []),
            device_fingerprint=event.get("device_fingerprint") or {},
            label=LabelType(event.get("label", "normal")),
        )
        db.add(raw_log)
        db.flush()  # populate raw_log.log_id for the FK below

        if alert is None:
            db.commit()
            return None

        processed = ProcessedStreamingLog(
            processed_id=uuid.uuid4(),
            raw_log_id=raw_log.log_id,
            entity_id=alert.entity_id,
            timestamp=alert.timestamp,
            anomaly_score=alert.risk_score,
            predicted_label=alert.alert_type,
            model_version="v1-hackathon",
            feature_vector=alert.details,
        )
        db.add(processed)
        db.flush()

        alert_row = AlertQueue(
            alert_id=uuid.UUID(alert.alert_id),
            **alert.to_alert_queue_kwargs(processed_id=processed.processed_id),
        )
        db.add(alert_row)
        db.commit()

        return {
            "type": "alert",
            "alert_id": str(alert_row.alert_id),
            "entity_id": alert.entity_id,
            "alert_type": alert.alert_type,
            "severity": alert.severity.value,
            "risk_score": round(alert.risk_score, 4),
            "summary": alert.summary,
            "contributing_factors": alert.contributing_factors,
            "timestamp": alert.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------
async def _stream_loop() -> None:
    """Repeatedly generates a short burst of fresh synthetic events from
    the *same* trained population (natural attack rate included), feeds
    each one through `AnomalyDetectionPipeline.process_event()`, persists
    it, and broadcasts it -- paced with a randomized delay so the
    dashboard sees a believable live trickle rather than a data dump."""
    assert state.pipeline is not None and state.generator is not None
    cursor = datetime.utcnow()

    while state.is_streaming:
        window_start = cursor
        cursor = window_start + timedelta(hours=settings.STREAM_BATCH_HOURS)
        days = max(1, settings.STREAM_BATCH_HOURS // 24 + 1)

        batch_df = await asyncio.to_thread(
            state.generator.generate, window_start, days, ATTACK_RATE_RANGE,
        )
        batch_df = batch_df.sort_values("timestamp").head(settings.STREAM_EVENTS_PER_BATCH)

        for _, row in batch_df.iterrows():
            if not state.is_streaming:
                break
            event = row.to_dict()

            alert = await asyncio.to_thread(state.pipeline.process_event, event)
            payload = await asyncio.to_thread(_persist_event_and_alert, event, alert)

            state.events_processed += 1
            ts = event["timestamp"]
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

            if payload is not None:
                state.alerts_raised += 1
                logger.info("ALERT %s -> %s (risk=%.2f)", payload["entity_id"], payload["alert_type"], payload["risk_score"])
                await manager.broadcast(payload)
            else:
                await manager.broadcast({
                    "type": "event",
                    "entity_id": event["entity_id"],
                    "entity_type": event.get("entity_type"),
                    "resource_accessed": event.get("resource_accessed", ""),
                    "timestamp": ts_str,
                })

            await asyncio.sleep(random.uniform(settings.STREAM_MIN_DELAY_SECONDS, settings.STREAM_MAX_DELAY_SECONDS))


# ---------------------------------------------------------------------------
# Public lifecycle hooks (called from main.py's lifespan)
# ---------------------------------------------------------------------------
async def start() -> None:
    if not settings.AUTO_TRAIN_ON_STARTUP:
        logger.info("AUTO_TRAIN_ON_STARTUP disabled -- skipping model training/streaming.")
        return

    state.pipeline = await asyncio.to_thread(_train_stack)
    state.is_trained = True
    await asyncio.to_thread(_persist_profiles, state.pipeline.entity_profiles.reset_index(drop=True))

    if settings.AUTO_STREAM_ON_STARTUP:
        state.is_streaming = True
        state.started_at = datetime.utcnow()
        state._task = asyncio.create_task(_stream_loop())
        logger.info("Streaming task started.")


async def stop() -> None:
    state.is_streaming = False
    if state._task is not None:
        state._task.cancel()
        try:
            await state._task
        except asyncio.CancelledError:
            pass
    logger.info("ML worker stopped.")
