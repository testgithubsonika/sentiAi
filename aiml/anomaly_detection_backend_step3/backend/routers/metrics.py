"""
routers/metrics.py
===================
GET /api/metrics -- system-wide summary counters for the dashboard header
(events processed, active anomalies, alert breakdowns, worker/ops state).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import ml_worker
import schemas
from database import get_db
from models import AlertQueue, AlertStatus, EntityProfile, RawAccessLog
from websocket_manager import manager

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("", response_model=schemas.MetricsResponse)
def get_metrics(db: Session = Depends(get_db)):
    total_events = db.execute(select(func.count()).select_from(RawAccessLog)).scalar_one()
    total_alerts = db.execute(select(func.count()).select_from(AlertQueue)).scalar_one()
    active_anomalies = db.execute(
        select(func.count()).select_from(AlertQueue).where(AlertQueue.status == AlertStatus.open)
    ).scalar_one()
    entities_monitored = db.execute(select(func.count()).select_from(EntityProfile)).scalar_one()

    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    events_last_hour = db.execute(
        select(func.count()).select_from(RawAccessLog).where(RawAccessLog.timestamp >= one_hour_ago)
    ).scalar_one()

    avg_open_risk = db.execute(
        select(func.avg(AlertQueue.anomaly_score)).where(AlertQueue.status == AlertStatus.open)
    ).scalar_one()

    severity_rows = db.execute(
        select(AlertQueue.severity, func.count())
        .where(AlertQueue.status == AlertStatus.open)
        .group_by(AlertQueue.severity)
    ).all()
    severity_counts = {sev.value: count for sev, count in severity_rows}

    type_rows = db.execute(
        select(AlertQueue.alert_type, func.count())
        .group_by(AlertQueue.alert_type)
        .order_by(func.count().desc())
        .limit(10)
    ).all()

    last_event_at = db.execute(select(func.max(RawAccessLog.timestamp))).scalar_one()

    return schemas.MetricsResponse(
        total_events_processed=total_events,
        total_alerts=total_alerts,
        active_anomalies=active_anomalies,
        entities_monitored=entities_monitored,
        events_last_hour=events_last_hour,
        avg_open_alert_risk_score=round(float(avg_open_risk), 4) if avg_open_risk is not None else 0.0,
        alerts_by_severity=schemas.SeverityBreakdown(**severity_counts),
        alerts_by_type=dict(type_rows),
        last_event_at=last_event_at,
        model_trained=ml_worker.state.is_trained,
        streaming=ml_worker.state.is_streaming,
        connected_dashboard_clients=manager.connection_count,
    )
