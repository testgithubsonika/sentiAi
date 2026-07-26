"""
routers/alerts.py
==================
GET   /api/alerts               -- ranked, filterable historical alerts
PATCH /api/alerts/{alert_id}    -- SOC triage: ack / resolve / mark false-positive
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

import schemas
from database import get_db
from models import AlertQueue, AlertSeverity, AlertStatus

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=schemas.AlertListResponse)
def list_alerts(
    status: Optional[AlertStatus] = Query(None),
    severity: Optional[AlertSeverity] = Query(None),
    entity_id: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None),
    sort_by: str = Query("risk_score", pattern="^(risk_score|detected_at)$"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Ranked historical alerts. Defaults to highest-risk first, which is
    what a SOC analyst triaging a queue wants to see; pass
    `sort_by=detected_at` for a chronological feed instead."""
    query = select(AlertQueue)

    if status is not None:
        query = query.where(AlertQueue.status == status)
    if severity is not None:
        query = query.where(AlertQueue.severity == severity)
    if entity_id is not None:
        query = query.where(AlertQueue.entity_id == entity_id)
    if alert_type is not None:
        query = query.where(AlertQueue.alert_type == alert_type)

    order_col = AlertQueue.anomaly_score if sort_by == "risk_score" else AlertQueue.detected_at
    query = query.order_by(desc(order_col))

    total = db.execute(select(func.count()).select_from(query.subquery())).scalar_one()

    rows = db.execute(query.limit(limit).offset(offset)).scalars().all()

    return schemas.AlertListResponse(
        total=total,
        limit=limit,
        offset=offset,
        alerts=[schemas.AlertOut.model_validate(row) for row in rows],
    )


@router.patch("/{alert_id}", response_model=schemas.AlertOut)
def update_alert_status(
    alert_id: UUID,
    update: schemas.AlertStatusUpdate,
    db: Session = Depends(get_db),
):
    """SOC triage action: acknowledge, resolve, or mark a false positive."""
    alert = db.get(AlertQueue, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Unknown alert_id '{alert_id}'")

    alert.status = AlertStatus(update.status)
    if update.assigned_to is not None:
        alert.assigned_to = update.assigned_to
    if alert.status in (AlertStatus.resolved, AlertStatus.false_positive):
        alert.resolved_at = datetime.utcnow()

    db.commit()
    db.refresh(alert)
    return schemas.AlertOut.model_validate(alert)
