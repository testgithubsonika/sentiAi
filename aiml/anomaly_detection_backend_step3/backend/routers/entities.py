"""
routers/entities.py
====================
GET /api/entities                    -- risk-ranked entity list, filterable
GET /api/entities/{entity_id}/history -- raw logs + alerts + live adaptive baseline
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

import ml_worker
import schemas
from database import get_db
from models import AlertQueue, AlertStatus, EntityProfile, EntityType, RawAccessLog

router = APIRouter(prefix="/api/entities", tags=["entities"])

# (inclusive lower bound, exclusive upper bound)
RISK_LEVEL_BANDS = {
    "low": (0.0, 0.35),
    "medium": (0.35, 0.6),
    "high": (0.6, 0.85),
    "critical": (0.85, 1.0001),
}


def _risk_level(score: float) -> str:
    for level, (lo, hi) in RISK_LEVEL_BANDS.items():
        if lo <= score < hi:
            return level
    return "critical"


@router.get("", response_model=schemas.EntityListResponse)
def list_entities(
    entity_type: Optional[EntityType] = Query(None, description="Filter by user / service_account / edge_device"),
    risk_level: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    min_risk_score: Optional[float] = Query(None, ge=0.0, le=1.0),
    q: Optional[str] = Query(None, description="Substring match on entity_id or display_name"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Entities ranked by current risk (max anomaly_score across their
    *open* alerts), highest risk first -- the primary SOC triage view."""

    # Current risk = strongest open alert per entity. Entities with no
    # open alerts simply don't appear in this subquery and default to 0.
    risk_subq = (
        select(
            AlertQueue.entity_id.label("entity_id"),
            func.max(AlertQueue.anomaly_score).label("current_risk_score"),
            func.count(AlertQueue.alert_id).label("open_alert_count"),
        )
        .where(AlertQueue.status == AlertStatus.open)
        .group_by(AlertQueue.entity_id)
        .subquery()
    )

    last_seen_subq = (
        select(
            RawAccessLog.entity_id.label("entity_id"),
            func.max(RawAccessLog.timestamp).label("last_seen"),
        )
        .group_by(RawAccessLog.entity_id)
        .subquery()
    )

    # Most recent alert (any status) per entity, for the "last alert
    # type" column -- ranked via ROW_NUMBER() so we only take the latest.
    ranked_alerts = select(
        AlertQueue.entity_id,
        AlertQueue.alert_type,
        func.row_number()
        .over(partition_by=AlertQueue.entity_id, order_by=desc(AlertQueue.detected_at))
        .label("rn"),
    ).subquery()
    latest_alert_subq = select(ranked_alerts).where(ranked_alerts.c.rn == 1).subquery()

    risk_score_expr = func.coalesce(risk_subq.c.current_risk_score, 0.0)

    base_query = (
        select(
            EntityProfile,
            risk_score_expr.label("current_risk_score"),
            func.coalesce(risk_subq.c.open_alert_count, 0).label("open_alert_count"),
            last_seen_subq.c.last_seen,
            latest_alert_subq.c.alert_type.label("last_alert_type"),
        )
        .outerjoin(risk_subq, EntityProfile.entity_id == risk_subq.c.entity_id)
        .outerjoin(last_seen_subq, EntityProfile.entity_id == last_seen_subq.c.entity_id)
        .outerjoin(latest_alert_subq, EntityProfile.entity_id == latest_alert_subq.c.entity_id)
    )

    if entity_type is not None:
        base_query = base_query.where(EntityProfile.entity_type == entity_type)
    if q:
        like = f"%{q}%"
        base_query = base_query.where(
            (EntityProfile.entity_id.ilike(like)) | (EntityProfile.display_name.ilike(like))
        )
    if min_risk_score is not None:
        base_query = base_query.where(risk_score_expr >= min_risk_score)
    if risk_level is not None:
        lo, hi = RISK_LEVEL_BANDS[risk_level]
        base_query = base_query.where(risk_score_expr >= lo, risk_score_expr < hi)

    total = db.execute(select(func.count()).select_from(base_query.subquery())).scalar_one()

    rows = db.execute(
        base_query.order_by(desc(risk_score_expr)).limit(limit).offset(offset)
    ).all()

    entities = [
        schemas.EntitySummary(
            entity_id=profile.entity_id,
            entity_type=profile.entity_type.value,
            display_name=profile.display_name,
            home_city=profile.home_city,
            home_country=profile.home_country,
            current_risk_score=round(float(risk_score), 4),
            risk_level=_risk_level(float(risk_score)),
            open_alert_count=int(open_alert_count),
            last_alert_type=last_alert_type,
            last_seen=last_seen,
        )
        for profile, risk_score, open_alert_count, last_seen, last_alert_type in rows
    ]

    return schemas.EntityListResponse(total=total, limit=limit, offset=offset, entities=entities)


@router.get("/{entity_id}/history", response_model=schemas.EntityHistoryResponse)
def get_entity_history(
    entity_id: str,
    log_limit: int = Query(100, ge=1, le=1000),
    alert_limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Historical access logs + alerts for one entity, plus (if the ML
    worker currently holds warm state for it) the live per-feature
    (mean, std) baseline the online River model is adapting."""
    profile = db.get(EntityProfile, entity_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown entity_id '{entity_id}'")

    logs = db.execute(
        select(RawAccessLog)
        .where(RawAccessLog.entity_id == entity_id)
        .order_by(desc(RawAccessLog.timestamp))
        .limit(log_limit)
    ).scalars().all()

    alerts = db.execute(
        select(AlertQueue)
        .where(AlertQueue.entity_id == entity_id)
        .order_by(desc(AlertQueue.detected_at))
        .limit(alert_limit)
    ).scalars().all()

    adaptive_baseline = None
    pipeline = ml_worker.get_pipeline()
    if pipeline is not None:
        raw_baseline = pipeline.behavioral_profiler.get_adaptive_baseline(entity_id)
        if raw_baseline is not None:
            adaptive_baseline = {
                feature: schemas.AdaptiveBaselineFeature(mean=mean, std=std)
                for feature, (mean, std) in raw_baseline.items()
            }

    return schemas.EntityHistoryResponse(
        entity=schemas.EntityProfileOut.model_validate(profile),
        recent_logs=[schemas.RawLogOut.model_validate(log) for log in logs],
        recent_alerts=[schemas.AlertOut.model_validate(alert) for alert in alerts],
        adaptive_baseline=adaptive_baseline,
    )
