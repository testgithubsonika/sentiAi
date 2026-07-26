"""
schemas.py
==========
Pydantic (v2) response/request models for the REST API. Kept separate
from `models.py` (the SQLAlchemy ORM layer) on purpose -- these are the
API's public contract and can evolve independently of the DB schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Shared / enums as plain strings (kept loose so the API doesn't break if
# `models.py` enums gain values -- validation still happens at the DB layer)
# ---------------------------------------------------------------------------
RiskLevel = str  # "low" | "medium" | "high" | "critical"


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
class EntitySummary(BaseModel):
    """One row in the entity risk table / dashboard grid."""

    model_config = ConfigDict(from_attributes=True)

    entity_id: str
    entity_type: str
    display_name: str
    home_city: str
    home_country: str
    current_risk_score: float = 0.0
    risk_level: str = "low"
    open_alert_count: int = 0
    last_alert_type: Optional[str] = None
    last_seen: Optional[datetime] = None


class EntityListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    entities: List[EntitySummary]


class EntityProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    entity_id: str
    entity_type: str
    display_name: str
    habitual_hour_start: int
    habitual_hour_end: int
    home_city: str
    home_country: str
    home_lat: float
    home_lon: float
    home_subnet: str
    typical_resources: List[str]
    typical_auth_method: str
    typical_os: str
    typical_mac: str
    typical_protocol: str
    mean_session_seconds: float
    std_session_seconds: float
    created_at: Optional[datetime] = None


class RawLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    log_id: UUID
    timestamp: datetime
    source_ip: str
    resource_accessed: str
    auth_method: str
    auth_result: str
    session_duration: float
    command_sequence: List[str]
    device_fingerprint: Dict[str, Any]
    label: str


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    alert_id: UUID
    processed_id: Optional[UUID] = None
    entity_id: str
    alert_type: str
    severity: str
    anomaly_score: float
    status: str
    details: Optional[Dict[str, Any]] = None
    assigned_to: Optional[str] = None
    detected_at: datetime
    resolved_at: Optional[datetime] = None


class AdaptiveBaselineFeature(BaseModel):
    mean: float
    std: float


class EntityHistoryResponse(BaseModel):
    entity: EntityProfileOut
    recent_logs: List[RawLogOut]
    recent_alerts: List[AlertOut]
    # Live, per-feature (mean, std) from the online River baseline, if the
    # ML worker currently holds warm state for this entity -- None if the
    # worker hasn't observed this entity yet (e.g. right after a restart).
    adaptive_baseline: Optional[Dict[str, AdaptiveBaselineFeature]] = None


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
class AlertListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    alerts: List[AlertOut]


class AlertStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(open|acknowledged|resolved|false_positive)$")
    assigned_to: Optional[str] = None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class SeverityBreakdown(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class MetricsResponse(BaseModel):
    total_events_processed: int
    total_alerts: int
    active_anomalies: int
    entities_monitored: int
    events_last_hour: int
    avg_open_alert_risk_score: float
    alerts_by_severity: SeverityBreakdown
    alerts_by_type: Dict[str, int]
    last_event_at: Optional[datetime] = None

    # live worker/ops state
    model_trained: bool
    streaming: bool
    connected_dashboard_clients: int
