"""
models.py
=========
SQLAlchemy 2.0 ORM models for the Behavioral Anomaly Detection system.

Tables
------
entity_profiles        Baseline behavioral profile per entity (user /
                        service_account / edge_device) used both to drive
                        the synthetic data generator and, in production,
                        to hold the "known-good" baseline for scoring.
raw_access_logs         Immutable, append-only landing table for every
                        access event (batch or streamed).
processed_streaming_logs  Feature-engineered / scored version of a raw log
                        row, produced by the stream-processing / ML layer.
alert_queue             Actionable alerts raised when a processed log
                        crosses an anomaly threshold, with a lifecycle
                        (open -> acknowledged -> resolved / false_positive).

Designed for PostgreSQL (uses JSONB, ARRAY, UUID). Run against another
engine and SQLAlchemy will fall back to portable types where possible,
but JSONB/ARRAY are Postgres-specific -- swap them for generic JSON/String
if you need cross-database support.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class EntityType(str, enum.Enum):
    user = "user"
    service_account = "service_account"
    edge_device = "edge_device"


class LabelType(str, enum.Enum):
    normal = "normal"
    brute_force = "brute_force"
    impossible_travel = "impossible_travel"
    credential_stuffing = "credential_stuffing"
    lateral_movement = "lateral_movement"
    device_spoofing = "device_spoofing"
    low_and_slow_exfiltration = "low_and_slow_exfiltration"
    insider_drift = "insider_drift"


class AlertSeverity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class AlertStatus(str, enum.Enum):
    open = "open"
    acknowledged = "acknowledged"
    resolved = "resolved"
    false_positive = "false_positive"


# ---------------------------------------------------------------------------
# entity_profiles
# ---------------------------------------------------------------------------
class EntityProfile(Base):
    """Baseline behavioral fingerprint for a user / service account / device."""

    __tablename__ = "entity_profiles"

    entity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_type: Mapped[EntityType] = mapped_column(Enum(EntityType), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)

    # Habitual access window (local hour-of-day, 0-23)
    habitual_hour_start: Mapped[int] = mapped_column(Integer, nullable=False)
    habitual_hour_end: Mapped[int] = mapped_column(Integer, nullable=False)

    # Home geo (baseline location)
    home_city: Mapped[str] = mapped_column(String(64), nullable=False)
    home_country: Mapped[str] = mapped_column(String(8), nullable=False)
    home_lat: Mapped[float] = mapped_column(Float, nullable=False)
    home_lon: Mapped[float] = mapped_column(Float, nullable=False)

    # Typical IP subnet, e.g. "10.14.0.0/24"
    home_subnet: Mapped[str] = mapped_column(String(32), nullable=False)

    # Behavioral baseline
    typical_resources: Mapped[list] = mapped_column(ARRAY(String), nullable=False)
    typical_auth_method: Mapped[str] = mapped_column(String(32), nullable=False)
    typical_os: Mapped[str] = mapped_column(String(64), nullable=False)
    typical_mac: Mapped[str] = mapped_column(String(32), nullable=False)
    typical_protocol: Mapped[str] = mapped_column(String(16), nullable=False)
    mean_session_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    std_session_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    raw_logs: Mapped[list["RawAccessLog"]] = relationship(back_populates="entity_profile")

    __table_args__ = (Index("ix_entity_profiles_entity_type", "entity_type"),)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<EntityProfile {self.entity_id} ({self.entity_type})>"


# ---------------------------------------------------------------------------
# raw_access_logs
# ---------------------------------------------------------------------------
class RawAccessLog(Base):
    """Append-only landing table for every raw access event."""

    __tablename__ = "raw_access_logs"

    log_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[str] = mapped_column(String(64), ForeignKey("entity_profiles.entity_id"), nullable=False)
    entity_type: Mapped[EntityType] = mapped_column(Enum(EntityType), nullable=False)

    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    source_ip: Mapped[str] = mapped_column(String(45), nullable=False)  # IPv4/IPv6

    # Geo location, stored both as JSONB (rich) and flattened lat/lon (fast filtering)
    geo_location: Mapped[dict] = mapped_column(JSONB, nullable=False)
    geo_lat: Mapped[float] = mapped_column(Float, nullable=False)
    geo_lon: Mapped[float] = mapped_column(Float, nullable=False)

    resource_accessed: Mapped[str] = mapped_column(String(128), nullable=False)
    auth_method: Mapped[str] = mapped_column(String(32), nullable=False)
    auth_result: Mapped[str] = mapped_column(String(16), nullable=False)  # success/failure

    session_duration: Mapped[float] = mapped_column(Float, nullable=False)  # seconds

    command_sequence: Mapped[list] = mapped_column(ARRAY(String), nullable=False)

    # device_fingerprint: {os, mac, protocol}
    device_fingerprint: Mapped[dict] = mapped_column(JSONB, nullable=False)

    label: Mapped[LabelType] = mapped_column(Enum(LabelType), nullable=False, default=LabelType.normal)

    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    entity_profile: Mapped["EntityProfile"] = relationship(back_populates="raw_logs")
    processed_log: Mapped["ProcessedStreamingLog"] = relationship(back_populates="raw_log", uselist=False)

    __table_args__ = (
        Index("ix_raw_access_logs_entity_ts", "entity_id", "timestamp"),
        Index("ix_raw_access_logs_label", "label"),
        Index("ix_raw_access_logs_source_ip", "source_ip"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RawAccessLog {self.log_id} entity={self.entity_id} label={self.label}>"


# ---------------------------------------------------------------------------
# processed_streaming_logs
# ---------------------------------------------------------------------------
class ProcessedStreamingLog(Base):
    """Feature-engineered / scored version of a raw log, produced downstream."""

    __tablename__ = "processed_streaming_logs"

    processed_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_log_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("raw_access_logs.log_id"), nullable=False, unique=True)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Engineered features consumed by the model
    seconds_since_last_event: Mapped[float] = mapped_column(Float, nullable=True)
    geo_velocity_kmh: Mapped[float] = mapped_column(Float, nullable=True)
    resource_rarity_score: Mapped[float] = mapped_column(Float, nullable=True)
    command_entropy: Mapped[float] = mapped_column(Float, nullable=True)
    auth_failure_rate_5min: Mapped[float] = mapped_column(Float, nullable=True)
    is_new_device: Mapped[bool] = mapped_column(Boolean, default=False)
    is_new_geo: Mapped[bool] = mapped_column(Boolean, default=False)

    feature_vector: Mapped[dict] = mapped_column(JSONB, nullable=True)  # full feature dump for model input

    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # 0-1
    predicted_label: Mapped[str] = mapped_column(String(48), nullable=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=True)

    processed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    raw_log: Mapped["RawAccessLog"] = relationship(back_populates="processed_log")
    alerts: Mapped[list["AlertQueue"]] = relationship(back_populates="processed_log")

    __table_args__ = (
        Index("ix_processed_logs_entity_ts", "entity_id", "timestamp"),
        Index("ix_processed_logs_score", "anomaly_score"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ProcessedStreamingLog {self.processed_id} score={self.anomaly_score:.3f}>"


# ---------------------------------------------------------------------------
# alert_queue
# ---------------------------------------------------------------------------
class AlertQueue(Base):
    """Actionable alert raised from a scored/processed log."""

    __tablename__ = "alert_queue"

    alert_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    processed_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("processed_streaming_logs.processed_id"), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)

    alert_type: Mapped[str] = mapped_column(String(48), nullable=False)  # e.g. "brute_force"
    severity: Mapped[AlertSeverity] = mapped_column(Enum(AlertSeverity), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)

    status: Mapped[AlertStatus] = mapped_column(Enum(AlertStatus), nullable=False, default=AlertStatus.open)
    details: Mapped[dict] = mapped_column(JSONB, nullable=True)  # supporting evidence / explainability

    assigned_to: Mapped[str] = mapped_column(String(64), nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    resolved_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    processed_log: Mapped["ProcessedStreamingLog"] = relationship(back_populates="alerts")

    __table_args__ = (
        Index("ix_alert_queue_status", "status"),
        Index("ix_alert_queue_entity", "entity_id"),
        UniqueConstraint("processed_id", "alert_type", name="uq_alert_per_processed_type"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AlertQueue {self.alert_id} type={self.alert_type} status={self.status}>"


def create_all(engine):
    """Convenience helper: create every table defined above."""
    Base.metadata.create_all(engine)


def drop_all(engine):
    """Convenience helper: drop every table defined above (destructive!)."""
    Base.metadata.drop_all(engine)
