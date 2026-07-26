"""
config.py
=========
Central, env-driven settings. Nothing fancy (no pydantic-settings
dependency) -- just `os.environ.get(...)` with sane hackathon defaults,
so the whole backend runs out of the box with zero configuration and
can be tuned via env vars for a real deployment.
"""

from __future__ import annotations

import os


def _bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    # -- Database ---------------------------------------------------------
    DATABASE_URL: str = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:Ms%408120150260@localhost:5432/anomaly_db",
    )
    DB_POOL_SIZE: int = int(os.environ.get("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.environ.get("DB_MAX_OVERFLOW", "20"))

    # -- CORS ---------------------------------------------------------------
    CORS_ORIGINS: list = [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
    ]

    # -- ML worker / training ------------------------------------------------
    AUTO_TRAIN_ON_STARTUP: bool = _bool("AUTO_TRAIN_ON_STARTUP", True)
    AUTO_STREAM_ON_STARTUP: bool = _bool("AUTO_STREAM_ON_STARTUP", True)

    TRAIN_NUM_USERS: int = int(os.environ.get("TRAIN_NUM_USERS", "60"))
    TRAIN_NUM_SERVICE_ACCOUNTS: int = int(os.environ.get("TRAIN_NUM_SERVICE_ACCOUNTS", "12"))
    TRAIN_NUM_DEVICES: int = int(os.environ.get("TRAIN_NUM_DEVICES", "18"))
    TRAIN_DAYS: int = int(os.environ.get("TRAIN_DAYS", "10"))
    BILSTM_EPOCHS: int = int(os.environ.get("BILSTM_EPOCHS", "4"))
    TRAIN_SEED: int = int(os.environ.get("TRAIN_SEED", "42"))
    MODEL_DIR: str = os.environ.get("MODEL_DIR", "models")

    STREAM_MIN_DELAY_SECONDS: float = float(os.environ.get("STREAM_MIN_DELAY_SECONDS", "0.4"))
    STREAM_MAX_DELAY_SECONDS: float = float(os.environ.get("STREAM_MAX_DELAY_SECONDS", "1.5"))
    STREAM_BATCH_HOURS: int = int(os.environ.get("STREAM_BATCH_HOURS", "6"))
    STREAM_EVENTS_PER_BATCH: int = int(os.environ.get("STREAM_EVENTS_PER_BATCH", "25"))

    SEVERITY_THRESHOLDS: tuple = (0.35, 0.6, 0.85)


settings = Settings()
