"""
anomaly_pipeline.py
=====================
The main orchestrator. `AnomalyDetectionPipeline.process_event()` takes a
single incoming raw access-log event and runs it through the full stack:

    a) Isolation Forest / online baseline check   (baseline_profiling.py)
    b) Bi-LSTM sequence evaluation, if the entity has enough recent
       history to form sequential context           (sequence_model.py)
    c) XGBoost multi-class attack classification, only if (a) or (b)
       flagged the event as anomalous                (attack_classifier.py)
    d) SHAP feature attribution -> human-readable contributing factors,
       a final risk score, and an `Alert` object      (explainability.py)

This mirrors how the system would run in production against a live
stream (e.g. a Kafka consumer writing into `raw_access_logs` and, for
flagged events, `processed_streaming_logs` / `alert_queue` per the
schema in `models.py`) -- one event in, at most one `Alert` out, with
the expensive per-model work (SHAP, XGBoost, Bi-LSTM) only happening
once an event has already been flagged by cheaper upstream checks.

See `run_pipeline_demo()` at the bottom for an end-to-end example: train
on a batch of synthetic data, then stream fresh events one at a time
through `process_event()`.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from attack_classifier import AttackFeatureBuilder, EventContext, XGBoostAttackClassifier
from baseline_profiling import FeatureVector, HybridBehavioralProfiler
from explainability import FeatureExplainer
from sequence_model import (
    BiLSTMTrainer,
    LABEL_ORDER,
    SequenceWindowBuilder,
    Vocabulary,
    WindowBatch,
)

try:
    from models import AlertSeverity, AlertStatus
except ImportError:  # pragma: no cover - standalone use
    import enum

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


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


# ---------------------------------------------------------------------------
# Alert object
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    """Final output of the pipeline for one flagged event. Shape mirrors
    `models.AlertQueue` closely enough that `to_alert_queue_kwargs()` can
    build the ORM row directly -- `processed_id` is left for the caller
    to fill in once the corresponding `ProcessedStreamingLog` row exists."""

    alert_id: str
    entity_id: str
    timestamp: datetime
    alert_type: str                          # predicted LabelType value, e.g. "brute_force"
    severity: "AlertSeverity"
    risk_score: float                        # 0-1
    summary: str                             # "flagged due to A + B + C"
    contributing_factors: List[Dict[str, Any]]
    details: Dict[str, Any] = field(default_factory=dict)
    status: "AlertStatus" = AlertStatus.open

    def to_alert_queue_kwargs(self, processed_id: Optional[str] = None) -> Dict[str, Any]:
        return dict(
            processed_id=processed_id,
            entity_id=self.entity_id,
            alert_type=self.alert_type,
            severity=self.severity,
            anomaly_score=self.risk_score,
            status=self.status,
            details={**self.details, "summary": self.summary, "contributing_factors": self.contributing_factors},
        )

    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.entity_id} @ {self.timestamp} "
            f"-> {self.alert_type} (risk={self.risk_score:.2f})\n"
            f"  {self.summary}"
        )


def _severity_from_risk(risk_score: float, thresholds=(0.35, 0.6, 0.85)) -> AlertSeverity:
    low_t, med_t, high_t = thresholds
    if risk_score >= high_t:
        return AlertSeverity.critical
    if risk_score >= med_t:
        return AlertSeverity.high
    if risk_score >= low_t:
        return AlertSeverity.medium
    return AlertSeverity.low


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class AnomalyDetectionPipeline:
    """Stateful, per-event orchestrator. Construct once (it owns the
    behavioral profiler's per-entity model state and rolling per-entity
    event history), then call `process_event()` once per incoming log
    line, in timestamp order.
    """

    def __init__(
        self,
        entity_profiles_df: pd.DataFrame,
        xgb_classifier: XGBoostAttackClassifier,
        bilstm_trainer: BiLSTMTrainer,
        resource_vocab: Vocabulary,
        command_vocab: Vocabulary,
        window_size: int = 10,
        behavioral_profiler: Optional[HybridBehavioralProfiler] = None,
        min_history_for_bilstm: int = 2,
        bilstm_normal_prob_threshold: float = 0.5,
        severity_thresholds: tuple = (0.35, 0.6, 0.85),
        random_state: int = 42,
    ):
        self.entity_profiles = entity_profiles_df.set_index("entity_id", drop=False)
        self.xgb_classifier = xgb_classifier
        self.bilstm_trainer = bilstm_trainer
        self.window_builder = SequenceWindowBuilder(resource_vocab, command_vocab, window_size=window_size)
        self.window_size = window_size
        self.min_history_for_bilstm = min_history_for_bilstm
        self.bilstm_normal_prob_threshold = bilstm_normal_prob_threshold
        self.severity_thresholds = severity_thresholds

        # Reuse a warmed-up profiler if given (e.g. the one used during
        # training, so per-entity baselines aren't cold on day one of
        # "live" traffic); otherwise start fresh.
        self.behavioral_profiler = behavioral_profiler or HybridBehavioralProfiler(random_state=random_state)
        self.explainer = FeatureExplainer(xgb_classifier)

        self._entity_history: Dict[str, List[dict]] = {}

    @classmethod
    def load_pretrained(
        cls,
        model_dir: str,
        entity_profiles_df: Optional[pd.DataFrame] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "AnomalyDetectionPipeline":
        """Build a ready-to-run `AnomalyDetectionPipeline` straight from
        artifacts written by `attack_pipeline.AttackClassificationPipeline.save()`
        -- no training/fit() involved anywhere in this path. This is the
        inference-only entry point: load once at process startup, then
        call `process_event()` per incoming log line.

        `**kwargs` are passed through to `__init__` (e.g.
        `bilstm_normal_prob_threshold`, `severity_thresholds`) so callers
        can still tune alerting behavior without retraining anything.
        """
        # Deferred import: attack_pipeline -> anomaly_pipeline would be
        # circular if imported at module level (attack_pipeline doesn't
        # import this module, but keeping the dependency one-directional
        # and load-time-only avoids ever having to care about the order).
        from attack_pipeline import AttackClassificationPipeline

        trained = AttackClassificationPipeline.load_pretrained(
            model_dir, entity_profiles_df=entity_profiles_df, device=device,
        )
        kwargs.setdefault("random_state", trained.random_state)
        return cls(
            entity_profiles_df=trained.entity_profiles,
            xgb_classifier=trained.xgb_classifier,
            bilstm_trainer=trained.bilstm_trainer,
            resource_vocab=trained.resource_vocab,
            command_vocab=trained.command_vocab,
            window_size=trained.window_size,
            behavioral_profiler=trained.behavioral_profiler,
            **kwargs,
        )

    def save_state(self, model_dir: str) -> None:
        """Checkpoint the *mutable* part of a running pipeline -- the
        behavioral profiler, whose per-entity baselines keep learning
        online as `process_event()` is called. The Bi-LSTM and XGBoost
        weights loaded via `load_pretrained()` don't change during
        inference, so there's nothing new to persist for them; this just
        lets a long-running process checkpoint its warmed-up baselines
        periodically (e.g. before a restart) without losing that state."""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "behavioral_profiler.joblib")
        self.behavioral_profiler.save(path)
        print(f"Checkpointed behavioral profiler state to {path}")

    # -- helpers ---------------------------------------------------------
    def _normalize_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        event = dict(raw_event)
        ts = event["timestamp"]
        event["timestamp"] = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)
        event.setdefault("command_sequence", [])
        event.setdefault("session_duration", 0.0)
        event.setdefault("auth_result", "success")
        event.setdefault("resource_accessed", "")

        entity_id = event["entity_id"]
        if "entity_type" not in event or event["entity_type"] is None:
            event["entity_type"] = (
                self.entity_profiles.loc[entity_id, "entity_type"] if entity_id in self.entity_profiles.index else "user"
            )

        geo = event.get("geo_location")
        if isinstance(geo, dict) and "lat" in geo:
            lat, lon = geo["lat"], geo["lon"]
        else:
            lat, lon = event.get("geo_lat"), event.get("geo_lon")

        if lat is not None and lon is not None and entity_id in self.entity_profiles.index:
            profile = self.entity_profiles.loc[entity_id]
            event["geo_distance_km"] = _haversine_km(lat, lon, profile["home_lat"], profile["home_lon"])
        else:
            event["geo_distance_km"] = 0.0
        return event

    def _bilstm_score(self, entity_id: str) -> Dict[str, float]:
        history = self._entity_history.get(entity_id, [])
        if len(history) < self.min_history_for_bilstm:
            return {"sequence_loss": 0.0, "normal_prob": 1.0}

        df = pd.DataFrame(history[-self.window_size:])
        df["label"] = "normal"  # placeholder; unused at inference, WindowBatch requires the column
        batch = self.window_builder.build(df)
        # `build()` returns one window per position in the (small) history
        # df; the *last* row is the window ending at the current event,
        # which is the one we want.
        last = WindowBatch(
            entity_ids=[batch.entity_ids[-1]],
            end_timestamps=[batch.end_timestamps[-1]],
            resource_ids=batch.resource_ids[-1:],
            command_ids=batch.command_ids[-1:],
            numeric_feats=batch.numeric_feats[-1:],
            lengths=batch.lengths[-1:],
            label_ids=batch.label_ids[-1:],
        )
        seq_features = self.bilstm_trainer.extract_features(last)
        return {
            "sequence_loss": float(seq_features.sequence_loss[0]),
            "normal_prob": float(seq_features.normal_probability[0]),
        }

    def _get_iforest_model_and_scaler(self, entity_id: str, source: str):
        batch_profiler = self.behavioral_profiler.batch_profiler
        if source == "entity_model":
            entity_model = batch_profiler.entities.get(entity_id)
            if entity_model is not None and entity_model.model is not None:
                return entity_model.model, entity_model.scaler
        elif source == "global_fallback":
            if batch_profiler.global_model.fitted:
                return batch_profiler.global_model.model, batch_profiler.global_model.scaler
        return None, None

    # -- main entry point --------------------------------------------------
    def process_event(self, raw_event: Dict[str, Any]) -> Optional[Alert]:
        """Run one event through the full pipeline. Returns an `Alert` if
        the event was ultimately classified as an attack, otherwise
        `None`. Always ingests the event into rolling state regardless
        of outcome, so later events benefit from it."""
        event = self._normalize_event(raw_event)
        entity_id, timestamp = event["entity_id"], event["timestamp"]

        # -- (a) Isolation Forest / online baseline check -----------------
        fv = FeatureVector(
            login_hour=float(timestamp.hour),
            session_duration=float(event["session_duration"]),
            geo_distance_km=float(event["geo_distance_km"]),
            failure_count=1.0 if event["auth_result"] != "success" else 0.0,
            extra={"num_commands": float(len(event["command_sequence"]))},
        )
        baseline_result = self.behavioral_profiler.observe(entity_id, fv)
        print(
            baseline_result.is_anomaly,
            baseline_result.combined_score,
            bilstm["normal_prob"] if "bilstm" in locals() else "NA",
        )

        # Record history *after* scoring (so the baseline was judged
        # against the past, not including this event), then update the
        # rolling window used for Bi-LSTM sequential context.
        history = self._entity_history.setdefault(entity_id, [])
        history.append(event)
        if len(history) > self.window_size:
            del history[: len(history) - self.window_size]

        # -- (b) Bi-LSTM sequence evaluation, if sequential context exists --
        bilstm = self._bilstm_score(entity_id)
        bilstm_flagged = bilstm["normal_prob"] < self.bilstm_normal_prob_threshold

        if not (baseline_result.is_anomaly or bilstm_flagged):
            return None  # neither upstream detector flagged this event -- no alert, no further work

        # -- (c) XGBoost multi-class attack classification -----------------
        context = EventContext(
            entity_id=entity_id,
            entity_type=event["entity_type"],
            timestamp=timestamp,
            session_duration=event["session_duration"],
            auth_result=event["auth_result"],
            command_sequence=event["command_sequence"],
            resource_accessed=event["resource_accessed"],
            geo_distance_km=event["geo_distance_km"],
            iforest_score=baseline_result.combined_score,
            iforest_is_anomaly=baseline_result.is_anomaly,
            bilstm_sequence_loss=bilstm["sequence_loss"],
            bilstm_normal_prob=bilstm["normal_prob"],
            label=None,
        )
        features_df = AttackFeatureBuilder().build([context])
        probs = self.xgb_classifier.predict_proba(features_df).iloc[0]
        predicted_label = probs.idxmax()
        print("Predicted:", predicted_label)
        predicted_label_id = LABEL_ORDER.index(predicted_label)
        attack_probability = 1.0 - float(probs["normal"])

        if predicted_label == "normal":
            # XGBoost is the final arbiter: the upstream detectors flagged
            # something, but the classifier -- trained on labeled attack
            # patterns -- doesn't recognize this as any of them. Treat as
            # a suppressed false positive rather than raising an alert.
            return None

        # -- (d) SHAP explainability -> contributing factors + risk score --
        iforest_model, iforest_scaler = self._get_iforest_model_and_scaler(entity_id, baseline_result.batch_source)
        iforest_factors = (
            self.explainer.explain_iforest(iforest_model, iforest_scaler, fv.to_array(), feature_names=fv.feature_names())
            if iforest_model is not None else []
        )
        xgb_factors = self.explainer.explain_xgb(features_df, predicted_label_id)
        combined = self.explainer.combine_and_summarize(iforest_factors, xgb_factors)

        risk_score = attack_probability
        severity = _severity_from_risk(risk_score, self.severity_thresholds)

        return Alert(
            alert_id=str(uuid.uuid4()),
            entity_id=entity_id,
            timestamp=timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp,
            alert_type=predicted_label,
            severity=severity,
            risk_score=risk_score,
            summary=combined["summary"],
            contributing_factors=combined["top_factors"],
            details={
                "iforest_score": baseline_result.combined_score,
                "iforest_batch_source": baseline_result.batch_source,
                "iforest_is_anomaly": baseline_result.is_anomaly,
                "bilstm_sequence_loss": bilstm["sequence_loss"],
                "bilstm_normal_prob": bilstm["normal_prob"],
                "xgb_class_probabilities": probs.round(4).to_dict(),
            },
        )


# ---------------------------------------------------------------------------
# End-to-end example
# ---------------------------------------------------------------------------
def run_pipeline_demo(
    num_users: int = 60,
    num_service_accounts: int = 10,
    num_devices: int = 15,
    train_days: int = 10,
    stream_days: int = 2,
    seed: int = 7,
    model_dir: str = "models",
    force_retrain: bool = False,
) -> List[Alert]:
    """If `model_dir` already has saved artifacts (from a prior
    `attack_pipeline.py` or `anomaly_pipeline.py` run), loads them via
    `AnomalyDetectionPipeline.load_pretrained()` and skips straight to
    streaming -- no training. Otherwise trains the full stack on a batch
    of synthetic data, saves it to `model_dir`, and then streams a fresh
    batch of synthetic events (including freshly-injected attacks)
    through `AnomalyDetectionPipeline.process_event()` one at a time, in
    timestamp order, printing every resulting alert."""
    from datetime import timedelta

    from attack_pipeline import AttackClassificationPipeline, models_exist
    from data_generator import ATTACK_RATE_RANGE, SyntheticDataGenerator

    gen = SyntheticDataGenerator(
        num_users=num_users, num_service_accounts=num_service_accounts,
        num_devices=num_devices, seed=seed,
    )
    profiles_df = gen.profiles_dataframe()

    if models_exist(model_dir) and not force_retrain:
        print(f"=== Found existing model artifacts in {model_dir}/ -- loading pretrained (no training) ===")
        live_pipeline = AnomalyDetectionPipeline.load_pretrained(model_dir, entity_profiles_df=profiles_df, random_state=seed)
        stream_start = datetime.utcnow() - timedelta(days=stream_days)
    else:
        print(f"=== No usable model artifacts in {model_dir}/ -- training on {train_days} days of synthetic data ===")
        train_start = datetime.utcnow() - timedelta(days=train_days + stream_days)
        train_df = gen.generate(train_start, train_days, attack_rate_range=ATTACK_RATE_RANGE)

        training_pipeline = AttackClassificationPipeline(profiles_df, bilstm_epochs=6, random_state=seed)
        training_pipeline.fit(train_df)
        training_pipeline.save(model_dir)

        live_pipeline = AnomalyDetectionPipeline(
            entity_profiles_df=profiles_df,
            xgb_classifier=training_pipeline.xgb_classifier,
            bilstm_trainer=training_pipeline.bilstm_trainer,
            resource_vocab=training_pipeline.resource_vocab,
            command_vocab=training_pipeline.command_vocab,
            window_size=training_pipeline.window_size,
            behavioral_profiler=training_pipeline.behavioral_profiler,  # warm-started, not cold
            random_state=seed,
        )
        stream_start = train_start + timedelta(days=train_days)

    print(f"\n=== Streaming {stream_days} day(s) of fresh events through AnomalyDetectionPipeline.process_event() ===")
    stream_df = gen.generate(stream_start, stream_days, attack_rate_range=ATTACK_RATE_RANGE).sort_values("timestamp")

    alerts: List[Alert] = []
    for _, row in stream_df.iterrows():
        event = row.to_dict()
        alert = live_pipeline.process_event(event)
        if alert is not None:
            alerts.append(alert)
            print(alert)
            print(f"  ground_truth_label={row['label']}\n")

    print(f"\n=== {len(alerts)} alerts raised out of {len(stream_df)} streamed events ===")

    # Checkpoint the (possibly further-adapted) online baseline state so
    # a subsequent run picks up warm rather than cold.
    live_pipeline.save_state(model_dir)
    return alerts


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Load (or train) the full anomaly-detection stack and stream synthetic events through it."
    )
    parser.add_argument("--num-users", type=int, default=60)
    parser.add_argument("--num-service-accounts", type=int, default=10)
    parser.add_argument("--num-devices", type=int, default=15)
    parser.add_argument("--train-days", type=int, default=10)
    parser.add_argument("--stream-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if model_dir already has saved artifacts")
    args = parser.parse_args()

    run_pipeline_demo(
        num_users=args.num_users,
        num_service_accounts=args.num_service_accounts,
        num_devices=args.num_devices,
        train_days=args.train_days,
        stream_days=args.stream_days,
        seed=args.seed,
        model_dir=args.model_dir,
        force_retrain=args.force_retrain,
    )


if __name__ == "__main__":
    main()