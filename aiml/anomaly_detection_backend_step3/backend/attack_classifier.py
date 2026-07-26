"""
attack_classifier.py
=====================
Multi-class attack classifier: takes the outputs of the two upstream
detectors -- the Isolation Forest / HalfSpaceTrees baseline in
`baseline_profiling.py` and the Bi-LSTM sequence model in
`sequence_model.py` -- plus raw session metadata, and classifies each
flagged event into one of the 8 `LabelType` classes (normal + 7 attack
patterns).

Why a second (XGBoost) stage on top of two detectors
-----------------------------------------------------
The IForest baseline and the Bi-LSTM are both *detectors*: they're good
at producing a scalar "how weird is this" signal but not at naming
*which* attack pattern is happening -- e.g. a high IForest score and a
high Bi-LSTM sequence loss look similar whether it's brute_force or
credential_stuffing. XGBoost, trained on labeled synthetic data, learns
to combine those two continuous anomaly signals with structured session
metadata (auth failures, geo velocity, resource sensitivity, entity
type, ...) into a calibrated multi-class decision -- this is the
signal that ultimately drives `alert_queue.alert_type` / `severity`.

Feature contract
----------------
`AttackFeatureBuilder.build()` produces one row per event with columns:
    iforest_score            -- HybridBehavioralProfiler.combined_score (or batch_score)
    iforest_is_anomaly       -- 0/1
    bilstm_sequence_loss     -- BiLSTMTrainer.extract_features(...).sequence_loss
    bilstm_normal_prob       -- 1 - (anomaly signal), convenience feature
    session_duration
    hour_of_day
    is_auth_failure
    num_commands
    is_sensitive_resource
    entity_type (one-hot: user / service_account / edge_device)
    geo_distance_km (optional, if supplied)
plus the ground-truth `label` column for training.

This module doesn't compute the IForest/Bi-LSTM scores itself -- see
`attack_pipeline.py` for the glue that runs `HybridBehavioralProfiler`
and `BiLSTMTrainer.extract_features` and calls `AttackFeatureBuilder`
with their outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb

try:
    from models import EntityType, LabelType
except ImportError:  # pragma: no cover - standalone use
    import enum

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

from sequence_model import LABEL_ORDER, LABEL_TO_ID

ENTITY_TYPE_ORDER: List[str] = [e.value for e in EntityType]

SENSITIVE_SUBSTRINGS = ("finance", "customer_db", "billing")

FEATURE_COLUMNS: List[str] = [
    "iforest_score",
    "iforest_is_anomaly",
    "bilstm_sequence_loss",
    "bilstm_normal_prob",
    "session_duration",
    "hour_sin",
    "hour_cos",
    "is_auth_failure",
    "num_commands",
    "is_sensitive_resource",
    "geo_distance_km",
] + [f"entity_type_{t}" for t in ENTITY_TYPE_ORDER]


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
@dataclass
class EventContext:
    """One row of raw signal to be turned into a feature vector.

    `iforest_score` / `iforest_is_anomaly` come from
    `HybridBehavioralProfiler.observe(...)` (see `baseline_profiling.py`).
    `bilstm_sequence_loss` / `bilstm_normal_prob` come from
    `BiLSTMTrainer.extract_features(...)` (see `sequence_model.py`),
    joined on (entity_id, timestamp).
    """

    entity_id: str
    entity_type: str
    timestamp: pd.Timestamp
    session_duration: float
    auth_result: str
    command_sequence: List[str]
    resource_accessed: str
    geo_distance_km: float
    iforest_score: float
    iforest_is_anomaly: bool
    bilstm_sequence_loss: float
    bilstm_normal_prob: float
    label: Optional[str] = None


class AttackFeatureBuilder:
    """Joins IForest + Bi-LSTM signals with session metadata into the
    flat feature table XGBoost trains/predicts on."""

    def build(self, contexts: List[EventContext]) -> pd.DataFrame:
        rows = []
        for ctx in contexts:
            hour = ctx.timestamp.hour + ctx.timestamp.minute / 60.0
            row: Dict[str, float] = {
                "iforest_score": float(ctx.iforest_score),
                "iforest_is_anomaly": float(bool(ctx.iforest_is_anomaly)),
                "bilstm_sequence_loss": float(ctx.bilstm_sequence_loss),
                "bilstm_normal_prob": float(ctx.bilstm_normal_prob),
                "session_duration": float(ctx.session_duration),
                "hour_sin": math.sin(2 * math.pi * hour / 24.0),
                "hour_cos": math.cos(2 * math.pi * hour / 24.0),
                "is_auth_failure": 1.0 if ctx.auth_result != "success" else 0.0,
                "num_commands": float(len(ctx.command_sequence or [])),
                "is_sensitive_resource": 1.0 if any(s in ctx.resource_accessed for s in SENSITIVE_SUBSTRINGS) else 0.0,
                "geo_distance_km": float(ctx.geo_distance_km),
            }
            for t in ENTITY_TYPE_ORDER:
                row[f"entity_type_{t}"] = 1.0 if ctx.entity_type == t else 0.0
            if ctx.label is not None:
                row["label"] = ctx.label
                row["_entity_id"] = ctx.entity_id
            rows.append(row)

        df = pd.DataFrame(rows)
        # Guarantee stable column order/presence even if a batch happens
        # to omit a class of entity_type etc.
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        return df

    @staticmethod
    def from_dataframe(
        events_df: pd.DataFrame,
        iforest_scores: pd.DataFrame,
        bilstm_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convenience path used by `attack_pipeline.py`: join three
        already-computed tables on (entity_id, timestamp) instead of
        constructing `EventContext` objects one at a time.

        - events_df: raw_access_logs-shaped rows (entity_id, entity_type,
          timestamp, session_duration, auth_result, command_sequence,
          resource_accessed, geo_lat/geo_lon or geo_distance_km, label)
        - iforest_scores: columns [entity_id, timestamp, iforest_score, iforest_is_anomaly]
        - bilstm_features: columns [entity_id, timestamp, bilstm_sequence_loss, bilstm_normal_prob]
        """
        merged = events_df.merge(iforest_scores, on=["entity_id", "timestamp"], how="left")
        merged = merged.merge(bilstm_features, on=["entity_id", "timestamp"], how="left")
        merged[["iforest_score", "iforest_is_anomaly", "bilstm_sequence_loss", "bilstm_normal_prob"]] = (
            merged[["iforest_score", "iforest_is_anomaly", "bilstm_sequence_loss", "bilstm_normal_prob"]].fillna(0.0)
        )
        if "geo_distance_km" not in merged.columns:
            merged["geo_distance_km"] = 0.0

        builder = AttackFeatureBuilder()
        contexts = [
            EventContext(
                entity_id=r["entity_id"],
                entity_type=r["entity_type"],
                timestamp=r["timestamp"],
                session_duration=r.get("session_duration", 0.0),
                auth_result=r.get("auth_result", "success"),
                command_sequence=r.get("command_sequence", []) or [],
                resource_accessed=r.get("resource_accessed", ""),
                geo_distance_km=r.get("geo_distance_km", 0.0),
                iforest_score=r["iforest_score"],
                iforest_is_anomaly=bool(r["iforest_is_anomaly"]),
                bilstm_sequence_loss=r["bilstm_sequence_loss"],
                bilstm_normal_prob=r["bilstm_normal_prob"],
                label=r.get("label"),
            )
            for _, r in merged.iterrows()
        ]
        return builder.build(contexts)


# ---------------------------------------------------------------------------
# XGBoost classifier
# ---------------------------------------------------------------------------
class XGBoostAttackClassifier:
    """Multi-class (8-way) attack classifier on top of assembled features.

    Label ids follow `sequence_model.LABEL_ORDER` (0 = normal), so this
    classifier's predictions line up 1:1 with the Bi-LSTM's own label
    space and with `models.LabelType`.
    """

    def __init__(
        self,
        num_classes: int = len(LABEL_ORDER),
        max_depth: int = 6,
        n_estimators: int = 300,
        learning_rate: float = 0.08,
        random_state: int = 42,
        **xgb_kwargs,
    ):
        self.feature_columns = FEATURE_COLUMNS
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
            **xgb_kwargs,
        )
        self.fitted = False

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Inverse-frequency sample weights -- attacks are 0.5-3% of
        volume each, so unweighted training would just predict 'normal'."""
        counts = np.bincount(y, minlength=len(LABEL_ORDER)).astype(np.float64)
        counts[counts == 0] = 1.0
        class_weight = counts.sum() / (len(LABEL_ORDER) * counts)
        return class_weight[y]

    def train(
        self,
        features_df: pd.DataFrame,
        label_col: str = "label",
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ) -> Dict[str, object]:
        """Train on an `AttackFeatureBuilder` output that also has the
        ground-truth `label` (string) column. Returns a dict with a held-out
        classification report and confusion matrix for quick sanity-checking."""
        X = features_df[self.feature_columns].values
        y = features_df[label_col].map(LABEL_TO_ID).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        sw_train = self._sample_weights(y_train)

        self.model.fit(X_train, y_train, sample_weight=sw_train)
        self.fitted = True

        y_pred = self.model.predict(X_test)
        report = classification_report(
            y_test, y_pred, labels=list(range(len(LABEL_ORDER))),
            target_names=LABEL_ORDER, output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(LABEL_ORDER))))

        if verbose:
            print(classification_report(
                y_test, y_pred, labels=list(range(len(LABEL_ORDER))),
                target_names=LABEL_ORDER, zero_division=0,
            ))

        return {"classification_report": report, "confusion_matrix": cm}

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Returns predicted `LabelType` string per row."""
        self._check_fitted()
        X = features_df[self.feature_columns].values
        pred_ids = self.model.predict(X)
        return np.array([LABEL_ORDER[i] for i in pred_ids])

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X = features_df[self.feature_columns].values
        probs = self.model.predict_proba(X)
        return pd.DataFrame(probs, columns=LABEL_ORDER, index=features_df.index)

    def feature_importances(self) -> pd.Series:
        self._check_fitted()
        return pd.Series(self.model.feature_importances_, index=self.feature_columns).sort_values(ascending=False)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("XGBoostAttackClassifier must be trained (or loaded) before predicting.")

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns}, path)

    @classmethod
    def load(cls, path: str) -> "XGBoostAttackClassifier":
        payload = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = payload["model"]
        obj.feature_columns = payload["feature_columns"]
        obj.fitted = True
        return obj
