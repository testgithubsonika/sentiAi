"""
explainability.py
===================
SHAP-based explainability layer. Turns the raw outputs of the two
tree-based models in the stack -- the per-entity/global Isolation Forest
in `baseline_profiling.py` and the XGBoost attack classifier in
`attack_classifier.py` -- into feature-attribution scores, and then into
short, human-readable "contributing factors" a SOC analyst can scan in
under a second (e.g. "flagged due to geo-velocity + new device
fingerprint + off-hours access").

Both models are tree ensembles, so `shap.TreeExplainer` (fast, exact,
no background-distribution sampling needed) is used for both -- no
KernelExplainer/approximation required anywhere in this module.

Design
------
- `FeatureExplainer.explain_iforest(...)`  -- attributions for *why* the
  Isolation Forest called an event anomalous, in the entity's own
  (`login_hour`, `session_duration`, `geo_distance_km`, `failure_count`,
  `num_commands`) feature space.
- `FeatureExplainer.explain_xgb(...)`      -- attributions for *why*
  XGBoost assigned the predicted attack class, in the
  `attack_classifier.FEATURE_COLUMNS` space (which already includes the
  IForest score and Bi-LSTM sequence loss as inputs, so this layer's
  attributions naturally surface "was it mostly the baseline deviation,
  the sequence anomaly, or the raw session metadata that drove this
  classification").
- `combine_and_summarize(...)`             -- merges both attribution
  sets, ranks by |SHAP value|, and renders the top-k into the
  `"flagged due to A + B + C"` string used in the alert object.

TreeExplainer objects are expensive to construct (they walk the full
tree structure once), so callers should build one `FeatureExplainer` per
trained model and reuse it across events -- see `AnomalyDetectionPipeline`
in `anomaly_pipeline.py`, which does exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import shap

from attack_classifier import FEATURE_COLUMNS, XGBoostAttackClassifier
from sequence_model import LABEL_ORDER

# Isolation-Forest-space feature order, matching
# `baseline_profiling.FeatureVector.feature_names()` when `extra` is
# built with a single "num_commands" key (as `attack_pipeline.py` does).
IFOREST_FEATURE_NAMES: List[str] = [
    "login_hour", "session_duration", "geo_distance_km", "failure_count", "num_commands",
]

# ---------------------------------------------------------------------------
# Human-readable phrasing
# ---------------------------------------------------------------------------
# Static, feature-agnostic fallback labels. Anything with custom logic
# (needs the raw value to phrase well) is handled in `_phrase()` instead.
_FRIENDLY_LABELS: Dict[str, str] = {
    "login_hour": "unusual login hour",
    "session_duration": "session duration deviating from baseline",
    "geo_distance_km": "geo-velocity / distance from home location",
    "failure_count": "authentication failures",
    "num_commands": "command burst",
    "iforest_score": "deviation from entity's historical baseline (Isolation Forest)",
    "iforest_is_anomaly": "flagged by baseline Isolation Forest",
    "bilstm_sequence_loss": "anomalous event sequence / temporal pattern (Bi-LSTM)",
    "bilstm_normal_prob": "sequence looks atypical for this entity (Bi-LSTM)",
    "session_duration_z": "session duration deviating from baseline",
    "hour_sin": "off-hours access timing",
    "hour_cos": "off-hours access timing",
    "is_auth_failure": "authentication failure on this event",
    "is_sensitive_resource": "access to a sensitive resource (finance/customer_db/billing)",
    "entity_type_user": "human-user account context",
    "entity_type_service_account": "service-account context",
    "entity_type_edge_device": "edge-device context",
}


def _phrase(feature_name: str, raw_value: Optional[float]) -> str:
    """Render one feature into an analyst-facing phrase, using the raw
    value where it makes the phrase materially clearer."""
    if feature_name == "geo_distance_km" and raw_value is not None:
        return f"unusual travel distance (~{raw_value:.0f} km from home location)"
    if feature_name == "failure_count" and raw_value is not None and raw_value > 0:
        return f"{int(raw_value)} authentication failure(s)"
    if feature_name == "num_commands" and raw_value is not None:
        return f"command burst ({int(raw_value)} commands in one event)"
    if feature_name == "bilstm_sequence_loss" and raw_value is not None:
        return f"anomalous event sequence (Bi-LSTM surprise score {raw_value:.2f})"
    if feature_name == "iforest_score" and raw_value is not None:
        return f"deviates from baseline (Isolation Forest score {raw_value:.2f})"
    if feature_name.startswith("entity_type_") and raw_value:
        return _FRIENDLY_LABELS.get(feature_name, feature_name)
    return _FRIENDLY_LABELS.get(feature_name, feature_name.replace("_", " "))


@dataclass
class ContributingFactor:
    """One SHAP-attributed feature, ready to render or sort on."""

    feature_name: str
    shap_value: float          # signed; positive = pushed toward the flagged/attack outcome
    raw_value: Optional[float]
    source: str                # "iforest" | "xgboost"
    description: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature_name,
            "shap_value": round(float(self.shap_value), 4),
            "raw_value": None if self.raw_value is None else round(float(self.raw_value), 4),
            "source": self.source,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------
class FeatureExplainer:
    """Owns SHAP TreeExplainers for the XGBoost classifier and (lazily,
    since there can be many) per-entity/global Isolation Forest models."""

    def __init__(self, xgb_classifier: XGBoostAttackClassifier):
        if not xgb_classifier.fitted:
            raise RuntimeError("XGBoostAttackClassifier must be trained before building a FeatureExplainer.")
        self.xgb_classifier = xgb_classifier
        self._xgb_explainer = shap.TreeExplainer(xgb_classifier.model)
        # keyed by id(sklearn IsolationForest) so each distinct fitted
        # per-entity/global model gets its own explainer, built once.
        self._iforest_explainers: Dict[int, shap.TreeExplainer] = {}

    # -- Isolation Forest ---------------------------------------------------
    def explain_iforest(
        self,
        pyod_iforest_model,
        scaler,
        x_raw: np.ndarray,
        feature_names: Sequence[str] = IFOREST_FEATURE_NAMES,
        top_k: int = 3,
    ) -> List[ContributingFactor]:
        """`pyod_iforest_model` is a fitted `pyod.models.iforest.IForest`
        (e.g. `baseline_profiler.entities[entity_id].model` or
        `baseline_profiler.global_model.model`); `scaler` is the matching
        fitted `StandardScaler` it was trained behind. `x_raw` is the
        *unscaled* feature vector (`FeatureVector.to_array()`)."""
        sklearn_model = pyod_iforest_model.detector_  # underlying sklearn IsolationForest
        key = id(sklearn_model)
        if key not in self._iforest_explainers:
            self._iforest_explainers[key] = shap.TreeExplainer(sklearn_model)
        explainer = self._iforest_explainers[key]

        x_scaled = scaler.transform(x_raw.reshape(1, -1))
        shap_values = explainer.shap_values(x_scaled)
        values = np.asarray(shap_values).reshape(-1)  # single-output model -> flat per-feature

        factors = [
            ContributingFactor(
                feature_name=name,
                shap_value=float(values[i]),
                raw_value=float(x_raw[i]),
                source="iforest",
                description=_phrase(name, float(x_raw[i])),
            )
            for i, name in enumerate(feature_names)
        ]
        factors.sort(key=lambda f: abs(f.shap_value), reverse=True)
        return factors[:top_k]

    # -- XGBoost --------------------------------------------------------------
    def explain_xgb(
        self,
        features_row: pd.DataFrame,
        predicted_label_id: int,
        top_k: int = 3,
    ) -> List[ContributingFactor]:
        """`features_row` is a single-row DataFrame with
        `attack_classifier.FEATURE_COLUMNS` (as produced by
        `AttackFeatureBuilder`)."""
        X = features_row[self.xgb_classifier.feature_columns].values
        raw_shap = self._xgb_explainer.shap_values(X)
        class_values = _select_class_shap(raw_shap, predicted_label_id, num_classes=len(LABEL_ORDER))
        row_values = class_values[0]  # single row

        raw_row = features_row[self.xgb_classifier.feature_columns].iloc[0]
        factors = [
            ContributingFactor(
                feature_name=name,
                shap_value=float(row_values[i]),
                raw_value=float(raw_row[name]),
                source="xgboost",
                description=_phrase(name, float(raw_row[name])),
            )
            for i, name in enumerate(self.xgb_classifier.feature_columns)
        ]
        factors.sort(key=lambda f: abs(f.shap_value), reverse=True)
        return factors[:top_k]

    # -- combined summary -----------------------------------------------------
    def combine_and_summarize(
        self,
        iforest_factors: List[ContributingFactor],
        xgb_factors: List[ContributingFactor],
        top_k: int = 3,
    ) -> Dict[str, object]:
        """Merge both attribution sets (de-duplicating near-identical
        signals like `iforest_score` appearing in both spaces), rank by
        |SHAP value|, and render the analyst-facing summary string."""
        all_factors = iforest_factors + xgb_factors
        seen_descriptions = set()
        deduped: List[ContributingFactor] = []
        for f in sorted(all_factors, key=lambda f: abs(f.shap_value), reverse=True):
            if f.description in seen_descriptions:
                continue
            seen_descriptions.add(f.description)
            deduped.append(f)

        top = deduped[:top_k]
        # Only surface factors that actually pushed *toward* the flagged
        # outcome (positive SHAP); a large negative contributor is
        # evidence *against* the alert and would confuse a one-line summary.
        positive_top = [f for f in top if f.shap_value > 0] or top[:1]
        summary = "flagged due to " + " + ".join(f.description for f in positive_top) if positive_top else "flagged (no single dominant factor)"

        return {
            "summary": summary,
            "top_factors": [f.to_dict() for f in top],
        }


def _select_class_shap(shap_values, class_id: int, num_classes: int) -> np.ndarray:
    """Normalizes across the several shapes `shap.TreeExplainer.shap_values()`
    has returned for multiclass models across SHAP versions:
      - list of `num_classes` arrays, each (n_samples, n_features)
      - ndarray (n_samples, n_features, n_classes)
      - ndarray (n_classes, n_samples, n_features)
      - ndarray (n_samples, n_features) for a binary/single-output model
    Always returns (n_samples, n_features) for the requested class.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_id])

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[-1] == num_classes:
            return arr[:, :, class_id]
        if arr.shape[0] == num_classes:
            return arr[class_id]
        # Fall back to last axis if shapes are ambiguous but sizes still match.
        raise ValueError(f"Unexpected SHAP output shape {arr.shape} for {num_classes} classes.")
    raise ValueError(f"Unexpected SHAP output shape {arr.shape}.")
