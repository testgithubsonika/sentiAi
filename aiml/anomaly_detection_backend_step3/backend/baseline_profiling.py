"""
baseline_profiling.py
======================
ML backend for behavioral baseline profiling and concept drift handling.

Two complementary layers, combined into one hybrid profiler:

1. Batch point-anomaly detection (PyOD `IForest`)
   -------------------------------------------------
   Each entity gets its own Isolation Forest, periodically refit on its
   rolling event history. Entities with sparse history (< `min_history`
   events, default 5) are cold-start: they are scored against a *global*
   Isolation Forest fit on pooled data across all entities, rather than
   against an unreliable few-sample model of their own.

2. Online concept-drift handling (River)
   -------------------------------------------------
   A fully online, always-learning layer per entity:
     - `river.preprocessing.StandardScaler` keeps an adaptive
       (streaming) mean/variance per feature, so "normal" is always
       relative to recent behavior, not a frozen snapshot.
     - `river.anomaly.HalfSpaceTrees` gives an anomaly score that
       updates every single event (useful between batch IForest
       retrains).
     - `river.drift.ADWIN` watches the online anomaly-score stream. When
       it fires, that's a signal the entity's behavior has *genuinely
       shifted* (new work hours, new legitimate device, new normal
       location) rather than a one-off intrusion. The hybrid profiler
       reacts by immediately retraining that entity's batch Isolation
       Forest on the recent window -- so the new normal is folded into
       the baseline instead of being flagged forever.

Author: ML backend, hackathon cybersecurity anomaly detection project.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler

from river import anomaly as river_anomaly
from river import drift as river_drift
from river import preprocessing as river_preprocessing
from river import stats as river_stats

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = ["login_hour", "session_duration", "geo_distance_km", "failure_count"]


@dataclass
class FeatureVector:
    """A single entity-event's feature vector.

    `extra` lets callers pass additional named features (e.g.
    `resource_rarity_score`, `command_entropy`) without changing this
    dataclass; they are appended after the core four in a stable order.
    """

    login_hour: float
    session_duration: float
    geo_distance_km: float
    failure_count: float
    extra: Dict[str, float] = field(default_factory=dict)

    def feature_names(self) -> List[str]:
        return FEATURE_NAMES + sorted(self.extra.keys())

    def to_array(self) -> np.ndarray:
        base = [self.login_hour, self.session_duration, self.geo_distance_km, self.failure_count]
        extra_vals = [self.extra[k] for k in sorted(self.extra.keys())]
        return np.array(base + extra_vals, dtype=float)

    def to_dict(self) -> Dict[str, float]:
        d = {
            "login_hour": self.login_hour,
            "session_duration": self.session_duration,
            "geo_distance_km": self.geo_distance_km,
            "failure_count": self.failure_count,
        }
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class BatchScoreResult:
    """Result of scoring one event against the batch (PyOD) layer."""

    entity_id: str
    score: Optional[float]        # PyOD decision_function value; higher = more abnormal. None if no model available.
    is_anomaly: bool
    source: str                   # "entity_model" | "global_fallback" | "cold_start_no_fallback"


@dataclass
class AnomalyResult:
    """Final, combined result returned by `HybridBehavioralProfiler.observe()`."""

    entity_id: str
    is_anomaly: bool
    combined_score: float
    batch_score: Optional[float]
    batch_source: str
    online_score: float           # River HalfSpaceTrees score, 0-1, higher = more abnormal
    drift_detected: bool          # True if ADWIN fired on this event (baseline just re-adapted)


# ---------------------------------------------------------------------------
# 1. Global fallback model (cold-start entities)
# ---------------------------------------------------------------------------
class GlobalFallbackModel:
    """A single Isolation Forest fit on pooled events across *all* entities.

    Used to score entities whose own history is too sparse (< `min_history`
    events) to train a reliable per-entity model. This avoids the classic
    cold-start failure mode where a brand-new entity's first few events
    either can't be scored at all, or get a degenerate/overfit model built
    from 1-4 data points.
    """

    def __init__(self, contamination: float = 0.02, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[IForest] = None
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X)
        self.model = IForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.model.fit(Xs)
        self.fitted = True

    def score(self, x: np.ndarray) -> Tuple[float, bool]:
        if not self.fitted or self.model is None:
            raise RuntimeError("GlobalFallbackModel.score() called before fit().")
        xs = self.scaler.transform(x.reshape(1, -1))
        score = float(self.model.decision_function(xs)[0])
        label = bool(self.model.predict(xs)[0])
        return score, label


# ---------------------------------------------------------------------------
# 2. Per-entity batch (PyOD) model
# ---------------------------------------------------------------------------
class EntityBaselineModel:
    """Rolling-window, periodically-refit Isolation Forest for one entity.

    Keeps up to `window_size` most recent feature vectors. Refits every
    `retrain_every` new events (or immediately, on demand, when the online
    drift layer signals a behavioral shift -- see `HybridBehavioralProfiler`).
    """

    def __init__(
        self,
        entity_id: str,
        min_history: int = 5,
        window_size: int = 500,
        retrain_every: int = 25,
        contamination: float = 0.03,
        random_state: int = 42,
    ):
        self.entity_id = entity_id
        self.min_history = min_history
        self.window_size = window_size
        self.retrain_every = retrain_every
        self.contamination = contamination
        self.random_state = random_state

        self.history: Deque[np.ndarray] = deque(maxlen=window_size)
        self.scaler = StandardScaler()
        self.model: Optional[IForest] = None
        self.events_since_fit = 0

    @property
    def is_cold_start(self) -> bool:
        """True while this entity has fewer than `min_history` events on record."""
        return len(self.history) < self.min_history

    def ingest(self, x: np.ndarray) -> None:
        self.history.append(x)
        self.events_since_fit += 1

    def should_retrain(self) -> bool:
        return (not self.is_cold_start) and (self.model is None or self.events_since_fit >= self.retrain_every)

    def fit(self) -> None:
        """(Re)fit this entity's Isolation Forest on its current rolling window."""
        X = np.vstack(self.history)
        Xs = self.scaler.fit_transform(X)
        self.model = IForest(
            contamination=self.contamination,
            n_estimators=100,
            random_state=self.random_state,
        )
        self.model.fit(Xs)
        self.events_since_fit = 0

    def score(self, x: np.ndarray) -> Tuple[float, bool]:
        if self.model is None:
            raise RuntimeError(f"EntityBaselineModel for {self.entity_id} has no fitted model yet.")
        xs = self.scaler.transform(x.reshape(1, -1))
        score = float(self.model.decision_function(xs)[0])
        label = bool(self.model.predict(xs)[0])
        return score, label


# ---------------------------------------------------------------------------
# 3. Batch profiler orchestrator (per-entity models + global fallback)
# ---------------------------------------------------------------------------
class BaselineProfiler:
    """Cold-start-aware, per-entity Isolation Forest baseline profiler.

    Usage
    -----
        profiler = BaselineProfiler()
        result = profiler.score(entity_id, feature_vector)   # score first
        profiler.ingest(entity_id, feature_vector)            # then ingest

    Scoring before ingesting means an event is always evaluated against
    the baseline as it stood *before* that event -- i.e. against past
    behavior, not behavior that includes the event itself.
    """

    def __init__(
        self,
        min_history: int = 5,
        window_size: int = 500,
        retrain_every: int = 25,
        contamination: float = 0.03,
        global_contamination: float = 0.02,
        global_min_events: int = 50,
        global_refit_every: int = 200,
        random_state: int = 42,
    ):
        self.min_history = min_history
        self.window_size = window_size
        self.retrain_every = retrain_every
        self.contamination = contamination
        self.random_state = random_state

        self.entities: Dict[str, EntityBaselineModel] = {}

        self.global_model = GlobalFallbackModel(contamination=global_contamination, random_state=random_state)
        self._global_history: Deque[np.ndarray] = deque(maxlen=20_000)
        self._global_min_events = global_min_events
        self._global_refit_every = global_refit_every
        self._events_since_global_fit = 0

    def _get_entity(self, entity_id: str) -> EntityBaselineModel:
        if entity_id not in self.entities:
            self.entities[entity_id] = EntityBaselineModel(
                entity_id=entity_id,
                min_history=self.min_history,
                window_size=self.window_size,
                retrain_every=self.retrain_every,
                contamination=self.contamination,
                random_state=self.random_state,
            )
        return self.entities[entity_id]

    def score(self, entity_id: str, feature_vector: FeatureVector) -> BatchScoreResult:
        """Score an event against the current baseline (cold-start aware)."""
        x = feature_vector.to_array()
        entity = self.entities.get(entity_id)

        if entity is None or entity.is_cold_start or entity.model is None:
            if not self.global_model.fitted:
                # Not even the global pool has enough data yet -- return a
                # neutral, non-anomalous result rather than a spurious score.
                return BatchScoreResult(entity_id, None, False, "cold_start_no_fallback")
            score, label = self.global_model.score(x)
            return BatchScoreResult(entity_id, score, label, "global_fallback")

        score, label = entity.score(x)
        return BatchScoreResult(entity_id, score, label, "entity_model")

    def ingest(self, entity_id: str, feature_vector: FeatureVector) -> None:
        """Record an event into entity + global history, retraining as needed."""
        x = feature_vector.to_array()
        entity = self._get_entity(entity_id)
        entity.ingest(x)

        self._global_history.append(x)
        self._events_since_global_fit += 1

        if entity.should_retrain():
            entity.fit()

        if (not self.global_model.fitted and len(self._global_history) >= self._global_min_events) or (
            self.global_model.fitted and self._events_since_global_fit >= self._global_refit_every
        ):
            self._fit_global()

    def _fit_global(self) -> None:
        X = np.vstack(self._global_history)
        self.global_model.fit(X)
        self._events_since_global_fit = 0

    def force_retrain(self, entity_id: str) -> bool:
        """Immediately refit an entity's model on its current window, if possible.
        Returns True if a retrain happened."""
        entity = self.entities.get(entity_id)
        if entity is not None and not entity.is_cold_start:
            entity.fit()
            return True
        return False


# ---------------------------------------------------------------------------
# 4. Online concept-drift layer (River)
# ---------------------------------------------------------------------------
class OnlineDriftBaseline:
    """Fully online, per-entity adaptive baseline + drift detector (River).

    This layer updates on *every* event (no periodic refit lag) and is
    what lets the system distinguish "this looks different because the
    baseline is stale" from "this looks different because it's an
    intrusion":

      - `river.preprocessing.StandardScaler` re-centers/re-scales features
        using running statistics, so scores track the entity's *current*
        normal, not a frozen historical one.
      - `river.anomaly.HalfSpaceTrees` is a streaming anomaly detector
        that scores and learns from every event in constant time/memory.
      - `river.drift.ADWIN` monitors the online anomaly-score stream for
        a sustained distributional shift. When it fires, that's the
        drift signal: the entity's behavior has durably changed (e.g. it
        shifted to a new work-hour pattern, or started using a new
        legitimate device consistently) rather than a single outlier
        event.
    """

    def __init__(
        self,
        entity_id: str,
        feature_names: Optional[List[str]] = None,
        drift_delta: float = 0.002,
        hst_n_trees: int = 25,
        hst_height: int = 8,
        hst_window_size: int = 250,
    ):
        self.entity_id = entity_id
        self.feature_names = feature_names or FEATURE_NAMES

        self.scaler = river_preprocessing.StandardScaler()
        self.hst = river_anomaly.HalfSpaceTrees(
            n_trees=hst_n_trees, height=hst_height, window_size=hst_window_size, seed=42
        )
        self.drift_detector = river_drift.ADWIN(delta=drift_delta)

        self.n_seen = 0
        self.drift_events: List[int] = []  # event indices at which drift fired

        # Adaptive per-feature running stats, exposed for explainability
        # (e.g. showing an analyst "typical login hour has drifted from
        # 9am to 7pm over the last two weeks").
        self._feature_mean: Dict[str, river_stats.Mean] = {f: river_stats.Mean() for f in self.feature_names}
        self._feature_var: Dict[str, river_stats.Var] = {f: river_stats.Var() for f in self.feature_names}

    def update_and_score(self, feature_vector: FeatureVector) -> Tuple[float, bool]:
        """Feed one event through the online pipeline.

        Returns (online_anomaly_score in [0, 1], drift_detected_this_event).
        """
        record = feature_vector.to_dict()

        self.scaler.learn_one(record)
        scaled = self.scaler.transform_one(record)

        score = self.hst.score_one(scaled)
        self.hst.learn_one(scaled)

        self.drift_detector.update(score)
        drift_detected = bool(self.drift_detector.drift_detected)
        if drift_detected:
            self.drift_events.append(self.n_seen)

        for name, val in record.items():
            if name in self._feature_mean:
                self._feature_mean[name].update(val)
                self._feature_var[name].update(val)

        self.n_seen += 1
        return float(score), drift_detected

    def adaptive_baseline(self) -> Dict[str, Tuple[float, float]]:
        """Current streaming (mean, std) per feature -- the 'live' baseline."""
        out = {}
        for name in self.feature_names:
            mean = self._feature_mean[name].get()
            var = self._feature_var[name].get()
            out[name] = (mean, math.sqrt(var) if var and var > 0 else 0.0)
        return out


# ---------------------------------------------------------------------------
# 5. Hybrid profiler: batch IForest + online River drift handling
# ---------------------------------------------------------------------------
class HybridBehavioralProfiler:
    """Production-facing entry point combining both layers.

    Call `observe(entity_id, feature_vector)` once per incoming event. It:
      1. Scores the event against the current batch baseline (PyOD),
         cold-start aware.
      2. Scores + learns the event online (River HalfSpaceTrees) and
         checks the ADWIN drift detector.
      3. Ingests the event into the batch profiler's rolling history
         (may trigger a periodic per-entity or global retrain).
      4. If drift just fired for this entity, immediately force-retrains
         its batch model on the recent window -- folding the new normal
         into the baseline right away instead of waiting for the next
         periodic retrain, and suppresses the anomaly flag for *this*
         event (a drift signal means "baseline shift", not "attack").
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        min_history: int = 5,
        window_size: int = 500,
        retrain_every: int = 25,
        contamination: float = 0.03,
        global_contamination: float = 0.02,
        drift_delta: float = 0.002,
        batch_weight: float = 0.6,
        anomaly_score_threshold: float = 0.0,
        random_state: int = 42,
    ):
        self.feature_names = feature_names or FEATURE_NAMES
        self.batch_profiler = BaselineProfiler(
            min_history=min_history,
            window_size=window_size,
            retrain_every=retrain_every,
            contamination=contamination,
            global_contamination=global_contamination,
            random_state=random_state,
        )
        self.online_baselines: Dict[str, OnlineDriftBaseline] = {}
        self.drift_delta = drift_delta
        self.batch_weight = batch_weight
        self.anomaly_score_threshold = anomaly_score_threshold

    def _get_online(self, entity_id: str) -> OnlineDriftBaseline:
        if entity_id not in self.online_baselines:
            self.online_baselines[entity_id] = OnlineDriftBaseline(
                entity_id, self.feature_names, drift_delta=self.drift_delta
            )
        return self.online_baselines[entity_id]

    def observe(self, entity_id: str, feature_vector: FeatureVector) -> AnomalyResult:
        # 1. Score against current (pre-event) batch baseline.
        batch_result = self.batch_profiler.score(entity_id, feature_vector)

        # 2. Online scoring + drift check (always-on, learns immediately).
        online = self._get_online(entity_id)
        online_score, drift_detected = online.update_and_score(feature_vector)

        # 3. Ingest into batch history (may trigger periodic retrain).
        self.batch_profiler.ingest(entity_id, feature_vector)

        # 4. On drift: immediately fold the new normal into the batch
        #    model, and don't flag this event as an anomaly -- it's the
        #    signal that behavior has legitimately shifted.
        if drift_detected:
            self.batch_profiler.force_retrain(entity_id)

        combined_score = self._combine_scores(batch_result.score, online_score)
        is_anomaly = (not drift_detected) and (
            batch_result.is_anomaly if batch_result.score is not None else (online_score > 0.7)
        )

        return AnomalyResult(
            entity_id=entity_id,
            is_anomaly=is_anomaly,
            combined_score=combined_score,
            batch_score=batch_result.score,
            batch_source=batch_result.source,
            online_score=online_score,
            drift_detected=drift_detected,
        )

    def _combine_scores(self, batch_score: Optional[float], online_score: float) -> float:
        """Heuristic blend of the two signals onto a single display score.

        `batch_score` (PyOD decision_function) is roughly zero-centered
        and unbounded; `online_score` (River HST) is in [0, 1]. We map
        online_score onto a comparable zero-centered range before
        blending. This is a simple, explainable combination suitable for
        a hackathon demo/dashboard -- swap in a calibrated ensemble for
        production use.
        """
        online_centered = (online_score * 2) - 1  # [0,1] -> [-1,1]
        if batch_score is None:
            return online_centered
        return self.batch_weight * batch_score + (1 - self.batch_weight) * online_centered

    def get_adaptive_baseline(self, entity_id: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """Expose the entity's current live (mean, std) per feature, for dashboards/alerts."""
        online = self.online_baselines.get(entity_id)
        return online.adaptive_baseline() if online else None

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the *entire* fitted state -- per-entity/global PyOD
        models + their scalers, and every per-entity River online
        baseline (adaptive scaler, HalfSpaceTrees, ADWIN) -- via a single
        `joblib.dump`. Everything on this object (dataclasses, sklearn,
        PyOD, River) is plain-Python-picklable, so whole-object pickling
        is simpler and less error-prone here than field-by-field
        serialization, and round-trips exactly (verified: entities dict,
        online baselines, and drift history all survive save/load)."""
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "HybridBehavioralProfiler":
        """Restore a profiler saved with `save()`, fully warmed up --
        `observe()` on the loaded object continues exactly where the
        saved one left off (same per-entity baselines, same drift
        history), no re-training or cold start required."""
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


# ---------------------------------------------------------------------------
# Sample test execution
# ---------------------------------------------------------------------------
def _demo():
    """Small end-to-end smoke test / demo you can run directly:
        python baseline_profiling.py
    Simulates:
      - a normal, well-behaved entity ('user-normal')
      - several cold-start entities (<5 events each) scored via global fallback
      - a point-anomaly event injected into an established entity
      - a gradually drifting entity whose work hours shift over time, showing
        that it stops being flagged once the drift detector adapts the baseline
    """
    import random as _random

    _random.seed(7)
    np.random.seed(7)

    profiler = HybridBehavioralProfiler(min_history=5, retrain_every=15, contamination=0.05)

    def make_fv(hour, duration, geo_km, failures):
        return FeatureVector(
            login_hour=hour, session_duration=duration, geo_distance_km=geo_km, failure_count=failures
        )

    print("=" * 80)
    print("1) Warming up global pool + an established 'normal' entity (60 events)")
    print("=" * 80)
    results = []
    for i in range(60):
        fv = make_fv(
            hour=np.random.normal(9, 0.7),
            duration=np.random.normal(600, 60),
            geo_km=np.random.normal(5, 2),
            failures=max(0, np.random.poisson(0.1)),
        )
        res = profiler.observe("user-normal", fv)
        results.append(res)
    print(f"Final 5 results for user-normal: {[ (round(r.combined_score,2), r.is_anomaly) for r in results[-5:] ]}")

    print()
    print("=" * 80)
    print("2) Cold-start entities (<5 events) -- scored via global fallback")
    print("=" * 80)
    for uid in ["svc-new-01", "svc-new-02"]:
        for i in range(3):
            fv = make_fv(hour=np.random.normal(9, 1), duration=np.random.normal(500, 50),
                         geo_km=np.random.normal(5, 2), failures=0)
            res = profiler.observe(uid, fv)
            print(f"  {uid} event {i+1}: source={res.batch_source:22s} score={res.combined_score:6.2f} anomaly={res.is_anomaly}")

    print()
    print("=" * 80)
    print("3) Point anomaly injected into 'user-normal' (odd hour, huge geo jump, many failures)")
    print("=" * 80)
    anomalous_fv = make_fv(hour=3.5, duration=45, geo_km=8200, failures=14)
    res = profiler.observe("user-normal", anomalous_fv)
    print(f"  score={res.combined_score:.2f}  batch_score={res.batch_score}  online_score={res.online_score:.2f}  "
          f"is_anomaly={res.is_anomaly}  drift_detected={res.drift_detected}")

    print()
    print("=" * 80)
    print("4) Gradual drift: 'user-shiftworker' slowly moves from 9am to 9pm logins over 80 events")
    print("=" * 80)
    flagged_count_first_half = 0
    flagged_count_second_half = 0
    for i in range(80):
        progress = i / 79
        target_hour = 9 + progress * 12  # 9am -> 9pm
        fv = make_fv(
            hour=np.random.normal(target_hour, 0.5),
            duration=np.random.normal(600, 60),
            geo_km=np.random.normal(5, 2),
            failures=0,
        )
        res = profiler.observe("user-shiftworker", fv)
        if i < 40 and res.is_anomaly:
            flagged_count_first_half += 1
        if i >= 40 and res.is_anomaly:
            flagged_count_second_half += 1
        if res.drift_detected:
            print(f"  [drift detected at event {i}] baseline re-adapted -> "
                  f"{profiler.get_adaptive_baseline('user-shiftworker')['login_hour']}")

    print(f"  Anomalies flagged in first half (early drift): {flagged_count_first_half}/40")
    print(f"  Anomalies flagged in second half (after adaptation): {flagged_count_second_half}/40")
    print(f"  Drift events fired at indices: {profiler.online_baselines['user-shiftworker'].drift_events}")
    print("  -> The goal: once ADWIN fires, the batch model is force-retrained on the")
    print("     recent window, so the new work-hour pattern stops being flagged as")
    print("     anomalous. Real-world results depend on ADWIN's `delta` sensitivity and")
    print("     the batch model's `retrain_every` cadence -- tune both against your")
    print("     actual event volume and drift speed.")


if __name__ == "__main__":
    _demo()