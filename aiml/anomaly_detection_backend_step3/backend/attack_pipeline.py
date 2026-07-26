"""
attack_pipeline.py
===================
End-to-end glue: generates/loads data with `data_generator.py`, runs the
IForest/HalfSpaceTrees baseline (`baseline_profiling.HybridBehavioralProfiler`)
event-by-event, builds sliding windows and trains the Bi-LSTM
(`sequence_model.py`), joins both signals with session metadata via
`attack_classifier.AttackFeatureBuilder`, and trains the final XGBoost
multi-class classifier (`attack_classifier.XGBoostAttackClassifier`).

Run standalone for a demo:
    python attack_pipeline.py --days 30 --num-users 200

Or import `AttackClassificationPipeline` and call `.fit(logs_df)` /
`.predict(logs_df)` from your own training/inference code (e.g. a
Prefect/Airflow job, or the streaming consumer that writes into
`processed_streaming_logs`).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from attack_classifier import AttackFeatureBuilder, XGBoostAttackClassifier
from baseline_profiling import FeatureVector, HybridBehavioralProfiler
from sequence_model import (
    BiLSTMSequenceModel,
    BiLSTMTrainer,
    SequenceWindowBuilder,
    Vocabulary,
    WindowDataset,
)

# Filenames used by both `.save()` / `.load_pretrained()` here and by
# `anomaly_pipeline.AnomalyDetectionPipeline`, so both modules agree on
# the on-disk layout of a `model_dir`.
BILSTM_FILENAME = "bilstm.pt"
XGB_FILENAME = "xgb_attack_classifier.joblib"
PROFILER_FILENAME = "behavioral_profiler.joblib"
PROFILES_FILENAME = "entity_profiles.csv"
METADATA_FILENAME = "pipeline_metadata.json"


def _dedupe_join_keys(logs_df: pd.DataFrame) -> pd.DataFrame:
    """`_score_baseline`, the Bi-LSTM window builder, and
    `AttackFeatureBuilder.from_dataframe` all join intermediate results
    back onto the raw events via (entity_id, timestamp). Bursty
    synthetic attacks (brute_force, credential_stuffing) can produce
    several events for the same entity at the exact same timestamp,
    which turns those joins into fan-outs instead of 1:1 matches. Nudge
    duplicates by a stable, order-preserving microsecond offset so
    (entity_id, timestamp) is always unique -- inconsequential for every
    downstream feature (hour-of-day, session duration, sequence order)
    but keeps every join exact."""
    df = logs_df.sort_values(["entity_id", "timestamp"], kind="stable").reset_index(drop=True)
    dup_rank = df.groupby(["entity_id", "timestamp"]).cumcount()
    if dup_rank.max() > 0:
        df["timestamp"] = df["timestamp"] + pd.to_timedelta(dup_rank, unit="us")
    return df


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    from math import asin, cos, radians, sin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


class AttackClassificationPipeline:
    """Owns the three model layers and the joins between them.

    Typical usage
    -------------
        pipeline = AttackClassificationPipeline(entity_profiles_df)
        pipeline.fit(raw_logs_df)                 # trains Bi-LSTM + XGBoost
        preds = pipeline.predict(new_logs_df)      # per-event attack label + probs
    """

    def __init__(
        self,
        entity_profiles_df: pd.DataFrame,
        window_size: int = 10,
        bilstm_epochs: int = 8,
        random_state: int = 42,
    ):
        self.entity_profiles = entity_profiles_df.set_index("entity_id", drop=False)
        self.window_size = window_size
        self.bilstm_epochs = bilstm_epochs
        self.random_state = random_state

        self.resource_vocab, self.command_vocab = Vocabulary.from_generator_pools()
        self.window_builder = SequenceWindowBuilder(
            self.resource_vocab, self.command_vocab, window_size=window_size,
        )
        self.behavioral_profiler = HybridBehavioralProfiler(random_state=random_state)
        self.bilstm_trainer: Optional[BiLSTMTrainer] = None
        self.xgb_classifier = XGBoostAttackClassifier(random_state=random_state)

    # -- stage 1: IForest / online baseline, scored event-by-event -------
    def _score_baseline(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Streams every event through `HybridBehavioralProfiler.observe()`
        in timestamp order (required -- it's a stateful online model) and
        returns a (entity_id, timestamp, iforest_score, iforest_is_anomaly)
        table to join back onto the raw logs."""
        df = logs_df.sort_values("timestamp").reset_index(drop=True)
        rows = []
        for _, row in df.iterrows():
            profile = self.entity_profiles.loc[row["entity_id"]] if row["entity_id"] in self.entity_profiles.index else None
            geo_km = 0.0
            if profile is not None:
                geo = row["geo_location"]
                lat = geo["lat"] if isinstance(geo, dict) else row.get("geo_lat")
                lon = geo["lon"] if isinstance(geo, dict) else row.get("geo_lon")
                if lat is not None and lon is not None:
                    geo_km = _haversine_km(lat, lon, profile["home_lat"], profile["home_lon"])

            cmds = row.get("command_sequence")
            if cmds is None or (isinstance(cmds, float) and pd.isna(cmds)):
                cmds = []
            elif isinstance(cmds, np.ndarray):
                cmds = cmds.tolist()
            fv = FeatureVector(
                login_hour=float(row["timestamp"].hour),
                session_duration=float(row["session_duration"]),
                geo_distance_km=float(geo_km),
                failure_count=1.0 if row.get("auth_result") != "success" else 0.0,
                extra={"num_commands": float(len(cmds))},
            )
            result = self.behavioral_profiler.observe(row["entity_id"], fv)
            rows.append({
                "entity_id": row["entity_id"],
                "timestamp": row["timestamp"],
                "iforest_score": result.combined_score,
                "iforest_is_anomaly": result.is_anomaly,
                "geo_distance_km": geo_km,
            })
        return pd.DataFrame(rows)

    # -- stage 2: Bi-LSTM sequence features -------------------------------
    def _fit_bilstm(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        batch = self.window_builder.build(logs_df)
        dataset = WindowDataset(batch)
        n = len(dataset)
        val_n = max(1, int(0.15 * n))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n - val_n, val_n],
            generator=torch.Generator().manual_seed(self.random_state),
        )
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

        model = BiLSTMSequenceModel(
            num_resources=len(self.resource_vocab), num_commands=len(self.command_vocab),
        )
        class_weights = BiLSTMTrainer.compute_class_weights(batch.label_ids)
        self.bilstm_trainer = BiLSTMTrainer(model, class_weights=class_weights)
        self.bilstm_trainer.train(train_loader, epochs=self.bilstm_epochs, val_loader=val_loader)

        seq_features = self.bilstm_trainer.extract_features(batch)
        return pd.DataFrame({
            "entity_id": seq_features.entity_ids,
            "timestamp": seq_features.end_timestamps,
            "bilstm_sequence_loss": seq_features.sequence_loss,
            "bilstm_normal_prob": seq_features.normal_probability,
        })

    def _bilstm_infer(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        if self.bilstm_trainer is None:
            raise RuntimeError("Bi-LSTM not trained yet -- call fit() first.")
        batch = self.window_builder.build(logs_df)
        seq_features = self.bilstm_trainer.extract_features(batch)
        return pd.DataFrame({
            "entity_id": seq_features.entity_ids,
            "timestamp": seq_features.end_timestamps,
            "bilstm_sequence_loss": seq_features.sequence_loss,
            "bilstm_normal_prob": seq_features.normal_probability,
        })

    # -- public API ---------------------------------------------------------
    def fit(self, logs_df: pd.DataFrame) -> Dict[str, object]:
        logs_df = _dedupe_join_keys(logs_df)
        logs_df = logs_df.merge(
            self.entity_profiles["entity_type"].rename("entity_type_lookup"),
            left_on="entity_id", right_index=True, how="left",
        )
        logs_df["entity_type"] = logs_df["entity_type"].fillna(logs_df.pop("entity_type_lookup"))

        iforest_scores = self._score_baseline(logs_df)
        bilstm_features = self._fit_bilstm(logs_df)
        features_df = AttackFeatureBuilder.from_dataframe(logs_df, iforest_scores, bilstm_features)
        return self.xgb_classifier.train(features_df)

    def predict(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        logs_df = _dedupe_join_keys(logs_df)
        iforest_scores = self._score_baseline(logs_df)
        bilstm_features = self._bilstm_infer(logs_df)
        features_df = AttackFeatureBuilder.from_dataframe(logs_df, iforest_scores, bilstm_features)
        preds = self.xgb_classifier.predict(features_df)
        probs = self.xgb_classifier.predict_proba(features_df)
        out = logs_df[["entity_id", "timestamp"]].copy()
        out["predicted_label"] = preds
        out = pd.concat([out.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
        return out

    # -- persistence -----------------------------------------------------
    def save(self, model_dir: str) -> None:
        """Persist every trained artifact needed to run inference later
        with zero retraining:
          - `bilstm.pt`    -- Bi-LSTM weights + architecture config +
                              the resource/command vocabularies (the
                              embedding tables are meaningless without
                              the exact vocab that produced their ids).
          - `xgb_attack_classifier.joblib` -- the trained XGBoost model.
          - `behavioral_profiler.joblib`   -- the *entire* fitted
                              IForest/River baseline state, so a loaded
                              pipeline picks up exactly where training
                              left off instead of cold-starting.
          - `entity_profiles.csv`          -- the entity baselines used
                              for geo-distance features, so callers
                              don't have to keep the original DataFrame
                              around just to call `.load_pretrained()`.
          - `pipeline_metadata.json`       -- the constructor args
                              (window_size, random_state, ...) needed to
                              reconstruct non-model pipeline state.
        Raises if `.fit()` hasn't been called yet (nothing to save).
        """
        if self.bilstm_trainer is None:
            raise RuntimeError("Nothing to save -- call fit() before save().")

        os.makedirs(model_dir, exist_ok=True)

        self.bilstm_trainer.save(
            os.path.join(model_dir, BILSTM_FILENAME),
            resource_vocab=self.resource_vocab,
            command_vocab=self.command_vocab,
        )
        self.xgb_classifier.save(os.path.join(model_dir, XGB_FILENAME))
        self.behavioral_profiler.save(os.path.join(model_dir, PROFILER_FILENAME))
        self.entity_profiles.to_csv(os.path.join(model_dir, PROFILES_FILENAME), index=False)

        with open(os.path.join(model_dir, METADATA_FILENAME), "w") as f:
            json.dump({"window_size": self.window_size, "random_state": self.random_state}, f)

        print(f"Saved pipeline artifacts to {model_dir}/ "
              f"({BILSTM_FILENAME}, {XGB_FILENAME}, {PROFILER_FILENAME}, {PROFILES_FILENAME}, {METADATA_FILENAME})")

    @classmethod
    def load_pretrained(
        cls,
        model_dir: str,
        entity_profiles_df: Optional[pd.DataFrame] = None,
        device: Optional[str] = None,
    ) -> "AttackClassificationPipeline":
        """Reconstruct a fully trained pipeline straight from a
        `model_dir` written by `.save()` -- no `.fit()` / training loop
        involved. `entity_profiles_df` is only needed if the saved
        `entity_profiles.csv` isn't available or you want to score
        against a different/updated entity roster than what was trained
        on (e.g. new entities onboarded since training)."""
        bilstm_path = os.path.join(model_dir, BILSTM_FILENAME)
        xgb_path = os.path.join(model_dir, XGB_FILENAME)
        profiler_path = os.path.join(model_dir, PROFILER_FILENAME)
        profiles_path = os.path.join(model_dir, PROFILES_FILENAME)
        metadata_path = os.path.join(model_dir, METADATA_FILENAME)

        for required in (bilstm_path, xgb_path):
            if not os.path.exists(required):
                raise FileNotFoundError(f"Expected {required} -- has this pipeline been saved with .save()?")

        bilstm_trainer, resource_vocab, command_vocab = BiLSTMTrainer.load_pretrained(bilstm_path, device=device)
        if resource_vocab is None or command_vocab is None:
            raise ValueError(
                f"{bilstm_path} was saved without vocabularies -- can't safely run inference "
                "(resource/command token ids would be undefined). Re-save via AttackClassificationPipeline.save()."
            )

        xgb_classifier = XGBoostAttackClassifier.load(xgb_path)

        if entity_profiles_df is None:
            if not os.path.exists(profiles_path):
                raise FileNotFoundError(
                    f"No entity_profiles_df supplied and {profiles_path} not found -- "
                    "pass entity_profiles_df explicitly."
                )
            entity_profiles_df = pd.read_csv(profiles_path)
            entity_profiles_df["typical_resources"] = entity_profiles_df["typical_resources"].apply(ast.literal_eval)

        behavioral_profiler = (
            HybridBehavioralProfiler.load(profiler_path)
            if os.path.exists(profiler_path)
            else HybridBehavioralProfiler()
        )

        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
        window_size = metadata.get("window_size", 10)
        random_state = metadata.get("random_state", 42)

        obj = cls.__new__(cls)  # bypass __init__ -- we're wiring up loaded components, not building fresh ones
        obj.entity_profiles = entity_profiles_df.set_index("entity_id", drop=False)
        obj.window_size = window_size
        obj.bilstm_epochs = 0  # not applicable -- this pipeline was loaded, not trained, in this process
        obj.random_state = random_state
        obj.resource_vocab = resource_vocab
        obj.command_vocab = command_vocab
        obj.window_builder = SequenceWindowBuilder(resource_vocab, command_vocab, window_size=window_size)
        obj.behavioral_profiler = behavioral_profiler
        obj.bilstm_trainer = bilstm_trainer
        obj.xgb_classifier = xgb_classifier
        return obj


def models_exist(model_dir: str) -> bool:
    """True if `model_dir` contains a complete enough set of artifacts
    for `AttackClassificationPipeline.load_pretrained()` to succeed
    (the two files it hard-requires; profiles/profiler/metadata all
    have fallbacks)."""
    return all(
        os.path.exists(os.path.join(model_dir, fname))
        for fname in (BILSTM_FILENAME, XGB_FILENAME)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train (or load pretrained) Bi-LSTM + XGBoost attack classification stack on synthetic data, "
                     "then run inference on a fresh batch and print a held-out/streaming report."
    )
    parser.add_argument("--num-users", type=int, default=100)
    parser.add_argument("--num-service-accounts", type=int, default=20)
    parser.add_argument("--num-devices", type=int, default=30)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--predict-days", type=int, default=2, help="Days of fresh data to run inference on after training/loading")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bilstm-epochs", type=int, default=8)
    parser.add_argument("--model-out-dir", type=str, default="models")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if models already has saved artifacts")
    args, _ = parser.parse_known_args()

    from datetime import datetime, timedelta

    from data_generator import ATTACK_RATE_RANGE, SyntheticDataGenerator

    gen = SyntheticDataGenerator(
        num_users=args.num_users, num_service_accounts=args.num_service_accounts,
        num_devices=args.num_devices, seed=args.seed,
    )
    profiles_df = gen.profiles_dataframe()

    if models_exist(args.model_out_dir) and not args.force_retrain:
        print(f"Found existing model artifacts in {args.model_out_dir}/ -- loading pretrained pipeline (no training).")
        pipeline = AttackClassificationPipeline.load_pretrained(args.model_out_dir, entity_profiles_df=profiles_df)
    else:
        print(f"No usable model artifacts in {args.model_out_dir}/ -- training from scratch.")
        start = datetime.utcnow() - timedelta(days=args.days + args.predict_days)
        train_df = gen.generate(start, args.days, attack_rate_range=ATTACK_RATE_RANGE)

        pipeline = AttackClassificationPipeline(profiles_df, bilstm_epochs=args.bilstm_epochs, random_state=args.seed)
        metrics = pipeline.fit(train_df)
        print("Held-out classification report:")
        print(metrics["classification_report"])

        pipeline.save(args.model_out_dir)

    # Either way (loaded or freshly trained), run inference on a fresh
    # batch to demonstrate `predict()` works standalone off the pipeline
    # we ended up with.
    print(f"\nRunning inference on {args.predict_days} fresh day(s) of synthetic data...")
    predict_start = datetime.utcnow() - timedelta(days=args.predict_days)
    predict_df = gen.generate(predict_start, args.predict_days, attack_rate_range=ATTACK_RATE_RANGE)
    predictions = pipeline.predict(predict_df)

    predicted_attacks = predictions[predictions["predicted_label"] != "normal"]
    print(f"Scored {len(predictions)} events -- {len(predicted_attacks)} flagged as an attack category.")
    if len(predicted_attacks):
        print(predicted_attacks["predicted_label"].value_counts())


if __name__ == "__main__":
    main()