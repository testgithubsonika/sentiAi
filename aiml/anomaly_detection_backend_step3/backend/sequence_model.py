"""
sequence_model.py
==================
Sequential / temporal modeling layer for the Behavioral Anomaly Detection
system: a Bi-LSTM that ingests sliding windows of an entity's recent
events (resource accessed + command issued + light numeric context per
event) and learns to recognize what a "normal" trajectory through that
space looks like.

Why this exists alongside the Isolation Forest baseline (baseline_profiling.py)
--------------------------------------------------------------------------
The IForest/HalfSpaceTrees baseline in `baseline_profiling.py` scores each
event *independently* -- it has no notion of order. That is exactly the
blind spot attacks like `lateral_movement` (rapid escalating hops across
resources) and `low_and_slow_exfiltration` (sparse pulls spread across
days, each individually unremarkable) are built to exploit. The Bi-LSTM
here looks at *sequences* of events per entity and produces a per-window
"sequence loss" -- how surprising this trajectory is under the model's
learned notion of normal -- which becomes one of the input features to
the downstream XGBoost multi-class attack classifier in
`attack_classifier.py`.

Pipeline
--------
1. `Vocabulary`            -- token <-> id maps for resources and commands.
2. `SequenceWindowBuilder` -- turns a raw_access_logs-shaped DataFrame
   into fixed-length, left-padded sliding windows per entity.
3. `WindowDataset`         -- torch Dataset wrapping the built windows.
4. `BiLSTMSequenceModel`   -- embeds resource/command tokens, concatenates
   numeric context, runs a bidirectional LSTM, and predicts a window-level
   label distribution (normal vs. the 7 attack types).
5. `BiLSTMTrainer`         -- training loop, evaluation, and the
   `extract_features()` inference method that produces the per-window
   "sequence loss" / anomaly-probability features consumed downstream.

Integrates with
----------------
- `models.py`     : `LabelType`, `EntityType` enums (window labels map
                    1:1 onto `LabelType`; label ids follow `LABEL_ORDER`
                    below, the same order used by `attack_classifier.py`).
- `data_generator.py` : `BENIGN_COMMANDS`, `PRIV_ESC_COMMANDS`,
                    `EXFIL_COMMANDS`, `RESOURCE_POOL` seed the vocabulary
                    so token ids are stable across train/inference even
                    for resources/commands not yet observed for a given
                    entity.
- `baseline_profiling.py` : consumed alongside (not by) this module --
                    `attack_classifier.py` is what joins IForest scores
                    with the sequence features produced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

try:
    # Ground-truth label enum + a stable class ordering shared with
    # attack_classifier.py so both modules agree on label <-> id mapping.
    from models import LabelType
except ImportError:  # pragma: no cover - allows standalone use/testing
    import enum

    class LabelType(str, enum.Enum):
        normal = "normal"
        brute_force = "brute_force"
        impossible_travel = "impossible_travel"
        credential_stuffing = "credential_stuffing"
        lateral_movement = "lateral_movement"
        device_spoofing = "device_spoofing"
        low_and_slow_exfiltration = "low_and_slow_exfiltration"
        insider_drift = "insider_drift"


# Fixed class ordering used everywhere in the sequence + classification
# stack. Index 0 is always "normal".
LABEL_ORDER: List[str] = [
    LabelType.normal.value,
    LabelType.brute_force.value,
    LabelType.impossible_travel.value,
    LabelType.credential_stuffing.value,
    LabelType.lateral_movement.value,
    LabelType.device_spoofing.value,
    LabelType.low_and_slow_exfiltration.value,
    LabelType.insider_drift.value,
]
LABEL_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(LABEL_ORDER)}
NORMAL_LABEL_ID = LABEL_TO_ID[LabelType.normal.value]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocabulary:
    """Simple token <-> id map with a reserved PAD (id 0) and UNK (id 1).

    Seed it with the known command/resource pools from `data_generator.py`
    at construction time so ids are stable regardless of what a given
    training slice happens to contain; anything unseen at inference falls
    back to UNK rather than crashing or silently reassigning ids.
    """

    def __init__(self, tokens: Optional[Sequence[str]] = None):
        self.token_to_id: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.id_to_token: List[str] = [PAD_TOKEN, UNK_TOKEN]
        if tokens:
            self.add_many(tokens)

    def add(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.token_to_id[token]

    def add_many(self, tokens: Sequence[str]) -> None:
        for t in tokens:
            self.add(t)

    def encode(self, token: Optional[str]) -> int:
        if token is None:
            return self.token_to_id[PAD_TOKEN]
        return self.token_to_id.get(token, self.token_to_id[UNK_TOKEN])

    def __len__(self) -> int:
        return len(self.id_to_token)

    def to_dict(self) -> Dict[str, object]:
        """Token order matters (ids are positional), so persist
        `id_to_token` as the source of truth and rebuild `token_to_id`
        from it on load rather than serializing both."""
        return {"id_to_token": self.id_to_token}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocabulary":
        vocab = cls.__new__(cls)
        vocab.id_to_token = list(data["id_to_token"])
        vocab.token_to_id = {tok: i for i, tok in enumerate(vocab.id_to_token)}
        return vocab

    @classmethod
    def from_generator_pools(cls) -> Tuple["Vocabulary", "Vocabulary"]:
        """Build (resource_vocab, command_vocab) seeded from data_generator's
        reference pools, falling back to a small built-in set if that
        module isn't importable (e.g. this file used standalone)."""
        try:
            from data_generator import COMMANDS_POOL, RESOURCE_POOL
            resources, commands = RESOURCE_POOL, COMMANDS_POOL
        except ImportError:
            resources = [f"resource/generic/{i:03d}" for i in range(50)]
            commands = ["ls", "cat", "whoami", "sudo", "scp", "curl", "kubectl"]
        return cls(resources), cls(commands)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
NUMERIC_FEATURE_NAMES = [
    "hour_sin", "hour_cos", "session_duration_z", "is_failure",
    "num_commands", "is_sensitive_resource",
]


@dataclass
class WindowBatch:
    """Container of encoded, padded sliding windows ready for the model."""

    entity_ids: List[str]
    end_timestamps: List[pd.Timestamp]
    resource_ids: np.ndarray      # (N, W) int64
    command_ids: np.ndarray       # (N, W) int64
    numeric_feats: np.ndarray     # (N, W, F) float32
    lengths: np.ndarray           # (N,) int64 -- true (unpadded) length
    label_ids: np.ndarray         # (N,) int64 -- window-level label


class SequenceWindowBuilder:
    """Turns a raw_access_logs-shaped DataFrame into fixed-length,
    left-padded sliding windows, one per entity per step.

    Expected input columns (matching `RawAccessLog` / `data_generator`
    output): entity_id, timestamp, resource_accessed, command_sequence
    (list[str]), session_duration, auth_result, label.

    Window label = the label of the *last* event in the window (i.e. "is
    the most recent event, in the context of its recent history, part of
    an attack"). This matches how the model will be used at inference: at
    time T you have the last W events and want to judge the one that just
    happened.
    """

    def __init__(
        self,
        resource_vocab: Vocabulary,
        command_vocab: Vocabulary,
        window_size: int = 10,
        stride: int = 1,
        sensitive_resource_substrings: Sequence[str] = ("finance", "customer_db", "billing"),
    ):
        self.resource_vocab = resource_vocab
        self.command_vocab = command_vocab
        self.window_size = window_size
        self.stride = stride
        self.sensitive_substrings = sensitive_resource_substrings

    def _numeric_row(self, row: pd.Series, duration_mean: float, duration_std: float) -> List[float]:
        ts = row["timestamp"]
        hour = ts.hour + ts.minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        dur_z = (row["session_duration"] - duration_mean) / (duration_std or 1.0)
        is_failure = 1.0 if str(row.get("auth_result", "success")) != "success" else 0.0
        cmds = row.get("command_sequence") or []
        num_commands = float(len(cmds))
        resource = str(row["resource_accessed"])
        is_sensitive = 1.0 if any(s in resource for s in self.sensitive_substrings) else 0.0
        return [hour_sin, hour_cos, dur_z, is_failure, num_commands, is_sensitive]

    def build(self, logs_df: pd.DataFrame) -> WindowBatch:
        df = logs_df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
        duration_mean = float(df["session_duration"].mean()) if len(df) else 0.0
        duration_std = float(df["session_duration"].std()) if len(df) else 1.0

        entity_ids: List[str] = []
        end_ts: List[pd.Timestamp] = []
        res_rows: List[List[int]] = []
        cmd_rows: List[List[int]] = []
        num_rows: List[List[List[float]]] = []
        lengths: List[int] = []
        label_rows: List[int] = []

        W = self.window_size
        for entity_id, group in df.groupby("entity_id", sort=False):
            group = group.reset_index(drop=True)
            n = len(group)
            for end in range(0, n, self.stride):
                start = max(0, end - W + 1)
                window = group.iloc[start:end + 1]
                true_len = len(window)

                res_ids = [self.resource_vocab.encode(str(r)) for r in window["resource_accessed"]]
                # primary command per event = first command in that event's
                # sequence (empty sequences -> PAD), a deliberate simplification
                # that keeps one token per timestep aligned with resource_ids.
                cmd_ids = [
                    self.command_vocab.encode(cmds[0]) if cmds else self.command_vocab.encode(None)
                    for cmds in window["command_sequence"]
                ]
                numeric = [self._numeric_row(r, duration_mean, duration_std) for _, r in window.iterrows()]

                pad_amt = W - true_len
                if pad_amt > 0:
                    res_ids = [0] * pad_amt + res_ids
                    cmd_ids = [0] * pad_amt + cmd_ids
                    numeric = [[0.0] * len(NUMERIC_FEATURE_NAMES)] * pad_amt + numeric

                entity_ids.append(entity_id)
                end_ts.append(window.iloc[-1]["timestamp"])
                res_rows.append(res_ids)
                cmd_rows.append(cmd_ids)
                num_rows.append(numeric)
                lengths.append(true_len)
                label_rows.append(LABEL_TO_ID.get(str(window.iloc[-1].get("label", "normal")), NORMAL_LABEL_ID))

        return WindowBatch(
            entity_ids=entity_ids,
            end_timestamps=end_ts,
            resource_ids=np.array(res_rows, dtype=np.int64) if res_rows else np.zeros((0, W), dtype=np.int64),
            command_ids=np.array(cmd_rows, dtype=np.int64) if cmd_rows else np.zeros((0, W), dtype=np.int64),
            numeric_feats=np.array(num_rows, dtype=np.float32) if num_rows else np.zeros((0, W, len(NUMERIC_FEATURE_NAMES)), dtype=np.float32),
            lengths=np.array(lengths, dtype=np.int64),
            label_ids=np.array(label_rows, dtype=np.int64),
        )


class WindowDataset(Dataset):
    """torch Dataset wrapping a `WindowBatch`."""

    def __init__(self, batch: WindowBatch):
        self.batch = batch

    def __len__(self) -> int:
        return len(self.batch.label_ids)

    def __getitem__(self, idx: int):
        b = self.batch
        return (
            torch.from_numpy(b.resource_ids[idx]),
            torch.from_numpy(b.command_ids[idx]),
            torch.from_numpy(b.numeric_feats[idx]),
            torch.tensor(b.label_ids[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BiLSTMSequenceModel(nn.Module):
    """Bidirectional LSTM over per-event (resource, command, numeric)
    tokens, pooled and classified into `LABEL_ORDER` classes.

    The classification head is trained on the synthetic ground-truth
    labels; at inference the *loss/negative-log-likelihood assigned to
    the "normal" class* (not the argmax label) is what's exposed as the
    "sequence loss" anomaly signal, since in production you don't have
    ground truth for new events -- see `BiLSTMTrainer.extract_features`.
    """

    def __init__(
        self,
        num_resources: int,
        num_commands: int,
        numeric_dim: int = len(NUMERIC_FEATURE_NAMES),
        resource_emb_dim: int = 32,
        command_emb_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = len(LABEL_ORDER),
        dropout: float = 0.2,
    ):
        super().__init__()
        # Stashed so `BiLSTMTrainer.save()` can persist the exact
        # architecture alongside the weights, and `.load()` can rebuild
        # an identical model before loading the state dict into it.
        self.init_kwargs = dict(
            num_resources=num_resources, num_commands=num_commands, numeric_dim=numeric_dim,
            resource_emb_dim=resource_emb_dim, command_emb_dim=command_emb_dim,
            hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes, dropout=dropout,
        )
        self.resource_emb = nn.Embedding(num_resources, resource_emb_dim, padding_idx=0)
        self.command_emb = nn.Embedding(num_commands, command_emb_dim, padding_idx=0)
        input_dim = resource_emb_dim + command_emb_dim + numeric_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Attention pooling over the bi-directional hidden states, so the
        # model can weight the timestep(s) that matter most (e.g. the
        # privilege-escalation hop in a lateral-movement window) rather
        # than relying solely on the final hidden state.
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, resource_ids: torch.Tensor, command_ids: torch.Tensor, numeric_feats: torch.Tensor):
        r = self.resource_emb(resource_ids)
        c = self.command_emb(command_ids)
        x = torch.cat([r, c, numeric_feats], dim=-1)
        out, _ = self.lstm(x)               # (B, W, 2*hidden)
        out = self.dropout(out)
        attn_scores = torch.softmax(self.attn(out).squeeze(-1), dim=-1)  # (B, W)
        pooled = torch.einsum("bwh,bw->bh", out, attn_scores)            # (B, 2*hidden)
        logits = self.classifier(pooled)
        return logits, attn_scores


# ---------------------------------------------------------------------------
# Trainer / inference
# ---------------------------------------------------------------------------
@dataclass
class SequenceFeatures:
    """Per-window features handed off to the XGBoost classifier."""

    entity_ids: List[str]
    end_timestamps: List[pd.Timestamp]
    true_label_ids: np.ndarray
    predicted_label_ids: np.ndarray
    predicted_probs: np.ndarray            # (N, num_classes)
    sequence_loss: np.ndarray              # NLL assigned to "normal" class; higher = more anomalous
    normal_probability: np.ndarray         # P(normal); convenience = 1 - anomaly signal


class BiLSTMTrainer:
    """Training loop + evaluation + feature-extraction wrapper around
    `BiLSTMSequenceModel`."""

    def __init__(
        self,
        model: BiLSTMSequenceModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        class_weights: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)

    @staticmethod
    def compute_class_weights(label_ids: np.ndarray, num_classes: int = len(LABEL_ORDER)) -> torch.Tensor:
        """Inverse-frequency class weights so the heavily-imbalanced
        normal/attack split doesn't collapse the model into always
        predicting 'normal'."""
        counts = np.bincount(label_ids, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (num_classes * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def train(self, loader, epochs: int = 10, val_loader=None, verbose: bool = True) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0
            for res, cmd, num, labels in loader:
                res, cmd, num, labels = (t.to(self.device) for t in (res, cmd, num, labels))
                self.optimizer.zero_grad()
                logits, _ = self.model(res, cmd, num)
                loss = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                running += loss.item() * len(labels)
            train_loss = running / max(1, len(loader.dataset))
            history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)

            if verbose:
                msg = f"[BiLSTM] epoch {epoch}/{epochs} train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.4f}"
                print(msg)
        return history

    @torch.no_grad()
    def evaluate(self, loader) -> float:
        self.model.eval()
        running = 0.0
        for res, cmd, num, labels in loader:
            res, cmd, num, labels = (t.to(self.device) for t in (res, cmd, num, labels))
            logits, _ = self.model(res, cmd, num)
            loss = self.criterion(logits, labels)
            running += loss.item() * len(labels)
        return running / max(1, len(loader.dataset))

    @torch.no_grad()
    def extract_features(self, batch: WindowBatch, batch_size: int = 512) -> SequenceFeatures:
        """Run inference over a `WindowBatch` and return the per-window
        features consumed by the XGBoost classifier: predicted class,
        full probability vector, and `sequence_loss` -- the negative
        log-likelihood the model assigns to the "normal" class for that
        window. A window whose recent trajectory looks nothing like
        normal behavior gets a high sequence_loss even if we don't know
        its true label, which is exactly what's needed at inference time
        in production (no ground truth available then).
        """
        self.model.eval()
        n = len(batch.label_ids)
        all_probs = np.zeros((n, len(LABEL_ORDER)), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            res = torch.from_numpy(batch.resource_ids[start:end]).to(self.device)
            cmd = torch.from_numpy(batch.command_ids[start:end]).to(self.device)
            num = torch.from_numpy(batch.numeric_feats[start:end]).to(self.device)
            logits, _ = self.model(res, cmd, num)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs[start:end] = probs

        eps = 1e-9
        normal_probability = all_probs[:, NORMAL_LABEL_ID]
        sequence_loss = -np.log(np.clip(normal_probability, eps, 1.0))
        predicted_label_ids = all_probs.argmax(axis=1)

        return SequenceFeatures(
            entity_ids=batch.entity_ids,
            end_timestamps=batch.end_timestamps,
            true_label_ids=batch.label_ids,
            predicted_label_ids=predicted_label_ids,
            predicted_probs=all_probs,
            sequence_loss=sequence_loss,
            normal_probability=normal_probability,
        )

    def save(self, path: str, resource_vocab: Optional[Vocabulary] = None, command_vocab: Optional[Vocabulary] = None) -> None:
        """Persist everything needed to reconstruct a ready-to-run
        trainer with no retraining: the architecture kwargs (so the
        model can be rebuilt before loading weights into it), the state
        dict, and -- if supplied -- the resource/command vocabularies
        that must line up with those weights' embedding tables.
        `AttackClassificationPipeline.save()` always passes the
        vocabularies; they're optional here only so this method still
        works standalone (e.g. saving mid-experiment)."""
        bundle = {
            "model_init_kwargs": self.model.init_kwargs,
            "state_dict": self.model.state_dict(),
            "resource_vocab": resource_vocab.to_dict() if resource_vocab is not None else None,
            "command_vocab": command_vocab.to_dict() if command_vocab is not None else None,
        }
        torch.save(bundle, path)

    def load(self, path: str) -> None:
        """In-place weight load into the trainer's existing model
        (architecture must already match, e.g. because the model was
        just constructed with the same kwargs). Prefer
        `BiLSTMTrainer.load_pretrained()` for the common case of building
        a trainer from scratch straight from a saved file."""
        bundle = torch.load(path, map_location=self.device)
        state_dict = bundle["state_dict"] if isinstance(bundle, dict) and "state_dict" in bundle else bundle
        self.model.load_state_dict(state_dict)

    @classmethod
    def load_pretrained(cls, path: str, device: Optional[str] = None) -> Tuple["BiLSTMTrainer", Optional[Vocabulary], Optional[Vocabulary]]:
        """Factory: rebuild the exact model architecture from the saved
        config, load its weights, and return `(trainer, resource_vocab,
        command_vocab)` -- no training loop, no dataset required. The
        returned trainer's model is set to eval mode; call `.train()`
        again only if you intend to fine-tune further."""
        bundle = torch.load(path, map_location=device or "cpu")
        if not isinstance(bundle, dict) or "model_init_kwargs" not in bundle:
            raise ValueError(
                f"{path} doesn't look like a bundle saved by BiLSTMTrainer.save() "
                "(missing 'model_init_kwargs') -- old-format weights-only file?"
            )
        model = BiLSTMSequenceModel(**bundle["model_init_kwargs"])
        model.load_state_dict(bundle["state_dict"])
        trainer = cls(model, device=device)
        trainer.model.eval()

        resource_vocab = Vocabulary.from_dict(bundle["resource_vocab"]) if bundle.get("resource_vocab") else None
        command_vocab = Vocabulary.from_dict(bundle["command_vocab"]) if bundle.get("command_vocab") else None
        return trainer, resource_vocab, command_vocab