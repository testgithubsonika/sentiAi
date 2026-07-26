# AI-Powered Behavioral Anomaly Detection — Synthetic Data Layer

This module generates realistic access-log data (with embedded per-entity
baselines and 7 injected attack patterns) and defines the PostgreSQL
schema the rest of the pipeline reads from.

## Files

| File                    | Purpose                                                              |
|--------------------------|-----------------------------------------------------------------------|
| `data_generator.py`     | Standalone generator: entity profiles, normal behavior, attack injection |
| `models.py`             | SQLAlchemy 2.0 ORM models (`entity_profiles`, `raw_access_logs`, `processed_streaming_logs`, `alert_queue`) |
| `generate_and_load.py`  | CLI: generate data and (optionally) load it straight into PostgreSQL |
| `baseline_profiling.py` | ML backend: PyOD Isolation Forest baseline profiling + River online concept-drift handling |
| `requirements.txt`      | Python dependencies |

## ML backend: baseline profiling & concept drift (`baseline_profiling.py`)

Run the built-in demo directly:

```bash
python baseline_profiling.py
```

It simulates a normal entity, cold-start entities scored via the global
fallback model, a point anomaly, and a gradually drifting "shift worker"
entity, printing scores and drift events at each stage.

Core classes:

- `FeatureVector` — one event's features (`login_hour`, `session_duration`,
  `geo_distance_km`, `failure_count`, plus optional `extra` features).
- `GlobalFallbackModel` — a single PyOD `IForest` fit on pooled events
  across all entities, used to score any entity with < `min_history`
  (default 5) events of its own.
- `EntityBaselineModel` / `BaselineProfiler` — per-entity PyOD `IForest`
  on a rolling window, periodically refit, cold-start aware.
- `OnlineDriftBaseline` — per-entity River pipeline: adaptive
  `StandardScaler`, streaming `HalfSpaceTrees` anomaly scorer, and an
  `ADWIN` drift detector on the online score stream.
- `HybridBehavioralProfiler` — the entry point. Call
  `observe(entity_id, feature_vector)` per event; it scores against both
  layers, and when ADWIN signals drift it force-retrains the entity's
  batch model immediately and suppresses the anomaly flag for that event,
  so a genuine behavioral shift (new hours, new device) gets folded into
  the baseline instead of being flagged forever.

Tune `min_history`, `retrain_every`, `contamination`, and `drift_delta`
against your actual event volume and how fast legitimate behavior is
expected to shift.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Generate data only (no database needed)

```bash
python data_generator.py \
    --num-users 200 \
    --num-service-accounts 40 \
    --num-devices 60 \
    --days 30 \
    --output data/access_logs.parquet \
    --profiles-output data/entity_profiles.csv
```

This prints a summary of event volume and label distribution, e.g.:

```
[SyntheticDataGenerator] entities=300 normal_events=22650 anomalous_events=2761
total=25411 anomaly_rate=10.87%
label
normal                       22650
brute_force                    597
low_and_slow_exfiltration      524
insider_drift                  511
lateral_movement               388
impossible_travel              298
device_spoofing                292
credential_stuffing            151
```

Each of the 7 attack types is independently sampled to occupy **0.5%–3%**
of total event volume (`ATTACK_RATE_RANGE` in `data_generator.py`) —
tune this constant or pass `attack_rate_range=(min, max)` to
`SyntheticDataGenerator.generate()` if you need a different mix.

Output columns: `entity_id, entity_type, timestamp, source_ip,
geo_location, resource_accessed, auth_method, auth_result,
session_duration, command_sequence, device_fingerprint, label`.
`geo_location`, `device_fingerprint`, and `command_sequence` are nested
JSON in CSV output and native list/struct columns in Parquet output.

## 3. Stand up PostgreSQL

Using Docker for a quick local instance:

```bash
docker run --name anomaly-pg -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=anomaly_db -p 5432:5432 -d postgres:16
```

## 4. Generate AND load into PostgreSQL

```bash
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/anomaly_db"

python generate_and_load.py \
    --num-users 200 --num-service-accounts 40 --num-devices 60 \
    --days 30 --create-tables --load-db
```

`--create-tables` runs `Base.metadata.create_all()` (idempotent — safe to
re-run). Drop `--create-tables` on subsequent runs once the schema
exists. Loading is batched (2,000 rows/commit) via
`Session.bulk_save_objects` for speed on large synthetic datasets.

## 5. Schema overview

- **`entity_profiles`** — one row per user/service_account/edge_device:
  habitual hours, home geo, home subnet, typical resources/device/auth
  method, session-duration baseline (mean/std). This is what the
  generator samples "normal" behavior from, and what a production
  scoring job would compare live events against.
- **`raw_access_logs`** — append-only landing zone for every access
  event, labeled `normal` or with the specific attack pattern (ground
  truth, useful for supervised model training/eval on synthetic data).
- **`processed_streaming_logs`** — one row per raw log after feature
  engineering (geo-velocity, resource rarity, command entropy, auth
  failure rate in a trailing window, etc.) plus a model `anomaly_score`.
- **`alert_queue`** — created when a processed log crosses an alerting
  threshold; carries severity, status lifecycle
  (open → acknowledged → resolved/false_positive), and a JSONB
  `details` field for explainability evidence.

## 6. Attack patterns modeled

| Label                        | Simulated technique |
|-------------------------------|----------------------|
| `brute_force`                | Burst of 15–60 rapid auth attempts against one entity from a single external IP, mostly failures |
| `impossible_travel`           | Two logins from geographically distant cities within a time delta implying >900 km/h travel |
| `credential_stuffing`         | Small pool of attacker IPs hitting 20–80 distinct entities, ~2% success rate |
| `lateral_movement`            | 6–18 rapid hops across resources, escalating into infra/sensitive resources with privilege-escalation commands (`sudo`, `kubectl`, `chmod`...) |
| `device_spoofing`             | Otherwise-normal event with OS/MAC/protocol that mismatches the entity's baseline fingerprint |
| `low_and_slow_exfiltration`   | 6–14 sparse pulls of sensitive resources spread across the full time window, with exfil-flavored commands (`scp`, `zip`, `export`...) |
| `insider_drift`               | Gradual (7–21 day) shift in an entity's active hours and increasing use of sensitive resources |

## 7. Notes for the anomaly-detection model layer

- `raw_access_logs.label` is **ground truth** for training/eval only —
  a real detector would not see it; it should be scored purely from
  `processed_streaming_logs` engineered features.
- `entity_profiles` doubles as the "expected baseline" a
  distance/z-score based detector (or the seed for an unsupervised
  model) can compare live events against.
- Reproducibility: both `data_generator.py` and `generate_and_load.py`
  accept `--seed` (default 42) so demo runs are repeatable.
