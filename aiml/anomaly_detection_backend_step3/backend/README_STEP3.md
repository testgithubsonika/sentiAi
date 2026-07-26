# Step 3 — FastAPI Backend

Wires the Step 1 DB schema (`models.py`) and Step 2 ML stack
(`baseline_profiling.py` -> `sequence_model.py` -> `attack_classifier.py`
-> `explainability.py`, orchestrated by `anomaly_pipeline.py`) into a
REST + WebSocket API with a background streaming worker.

## New files (Step 3)

| File                     | Purpose |
|--------------------------|---------|
| `config.py`              | Env-driven settings (DB URL, CORS, training/streaming tuning) |
| `database.py`            | SQLAlchemy engine/session + `get_db()` FastAPI dependency |
| `schemas.py`             | Pydantic request/response models |
| `websocket_manager.py`   | `ConnectionManager` for `/ws/stream` clients |
| `ml_worker.py`           | Trains the stack, then streams simulated events through `AnomalyDetectionPipeline`, persisting + broadcasting alerts |
| `routers/entities.py`    | `GET /api/entities`, `GET /api/entities/{entity_id}/history` |
| `routers/alerts.py`      | `GET /api/alerts`, `PATCH /api/alerts/{alert_id}` (triage) |
| `routers/metrics.py`     | `GET /api/metrics` |
| `main.py`                | FastAPI app, CORS, lifespan (train + start streaming), `/ws/stream`, `/api/health` |

## Run it

```bash
pip install -r requirements.txt

# point at your Postgres instance (see Step 1 README for docker run cmd)
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/anomaly_db"

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On startup the app will:
1. Create any missing tables (`Base.metadata.create_all`).
2. Train IForest + Bi-LSTM + XGBoost on a fresh synthetic batch
   (~1-2 min depending on `TRAIN_DAYS` / `BILSTM_EPOCHS` / hardware).
3. Persist the synthetic entity population into `entity_profiles`.
4. Start streaming simulated live events through the trained pipeline,
   persisting flagged alerts and broadcasting everything over
   `/ws/stream`.

Docs at `http://localhost:8000/docs`. Health/worker status at
`GET /api/health`.

## Key env vars (all optional, see `config.py` for the full list)

| Var                        | Default | Meaning |
|-----------------------------|---------|---------|
| `DATABASE_URL`              | local Postgres | SQLAlchemy connection string |
| `CORS_ORIGINS`              | `*`     | Comma-separated allowed origins |
| `AUTO_TRAIN_ON_STARTUP`     | `true`  | Set `false` for a REST-only instance against an already-populated DB |
| `AUTO_STREAM_ON_STARTUP`    | `true`  | Set `false` to train but not start the live stream |
| `TRAIN_DAYS` / `BILSTM_EPOCHS` | `10` / `4` | Trade training time vs. model quality for the demo |
| `STREAM_MIN_DELAY_SECONDS` / `STREAM_MAX_DELAY_SECONDS` | `0.4` / `1.5` | Pacing between simulated live events |

## API summary

- `GET /api/entities?risk_level=high&entity_type=user&q=...&limit=&offset=`
  Risk-ranked entity list (risk = strongest *open* alert per entity).
- `GET /api/entities/{entity_id}/history`
  Recent raw logs + alerts + the live River adaptive baseline
  (per-feature mean/std) if the worker currently holds warm state for
  that entity.
- `GET /api/alerts?status=open&severity=critical&sort_by=risk_score`
  Ranked, filterable historical alerts.
- `PATCH /api/alerts/{alert_id}` `{"status": "acknowledged"}`
  SOC triage action (bonus endpoint beyond the original spec, since a
  read-only alert queue isn't very actionable in a live demo).
- `GET /api/metrics`
  Dashboard header counters: total events, active anomalies, severity/
  type breakdowns, worker + WebSocket client status.
- `WS /ws/stream`
  JSON text frames: `{"type": "event", ...}` for every streamed event,
  `{"type": "alert", ...}` for every flagged one (includes SHAP-derived
  `summary` + `contributing_factors`).

## Notes / design choices

- All blocking ML/DB work in `ml_worker.py` runs via `asyncio.to_thread`
  so the event loop stays responsive to REST/WebSocket traffic during
  the ~1-2 min startup training and the per-event inference in the
  streaming loop.
- The streaming loop reuses the *same* trained population
  (`state.generator`), so the "live" entities are the exact ones just
  trained on — the online River baseline stays warm and
  `get_adaptive_baseline()` returns meaningful values immediately.
- `models.py`'s `JSONB`/`ARRAY`/`UUID` columns are Postgres-specific
  (per the Step 1 README) — this backend assumes Postgres, not SQLite.
