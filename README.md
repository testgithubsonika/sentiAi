"# sentiAi

SentiAi is an end-to-end behavioral anomaly detection demo that combines:

- a synthetic data generation layer for access-log and entity-profile data,
- an AI/ML pipeline for baseline profiling, sequence modeling, attack classification, and explainability,
- a FastAPI backend that exposes REST and WebSocket APIs for live anomaly streaming,
- a React + Tailwind dashboard for monitoring alerts, entities, and metrics in real time.

## Project structure

- [aiml/README.md](aiml/README.md) — synthetic data generation and the ML pipeline foundation
- [aiml/anomaly_detection_backend_step3/backend/README_STEP3.md](aiml/anomaly_detection_backend_step3/backend/README_STEP3.md) — FastAPI backend and streaming worker
- [sentinel_soc_frontend/frontend/README_FRONTEND.md](sentinel_soc_frontend/frontend/README_FRONTEND.md) — SOC dashboard frontend

## 1. Data and ML layer

The AI module lives under [aiml](aiml) and includes:

- synthetic data generation and PostgreSQL schema setup,
- baseline profiling and drift detection logic,
- training components for the anomaly detection stack.

Typical setup:

```bash
cd aiml
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

To generate synthetic data only:

```bash
python data_generator.py \
  --num-users 200 \
  --num-service-accounts 40 \
  --num-devices 60 \
  --days 30 \
  --output data/access_logs.parquet \
  --profiles-output data/entity_profiles.csv
```

To load into PostgreSQL:

```bash
docker run --name anomaly-pg -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=anomaly_db -p 5432:5432 -d postgres:16

export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/anomaly_db"

python generate_and_load.py \
  --num-users 200 --num-service-accounts 40 --num-devices 60 \
  --days 30 --create-tables --load-db
```

## 2. Backend API

The  backend is implemented in [aiml/anomaly_detection_backend_step3/backend](aiml/anomaly_detection_backend_step3/backend).

Run it with:

```bash
cd aiml/anomaly_detection_backend_step3/backend
pip install -r requirements.txt

export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/anomaly_db"

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

It provides:

- REST endpoints such as /api/alerts, /api/entities, and /api/metrics,
- a live WebSocket stream at /ws/stream,
- startup training and streaming of simulated anomalies.

## 3. Frontend dashboard

The dashboard frontend is in [sentinel_soc_frontend/frontend](sentinel_soc_frontend/frontend).

Run it with:

```bash
cd sentinel_soc_frontend/frontend
npm install
cp .env.example .env
npm run dev
```

Make sure the FastAPI backend is running and that VITE_API_BASE_URL in the frontend env points to the backend.

## 4. Quick start summary

1. Start PostgreSQL and load synthetic data.
2. Launch the backend API.
3. Start the frontend dashboard.
4. Open the UI and watch live alerts stream in from the backend.

For more detailed setup and architecture notes, see the README files in the subdirectories above." 
