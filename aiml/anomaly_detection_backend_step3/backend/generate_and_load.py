"""
generate_and_load.py
=====================
CLI entrypoint that ties the synthetic data generator to the PostgreSQL
schema defined in models.py.

Typical usage
-------------
# Generate only, write CSV/Parquet to disk (no DB required):
python generate_and_load.py --num-users 200 --days 30 --output data/access_logs.parquet

# Generate AND load straight into Postgres:
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/anomaly_db"
python generate_and_load.py --days 30 --load-db --create-tables
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from data_generator import ATTACK_RATE_RANGE, SyntheticDataGenerator
from models import Base, EntityProfile, EntityType, LabelType, RawAccessLog, create_all


def load_profiles(session: Session, profiles_df) -> None:
    objects = []
    for _, row in profiles_df.iterrows():
        objects.append(EntityProfile(
            entity_id=row["entity_id"],
            entity_type=EntityType(row["entity_type"]),
            display_name=row["display_name"],
            habitual_hour_start=int(row["habitual_hour_start"]),
            habitual_hour_end=int(row["habitual_hour_end"]),
            home_city=row["home_city"],
            home_country=row["home_country"],
            home_lat=float(row["home_lat"]),
            home_lon=float(row["home_lon"]),
            home_subnet=row["home_subnet"],
            typical_resources=list(row["typical_resources"]),
            typical_auth_method=row["typical_auth_method"],
            typical_os=row["typical_os"],
            typical_mac=row["typical_mac"],
            typical_protocol=row["typical_protocol"],
            mean_session_seconds=float(row["mean_session_seconds"]),
            std_session_seconds=float(row["std_session_seconds"]),
        ))
    session.bulk_save_objects(objects)
    session.commit()
    print(f"Loaded {len(objects)} entity profiles into entity_profiles")


def load_raw_logs(session: Session, df, batch_size: int = 2000) -> None:
    buffer = []
    total = 0
    for _, row in df.iterrows():
        buffer.append(RawAccessLog(
            entity_id=row["entity_id"],
            entity_type=EntityType(row["entity_type"]),
            timestamp=row["timestamp"],
            source_ip=row["source_ip"],
            geo_location=row["geo_location"],
            geo_lat=float(row["geo_location"]["lat"]),
            geo_lon=float(row["geo_location"]["lon"]),
            resource_accessed=row["resource_accessed"],
            auth_method=row["auth_method"],
            auth_result=row["auth_result"],
            session_duration=float(row["session_duration"]),
            command_sequence=list(row["command_sequence"]),
            device_fingerprint=row["device_fingerprint"],
            label=LabelType(row["label"]),
        ))
        if len(buffer) >= batch_size:
            session.bulk_save_objects(buffer)
            session.commit()
            total += len(buffer)
            print(f"  ...inserted {total} raw log rows")
            buffer = []
    if buffer:
        session.bulk_save_objects(buffer)
        session.commit()
        total += len(buffer)
    print(f"Loaded {total} raw access log rows into raw_access_logs")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic logs and optionally load into PostgreSQL.")
    parser.add_argument("--num-users", type=int, default=200)
    parser.add_argument("--num-service-accounts", type=int, default=40)
    parser.add_argument("--num-devices", type=int, default=60)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/access_logs.parquet")
    parser.add_argument("--profiles-output", type=str, default="data/entity_profiles.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-db", action="store_true", help="Also load generated data into PostgreSQL")
    parser.add_argument("--create-tables", action="store_true", help="Run create_all() before loading")
    parser.add_argument("--database-url", type=str, default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()

    start_date = (datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date
                  else datetime.utcnow() - timedelta(days=args.days))
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    gen = SyntheticDataGenerator(
        num_users=args.num_users,
        num_service_accounts=args.num_service_accounts,
        num_devices=args.num_devices,
        seed=args.seed,
    )
    df = gen.generate(start_date, args.days, attack_rate_range=ATTACK_RATE_RANGE)
    profiles_df = gen.profiles_dataframe()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.output.endswith(".parquet"):
        SyntheticDataGenerator.to_parquet(df, args.output)
    else:
        SyntheticDataGenerator.to_csv(df, args.output)
    print(f"Wrote {len(df)} rows to {args.output}")

    os.makedirs(os.path.dirname(args.profiles_output) or ".", exist_ok=True)
    profiles_df.to_csv(args.profiles_output, index=False)

    if args.load_db:
        if not args.database_url:
            raise SystemExit("--load-db requires --database-url or a DATABASE_URL env var")
        engine = create_engine(args.database_url)
        if args.create_tables:
            create_all(engine)
            print("Created tables (if not already present).")
        with Session(engine) as session:
            load_profiles(session, profiles_df)
            load_raw_logs(session, df)


if __name__ == "__main__":
    main()
