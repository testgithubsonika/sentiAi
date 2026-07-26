"""
data_generator.py
==================
Synthetic access-log generator for the Behavioral Anomaly Detection
hackathon project.

Pipeline
--------
1. EntityProfileFactory  -> builds a realistic baseline (habitual hours,
   home geo, typical resources, typical device fingerprint) for every
   entity (user / service_account / edge_device).
2. NormalBehaviorSimulator -> samples "normal" access events from each
   entity's baseline with Gaussian noise, across a configurable date
   range.
3. AttackInjector -> injects seven distinct attack patterns at a
   configurable rate (0.5%-3% of total normal volume each):
       - brute_force
       - impossible_travel
       - credential_stuffing
       - lateral_movement
       - device_spoofing
       - low_and_slow_exfiltration
       - insider_drift
4. SyntheticDataGenerator -> orchestrates 1-3, merges + shuffles the
   result into a single time-ordered pandas DataFrame, and can export
   to CSV/Parquet or load straight into PostgreSQL via SQLAlchemy.

Run standalone:
    python data_generator.py --num-users 200 --num-service-accounts 40 \
        --num-devices 60 --days 30 --output data/access_logs.csv
"""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Reference / vocabulary data
# ---------------------------------------------------------------------------
ENTITY_TYPES = ["user", "service_account", "edge_device"]
ENTITY_TYPE_WEIGHTS = [0.70, 0.20, 0.10]

AUTH_METHODS = ["password", "mfa", "sso", "api_key", "certificate"]
OS_LIST = ["Windows 11", "Windows Server 2022", "Ubuntu 22.04", "macOS Sonoma", "RHEL 9", "IoT-RTOS"]
PROTOCOLS = ["HTTPS", "SSH", "RDP", "SMB", "MQTT", "SFTP"]

RESOURCE_CATEGORIES = ["finance", "hr", "engineering", "customer_db", "infra", "iot_gateway", "billing", "devops"]
RESOURCE_POOL = [f"resource/{cat}/{i:03d}" for cat in RESOURCE_CATEGORIES for i in range(1, 26)]
SENSITIVE_RESOURCES = [r for r in RESOURCE_POOL if any(c in r for c in ("finance", "customer_db", "billing"))]
INFRA_RESOURCES = [r for r in RESOURCE_POOL if any(c in r for c in ("infra", "devops", "iot_gateway"))]

BENIGN_COMMANDS = ["ls", "cat", "whoami", "ps", "git pull", "git status", "netstat", "df -h"]
PRIV_ESC_COMMANDS = ["sudo", "chmod", "chown", "systemctl", "docker", "kubectl", "usermod"]
EXFIL_COMMANDS = ["scp", "curl", "wget", "zip", "export", "select *", "rsync"]
COMMANDS_POOL = BENIGN_COMMANDS + PRIV_ESC_COMMANDS + EXFIL_COMMANDS

# (city, country, lat, lon)
GEO_CITIES = [
    ("New York", "US", 40.7128, -74.0060),
    ("San Francisco", "US", 37.7749, -122.4194),
    ("London", "UK", 51.5074, -0.1278),
    ("Frankfurt", "DE", 50.1109, 8.6821),
    ("Singapore", "SG", 1.3521, 103.8198),
    ("Sydney", "AU", -33.8688, 151.2093),
    ("Tokyo", "JP", 35.6762, 139.6503),
    ("Sao Paulo", "BR", -23.5505, -46.6333),
    ("Mumbai", "IN", 19.0760, 72.8777),
    ("Toronto", "CA", 43.6532, -79.3832),
    ("Moscow", "RU", 55.7558, 37.6173),
    ("Lagos", "NG", 6.5244, 3.3792),
]

# Default per-attack injection rate range (fraction of normal event volume).
# Each is independently sampled within [0.5%, 3%] unless overridden.
ATTACK_RATE_RANGE = (0.005, 0.03)

ATTACK_TYPES = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfiltration",
    "insider_drift",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def random_subnet_ip(subnet_prefix: str) -> str:
    """Generate a plausible internal-looking IP within a /24 style prefix, e.g. '10.14.0'."""
    return f"{subnet_prefix}.{random.randint(2, 254)}"


def jitter_geo(lat: float, lon: float, max_km: float = 15.0) -> Tuple[float, float]:
    """Small Gaussian jitter around a home location (stays within the same metro area)."""
    dlat = np.random.normal(0, max_km / 111.0 / 2)
    dlon = np.random.normal(0, max_km / 111.0 / 2)
    return lat + dlat, lon + dlon


def sample_command_sequence(pool: List[str], min_len=1, max_len=8) -> List[str]:
    n = np.random.poisson(3) + min_len
    n = int(np.clip(n, min_len, max_len))
    return list(np.random.choice(pool, size=n, replace=True))


# ---------------------------------------------------------------------------
# Entity profile
# ---------------------------------------------------------------------------
@dataclass
class EntityProfile:
    entity_id: str
    entity_type: str
    display_name: str
    habitual_hour_start: int
    habitual_hour_end: int
    home_city: str
    home_country: str
    home_lat: float
    home_lon: float
    home_subnet: str  # e.g. "10.14.3" (a /24-style prefix)
    typical_resources: List[str]
    typical_auth_method: str
    typical_os: str
    typical_mac: str
    typical_protocol: str
    mean_session_seconds: float
    std_session_seconds: float

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


class EntityProfileFactory:
    """Builds a population of entity baselines."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    def _build_one(self, entity_type: str, idx: int) -> EntityProfile:
        geo = GEO_CITIES[self.rng.integers(0, len(GEO_CITIES))]
        city, country, lat, lon = geo

        # Service accounts / edge devices skew toward 24/7 or off-hours automation;
        # human users skew toward a ~9-hour workday window.
        if entity_type == "user":
            start_hour = int(self.rng.integers(6, 10))
            window = int(self.rng.integers(7, 10))
        elif entity_type == "service_account":
            start_hour = int(self.rng.integers(0, 24))
            window = int(self.rng.integers(6, 24))
        else:  # edge_device
            start_hour = 0
            window = 24

        end_hour = (start_hour + window) % 24

        n_resources = int(self.rng.integers(3, 9))
        typical_resources = list(self.rng.choice(RESOURCE_POOL, size=n_resources, replace=False))

        entity_id = f"{entity_type[:3].upper()}-{idx:05d}"
        display_name = fake.name() if entity_type == "user" else f"{entity_type}-{fake.user_name()}"

        return EntityProfile(
            entity_id=entity_id,
            entity_type=entity_type,
            display_name=display_name,
            habitual_hour_start=start_hour,
            habitual_hour_end=end_hour,
            home_city=city,
            home_country=country,
            home_lat=float(lat),
            home_lon=float(lon),
            home_subnet=f"10.{self.rng.integers(0, 255)}.{self.rng.integers(0, 255)}",
            typical_resources=typical_resources,
            typical_auth_method=str(self.rng.choice(AUTH_METHODS)),
            typical_os=str(self.rng.choice(OS_LIST)),
            typical_mac=fake.mac_address(),
            typical_protocol=str(self.rng.choice(PROTOCOLS)),
            mean_session_seconds=float(self.rng.uniform(120, 1800)),
            std_session_seconds=float(self.rng.uniform(20, 200)),
        )

    def build_population(self, num_users: int, num_service_accounts: int, num_devices: int) -> List[EntityProfile]:
        profiles = []
        idx = 1
        for _ in range(num_users):
            profiles.append(self._build_one("user", idx)); idx += 1
        for _ in range(num_service_accounts):
            profiles.append(self._build_one("service_account", idx)); idx += 1
        for _ in range(num_devices):
            profiles.append(self._build_one("edge_device", idx)); idx += 1
        return profiles


# ---------------------------------------------------------------------------
# Normal behavior simulation
# ---------------------------------------------------------------------------
class NormalBehaviorSimulator:
    """Generates habitual, low-noise access events for each entity profile."""

    def __init__(self, profiles: List[EntityProfile], seed: int = RANDOM_SEED):
        self.profiles = profiles
        self.rng = np.random.default_rng(seed)

    def _sample_timestamp_in_window(self, day: datetime, start_hour: int, end_hour: int) -> datetime:
        """Sample a timestamp within an entity's habitual hour window (handles wraparound)."""
        if start_hour <= end_hour:
            hour = self.rng.uniform(start_hour, end_hour)
        else:  # wraps past midnight
            span = (24 - start_hour) + end_hour
            offset = self.rng.uniform(0, span)
            hour = (start_hour + offset) % 24
        minute_frac = hour - int(hour)
        return day.replace(hour=int(hour), minute=int(minute_frac * 60), second=int(self.rng.integers(0, 60)), microsecond=0)

    def _row_for_event(self, profile: EntityProfile, ts: datetime) -> dict:
        # Small chance the entity explores a resource outside its typical set
        # (Gaussian-noise-equivalent behavioral variance).
        if self.rng.random() < 0.08:
            resource = str(self.rng.choice(RESOURCE_POOL))
        else:
            resource = str(self.rng.choice(profile.typical_resources))

        lat, lon = jitter_geo(profile.home_lat, profile.home_lon, max_km=15.0)
        session_duration = float(max(5.0, self.rng.normal(profile.mean_session_seconds, profile.std_session_seconds)))

        commands = sample_command_sequence(BENIGN_COMMANDS, min_len=1, max_len=6)

        return {
            "log_id": str(uuid.uuid4()),
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": random_subnet_ip(profile.home_subnet),
            "geo_location": {"city": profile.home_city, "country": profile.home_country, "lat": lat, "lon": lon},
            "resource_accessed": resource,
            "auth_method": profile.typical_auth_method,
            "auth_result": "success",
            "session_duration": round(session_duration, 2),
            "command_sequence": commands,
            "device_fingerprint": {"os": profile.typical_os, "mac": profile.typical_mac, "protocol": profile.typical_protocol},
            "label": "normal",
        }

    def generate(self, start_date: datetime, days: int, events_per_day_range: Tuple[int, int] = (1, 6)) -> pd.DataFrame:
        rows = []
        for profile in self.profiles:
            for d in range(days):
                day = start_date + timedelta(days=d)
                # weekends: users mostly quiet, service accounts/devices largely unaffected
                is_weekend = day.weekday() >= 5
                if profile.entity_type == "user" and is_weekend and self.rng.random() < 0.85:
                    continue
                n_events = int(self.rng.integers(*events_per_day_range))
                for _ in range(n_events):
                    ts = self._sample_timestamp_in_window(day, profile.habitual_hour_start, profile.habitual_hour_end)
                    rows.append(self._row_for_event(profile, ts))
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Attack injection
# ---------------------------------------------------------------------------
class AttackInjector:
    """Generates rows for each of the seven modeled attack patterns."""

    def __init__(self, profiles: List[EntityProfile], seed: int = RANDOM_SEED):
        self.profiles = profiles
        self.rng = np.random.default_rng(seed)

    def _random_profile(self, entity_type: Optional[str] = None) -> EntityProfile:
        pool = [p for p in self.profiles if entity_type is None or p.entity_type == entity_type]
        return pool[self.rng.integers(0, len(pool))]

    def _base_row(self, profile: EntityProfile, ts: datetime, label: str, **overrides) -> dict:
        row = {
            "log_id": str(uuid.uuid4()),
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": random_subnet_ip(profile.home_subnet),
            "geo_location": {"city": profile.home_city, "country": profile.home_country,
                              "lat": profile.home_lat, "lon": profile.home_lon},
            "resource_accessed": str(self.rng.choice(profile.typical_resources)),
            "auth_method": profile.typical_auth_method,
            "auth_result": "success",
            "session_duration": float(max(5.0, self.rng.normal(profile.mean_session_seconds, profile.std_session_seconds))),
            "command_sequence": sample_command_sequence(BENIGN_COMMANDS),
            "device_fingerprint": {"os": profile.typical_os, "mac": profile.typical_mac, "protocol": profile.typical_protocol},
            "label": label,
        }
        row.update(overrides)
        return row

    # -- 1. Brute Force ----------------------------------------------------
    def brute_force(self, n_incidents: int, start_date: datetime, days: int) -> List[dict]:
        rows = []
        for _ in range(n_incidents):
            target = self._random_profile()
            attacker_ip = fake.ipv4_public()
            burst_start = start_date + timedelta(days=int(self.rng.integers(0, days)),
                                                  seconds=int(self.rng.integers(0, 86400)))
            n_attempts = int(self.rng.integers(15, 60))
            for i in range(n_attempts):
                ts = burst_start + timedelta(seconds=int(i * self.rng.uniform(1, 8)))
                success = (i == n_attempts - 1) and self.rng.random() < 0.3  # occasional eventual breach
                rows.append(self._base_row(
                    target, ts, "brute_force",
                    source_ip=attacker_ip,
                    auth_result="success" if success else "failure",
                    auth_method="password",
                    session_duration=float(self.rng.uniform(1, 5)),
                    command_sequence=[],
                    geo_location={"city": "unknown", "country": "XX",
                                  "lat": float(self.rng.uniform(-60, 60)), "lon": float(self.rng.uniform(-180, 180))},
                ))
        return rows

    # -- 2. Impossible Travel ----------------------------------------------
    def impossible_travel(self, n_incidents: int, start_date: datetime, days: int) -> List[dict]:
        rows = []
        for _ in range(n_incidents):
            target = self._random_profile()
            t1 = start_date + timedelta(days=int(self.rng.integers(0, days)), seconds=int(self.rng.integers(0, 86400)))
            # Second login within 5-45 minutes, from a far-away city (impossible physical travel speed)
            delta_minutes = self.rng.uniform(5, 45)
            t2 = t1 + timedelta(minutes=delta_minutes)

            other_geo = self._random_profile().home_city  # just to vary; pick real far city below
            far_city = GEO_CITIES[self.rng.integers(0, len(GEO_CITIES))]
            while far_city[0] == target.home_city:
                far_city = GEO_CITIES[self.rng.integers(0, len(GEO_CITIES))]

            dist_km = haversine_km(target.home_lat, target.home_lon, far_city[2], far_city[3])
            implied_speed = dist_km / max(delta_minutes / 60.0, 1e-6)
            # Only keep it if implied speed exceeds commercial flight speed (~900 km/h) -> guaranteed "impossible"
            if implied_speed < 900:
                t2 = t1 + timedelta(minutes=self.rng.uniform(1, 4))

            rows.append(self._base_row(target, t1, "impossible_travel"))
            rows.append(self._base_row(
                target, t2, "impossible_travel",
                source_ip=fake.ipv4_public(),
                geo_location={"city": far_city[0], "country": far_city[1], "lat": far_city[2], "lon": far_city[3]},
            ))
        return rows

    # -- 3. Credential Stuffing ---------------------------------------------
    def credential_stuffing(self, n_incidents: int, start_date: datetime, days: int) -> List[dict]:
        rows = []
        for _ in range(n_incidents):
            # small pool of attacker IPs hitting many distinct entities
            attacker_ips = [fake.ipv4_public() for _ in range(int(self.rng.integers(2, 5)))]
            n_targets = int(self.rng.integers(20, 80))
            burst_start = start_date + timedelta(days=int(self.rng.integers(0, days)), seconds=int(self.rng.integers(0, 86400)))
            for i in range(n_targets):
                target = self._random_profile()
                ts = burst_start + timedelta(seconds=int(i * self.rng.uniform(0.5, 4)))
                success = self.rng.random() < 0.02  # very low hit rate, characteristic of stuffing
                rows.append(self._base_row(
                    target, ts, "credential_stuffing",
                    source_ip=str(self.rng.choice(attacker_ips)),
                    auth_result="success" if success else "failure",
                    auth_method="password",
                    session_duration=float(self.rng.uniform(1, 4)),
                    command_sequence=[],
                    geo_location={"city": "unknown", "country": "XX",
                                  "lat": float(self.rng.uniform(-60, 60)), "lon": float(self.rng.uniform(-180, 180))},
                ))
        return rows

    # -- 4. Lateral Movement -------------------------------------------------
    def lateral_movement(self, n_incidents: int, start_date: datetime, days: int) -> List[dict]:
        rows = []
        for _ in range(n_incidents):
            target = self._random_profile(entity_type=self.rng.choice(["user", "service_account"]))
            burst_start = start_date + timedelta(days=int(self.rng.integers(0, days)), seconds=int(self.rng.integers(0, 86400)))
            n_hops = int(self.rng.integers(6, 18))
            # escalate from typical resources toward infra/sensitive resources
            escalation_pool = target.typical_resources + INFRA_RESOURCES + SENSITIVE_RESOURCES
            for i in range(n_hops):
                ts = burst_start + timedelta(minutes=int(i * self.rng.uniform(1, 6)))
                resource = INFRA_RESOURCES[self.rng.integers(0, len(INFRA_RESOURCES))] if i > n_hops * 0.4 \
                    else str(self.rng.choice(escalation_pool))
                commands = sample_command_sequence(PRIV_ESC_COMMANDS, min_len=2, max_len=6)
                rows.append(self._base_row(
                    target, ts, "lateral_movement",
                    resource_accessed=resource,
                    command_sequence=commands,
                    session_duration=float(self.rng.uniform(60, 400)),
                ))
        return rows

    # -- 5. Device Spoofing ---------------------------------------------------
    def device_spoofing(self, n_incidents: int, start_date: datetime, days: int) -> List[dict]:
        rows = []
        for _ in range(n_incidents):
            target = self._random_profile()
            ts = start_date + timedelta(days=int(self.rng.integers(0, days)), seconds=int(self.rng.integers(0, 86400)))
            spoof_os = str(self.rng.choice([o for o in OS_LIST if o != target.typical_os]))
            spoof_protocol = str(self.rng.choice([p for p in PROTOCOLS if p != target.typical_protocol]))
            spoof_mac = fake.mac_address()
            rows.append(self._base_row(
                target, ts, "device_spoofing",
                device_fingerprint={"os": spoof_os, "mac": spoof_mac, "protocol": spoof_protocol},
            ))
        return rows

    # -- 6. Low-and-Slow Exfiltration ------------------------------------------
    def low_and_slow_exfiltration(self, n_incidents: int, start_date: datetime, days: int) -> List[dict]:
        rows = []
        for _ in range(n_incidents):
            target = self._random_profile()
            # spread a handful of sensitive-resource pulls thinly over most of the window
            n_pulls = int(self.rng.integers(6, 14))
            pull_days = sorted(self.rng.choice(range(days), size=min(n_pulls, days), replace=False))
            for d in pull_days:
                ts = start_date + timedelta(days=int(d), seconds=int(self.rng.integers(0, 86400)))
                resource = str(self.rng.choice(SENSITIVE_RESOURCES))
                commands = sample_command_sequence(EXFIL_COMMANDS, min_len=1, max_len=3)
                rows.append(self._base_row(
                    target, ts, "low_and_slow_exfiltration",
                    resource_accessed=resource,
                    command_sequence=commands,
                    session_duration=float(self.rng.uniform(200, 900)),
                ))
        return rows

    # -- 7. Insider Drift -------------------------------------------------------
    def insider_drift(self, n_incidents: int, start_date: datetime, days: int) -> List[dict]:
        rows = []
        for _ in range(n_incidents):
            target = self._random_profile(entity_type="user")
            drift_start_day = int(self.rng.integers(0, max(days - 7, 1)))
            drift_len = min(days - drift_start_day, int(self.rng.integers(7, 21)))
            for d in range(drift_len):
                day = start_date + timedelta(days=drift_start_day + d)
                progress = d / max(drift_len - 1, 1)  # 0 -> 1 over the drift window
                # habitual hours slowly shift later; resource set slowly expands into sensitive territory
                hour_shift = int(progress * 6)
                hour = (target.habitual_hour_end + hour_shift) % 24
                ts = day.replace(hour=hour, minute=int(self.rng.integers(0, 60)), second=0, microsecond=0)
                use_sensitive = self.rng.random() < progress * 0.6
                resource = str(self.rng.choice(SENSITIVE_RESOURCES)) if use_sensitive else str(self.rng.choice(target.typical_resources))
                rows.append(self._base_row(
                    target, ts, "insider_drift",
                    resource_accessed=resource,
                    session_duration=float(self.rng.uniform(target.mean_session_seconds, target.mean_session_seconds * 2)),
                ))
        return rows

    def inject_all(self, normal_event_count: int, start_date: datetime, days: int,
                    rate_range: Tuple[float, float] = ATTACK_RATE_RANGE) -> List[dict]:
        """Run every attack generator; each type's incident count is derived from
        an independently sampled rate in `rate_range` applied to normal volume."""
        all_rows: List[dict] = []
        generators = {
            "brute_force": self.brute_force,
            "impossible_travel": self.impossible_travel,
            "credential_stuffing": self.credential_stuffing,
            "lateral_movement": self.lateral_movement,
            "device_spoofing": self.device_spoofing,
            "low_and_slow_exfiltration": self.low_and_slow_exfiltration,
            "insider_drift": self.insider_drift,
        }
        for attack_type, gen_fn in generators.items():
            rate = self.rng.uniform(*rate_range)
            target_event_volume = max(1, int(normal_event_count * rate))
            # translate target event volume into a rough "number of incidents"
            # (each incident produces several rows); tuned per attack type.
            per_incident_rows = {
                "brute_force": 30, "impossible_travel": 2, "credential_stuffing": 40,
                "lateral_movement": 10, "device_spoofing": 1, "low_and_slow_exfiltration": 9,
                "insider_drift": 12,
            }[attack_type]
            n_incidents = max(1, target_event_volume // per_incident_rows)
            rows = gen_fn(n_incidents, start_date, days)
            all_rows.extend(rows)
        return all_rows


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class SyntheticDataGenerator:
    def __init__(self, num_users=200, num_service_accounts=40, num_devices=60, seed: int = RANDOM_SEED):
        self.factory = EntityProfileFactory(seed=seed)
        self.profiles = self.factory.build_population(num_users, num_service_accounts, num_devices)
        self.simulator = NormalBehaviorSimulator(self.profiles, seed=seed)
        self.injector = AttackInjector(self.profiles, seed=seed)

    def generate(self, start_date: datetime, days: int,
                 events_per_day_range: Tuple[int, int] = (1, 6),
                 attack_rate_range: Tuple[float, float] = ATTACK_RATE_RANGE) -> pd.DataFrame:
        normal_df = self.simulator.generate(start_date, days, events_per_day_range)
        attack_rows = self.injector.inject_all(len(normal_df), start_date, days, attack_rate_range)
        attack_df = pd.DataFrame(attack_rows)

        full_df = pd.concat([normal_df, attack_df], ignore_index=True)
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)

        total = len(full_df)
        anomaly_count = len(attack_df)
        print(f"[SyntheticDataGenerator] entities={len(self.profiles)} "
              f"normal_events={len(normal_df)} anomalous_events={anomaly_count} "
              f"total={total} anomaly_rate={anomaly_count/total:.2%}")
        print(full_df["label"].value_counts())
        return full_df

    def profiles_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([p.as_dict() for p in self.profiles])

    @staticmethod
    def to_csv(df: pd.DataFrame, path: str):
        out = df.copy()
        out["geo_location"] = out["geo_location"].apply(json.dumps)
        out["device_fingerprint"] = out["device_fingerprint"].apply(json.dumps)
        out["command_sequence"] = out["command_sequence"].apply(json.dumps)
        out.to_csv(path, index=False)

    @staticmethod
    def to_parquet(df: pd.DataFrame, path: str):
        # Parquet handles nested dict/list columns natively via pyarrow.
        df.to_parquet(path, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic behavioral access logs.")
    p.add_argument("--num-users", type=int, default=200)
    p.add_argument("--num-service-accounts", type=int, default=40)
    p.add_argument("--num-devices", type=int, default=60)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD, defaults to `days` ago from today")
    p.add_argument("--output", type=str, default="data/access_logs.csv", help="Output path (.csv or .parquet)")
    p.add_argument("--profiles-output", type=str, default="data/entity_profiles.csv")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p


def main():
    args = build_arg_parser().parse_args()

    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = datetime.utcnow() - timedelta(days=args.days)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    gen = SyntheticDataGenerator(
        num_users=args.num_users,
        num_service_accounts=args.num_service_accounts,
        num_devices=args.num_devices,
        seed=args.seed,
    )
    df = gen.generate(start_date, args.days)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.output.endswith(".parquet"):
        SyntheticDataGenerator.to_parquet(df, args.output)
    else:
        SyntheticDataGenerator.to_csv(df, args.output)
    print(f"Wrote {len(df)} rows to {args.output}")

    os.makedirs(os.path.dirname(args.profiles_output) or ".", exist_ok=True)
    gen.profiles_dataframe().to_csv(args.profiles_output, index=False)
    print(f"Wrote {len(gen.profiles)} entity profiles to {args.profiles_output}")


if __name__ == "__main__":
    main()
