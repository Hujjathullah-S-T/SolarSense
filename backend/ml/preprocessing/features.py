from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "irradiation_ratio",
    "ambient_temperature",
    "module_temperature",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


def add_time_features(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Apply the canonical cyclical calendar transforms used in fit and predict."""
    data = frame.copy()
    timestamp = pd.to_datetime(data[timestamp_column])
    hour = timestamp.dt.hour + timestamp.dt.minute / 60
    day = timestamp.dt.dayofyear
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    data["day_sin"] = np.sin(2 * np.pi * day / 365.25)
    data["day_cos"] = np.cos(2 * np.pi * day / 365.25)
    return data


def build_feature_frame(irradiance_wm2: float, ambient_temperature: float, module_temperature: float, when: datetime, timezone_name: Optional[str]) -> pd.DataFrame:
    """Construct one inference row with precisely the training feature definitions."""
    try:
        zone = ZoneInfo(timezone_name) if timezone_name else timezone.utc
    except Exception:
        zone = timezone.utc
    local_time = when.astimezone(zone) if when.tzinfo else when.replace(tzinfo=zone)
    timestamp_frame = pd.DataFrame({"timestamp": [local_time.replace(tzinfo=None)]})
    timestamp_frame = add_time_features(timestamp_frame)
    return pd.DataFrame([{
        "irradiation_ratio": max(0.0, float(irradiance_wm2)) / 1000,
        "ambient_temperature": float(ambient_temperature),
        "module_temperature": float(module_temperature),
        "hour_sin": timestamp_frame.loc[0, "hour_sin"],
        "hour_cos": timestamp_frame.loc[0, "hour_cos"],
        "day_sin": timestamp_frame.loc[0, "day_sin"],
        "day_cos": timestamp_frame.loc[0, "day_cos"],
    }])[FEATURE_NAMES]
