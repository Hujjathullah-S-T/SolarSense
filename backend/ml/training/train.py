"""Train and chronologically evaluate solar output models on local plant data."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ml.preprocessing.features import FEATURE_NAMES, add_time_features

PROJECT_ROOT = next((parent for parent in Path(__file__).resolve().parents if (parent / "dataset").is_dir()), Path(__file__).resolve().parents[3])
DATASET_DIR = PROJECT_ROOT / "dataset"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
FEATURES = FEATURE_NAMES
DATASET_FILES = [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv",
]


def load_joined_dataset(dataset_dir: Path = DATASET_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    input_counts = {}
    for plant_number in (1, 2):
        generation_path = dataset_dir / f"Plant_{plant_number}_Generation_Data.csv"
        weather_path = dataset_dir / f"Plant_{plant_number}_Weather_Sensor_Data.csv"
        if not generation_path.is_file() or not weather_path.is_file():
            raise FileNotFoundError(f"Expected both plant {plant_number} generation and weather CSV files under {dataset_dir}")

        generation = pd.read_csv(generation_path)
        weather = pd.read_csv(weather_path)
        generation["timestamp"] = pd.to_datetime(generation["DATE_TIME"], format="mixed", dayfirst=plant_number == 1, errors="coerce")
        weather["timestamp"] = pd.to_datetime(weather["DATE_TIME"], format="mixed", errors="coerce")
        generation = generation.dropna(subset=["timestamp", "PLANT_ID", "AC_POWER"])
        weather = weather.dropna(subset=["timestamp", "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"])

        # Aggregate inverter measurements to one measured AC power label per plant interval.
        measured = generation.groupby(["PLANT_ID", "timestamp"], as_index=False)["AC_POWER"].sum()
        measured = measured.rename(columns={"AC_POWER": "actual_power_kw"})
        conditions = weather.groupby(["PLANT_ID", "timestamp"], as_index=False).agg(
            ambient_temperature=("AMBIENT_TEMPERATURE", "mean"),
            module_temperature=("MODULE_TEMPERATURE", "mean"),
            irradiation_ratio=("IRRADIATION", "mean"),
        )
        joined = measured.merge(conditions, on=["PLANT_ID", "timestamp"], how="inner", validate="one_to_one")
        frames.append(joined)
        input_counts[f"plant_{plant_number}"] = {
            "generation_rows": int(len(generation)),
            "weather_rows": int(len(weather)),
            "matched_intervals": int(len(joined)),
        }

    data = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_power_kw"] + FEATURES[:3])
    data = data[(data["actual_power_kw"] >= 0) & (data["irradiation_ratio"] >= 0)]
    data = add_time_features(data)

    input_counts["raw_generation_rows"] = sum(x["generation_rows"] for x in input_counts.values())
    input_counts["matched_intervals"] = int(len(data))
    input_counts["missing_feature_rows_rejected"] = int(data[FEATURES + ["actual_power_kw"]].isna().any(axis=1).sum())
    return data.reset_index(drop=True), input_counts


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    predicted = np.clip(predicted, 0, 1.25)
    error = actual - predicted
    centered = actual - actual.mean()
    return {
        "mae_normalized": float(np.mean(np.abs(error))),
        "rmse_normalized": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(1 - np.sum(np.square(error)) / np.sum(np.square(centered))) if np.sum(np.square(centered)) else 0.0,
    }


def train_models(dataset_dir: Path = DATASET_DIR, artifact_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    try:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise RuntimeError("Optional ML training packages are missing. Install backend/requirements-ml.txt to train a model.") from exc
    try:
        from xgboost import XGBRegressor
        xgboost_import_error = None
    except Exception as exc:  # Optional system OpenMP runtime is not present on all developer machines.
        XGBRegressor = None
        xgboost_import_error = next((line.strip() for line in str(exc).splitlines() if line.strip()), str(exc))
    data, counts = load_joined_dataset(dataset_dir)
    timestamps = np.array(sorted(data["timestamp"].unique()))
    if len(timestamps) < 60:
        raise ValueError(f"Only {len(timestamps)} unique time intervals were found; at least 60 are required.")

    train_cut = timestamps[int(len(timestamps) * 0.70)]
    validation_cut = timestamps[int(len(timestamps) * 0.85)]
    train_mask = data["timestamp"] < train_cut
    validation_mask = (data["timestamp"] >= train_cut) & (data["timestamp"] < validation_cut)
    test_mask = data["timestamp"] >= validation_cut
    if min(int(train_mask.sum()), int(validation_mask.sum()), int(test_mask.sum())) == 0:
        raise ValueError("Chronological train, validation, and test partitions must each contain records.")

    # Normalize each plant by its training-only observed peak. This avoids using test-period
    # production to establish the label scale and permits scaling the output to the configured site.
    train_peaks = data.loc[train_mask].groupby("PLANT_ID")["actual_power_kw"].max().to_dict()
    data["normalized_output"] = data.apply(
        lambda row: row["actual_power_kw"] / train_peaks[row["PLANT_ID"]]
        if row["PLANT_ID"] in train_peaks and train_peaks[row["PLANT_ID"]] > 0
        else np.nan,
        axis=1,
    )
    data = data.dropna(subset=["normalized_output"])
    train_mask = data["timestamp"] < train_cut
    validation_mask = (data["timestamp"] >= train_cut) & (data["timestamp"] < validation_cut)
    test_mask = data["timestamp"] >= validation_cut

    x_train, y_train = data.loc[train_mask, FEATURES], data.loc[train_mask, "normalized_output"]
    x_validation, y_validation = data.loc[validation_mask, FEATURES], data.loc[validation_mask, "normalized_output"]
    x_test, y_test = data.loc[test_mask, FEATURES], data.loc[test_mask, "normalized_output"]

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42),
    }
    if XGBRegressor is not None:
        candidates["xgboost"] = XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror", n_jobs=4, random_state=42)
    validation_metrics = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        fitted[name] = model
        validation_metrics[name] = _metrics(y_validation.to_numpy(), model.predict(x_validation))

    selected_name = min(validation_metrics, key=lambda name: validation_metrics[name]["mae_normalized"])
    selected_model = fitted[selected_name]
    test_metrics = _metrics(y_test.to_numpy(), selected_model.predict(x_test))
    metadata = {
        "status": "trained",
        "model_version": f"{selected_name}-{uuid.uuid4().hex[:12]}",
        "feature_engineering_version": "solar-features-v1",
        "selected_model": selected_name,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "record_counts": {
            **counts,
            "train": int(len(x_train)),
            "validation": int(len(x_validation)),
            "test": int(len(x_test)),
        },
        "date_range": {"start": data["timestamp"].min().isoformat(), "end": data["timestamp"].max().isoformat()},
        "features": FEATURES,
        "target": "Measured plant AC output normalized by that plant's training-period peak; simulated output is not used.",
        "target_unit": "kW",
        "target_definition": "Sum of inverter-level AC_POWER (kW) by plant and timestamp; source readings are sampled at 15-minute intervals.",
        "target_normalization": "actual_power_kw / that plant's training-period maximum actual_power_kw; retained for optional training-artifact evaluation only.",
        "inference_pipeline_version": "not-used-for-runtime-prediction",
        "runtime_prediction_engine": "pvlib-pvwatts-v1",
        "normalization_peaks_kw": {str(key): float(value) for key, value in train_peaks.items()},
        "dataset_files": DATASET_FILES,
        "split": "Chronological 70% train / 15% validation / 15% test by interval timestamp.",
        "skipped_models": {"xgboost": xgboost_import_error} if xgboost_import_error else {},
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_tmp = artifact_dir / "solar_output.joblib.tmp"
    metadata_tmp = artifact_dir / "solar_output.json.tmp"
    joblib.dump({"model": selected_model, "features": FEATURES, "metadata": metadata}, artifact_tmp)
    metadata_tmp.write_text(json.dumps(metadata, indent=2))
    os.replace(artifact_tmp, artifact_dir / "solar_output.joblib")
    os.replace(metadata_tmp, artifact_dir / "solar_output.json")
    metadata["artifact_sha256"] = __import__("hashlib").sha256((artifact_dir / "solar_output.joblib").read_bytes()).hexdigest()
    (artifact_dir / "solar_output.json").write_text(json.dumps(metadata, indent=2))
    return metadata
