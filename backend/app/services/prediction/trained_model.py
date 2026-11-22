"""Optional ML training-artifact metadata; never used for runtime predictions."""
import hashlib
import json
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "ml" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "solar_output.joblib"
METADATA_PATH = ARTIFACT_DIR / "solar_output.json"


def model_status():
    if not METADATA_PATH.is_file():
        return {"status": "pvlib_primary", "trained": False,
                "message": "pvlib PVWatts is active. No optional ML training metadata is available."}
    try:
        metadata = json.loads(METADATA_PATH.read_text())
        return {
            "status": "trained_artifact_available",
            "trained": True,
            "runtime": False,
            "model_version": metadata.get("model_version"),
            "feature_engineering_version": metadata.get("feature_engineering_version"),
            "inference_pipeline_version": "not-used-for-runtime-prediction",
            "target_unit": metadata.get("target_unit", "kW"),
            "artifact_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() if MODEL_PATH.is_file() else None,
            "selected_model": metadata.get("selected_model"),
            "validation_metrics": metadata.get("validation_metrics"),
            "test_metrics": metadata.get("test_metrics"),
            "record_counts": metadata.get("record_counts"),
            "date_range": metadata.get("date_range"),
            "trained_at_utc": metadata.get("trained_at_utc"),
            "dataset_files": metadata.get("dataset_files"),
            "target": metadata.get("target"),
            "split": metadata.get("split"),
            "skipped_models": metadata.get("skipped_models", {}),
            "message": "Optional ML evaluation artifact is present. Runtime prediction remains pvlib PVWatts.",
        }
    except Exception as exc:
        return {"status": "pvlib_primary", "trained": False,
                "message": f"pvlib PVWatts remains active; optional ML metadata could not be read: {exc}"}


def model_version():
    try:
        metadata = json.loads(METADATA_PATH.read_text())
        return metadata.get("model_version")
    except Exception:
        return None
