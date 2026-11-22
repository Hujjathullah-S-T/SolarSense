from datetime import datetime, timezone
import json

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
import pytest

from app import main
from app.models.records import Installation
from app.services.prediction.engine import predict_installation_power
from app.services.prediction import trained_model
from app.services.production.units import energy_to_average_power_kw, power_to_energy_kwh
from ml.training.train import load_joined_dataset


def test_kaggle_target_is_aggregated_inverter_ac_power_kw():
    data, counts = load_joined_dataset()
    assert {"AC_POWER", "SOURCE_KEY", "PLANT_ID"}.issubset(
        set(pd.read_csv(Path(__file__).resolve().parents[2] / "dataset/Plant_1_Generation_Data.csv", nrows=1).columns)
    )
    raw = pd.read_csv(Path(__file__).resolve().parents[2] / "dataset/Plant_1_Generation_Data.csv")
    timestamp = data.iloc[0]["timestamp"]
    plant = int(data.iloc[0]["PLANT_ID"])
    grouped = raw[(raw.PLANT_ID == plant) & (pd.to_datetime(raw.DATE_TIME, dayfirst=True) == timestamp)].AC_POWER.sum()
    assert data.iloc[0]["actual_power_kw"] == pytest.approx(grouped)
    assert counts["matched_intervals"] > 0
    assert 0 <= data["actual_power_kw"].max() < 50_000


def test_power_energy_conversions_respect_interval_and_do_not_double_convert():
    assert power_to_energy_kwh(4, 15) == 1
    assert energy_to_average_power_kw(1, 15) == 4


def test_physics_prediction_is_primary_and_respects_installation_dc_capacity():
    site = Installation(capacity_kw=10, inverter_efficiency=.9, losses=.1, timezone="UTC", latitude=11, longitude=76)
    result = predict_installation_power(site, 800, 25, datetime(2026, 11, 17, 10, tzinfo=timezone.utc), 700, 100, 5)
    assert result.expected_power_kw > 0
    assert result.method == "pvlib_pvwatts"
    assert result.model_version is None
    assert 0 <= result.dc_power_kw <= site.capacity_kw

    site.capacity_kw = 0
    with pytest.raises(ValueError, match="capacity"):
        predict_installation_power(site, 800, 25, datetime(2026, 11, 17, 10, tzinfo=timezone.utc), 700, 100, 5)


def test_prediction_respects_configured_inverter_ac_capacity():
    site = Installation(capacity_kw=10, inverter_capacity_kw=4, inverter_efficiency=.98, losses=0, timezone="UTC")
    prediction = predict_installation_power(site, 1000, 25, datetime(2026, 11, 17, 10, tzinfo=timezone.utc), 900, 100, 3)
    assert prediction.expected_power_kw <= 4


def test_prediction_requires_real_ghi_and_does_not_substitute_zero():
    site = Installation(capacity_kw=10, timezone="UTC")
    with pytest.raises(ValueError, match="GHI is missing"):
        predict_installation_power(site, None, 25, datetime(2026, 11, 17, 10, tzinfo=timezone.utc))


def test_runtime_prediction_does_not_require_ml_artifact_or_joblib_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(trained_model, "MODEL_PATH", tmp_path / "absent.joblib")
    monkeypatch.setattr(trained_model, "METADATA_PATH", tmp_path / "absent.json")
    site = Installation(capacity_kw=6, timezone="UTC", latitude=11, longitude=76)
    when = datetime(2026, 11, 17, 10, tzinfo=timezone.utc)
    prediction = predict_installation_power(site, 700, 25, when, 600, 100, 5)
    assert prediction.method == "pvlib_pvwatts"
    assert prediction.expected_power_kw > 0
    assert trained_model.model_status()["status"] == "pvlib_primary"

    (tmp_path / "metadata.json").write_text(json.dumps({"model_version": "optional-test", "selected_model": "random_forest"}))
    monkeypatch.setattr(trained_model, "METADATA_PATH", tmp_path / "metadata.json")
    status = trained_model.model_status()
    assert status["runtime"] is False
    assert status["artifact_sha256"] is None


def test_legacy_rows_keep_original_values_and_get_only_defensible_canonical_fields():
    engine = create_engine("sqlite://")
    with engine.begin() as db:
        db.execute(text("CREATE TABLE observations (id INTEGER PRIMARY KEY, timestamp DATETIME, expected_kwh FLOAT, actual_kwh FLOAT, unit VARCHAR(8), interval_minutes INTEGER, source VARCHAR(30), status VARCHAR(30))"))
        db.execute(text("INSERT INTO observations VALUES (1,'2026-11-15 10:00',14.5,10.25,'kW',NULL,'simulation','watch')"))
        db.execute(text("INSERT INTO observations VALUES (2,'2026-11-15 11:00',2.0,1.5,'kWh',30,'manual','normal')"))
        columns = {row["name"] for row in main.inspect(db).get_columns("observations")}
        main.add_observation_columns(db, columns)
        power_row = db.execute(text("SELECT expected_kwh,actual_kwh,expected_power_kw,actual_power_kw,expected_energy_kwh,requires_interval_review,raw_measurement_unit FROM observations WHERE id=1")).one()
        assert tuple(power_row) == (14.5,10.25,14.5,10.25,None,1,"kW")
        energy_row = db.execute(text("SELECT expected_kwh,actual_kwh,expected_energy_kwh,actual_energy_kwh,expected_power_kw,actual_power_kw,requires_interval_review FROM observations WHERE id=2")).one()
        assert tuple(energy_row) == (2.0,1.5,2.0,1.5,4.0,3.0,0)
