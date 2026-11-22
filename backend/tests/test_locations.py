import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.database.session import Base, get_db
from app.models.records import Installation, Observation
from app.services.prediction.engine import PowerPrediction


LOCATIONS = [
    {"id": 1, "city": "Coimbatore", "administrative_region": "Tamil Nadu", "country": "India", "latitude": 11.0168, "longitude": 76.9558, "timezone": "Asia/Kolkata", "population": 1000000},
    {"id": 2, "city": "Chennai", "administrative_region": "Tamil Nadu", "country": "India", "latitude": 13.0878, "longitude": 80.2785, "timezone": "Asia/Kolkata", "population": 7000000},
    {"id": 3, "city": "Springfield", "administrative_region": "Illinois", "country": "United States", "latitude": 39.8, "longitude": -89.6, "timezone": "America/Chicago", "population": 100000},
    {"id": 4, "city": "Springfield", "administrative_region": "Oregon", "country": "United States", "latitude": 44.0, "longitude": -123.0, "timezone": "America/Los_Angeles", "population": 60000},
]


@pytest.fixture
def client(monkeypatch, tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine, expire_on_commit=False)
    with TestingSession() as db:
        db.add(Installation(name="Test array", latitude=11.0, longitude=76.0, timezone="UTC", capacity_kw=5))
        db.commit()

    def override_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(main, "search_locations", lambda q: asyncio.sleep(0, result=[x for x in LOCATIONS if q.lower() in x["city"].lower()]))
    monkeypatch.setattr(main, "resolve_location", lambda location_id, city_hint="": asyncio.sleep(0, result=next((x for x in LOCATIONS if x["id"] == location_id and x["city"].lower() == city_hint.lower()), None)))
    main.app.dependency_overrides[get_db] = override_db
    api_client = TestClient(main.app)
    api_client.testing_session = TestingSession
    yield api_client
    main.app.dependency_overrides.clear()
    engine.dispose()


def test_city_search_and_duplicate_results(client):
    coimbatore = client.get("/api/locations/search", params={"q": "Coimbatore"})
    assert coimbatore.status_code == 200
    assert coimbatore.json()["results"][0]["administrative_region"] == "Tamil Nadu"
    assert client.get("/api/locations/search", params={"q": "Chennai"}).json()["results"][0]["city"] == "Chennai"
    assert len(client.get("/api/locations/search", params={"q": "Springfield"}).json()["results"]) == 2


def test_invalid_query_and_geocoding_failure(client, monkeypatch):
    assert client.get("/api/locations/search", params={"q": "zzznocity"}).json() == {"results": []}

    async def unavailable(_query):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(main, "search_locations", unavailable)
    assert client.get("/api/locations/search", params={"q": "Chennai"}).status_code == 502


def test_selection_persists_and_weather_uses_resolved_coordinates(client, monkeypatch):
    original = client.get("/api/installation").json()
    saved_config = client.put("/api/installation", json={"name": "Renamed array"})
    assert saved_config.json()["latitude"] == original["latitude"]
    assert saved_config.json()["longitude"] == original["longitude"]
    found = client.get("/api/locations/search", params={"q": "Coimbatore"}).json()["results"][0]
    saved = client.put("/api/installation", json={"location_id": found["id"], "city": found["city"]})
    assert saved.status_code == 200
    assert saved.json()["latitude"] == found["latitude"]
    assert saved.json()["longitude"] == found["longitude"]
    assert saved.json()["timezone"] == "Asia/Kolkata"
    refreshed = client.get("/api/installation").json()
    assert refreshed["city"] == "Coimbatore"
    assert refreshed["administrative_region"] == "Tamil Nadu"

    seen = {}

    async def fake_forecast(lat, lon):
        seen["coords"] = (lat, lon)
        return {"timezone": "Asia/Kolkata", "current": {"temperature_2m": 30, "shortwave_radiation": 500, "is_day": 1}}

    monkeypatch.setattr(main, "forecast", fake_forecast)
    overview = client.get("/api/overview").json()
    assert overview["installation"]["location_label"] == "Coimbatore, Tamil Nadu, India"
    assert seen["coords"] == (found["latitude"], found["longitude"])


def test_manual_energy_entry_is_saved_as_measured_kwh(client, monkeypatch):
    async def fake_weather(_lat, _lon, _timestamp, _timezone):
        return {"timestamp": "2026-11-15T10:00", "irradiance": 600, "temperature": 25, "timezone": "UTC"}

    monkeypatch.setattr(main, "predict_installation_power", lambda *args, **kwargs: PowerPrediction(3.25, "pvlib_pvwatts", None))
    monkeypatch.setattr(main, "historical_weather", fake_weather)
    response = client.post("/api/production/manual", json={
        "timestamp": "2026-11-15T10:30:00+05:30",
        "actual_kwh": 1.425,
        "interval_minutes": 30,
    })
    assert response.status_code == 200
    assert response.json()["actual_kwh"] == 1.425
    assert response.json()["expected_kwh"] == 1.625
    assert response.json()["interval_minutes"] == 30
    assert response.json()["prediction_method"] == "pvlib_pvwatts"
    assert response.json()["model_version"] is None
    assert response.json()["weather_timestamp"] == "2026-11-15T10:00"
    assert response.json()["status"] == "normal"
    assert response.json()["unit"] == "kWh"
    assert response.json()["source"] == "manual"
    row = client.get("/api/observations").json()[0]
    assert row["source"] == "manual"
    assert row["unit"] == "kWh"
    assert row["interval_minutes"] == 30
    assert row["weather_source"] == "Open-Meteo hourly historical/forecast"


def test_weather_failure_keeps_manual_reading_retryable(client, monkeypatch):
    async def unavailable(*_args):
        raise RuntimeError("historical weather unavailable")

    monkeypatch.setattr(main, "historical_weather", unavailable)
    pending = client.post("/api/production/manual", json={
        "timestamp": "2026-11-15T10:30:00Z",
        "actual_kwh": 1.0,
        "interval_minutes": 60,
    }).json()
    assert pending["status"] == "pending"

    async def restored(*_args):
        return {"timestamp": "2026-11-15T10:00", "irradiance": 500, "temperature": 25, "timezone": "UTC"}

    monkeypatch.setattr(main, "historical_weather", restored)
    monkeypatch.setattr(main, "predict_installation_power", lambda *args, **kwargs: PowerPrediction(2.0, "pvlib_pvwatts", None))
    analyzed = client.post(f"/api/observations/{pending['id']}/analyze")
    assert analyzed.status_code == 200
    assert analyzed.json()["expected_kwh"] > 0
    assert analyzed.json()["status"] != "pending"


def test_virtual_inverter_uses_pvlib_prediction_without_a_trained_model(client, monkeypatch):
    async def fake_forecast(_lat, _lon):
        return {"current": {"time": "2026-11-15T10:00", "shortwave_radiation": 600, "temperature_2m": 25, "is_day": 1}}

    monkeypatch.setattr(main, "forecast", fake_forecast)
    response = client.post("/api/simulator/run?dust=0.2")
    assert response.status_code == 200
    assert response.json()["expected_kw"] > 0
    assert response.json()["prediction_method"] == "pvlib_pvwatts"
    assert response.json()["model_version"] is None
    assert response.json()["weather_timestamp"].startswith("2026-11-15T10:00")
    assert response.json()["unit"] == "kWh"
    assert response.json()["expected_energy_kwh"] == response.json()["expected_kw"]
    assert response.json()["simulated_actual_energy_kwh"] < response.json()["expected_energy_kwh"]
    duplicate = client.post("/api/simulator/run?dust=0.2")
    assert duplicate.json()["duplicate_run"] is True


def test_manual_energy_entry_rejects_negative_values(client):
    response = client.post("/api/production/manual", json={
        "timestamp": "2026-11-15T10:30:00Z",
        "actual_kwh": -0.25,
    })
    assert response.status_code == 422


def test_observations_are_newest_first_and_reset_clears_diagnosis(client):
    from datetime import datetime

    with client.testing_session() as db:
        db.add_all([
            Observation(timestamp=datetime(2026, 11, 14, 10), expected_kwh=1, actual_kwh=.8, unit="kWh", source="manual", status="pending"),
            Observation(timestamp=datetime(2026, 11, 15, 10), expected_kwh=1, actual_kwh=.9, unit="kWh", source="csv", status="pending"),
        ])
        db.commit()
    rows = client.get("/api/observations").json()
    assert rows[0]["timestamp"].startswith("2026-11-15")
    reset = client.delete("/api/observations")
    assert reset.status_code == 200
    assert reset.json()["deleted"] == 2
    assert client.get("/api/observations").json() == []
    assert client.get("/api/diagnosis").json()["status"] == "insufficient_data"


def test_old_simulation_provenance_and_estimates_are_preserved(client, monkeypatch):
    from datetime import datetime

    with client.testing_session() as db:
        old = Observation(timestamp=datetime(2026, 11, 14, 10), expected_kwh=14.5, actual_kwh=10.25,
                          unit="kW", source="simulation", status="watch",
                          prediction_method="physics_fallback", weather_source="Open-Meteo current")
        db.add(old)
        db.commit()
        old_id = old.id

    async def fake_forecast(_lat, _lon):
        return {"current": {"time": "2026-11-15T10:00", "shortwave_radiation": 600,
                            "temperature_2m": 25, "is_day": 1}}

    monkeypatch.setattr(main, "forecast", fake_forecast)
    monkeypatch.setattr(main, "predict_installation_power", lambda *args, **kwargs: PowerPrediction(3.25, "trained_ml", "new-model-v2"))
    client.post("/api/simulator/run")
    rows = {r["id"]: r for r in client.get("/api/observations").json()}
    legacy = rows[old_id]
    assert legacy["expected_kwh"] == 14.5
    assert legacy["actual_kwh"] == 10.25
    assert legacy["unit"] == "kW"
    assert legacy["prediction_method"] == "physics_fallback"
    assert legacy["model_version"] is None


def test_reset_history_does_not_change_model_metadata(client, monkeypatch):
    from datetime import datetime

    model_info = {"status": "trained", "model_version": "persistent-artifact-v9", "metrics": {"mae": 0.12}}
    monkeypatch.setattr(main, "model_status", lambda: model_info)
    with client.testing_session() as db:
        db.add(Observation(timestamp=datetime(2026, 11, 15), expected_kwh=1, actual_kwh=.9,
                           unit="kWh", source="manual", status="normal"))
        db.commit()
    assert client.get("/api/models/status").json() == model_info
    assert client.delete("/api/observations").status_code == 200
    assert client.get("/api/models/status").json() == model_info


def test_csv_naive_timestamps_use_site_timezone_and_interval_cadence(client):
    response = client.post("/api/production/csv", files={
        "file": ("readings.csv", b"timestamp,actual_kwh\n2026-11-15 10:00,1.2\n2026-11-15 10:30,0.8\n", "text/csv")
    })
    assert response.status_code == 200
    rows = client.get("/api/observations").json()
    assert all(row["interval_minutes"] == 30 for row in rows)
    # The test installation has no named timezone, so UTC is the defined fallback.
    assert any(row["timestamp"].startswith("2026-11-15T10:00") for row in rows)
    assert any(row["actual_energy_kwh"] == 1.2 for row in rows)


def test_manual_and_simulation_use_the_same_physics_engine(client, monkeypatch):
    async def fake_weather(_lat, _lon, _timestamp, _timezone):
        return {"timestamp": "2026-11-15T10:00", "irradiance": 600, "temperature": 25,
                "timezone": "UTC", "source": "Open-Meteo archive hourly"}

    async def fake_forecast(_lat, _lon):
        return {"current": {"time": "2026-11-15T10:00", "shortwave_radiation": 600,
                            "temperature_2m": 25, "is_day": 1}}

    monkeypatch.setattr(main, "historical_weather", fake_weather)
    monkeypatch.setattr(main, "forecast", fake_forecast)
    manual = client.post("/api/production/manual", json={
        "timestamp": "2026-11-15T10:00:00Z", "actual_kwh": 3.25, "interval_minutes": 60,
    }).json()
    simulation = client.post("/api/simulator/run").json()
    assert manual["prediction_method"] == simulation["prediction_method"] == "pvlib_pvwatts"
    assert manual["model_version"] is simulation["model_version"] is None
    assert manual["unit"] == "kWh" and manual["expected_kwh"] > 0
    assert manual["actual_power_kw"] == 3.25
    assert simulation["unit"] == "kWh" and simulation["expected_kw"] > 0
    assert simulation["expected_energy_kwh"] == simulation["expected_kw"]
    assert simulation["simulated_actual_energy_kwh"] == simulation["expected_energy_kwh"]
    assert simulation["duplicate_run"] is False


def test_csv_power_is_converted_once_and_kaggle_rows_are_aggregated(client):
    direct = client.post("/api/production/csv", files={
        "file": ("power.csv", b"timestamp,interval_average_power_kw,interval_minutes\n2026-11-15T10:00:00Z,4,15\n", "text/csv")
    })
    assert direct.status_code == 200
    row = client.get("/api/observations").json()[0]
    assert row["raw_measurement_value"] == 4
    assert row["raw_measurement_unit"] == "kW"
    assert row["actual_power_kw"] == 4
    assert row["actual_energy_kwh"] == 1
    assert row["measurement_semantics"] == "interval_average_power"

    kaggle = client.post("/api/production/csv", files={
        "file": ("Plant_1_Generation_Data.csv", b"DATE_TIME,PLANT_ID,SOURCE_KEY,AC_POWER,interval_minutes\n15-05-2020 11:00:00,7,a,1.5,15\n15-05-2020 11:00:00,7,b,2.5,15\n", "text/csv")
    })
    assert kaggle.status_code == 200
    all_rows = client.get("/api/observations").json()
    grouped = next(r for r in all_rows if r["timestamp"].startswith("2020-05-15T11:00"))
    assert grouped["raw_measurement_value"] == 4
    assert '"source_key": "a"' in grouped["raw_measurement_components"]
    assert '"source_key": "b"' in grouped["raw_measurement_components"]
    assert grouped["actual_power_kw"] == 4
    assert grouped["actual_energy_kwh"] is None
    assert grouped["unit"] == "kW"
    assert grouped["measurement_semantics"] == "instantaneous_power_sample"
    assert grouped["requires_interval_review"] is True


def test_simulator_timestamp_uses_requested_weather_time_and_duplicates_are_idempotent(client, monkeypatch):
    async def fake_weather(_lat, _lon, timestamp, zone):
        assert timestamp.hour == 10
        assert zone == "UTC"
        return {"timestamp": "2026-11-15T10:00", "irradiance": 400, "temperature": 25,
                "source": "Open-Meteo archive hourly"}

    monkeypatch.setattr(main, "historical_weather", fake_weather)
    params = {"timestamp": "2026-11-15T10:00:00Z", "interval_minutes": 15, "dust": .25}
    first = client.post("/api/simulator/run", params=params)
    second = client.post("/api/simulator/run", params=params)
    assert first.status_code == second.status_code == 200
    assert first.json()["weather_timestamp"] == "2026-11-15T10:00:00"
    assert first.json()["expected_energy_kwh"] == pytest.approx(first.json()["expected_kw"] * .25, abs=.001)
    assert first.json()["simulated_actual_energy_kwh"] < first.json()["expected_energy_kwh"]
    assert first.json()["expected_kw"] > 0
    assert first.json()["simulated_actual_kw"] < first.json()["expected_kw"]
    assert first.json()["duplicate_run"] is False
    assert second.json()["duplicate_run"] is True
    records = [r for r in client.get("/api/observations").json() if r["source"] == "simulation"]
    assert len(records) == 1


def test_installation_capacity_validation(client):
    assert client.put("/api/installation", json={"capacity_kw": 0}).status_code == 422
    assert client.put("/api/installation", json={"capacity_kw": 1_000_001}).status_code == 422
    assert client.put("/api/installation", json={"tilt": 91}).status_code == 422
    assert client.put("/api/installation", json={"azimuth": 361}).status_code == 422


def test_instantaneous_power_sample_is_not_converted_to_energy_on_analysis(client, monkeypatch):
    uploaded = client.post("/api/production/csv", files={
        "file": ("sample.csv", b"timestamp,power_kw,interval_minutes\n2026-11-15T10:00:00Z,4,15\n", "text/csv")
    }).json()
    row_id = client.get("/api/observations").json()[0]["id"]
    async def fake_weather(*_args):
        return {"timestamp":"2026-11-15T10:00","irradiance":500,"temperature":25,"source":"Open-Meteo archive hourly"}
    monkeypatch.setattr(main,"historical_weather",fake_weather)
    monkeypatch.setattr(main,"predict_installation_power",lambda *args,**kwargs: PowerPrediction(3,"pvlib_pvwatts",None))
    analyzed=client.post(f"/api/observations/{row_id}/analyze")
    assert analyzed.status_code==200
    assert analyzed.json()["unit"]=="kW"
    assert analyzed.json()["expected_power_kw"]==3
    assert analyzed.json()["actual_power_kw"]==4
    assert analyzed.json()["expected_energy_kwh"] is None
    assert analyzed.json()["actual_energy_kwh"] is None
