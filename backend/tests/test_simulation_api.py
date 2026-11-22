import asyncio
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.session import Base, get_db
from app.main import app
from app.models.records import Installation, Observation, SimulationSession, TelemetryReading, VirtualDevice
from app.routers import simulation


def test_controlled_simulation_persists_separate_energy_and_device_telemetry(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with TestSession() as db:
        site = Installation(name="Test array", timezone="UTC", latitude=11, longitude=76, capacity_kw=10, inverter_capacity_kw=9)
        db.add(site); db.flush()
        for suffix, name, kind in [("panel", "Panel array", "panel_array"), ("inv", "Inverter", "inverter"), ("env", "Environment", "environment"), ("gw", "Gateway", "gateway")]:
            db.add(VirtualDevice(installation_id=site.id, device_id=suffix, name=name, device_type=kind, status="online"))
        db.commit()

    def override_db():
        with TestSession() as db:
            yield db

    async def weather(*_args):
        return {"timestamp": "2026-11-17T10:00", "irradiance": 700.0, "temperature": 25.0,
                "humidity": 45, "cloud_cover": 20, "wind_speed": 4, "precipitation": 0,
                "source": "Open-Meteo forecast hourly"}

    monkeypatch.setattr(simulation, "historical_weather", weather)
    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post("/api/simulation/start", json={"scenario": "inverter_failure", "start_time": "2026-11-17T10:00:00Z", "duration_minutes": 60, "interval_minutes": 15})
        assert response.status_code == 200, response.text
        session_id = response.json()["id"]
        assert client.post(f"/api/simulation/{session_id}/step").status_code == 200
        result = client.post(f"/api/simulation/{session_id}/step").json()["generated"][0]
        assert result["actual_kw"] == 0
        assert result["expected_kwh"] > 0
        assert result["poa_irradiance_wm2"] > 0
        assert result["cell_temperature_c"] > 25
        assert result["prediction_method"] == "pvlib_pvwatts"
        assert result["actual_kwh"] == 0
        assert result["status"] == "possible_system_fault"
        with TestSession() as db:
            observation = db.scalar(select(Observation).where(Observation.simulation_session_id == session_id))
            readings = db.scalars(select(TelemetryReading).where(TelemetryReading.session_id == session_id)).all()
            assert observation.source == "simulation"
            assert observation.unit == "kWh"
            assert observation.expected_energy_kwh == observation.expected_power_kw * 0.25
            assert {x.device_id for x in readings} == {"panel", "inv", "env", "gw"}
            assert {x.source for x in readings} == {"weather_api", "simulated"}
            assert all(x.valid for x in readings)
        assert client.post(f"/api/simulation/{session_id}/control/pause").json()["status"] == "paused"
        assert client.post(f"/api/simulation/{session_id}/control/resume").json()["status"] == "running"
        assert client.post(f"/api/simulation/{session_id}/control/stop").json()["status"] == "stopped"
        diagnosis = client.get("/api/diagnosis").json()
        assert diagnosis["status"] == "critical_fault"
        assert diagnosis["evidence"]["latest_prediction_method"] == "pvlib_pvwatts"

        dust = client.post("/api/simulation/start", json={"scenario": "dust", "start_time": "2026-11-17T10:00:00Z", "duration_minutes": 60, "interval_minutes": 15})
        dust_id = dust.json()["id"]
        client.post(f"/api/simulation/{dust_id}/step")
        client.post(f"/api/simulation/{dust_id}/step")
        cleaned = client.post(f"/api/simulation/{dust_id}/clean")
        assert cleaned.status_code == 200
        assert cleaned.json()["source"] == "simulated"
        assert cleaned.json()["soiling_loss_before"] > 0
        assert cleaned.json()["soiling_loss_after"] == 0
        client.post(f"/api/simulation/{dust_id}/step")
        with TestSession() as db:
            last_soiling = db.scalar(select(TelemetryReading).where(TelemetryReading.session_id == dust_id, TelemetryReading.measurement == "soiling_loss").order_by(TelemetryReading.timestamp.desc()))
            assert last_soiling.value == 0
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_gateway_saves_invalid_physical_readings_as_invalid():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with TestSession() as db:
        site = Installation(name="Test array")
        db.add(site); db.flush()
        db.add(VirtualDevice(installation_id=site.id, device_id="gw", name="Gateway", device_type="gateway", status="offline"))
        db.commit()
    def override_db():
        with TestSession() as db:
            yield db
    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post("/api/gateway/ingest", json={"device_id": "gw", "timestamp": "2026-11-17T10:00:00Z", "readings": [{"measurement": "irradiance", "value": -2, "unit": "W/m2"}]})
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"
        with TestSession() as db:
            reading = db.scalar(select(TelemetryReading))
            assert reading.valid is False
            assert reading.value == -2
    finally:
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(engine)
        engine.dispose()
