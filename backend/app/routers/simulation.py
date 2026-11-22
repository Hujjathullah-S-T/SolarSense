from datetime import datetime, timezone, timedelta
import json
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal
from sqlalchemy import select, delete
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.records import Installation, Observation, SimulationSession, TelemetryReading, VirtualDevice, CleaningEvent
from app.services.monitoring.logic import classify
from app.services.prediction.engine import predict_installation_power
from app.services.solar.physics import dc_to_ac
from app.services.production.units import power_to_energy_kwh
from app.services.simulation.engine import SCENARIOS, scenario_losses, telemetry_values
from app.services.weather.client import historical_weather

router = APIRouter(prefix="/api", tags=["virtual simulation"])

class SimulationStartInput(BaseModel):
    model_config = {"extra": "forbid"}
    scenario: str = "normal"
    start_time: Optional[datetime] = None
    duration_minutes: int = Field(default=240, ge=1, le=10080)
    interval_minutes: int = Field(default=15, ge=1, le=1440)
    speed: int = Field(default=1, ge=1, le=60)

class GatewayReading(BaseModel):
    measurement: str = Field(min_length=1, max_length=64)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=16)

class GatewayPayload(BaseModel):
    device_id: str
    timestamp: datetime
    source: Literal["simulated", "weather_api"] = "simulated"
    readings: list[GatewayReading] = Field(min_length=1, max_length=100)

def _installation(db: Session) -> Installation:
    site = db.scalar(select(Installation).order_by(Installation.id))
    if not site:
        raise HTTPException(404, "Configure an installation first")
    return site

def _view(row: SimulationSession):
    return {"id": row.id, "installation_id": row.installation_id, "status": row.status, "scenario": row.scenario, "start_time": row.start_time.isoformat(), "cursor_time": row.cursor_time.isoformat(), "end_time": row.end_time.isoformat() if row.end_time else None, "duration_minutes": row.duration_minutes, "interval_minutes": row.interval_minutes, "speed": row.speed, "configuration": json.loads(row.config_json or "{}")}

@router.get("/simulation/state")
def simulation_state(db: Session = Depends(get_db)):
    site = _installation(db)
    session = db.scalar(select(SimulationSession).where(SimulationSession.installation_id == site.id).order_by(SimulationSession.id.desc()))
    devices = db.scalars(select(VirtualDevice).where(VirtualDevice.installation_id == site.id).order_by(VirtualDevice.id)).all()
    telemetry = db.scalars(select(TelemetryReading).where(TelemetryReading.installation_id == site.id).order_by(TelemetryReading.timestamp.desc(), TelemetryReading.id.desc()).limit(200)).all()
    return {"session": _view(session) if session else None,
            "devices": [{"device_id": d.device_id, "name": d.name, "type": d.device_type, "status": d.status, "last_seen": d.last_seen.isoformat() if d.last_seen else None} for d in devices],
            "telemetry": [{"id": t.id, "device_id": t.device_id, "session_id": t.session_id, "timestamp": t.timestamp.isoformat(), "source": t.source, "measurement": t.measurement, "value": t.value, "unit": t.unit, "valid": t.valid} for t in reversed(telemetry)]}

@router.post("/simulation/start")
def simulation_start(payload: SimulationStartInput, db: Session = Depends(get_db)):
    site = _installation(db)
    if payload.scenario not in SCENARIOS:
        raise HTTPException(422, f"Scenario must be one of: {', '.join(sorted(SCENARIOS))}.")
    try: zone = ZoneInfo(site.timezone) if site.timezone else timezone.utc
    except Exception: zone = timezone.utc
    requested = payload.start_time or datetime.now(zone)
    local = requested.astimezone(zone) if requested.tzinfo else requested.replace(tzinfo=zone)
    start_utc = local.astimezone(timezone.utc).replace(tzinfo=None)
    active = db.scalar(select(SimulationSession).where(SimulationSession.installation_id == site.id, SimulationSession.status.in_(["running", "paused"])).order_by(SimulationSession.id.desc()))
    if active:
        active.status = "stopped"; active.end_time = datetime.utcnow()
    config = {"capacity_kwp": site.capacity_kw, "panel_count": site.panel_count, "panel_efficiency": site.panel_efficiency, "tilt_degrees": site.tilt, "azimuth_degrees": site.azimuth, "inverter_capacity_kw": site.inverter_capacity_kw or site.capacity_kw, "weather_source": "Open-Meteo", "simulation_kind": "historical" if payload.start_time else "live_style"}
    row = SimulationSession(installation_id=site.id, status="running", scenario=payload.scenario, start_time=start_utc, cursor_time=start_utc, duration_minutes=payload.duration_minutes, interval_minutes=payload.interval_minutes, speed=payload.speed, config_json=json.dumps(config))
    db.add(row); db.commit(); db.refresh(row)
    return _view(row)

@router.post("/simulation/{session_id}/control/{action}")
def simulation_control(session_id: int, action: str, db: Session = Depends(get_db)):
    row = db.get(SimulationSession, session_id)
    if not row: raise HTTPException(404, "Simulation session not found")
    if action == "pause" and row.status == "running": row.status = "paused"
    elif action == "resume" and row.status == "paused": row.status = "running"
    elif action == "stop" and row.status in {"running", "paused"}: row.status = "stopped"; row.end_time = datetime.utcnow()
    elif action == "reset":
        db.execute(delete(TelemetryReading).where(TelemetryReading.session_id == row.id))
        db.execute(delete(Observation).where(Observation.simulation_session_id == row.id))
        row.status = "stopped"; row.cursor_time = row.start_time; row.end_time = None
    else: raise HTTPException(409, f"Cannot {action} a {row.status} session.")
    db.commit(); db.refresh(row)
    return _view(row)

@router.post("/simulation/{session_id}/step")
async def simulation_step(session_id: int, count: int = Query(1, ge=1, le=24), db: Session = Depends(get_db)):
    row = db.get(SimulationSession, session_id)
    if not row: raise HTTPException(404, "Simulation session not found")
    if row.status != "running": raise HTTPException(409, "Simulation must be running to generate telemetry.")
    site = db.get(Installation, row.installation_id)
    devices = {d.device_type: d for d in db.scalars(select(VirtualDevice).where(VirtualDevice.installation_id == site.id))}
    generated = []
    try: zone = ZoneInfo(site.timezone) if site.timezone else timezone.utc
    except Exception: zone = timezone.utc
    for _ in range(count):
        elapsed = (row.cursor_time - row.start_time).total_seconds() / 60
        if elapsed >= row.duration_minutes:
            row.status = "completed"; row.end_time = datetime.utcnow(); break
        local = row.cursor_time.replace(tzinfo=timezone.utc).astimezone(zone)
        try: weather = await historical_weather(site.latitude, site.longitude, local, site.timezone or "UTC")
        except Exception as exc: raise HTTPException(502, f"Weather unavailable for simulation timestamp {local.isoformat()}: {exc}")
        irradiance, ambient = weather["irradiance"], weather["temperature"]
        try:
            prediction = predict_installation_power(site, weather.get("ghi", irradiance), ambient, when=local, dni_wm2=weather.get("dni"), dhi_wm2=weather.get("dhi"), wind_speed_kmh=weather.get("wind_speed"))
        except (ValueError, TypeError) as exc:
            raise HTTPException(502, f"Physics prediction unavailable for aligned weather: {exc}")
        config = json.loads(row.config_json or "{}")
        cleaned_at = float(config.get("cleaned_at_elapsed_minutes", 0))
        scenario_elapsed = max(0.0, elapsed - cleaned_at)
        scenario_duration = max(1.0, row.duration_minutes - cleaned_at)
        values = telemetry_values(site, prediction, ambient, row.scenario, scenario_elapsed, scenario_duration)
        for name, value, unit in [("humidity", weather.get("humidity"), "%"), ("cloud_cover", weather.get("cloud_cover"), "%"), ("wind_speed", weather.get("wind_speed"), "km/h"), ("precipitation", weather.get("precipitation"), "mm")]:
            if value is not None: values[name] = (float(value), unit)
        expected_kw, actual_kw = prediction.expected_power_kw, values["ac_power"][0]
        expected_kwh = power_to_energy_kwh(expected_kw, row.interval_minutes)
        actual_kwh = power_to_energy_kwh(actual_kw, row.interval_minutes)
        fault = row.scenario == "inverter_failure"
        status, explanation = classify(expected_kw, actual_kw, irradiance, irradiance >= 25)
        if fault and irradiance >= 25: status, explanation = "possible_system_fault", "Simulated inverter failure: the virtual inverter is producing zero AC power."
        weather_stamp = datetime.fromisoformat(weather["timestamp"])
        if weather_stamp.tzinfo is None: weather_stamp = weather_stamp.replace(tzinfo=zone)
        observation = Observation(timestamp=row.cursor_time, installation_id=site.id, simulation_session_id=row.id, expected_kwh=expected_kwh, actual_kwh=actual_kwh, unit="kWh", interval_minutes=row.interval_minutes, irradiance=irradiance, temperature=ambient, source="simulation", status=status, prediction_method=prediction.method, model_version=prediction.model_version, weather_timestamp=weather_stamp.astimezone(timezone.utc).replace(tzinfo=None), weather_source=weather["source"], expected_power_kw=expected_kw, actual_power_kw=actual_kw, expected_energy_kwh=expected_kwh, actual_energy_kwh=actual_kwh, measurement_semantics="simulated_power_assumed_constant_over_interval", scenario_loss_fraction=max(0.0, 1 - actual_kw / expected_kw) if expected_kw > 0 else 0.0, scenario_fault=fault, requires_unit_review=False)
        db.add(observation); db.flush()
        groups = [("panel_array", ["irradiance", "dni", "dhi", "poa_irradiance", "solar_zenith", "solar_azimuth", "panel_temperature", "soiling_loss", "shading_loss"]), ("environment", ["ambient_temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"]), ("inverter", ["dc_power", "ac_power", "inverter_efficiency", "expected_dc_power", "expected_power"]), ("gateway", ["ac_power"])]
        for device_type, names in groups:
            device = devices.get(device_type)
            if not device: continue
            for name in names:
                if name in values:
                    value, unit = values[name]
                    source = "weather_api" if name in {"irradiance", "dni", "dhi", "ambient_temperature", "humidity", "cloud_cover", "wind_speed", "precipitation"} else "simulated"
                    db.add(TelemetryReading(device_id=device.device_id, installation_id=site.id, session_id=row.id, timestamp=row.cursor_time, source=source, measurement=name, value=value, unit=unit, valid=True))
            device.last_seen = row.cursor_time; device.status = "online"
        generated.append({"timestamp": row.cursor_time.isoformat(), "expected_kw": round(expected_kw, 4), "actual_kw": round(actual_kw, 4), "expected_kwh": round(expected_kwh, 4), "actual_kwh": round(actual_kwh, 4), "dc_power_kw": round(prediction.dc_power_kw, 4), "poa_irradiance_wm2": round(prediction.poa_global_wm2, 2), "cell_temperature_c": round(prediction.cell_temperature_c, 2), "irradiance_wm2": irradiance, "temperature_c": ambient, "scenario": row.scenario, "status": status, "explanation": explanation, "prediction_method": prediction.method, "model_version": prediction.model_version, "weather_source": weather["source"], "weather_timestamp": weather["timestamp"], "irradiance_method": prediction.irradiance_method, "assumptions": prediction.assumptions})
        row.cursor_time += timedelta(minutes=row.interval_minutes * row.speed)
    if row.status == "running" and (row.cursor_time - row.start_time).total_seconds() / 60 >= row.duration_minutes:
        row.status = "completed"; row.end_time = datetime.utcnow()
    db.commit(); db.refresh(row)
    return {"session": _view(row), "generated": generated}

@router.post("/simulation/{session_id}/clean")
def simulate_cleaning(session_id: int, db: Session = Depends(get_db)):
    row = db.get(SimulationSession, session_id)
    if not row: raise HTTPException(404, "Simulation session not found")
    if row.status not in {"running", "paused"}: raise HTTPException(409, "Cleaning simulation requires an active session.")
    if row.scenario != "dust": raise HTTPException(409, "This cleaning simulation applies to the dust accumulation scenario.")
    config = json.loads(row.config_json or "{}")
    elapsed = max(0.0, (row.cursor_time - row.start_time).total_seconds() / 60)
    previous_clean = float(config.get("cleaned_at_elapsed_minutes", 0))
    before_loss = scenario_losses("dust", max(0.0, elapsed - previous_clean), max(1.0, row.duration_minutes - previous_clean))[0]
    site = db.get(Installation, row.installation_id)
    irradiance_row = db.scalar(select(TelemetryReading).where(TelemetryReading.session_id == row.id, TelemetryReading.measurement == "irradiance").order_by(TelemetryReading.timestamp.desc()))
    temperature_row = db.scalar(select(TelemetryReading).where(TelemetryReading.session_id == row.id, TelemetryReading.measurement == "ambient_temperature").order_by(TelemetryReading.timestamp.desc()))
    dni_row = db.scalar(select(TelemetryReading).where(TelemetryReading.session_id == row.id, TelemetryReading.measurement == "dni").order_by(TelemetryReading.timestamp.desc()))
    dhi_row = db.scalar(select(TelemetryReading).where(TelemetryReading.session_id == row.id, TelemetryReading.measurement == "dhi").order_by(TelemetryReading.timestamp.desc()))
    wind_row = db.scalar(select(TelemetryReading).where(TelemetryReading.session_id == row.id, TelemetryReading.measurement == "wind_speed").order_by(TelemetryReading.timestamp.desc()))
    recovery = None
    if irradiance_row and temperature_row:
        prediction = predict_installation_power(site, irradiance_row.value, temperature_row.value, when=row.cursor_time.replace(tzinfo=timezone.utc), dni_wm2=dni_row.value if dni_row else None, dhi_wm2=dhi_row.value if dhi_row else None, wind_speed_kmh=wind_row.value if wind_row else None)
        recovery = prediction.expected_power_kw - dc_to_ac(site, prediction.dc_power_kw * (1.0 - before_loss))
    event = CleaningEvent(notes="Virtual panel cleaning; simulated soiling reset.", source="simulated", installation_id=site.id, simulation_session_id=row.id, estimated_recovery_kw=recovery)
    config["cleaned_at_elapsed_minutes"] = elapsed
    row.config_json = json.dumps(config)
    db.add(event); db.flush()
    db.commit(); db.refresh(event); db.refresh(row)
    return {"event_id": event.id, "source": "simulated", "timestamp": event.timestamp.isoformat(), "session": _view(row), "soiling_loss_before": before_loss, "soiling_loss_after": 0.0, "estimated_recovery_kw": round(recovery, 4) if recovery is not None else None, "message": "Simulated soiling reset. The next telemetry interval shows recovery; this is not measured recovery."}

@router.get("/devices")
def virtual_devices(db: Session = Depends(get_db)):
    site = _installation(db)
    rows = db.scalars(select(VirtualDevice).where(VirtualDevice.installation_id == site.id).order_by(VirtualDevice.id)).all()
    return [{"device_id": d.device_id, "installation_id": d.installation_id, "name": d.name, "type": d.device_type, "status": d.status, "last_seen": d.last_seen.isoformat() if d.last_seen else None} for d in rows]

@router.post("/gateway/ingest")
def gateway_ingest(payload: GatewayPayload, db: Session = Depends(get_db)):
    device = db.scalar(select(VirtualDevice).where(VirtualDevice.device_id == payload.device_id))
    if not device: raise HTTPException(404, "Unknown virtual device ID")
    units = {"W/m2", "kW", "kWh", "C", "%", "km/h", "mm", "ratio"}
    timestamp = payload.timestamp.astimezone(timezone.utc).replace(tzinfo=None) if payload.timestamp.tzinfo else payload.timestamp
    accepted, rejected = [], []
    for item in payload.readings:
        reason = None
        if item.unit not in units:
            reason = "Unsupported unit"
        if item.measurement in {"irradiance", "dc_power", "ac_power", "expected_power", "wind_speed", "precipitation"} and item.value < 0:
            reason = "Value is below the physical minimum"
        if item.unit == "ratio" and not 0 <= item.value <= 1:
            reason = "Ratio must be from 0 to 1"
        if item.measurement in {"humidity", "cloud_cover"} and not 0 <= item.value <= 100:
            reason = "Percentage must be from 0 to 100"
        db.add(TelemetryReading(device_id=device.device_id, installation_id=device.installation_id, session_id=None, timestamp=timestamp, source=payload.source, measurement=item.measurement, value=item.value, unit=item.unit, valid=reason is None))
        if reason:
            rejected.append({"measurement": item.measurement, "reason": reason})
        else:
            accepted.append(item.measurement)
    device.last_seen = timestamp; device.status = "online"; db.commit()
    state = "partial" if accepted and rejected else "rejected" if rejected else "accepted"
    return {"device_id": device.device_id, "timestamp": timestamp.isoformat(), "accepted": len(accepted), "accepted_measurements": accepted, "rejected": rejected, "status": state}
