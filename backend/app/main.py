from datetime import datetime, timezone
import io, os, json
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session
from sqlalchemy import select, func, delete as sql_delete
from sqlalchemy import inspect, text
from app.core.config import settings
from app.database.session import Base, engine, get_db, SessionLocal
from app.models.records import Installation, Observation, DatasetRun, CleaningEvent, VirtualDevice, SimulationSession, TelemetryReading
from app.services.weather.client import forecast, historical_weather
from app.services.monitoring.logic import classify
from app.services.location.geocoding import search_locations, resolve_location
from app.services.prediction.trained_model import model_status
from app.services.prediction.engine import predict_installation_power
from app.services.production.units import power_to_energy_kwh, energy_to_average_power_kw
from app.services.solar.physics import dc_to_ac
import httpx

Base.metadata.create_all(bind=engine)
app=FastAPI(title=settings.app_name, version="1.0.0", description="Weather-aware solar generation monitoring and diagnosis")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
from app.routers.simulation import router as simulation_router
app.include_router(simulation_router)

class ManualProductionInput(BaseModel):
    timestamp: datetime
    actual_kwh: float = Field(ge=0, allow_inf_nan=False)
    interval_minutes: int = Field(default=60, ge=1, le=1440)

class GatewayReading(BaseModel):
    measurement: str = Field(min_length=1, max_length=64)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=16)

class GatewayPayload(BaseModel):
    device_id: str
    timestamp: datetime
    readings: list[GatewayReading] = Field(min_length=1, max_length=100)

def add_observation_columns(connection, observation_columns):
    additions = {
        "installation_id": "INTEGER",
        "simulation_session_id": "INTEGER",
        "raw_measurement_value": "FLOAT",
        "raw_measurement_unit": "VARCHAR(12)",
        "raw_measurement_components": "TEXT",
        "measurement_semantics": "VARCHAR(40)",
        "scenario_loss_fraction": "FLOAT",
        "scenario_fault": "BOOLEAN",
        "expected_power_kw": "FLOAT",
        "actual_power_kw": "FLOAT",
        "expected_energy_kwh": "FLOAT",
        "actual_energy_kwh": "FLOAT",
        "requires_unit_review": "BOOLEAN NOT NULL DEFAULT 0",
        "requires_interval_review": "BOOLEAN NOT NULL DEFAULT 0",
    }
    for column, sql_type in additions.items():
        if column not in observation_columns:
            connection.execute(text(f"ALTER TABLE observations ADD COLUMN {column} {sql_type}"))

    # Preserve legacy columns verbatim and add canonical values only where their saved
    # unit and interval make the conversion unambiguous.
    connection.execute(text("""
        UPDATE observations
        SET expected_power_kw=expected_kwh, actual_power_kw=actual_kwh,
            raw_measurement_value=actual_kwh, raw_measurement_unit='kW',
            requires_interval_review=CASE WHEN interval_minutes IS NULL OR interval_minutes <= 0 THEN 1 ELSE 0 END
        WHERE unit='kW' AND expected_power_kw IS NULL AND actual_power_kw IS NULL
    """))
    connection.execute(text("""
        UPDATE observations
        SET raw_measurement_value=actual_kwh, raw_measurement_unit='kWh',
            actual_energy_kwh=actual_kwh,
            expected_energy_kwh=CASE WHEN status='pending' THEN NULL ELSE expected_kwh END,
            actual_power_kw=CASE WHEN interval_minutes > 0 THEN actual_kwh / (interval_minutes / 60.0) ELSE NULL END,
            expected_power_kw=CASE WHEN interval_minutes > 0 AND status!='pending' THEN expected_kwh / (interval_minutes / 60.0) ELSE NULL END,
            requires_interval_review=CASE WHEN interval_minutes IS NULL OR interval_minutes <= 0 THEN 1 ELSE 0 END
        WHERE unit='kWh' AND actual_energy_kwh IS NULL
    """))
    connection.execute(text("UPDATE observations SET requires_unit_review=1 WHERE unit NOT IN ('kWh','kW')"))
    connection.execute(text("UPDATE observations SET measurement_semantics='legacy_inferred_interval_energy' WHERE unit='kWh' AND measurement_semantics IS NULL"))
    connection.execute(text("UPDATE observations SET measurement_semantics='legacy_power_sample' WHERE unit='kW' AND measurement_semantics IS NULL"))

@app.on_event("startup")
def initialize():
    # Additive migration for existing installations: preserve their saved coordinates.
    columns = {column["name"] for column in inspect(engine).get_columns("installations")}
    additions = {
        "city": "VARCHAR(120)",
        "administrative_region": "VARCHAR(120)",
        "country": "VARCHAR(120)",
        "timezone": "VARCHAR(80)",
        "panel_count": "INTEGER NOT NULL DEFAULT 13",
        "panel_efficiency": "FLOAT NOT NULL DEFAULT 0.20",
        "inverter_capacity_kw": "FLOAT",
    }
    observation_columns = {column["name"] for column in inspect(engine).get_columns("observations")}
    cleaning_columns = {column["name"] for column in inspect(engine).get_columns("cleaning_events")}
    with engine.begin() as connection:
        for column, sql_type in additions.items():
            if column not in columns:
                connection.execute(text(f"ALTER TABLE installations ADD COLUMN {column} {sql_type}"))
        if "unit" not in observation_columns:
            connection.execute(text("ALTER TABLE observations ADD COLUMN unit VARCHAR(8) NOT NULL DEFAULT 'kWh'"))
            connection.execute(text("UPDATE observations SET unit = 'kW' WHERE source = 'simulation'"))
        if "interval_minutes" not in observation_columns:
            connection.execute(text("ALTER TABLE observations ADD COLUMN interval_minutes INTEGER"))
        for column, sql_type in {
            "prediction_method": "VARCHAR(32)",
            "model_version": "VARCHAR(80)",
            "weather_timestamp": "DATETIME",
            "weather_source": "VARCHAR(80)",
        }.items():
            if column not in observation_columns:
                connection.execute(text(f"ALTER TABLE observations ADD COLUMN {column} {sql_type}"))
        add_observation_columns(connection, observation_columns)
        for column, sql_type in {"installation_id": "INTEGER", "simulation_session_id": "INTEGER", "estimated_recovery_kw": "FLOAT"}.items():
            if column not in cleaning_columns:
                connection.execute(text(f"ALTER TABLE cleaning_events ADD COLUMN {column} {sql_type}"))
        # Previous simulator code used the physics baseline. Preserve all old expected/actual values.
        connection.execute(text("UPDATE observations SET prediction_method='physics_fallback' WHERE source='simulation' AND prediction_method IS NULL"))
        connection.execute(text("UPDATE observations SET weather_source='Open-Meteo current' WHERE source='simulation' AND weather_source IS NULL"))
    with SessionLocal() as db:
        if not db.scalar(select(Installation.id)):
            db.add(Installation()); db.commit()
        installation_ids=db.scalars(select(Installation.id).order_by(Installation.id)).all()
        if len(installation_ids)==1:
            db.execute(text("UPDATE observations SET installation_id=:installation_id WHERE installation_id IS NULL"),{"installation_id":installation_ids[0]})
            db.commit()
        device_templates = [("panel-array-01", "Solar panel array", "panel_array"), ("inverter-01", "Virtual inverter", "inverter"), ("environment-01", "Environmental sensors", "environment"), ("gateway-01", "Virtual IoT gateway", "gateway")]
        for installation_id in installation_ids:
            for device_id, name, device_type in device_templates:
                if not db.scalar(select(VirtualDevice.id).where(VirtualDevice.device_id == f"{installation_id}:{device_id}")):
                    db.add(VirtualDevice(installation_id=installation_id, device_id=f"{installation_id}:{device_id}", name=name, device_type=device_type, status="online"))
        db.commit()

def installation(db):
    row=db.scalar(select(Installation).order_by(Installation.id))
    if not row: raise HTTPException(404,"Configure an installation first")
    return row

@app.get("/api/health")
def health(db:Session=Depends(get_db)):
    return {"status":"ok","database":"connected","installation_count":db.scalar(select(func.count(Installation.id))),"model":model_status()}

@app.get("/api/models/status")
def model_readiness():
    return model_status()

@app.post("/api/models/train")
def train_model():
    from ml.training.train import train_models
    try:
        return train_models()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Model training failed: {exc}")

@app.get("/api/installation")
def get_installation(db:Session=Depends(get_db)):
    x=installation(db); return {k:getattr(x,k) for k in ["id","name","city","administrative_region","country","timezone","latitude","longitude","capacity_kw","tilt","azimuth","inverter_efficiency","losses","panel_count","panel_efficiency","inverter_capacity_kw"]}

@app.get("/api/locations/search")
async def location_search(q: str = Query(..., min_length=2, max_length=100)):
    try:
        return {"results": await search_locations(q)}
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Open-Meteo geocoding is unavailable: {exc}")
    except (ValueError, KeyError) as exc:
        raise HTTPException(502, f"Could not read geocoding results: {exc}")

@app.put("/api/installation")
async def put_installation(payload:dict, db:Session=Depends(get_db)):
    x=installation(db)
    if "location_id" in payload:
        try:
            location_id = int(payload["location_id"])
        except (TypeError, ValueError):
            raise HTTPException(422, "Select a location from the city search results.")
        try:
            selected = await resolve_location(location_id, str(payload.get("city", "")))
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Open-Meteo geocoding is unavailable: {exc}")
        if not selected:
            raise HTTPException(422, "The selected location could not be verified. Search again and reselect it.")
        x.city = selected["city"]
        x.administrative_region = selected["administrative_region"] or None
        x.country = selected["country"] or None
        x.latitude = selected["latitude"]
        x.longitude = selected["longitude"]
        x.timezone = selected["timezone"] or None
    for k in ["name","capacity_kw","tilt","azimuth","inverter_efficiency","losses","panel_count","panel_efficiency","inverter_capacity_kw"]:
        if k in payload:
            if k == "capacity_kw" and (not isinstance(payload[k], (int, float)) or not 0 < payload[k] <= 1_000_000):
                raise HTTPException(422, "DC capacity must be greater than 0 and no more than 1,000,000 kWp.")
            if k == "inverter_efficiency" and (not isinstance(payload[k], (int, float)) or not 0 < payload[k] <= 1):
                raise HTTPException(422, "Inverter efficiency must be greater than 0 and at most 1.")
            if k == "tilt" and (not isinstance(payload[k], (int, float)) or not 0 <= payload[k] <= 90):
                raise HTTPException(422, "Panel tilt must be between 0 and 90 degrees.")
            if k == "azimuth" and (not isinstance(payload[k], (int, float)) or not 0 <= payload[k] <= 360):
                raise HTTPException(422, "Panel azimuth must be between 0 and 360 degrees (0=N, 180=S).")
            if k == "losses" and (not isinstance(payload[k], (int, float)) or not 0 <= payload[k] < 1):
                raise HTTPException(422, "System losses must be from 0 up to (but not including) 1.")
            if k == "panel_count" and (not isinstance(payload[k], int) or not 1 <= payload[k] <= 10_000_000):
                raise HTTPException(422, "Panel count must be a positive whole number.")
            if k == "panel_efficiency" and (not isinstance(payload[k], (int, float)) or not 0 < payload[k] <= 1):
                raise HTTPException(422, "Panel efficiency must be greater than 0 and at most 1.")
            if k == "inverter_capacity_kw" and payload[k] is not None and (not isinstance(payload[k], (int, float)) or not 0 < payload[k] <= 1_000_000):
                raise HTTPException(422, "Inverter capacity must be greater than 0 and no more than 1,000,000 kW.")
            setattr(x,k,payload[k])
    db.commit(); return get_installation(db)

@app.get("/api/weather")
async def get_weather(db:Session=Depends(get_db)):
    x=installation(db)
    try:
        raw=await forecast(x.latitude,x.longitude); current=raw["current"]
        return {"source":"Open-Meteo","stale":False,"timezone":raw.get("timezone"),"current":current,"hourly":raw.get("hourly",{})}
    except Exception as exc: raise HTTPException(502,f"Open-Meteo unavailable: {exc}")

@app.get("/api/overview")
async def overview(db:Session=Depends(get_db)):
    x=installation(db)
    weather_error=None
    try:
        weather=await forecast(x.latitude,x.longitude); w=weather["current"]
    except Exception as exc:
        weather_error=str(exc); w={"temperature_2m":None,"shortwave_radiation":None,"cloud_cover":None,"is_day":0}; weather={"timezone":"unknown"}
    irr=w.get("shortwave_radiation"); temp=w.get("temperature_2m")
    live_prediction=None
    if irr is not None and temp is not None:
        try:
            live_prediction=predict_installation_power(x,irr,temp,when=datetime.fromisoformat(w["time"]) if w.get("time") else None,dni_wm2=w.get("direct_normal_irradiance"),dhi_wm2=w.get("diffuse_radiation"),wind_speed_kmh=w.get("wind_speed_10m"))
        except (ValueError,TypeError) as exc:
            weather_error=str(exc)
    expected=live_prediction.expected_power_kw if live_prediction else None
    latest=db.scalar(select(Observation).order_by(Observation.timestamp.desc()))
    actual=0.0
    actual_unit="kWh"
    if latest:
        if latest.actual_energy_kwh is not None:
            actual=latest.actual_energy_kwh
            actual_unit="kWh"
        elif latest.actual_power_kw is not None:
            actual=latest.actual_power_kw
            actual_unit="kW"
        else:
            actual=latest.actual_kwh
            actual_unit=latest.unit
    if latest is None:
        status, explanation = "insufficient_data", "No actual production records are available yet."
    elif latest.status == "pending":
        status, explanation = "insufficient_data", "This energy reading is waiting for weather data to calculate its expected interval energy."
    elif latest.expected_energy_kwh is not None and latest.actual_energy_kwh is not None and latest.interval_minutes:
        status, explanation = classify(energy_to_average_power_kw(latest.expected_energy_kwh, latest.interval_minutes), energy_to_average_power_kw(latest.actual_energy_kwh, latest.interval_minutes), latest.irradiance, latest.irradiance >= 25)
    elif latest.expected_power_kw is not None and latest.actual_power_kw is not None:
        status, explanation = classify(latest.expected_power_kw, latest.actual_power_kw, latest.irradiance, latest.irradiance >= 25)
    else:
        status, explanation = "insufficient_data", "The latest historical record lacks comparable power or interval-energy values."
    performance_ratio = None
    if latest and latest.expected_energy_kwh and latest.actual_energy_kwh is not None:
        performance_ratio = round(latest.actual_energy_kwh / latest.expected_energy_kwh, 3)
    elif latest and latest.expected_power_kw and latest.actual_power_kw is not None:
        performance_ratio = round(latest.actual_power_kw / latest.expected_power_kw, 3)
    readiness = model_status()
    prediction_method=live_prediction.method if live_prediction else "unavailable"
    return {"installation":{"id":x.id,"name":x.name,"city":x.city,"administrative_region":x.administrative_region,"country":x.country,"timezone":x.timezone,"location_label":", ".join(part for part in [x.city,x.administrative_region,x.country] if part),"capacity_kw":x.capacity_kw},"weather":{"source":"Open-Meteo","stale":False,"available":live_prediction is not None,"error":weather_error,"current":w,"timestamp":w.get("time"),"timezone":weather.get("timezone") or x.timezone},"prediction":{"expected_kw":round(expected,3) if expected is not None else None,"method":prediction_method,"pipeline_version":live_prediction.pipeline_version if live_prediction else "pvlib-pvwatts-v1","model_version":None,"dc_kw":round(live_prediction.dc_power_kw,3) if live_prediction else None,"poa_wm2":round(live_prediction.poa_global_wm2,1) if live_prediction else None,"cell_temperature_c":round(live_prediction.cell_temperature_c,2) if live_prediction else None,"irradiance_method":live_prediction.irradiance_method if live_prediction else None,"assumptions":live_prediction.assumptions if live_prediction else [],"weather_timestamp":w.get("time"),"weather_source":"Open-Meteo current","measured":False},"actual":{"value":actual,"unit":actual_unit,"interval_minutes":latest.interval_minutes if latest else None,"source":latest.source if latest else "no production data","source_kind":"simulated" if latest and latest.source=="simulation" else "measured" if latest else "none","prediction_method":latest.prediction_method if latest else None,"model_version":latest.model_version if latest else None,"weather_timestamp":latest.weather_timestamp.isoformat() if latest and latest.weather_timestamp else None,"weather_source":latest.weather_source if latest else None,"expected_power_kw":latest.expected_power_kw if latest else None,"actual_power_kw":latest.actual_power_kw if latest else None,"expected_energy_kwh":latest.expected_energy_kwh if latest else None,"actual_energy_kwh":latest.actual_energy_kwh if latest else None,"requires_unit_review":latest.requires_unit_review if latest else False,"requires_interval_review":latest.requires_interval_review if latest else False,"timestamp":latest.timestamp.isoformat() if latest else None},"performance":{"status":status,"ratio":performance_ratio,"explanation":explanation},"model":readiness}

@app.get("/api/observations")
def observations(limit:int=168,db:Session=Depends(get_db)):
    rows=db.scalars(select(Observation).order_by(Observation.timestamp.desc()).limit(min(limit,1000))).all()
    return [{"id":r.id,"installation_id":r.installation_id,"timestamp":r.timestamp.isoformat(),"expected_kwh":r.expected_kwh,"actual_kwh":r.actual_kwh,"unit":r.unit,"interval_minutes":r.interval_minutes,"raw_measurement_value":r.raw_measurement_value,"raw_measurement_unit":r.raw_measurement_unit,"raw_measurement_components":r.raw_measurement_components,"measurement_semantics":r.measurement_semantics,"expected_power_kw":r.expected_power_kw,"actual_power_kw":r.actual_power_kw,"expected_energy_kwh":r.expected_energy_kwh,"actual_energy_kwh":r.actual_energy_kwh,"requires_unit_review":r.requires_unit_review,"requires_interval_review":r.requires_interval_review,"irradiance":r.irradiance,"temperature":r.temperature,"source":r.source,"source_kind":"simulated" if r.source=="simulation" else "measured","prediction_method":r.prediction_method,"model_version":r.model_version,"weather_timestamp":r.weather_timestamp.isoformat() if r.weather_timestamp else None,"weather_source":r.weather_source,"status":r.status} for r in rows]

@app.delete("/api/observations")
def reset_observations(db:Session=Depends(get_db)):
    count = db.scalar(select(func.count(Observation.id))) or 0
    db.execute(sql_delete(Observation))
    db.commit()
    return {"deleted":count,"message":"Performance history and derived diagnosis have been reset."}

@app.post("/api/production/csv")
async def upload_csv(file:UploadFile=File(...),db:Session=Depends(get_db)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload a CSV file")
    content=await file.read(10_000_001)
    if len(content)>10_000_000: raise HTTPException(413,"CSV limit is 10 MB")
    try: df=pd.read_csv(io.BytesIO(content))
    except Exception as exc: raise HTTPException(400,f"Could not parse CSV: {exc}")
    normalized={c.lower().strip():c for c in df.columns}
    time_col=next((normalized[k] for k in ["timestamp","time","datetime","date","date_time"] if k in normalized),None)
    energy_col=next((normalized[k] for k in ["actual_kwh","energy_kwh","generation_kwh","kwh"] if k in normalized),None)
    average_power_col=next((normalized[k] for k in ["interval_average_power_kw","average_power_kw"] if k in normalized),None)
    instantaneous_power_col=next((normalized[k] for k in ["actual_power_kw","power_kw","ac_power_kw"] if k in normalized),None)
    kaggle_ac_col=normalized.get("ac_power")
    power_col=average_power_col or instantaneous_power_col or kaggle_ac_col
    is_interval_average_power=bool(average_power_col)
    if not time_col or (not energy_col and not power_col):
        raise HTTPException(400,"CSV needs a timestamp and either interval energy (kWh), interval-average power (interval_average_power_kw), or sampled power (kW). Kaggle AC_POWER is treated as a power sample.")
    site=installation(db)
    try:
        site_zone=ZoneInfo(site.timezone) if site.timezone else timezone.utc
    except Exception:
        site_zone=timezone.utc
    interval_col=next((normalized[k] for k in ["interval_minutes","duration_minutes"] if k in normalized),None)
    working=df.copy()
    working["_timestamp"] = pd.to_datetime(working[time_col], errors="coerce", format="mixed", dayfirst=True)
    if energy_col:
        working["_measurement"] = pd.to_numeric(working[energy_col], errors="coerce")
        measure_kind="energy"
        raw_unit="kWh"
    else:
        working["_measurement"] = pd.to_numeric(working[power_col], errors="coerce")
        measure_kind="interval_average_power" if is_interval_average_power else "power_sample"
        raw_unit="kW"
        # The supplied Kaggle dataset is inverter-level: preserve the raw rows while
        # aggregate AC_POWER to plant/timestamp power, matching the training target.
        plant_col=normalized.get("plant_id")
        source_key_col=normalized.get("source_key")
        if kaggle_ac_col and power_col==kaggle_ac_col and source_key_col:
            working["_raw_components"] = working.apply(lambda r: [{"source_key": str(r[source_key_col]), "value": float(r["_measurement"])}] if pd.notna(r["_measurement"]) else [], axis=1)
            group_cols=([plant_col] if plant_col else [])+["_timestamp"]
            working=working.dropna(subset=["_timestamp","_measurement"]).groupby(group_cols,as_index=False).agg(
                _measurement=("_measurement","sum"),
                _raw_components=("_raw_components",lambda groups: json.dumps([item for group in groups for item in group])),
                **({interval_col:(interval_col,"first")} if interval_col else {}),
            )
    if "_raw_components" not in working.columns:
        working["_raw_components"] = working["_measurement"].apply(lambda value: json.dumps([{"value": float(value)}]) if pd.notna(value) else "[]")
    parsed=[]
    for _,row in working.iterrows():
        ts=row["_timestamp"]
        val=row["_measurement"]
        if pd.isna(ts) or pd.isna(val) or val<0: continue
        recorded=ts.to_pydatetime()
        if recorded.tzinfo is None:
            recorded=recorded.replace(tzinfo=site_zone)
        stored=recorded.astimezone(timezone.utc).replace(tzinfo=None)
        explicit=pd.to_numeric(row[interval_col],errors="coerce") if interval_col and interval_col in row.index else None
        valid_explicit=int(explicit) if explicit is not None and not pd.isna(explicit) and 1<=explicit<=1440 else None
        components=row["_raw_components"]
        components=json.dumps(components) if not isinstance(components,str) else components
        parsed.append((stored,float(val),valid_explicit,components))
    ordered=sorted(item[0] for item in parsed)
    gaps=[int((b-a).total_seconds()/60) for a,b in zip(ordered,ordered[1:]) if 1 <= (b-a).total_seconds()/60 <= 1440]
    inferred=int(pd.Series(gaps).median()) if gaps else 60
    accepted=0
    for stored,val,explicit,components in parsed:
        interval=explicit or inferred
        if measure_kind=="energy":
            energy_kwh=val
            power_kw=energy_to_average_power_kw(val,interval)
            semantics="interval_energy"
        elif measure_kind=="interval_average_power":
            power_kw=val
            energy_kwh=power_to_energy_kwh(val,interval)
            semantics="interval_average_power"
        else:
            power_kw=val
            energy_kwh=None
            semantics="instantaneous_power_sample"
        db.add(Observation(timestamp=stored,installation_id=site.id,expected_kwh=energy_kwh if energy_kwh is not None else power_kw,actual_kwh=energy_kwh if energy_kwh is not None else power_kw,unit="kWh" if energy_kwh is not None else "kW",interval_minutes=interval,source="csv",status="pending",raw_measurement_value=val,raw_measurement_unit=raw_unit,raw_measurement_components=components,measurement_semantics=semantics,actual_power_kw=power_kw,actual_energy_kwh=energy_kwh,requires_unit_review=False,requires_interval_review=energy_kwh is None)); accepted+=1
    db.commit(); return {"accepted":accepted,"rejected":len(df)-accepted,"source":"user CSV measured production"}

async def analyze_energy_observation(site:Installation,row:Observation,recorded_at:datetime):
    try:
        site_zone=ZoneInfo(site.timezone) if site.timezone else timezone.utc
    except Exception:
        site_zone=timezone.utc
    local_time=recorded_at.astimezone(site_zone) if recorded_at.tzinfo else recorded_at.replace(tzinfo=site_zone)
    weather=await historical_weather(site.latitude,site.longitude,local_time,site.timezone or "UTC")
    irradiance=weather.get("ghi",weather.get("irradiance"))
    if irradiance is None: raise ValueError("Open-Meteo GHI is missing for this production timestamp.")
    irradiance=float(irradiance)
    temperature=float(weather["temperature"])
    prediction=predict_installation_power(site,irradiance,temperature,when=local_time,dni_wm2=weather.get("dni"),dhi_wm2=weather.get("dhi"),wind_speed_kmh=weather.get("wind_speed"))
    expected_kw=prediction.expected_power_kw
    interval=row.interval_minutes or 60
    is_energy_measurement=row.actual_energy_kwh is not None or row.unit=="kWh"
    if is_energy_measurement:
        expected_energy=power_to_energy_kwh(expected_kw,interval)
        if row.actual_energy_kwh is None:
            row.actual_energy_kwh=row.actual_kwh
        actual_power=energy_to_average_power_kw(row.actual_energy_kwh,interval)
        status,explanation=classify(expected_kw,actual_power,irradiance,irradiance>=25)
        row.expected_energy_kwh=expected_energy
        row.actual_power_kw=actual_power
        row.expected_kwh=expected_energy
        row.actual_kwh=row.actual_energy_kwh
        row.requires_interval_review=False
    else:
        # A sampled kW value remains power-only. Cadence is not proof that it is
        # an interval-average, so do not invent kWh by multiplying it by duration.
        expected_energy=None
        actual_power=row.actual_power_kw if row.actual_power_kw is not None else row.actual_kwh
        status,explanation=classify(expected_kw,actual_power,irradiance,irradiance>=25)
        row.expected_energy_kwh=None
        row.actual_energy_kwh=None
        row.expected_kwh=expected_kw
        row.actual_kwh=actual_power
        row.actual_power_kw=actual_power
        row.requires_interval_review=True
    row.expected_power_kw=expected_kw
    row.raw_measurement_value=row.raw_measurement_value if row.raw_measurement_value is not None else row.actual_kwh
    row.raw_measurement_unit=row.raw_measurement_unit or row.unit
    row.measurement_semantics=row.measurement_semantics or ("interval_energy" if is_energy_measurement else "instantaneous_power_sample")
    row.irradiance=irradiance
    row.temperature=temperature
    row.status=status
    raw_weather_time=datetime.fromisoformat(weather["timestamp"])
    if raw_weather_time.tzinfo is None:
        raw_weather_time=raw_weather_time.replace(tzinfo=site_zone)
    row.weather_timestamp=raw_weather_time.astimezone(timezone.utc).replace(tzinfo=None)
    row.weather_source=weather.get("source","Open-Meteo hourly historical/forecast")
    row.prediction_method=prediction.method
    row.model_version=prediction.model_version
    return {"status":status,"explanation":explanation,"expected_kwh":round(expected_energy,3) if expected_energy is not None else round(expected_kw,3),"expected_energy_kwh":round(expected_energy,3) if expected_energy is not None else None,"actual_energy_kwh":row.actual_energy_kwh,"expected_power_kw":round(expected_kw,3),"prediction_method":prediction.method,"model_version":row.model_version,"weather_timestamp":weather.get("timestamp"),"weather_source":row.weather_source}

@app.post("/api/production/manual")
async def add_manual_production(payload:ManualProductionInput,db:Session=Depends(get_db)):
    site=installation(db)
    timestamp=payload.timestamp
    try:
        site_zone=ZoneInfo(site.timezone) if site.timezone else timezone.utc
    except Exception:
        site_zone=timezone.utc
    recorded_at=timestamp.astimezone(site_zone) if timestamp.tzinfo else timestamp.replace(tzinfo=site_zone)
    stored_timestamp=recorded_at.astimezone(timezone.utc).replace(tzinfo=None)
    actual_power=energy_to_average_power_kw(payload.actual_kwh,payload.interval_minutes)
    row=Observation(timestamp=stored_timestamp,installation_id=site.id,expected_kwh=0,actual_kwh=payload.actual_kwh,unit="kWh",interval_minutes=payload.interval_minutes,source="manual",status="pending",raw_measurement_value=payload.actual_kwh,raw_measurement_unit="kWh",measurement_semantics="interval_energy",actual_power_kw=actual_power,actual_energy_kwh=payload.actual_kwh,requires_unit_review=False)
    try:
        analysis=await analyze_energy_observation(site,row,recorded_at)
        message="Production saved and weather-adjusted analysis completed."
    except Exception as exc:
        analysis={"status":"pending","explanation":str(exc),"expected_kwh":None,"weather_source":None}
        message=f"Production saved, but weather analysis is pending: {exc}"
    db.add(row);db.commit();db.refresh(row)
    return {"id":row.id,"installation_id":row.installation_id,"timestamp":row.timestamp.isoformat(),"actual_kwh":row.actual_energy_kwh,"expected_kwh":analysis["expected_kwh"],"actual_power_kw":row.actual_power_kw,"expected_power_kw":analysis.get("expected_power_kw"),"actual_energy_kwh":row.actual_energy_kwh,"expected_energy_kwh":row.expected_energy_kwh,"raw_measurement_value":row.raw_measurement_value,"raw_measurement_unit":row.raw_measurement_unit,"measurement_semantics":row.measurement_semantics,"unit":row.unit,"interval_minutes":row.interval_minutes,"source":row.source,"source_kind":"measured","status":row.status,"prediction_method":analysis.get("prediction_method"),"model_version":analysis.get("model_version"),"weather_timestamp":analysis.get("weather_timestamp"),"weather_source":analysis.get("weather_source"),"explanation":analysis["explanation"],"message":message}

@app.post("/api/observations/{observation_id}/analyze")
async def reanalyze_observation(observation_id:int,db:Session=Depends(get_db)):
    row=db.get(Observation,observation_id)
    if not row: raise HTTPException(404,"Production reading not found")
    if row.unit not in {"kWh","kW"}: raise HTTPException(400,"This record has an unknown unit and requires review before analysis.")
    try:
        site=installation(db)
        try:
            site_zone=ZoneInfo(site.timezone) if site.timezone else timezone.utc
        except Exception:
            site_zone=timezone.utc
        analysis=await analyze_energy_observation(site,row,row.timestamp.replace(tzinfo=timezone.utc).astimezone(site_zone))
    except Exception as exc:
        raise HTTPException(502,f"Weather analysis is still unavailable: {exc}")
    db.commit()
    return {"id":row.id,"status":row.status,"unit":row.unit,"expected_kwh":analysis["expected_kwh"],"expected_energy_kwh":row.expected_energy_kwh,"actual_energy_kwh":row.actual_energy_kwh,"expected_power_kw":row.expected_power_kw,"actual_power_kw":row.actual_power_kw,"interval_minutes":row.interval_minutes,"prediction_method":analysis["prediction_method"],"model_version":analysis.get("model_version"),"weather_timestamp":analysis.get("weather_timestamp"),"explanation":analysis["explanation"],"weather_source":analysis["weather_source"]}

@app.post("/api/simulator/run")
async def simulator_run(dust:float=Query(0.0,ge=0,le=0.9),fault:bool=False,interval_minutes:int=Query(60,ge=1,le=1440),timestamp:Optional[datetime]=None,db:Session=Depends(get_db)):
    x=installation(db)
    try:
        site_zone=ZoneInfo(x.timezone) if x.timezone else timezone.utc
    except Exception:
        site_zone=timezone.utc
    try:
        if timestamp is None:
            weather_payload=await forecast(x.latitude,x.longitude)
            w=weather_payload["current"]
            if not w.get("time"):
                raise ValueError("Weather response did not include a simulation timestamp.")
            simulation_local=datetime.fromisoformat(w["time"])
            if simulation_local.tzinfo is None:
                simulation_local=simulation_local.replace(tzinfo=site_zone)
            weather_source="Open-Meteo current"
            weather_stamp=simulation_local
        else:
            simulation_local=timestamp.astimezone(site_zone) if timestamp.tzinfo else timestamp.replace(tzinfo=site_zone)
            historical=await historical_weather(x.latitude,x.longitude,simulation_local,x.timezone or "UTC")
            historical_ghi=historical.get("ghi",historical.get("irradiance"))
            w={"time":historical["timestamp"],"shortwave_radiation":historical_ghi,"temperature_2m":historical["temperature"],"direct_normal_irradiance":historical.get("dni"),"diffuse_radiation":historical.get("dhi"),"wind_speed_10m":historical.get("wind_speed"),"is_day":int(historical_ghi>=25)}
            weather_stamp=datetime.fromisoformat(historical["timestamp"])
            if weather_stamp.tzinfo is None:
                weather_stamp=weather_stamp.replace(tzinfo=site_zone)
            weather_source=historical.get("source","Open-Meteo hourly historical/forecast")
    except Exception as exc:
        raise HTTPException(502,f"Simulator could not retrieve weather aligned to its timestamp: {exc}")
    irr=w.get("shortwave_radiation"); temp=w.get("temperature_2m")
    if irr is None or temp is None:
        raise HTTPException(502,"Open-Meteo GHI or ambient temperature is missing for the requested simulation timestamp.")
    simulation_utc=simulation_local.astimezone(timezone.utc)
    simulation_time=simulation_utc.replace(tzinfo=None)
    prediction=predict_installation_power(x,irr,temp,when=simulation_local,dni_wm2=w.get("direct_normal_irradiance"),dhi_wm2=w.get("diffuse_radiation"),wind_speed_kmh=w.get("wind_speed_10m"))
    expected_power=prediction.expected_power_kw
    scenario_loss=max(0.0,min(float(dust),0.9))
    actual_power=0.0 if fault else dc_to_ac(x,prediction.dc_power_kw*(1-scenario_loss))
    expected_energy=power_to_energy_kwh(expected_power,interval_minutes)
    actual_energy=power_to_energy_kwh(actual_power,interval_minutes)
    status,explanation=classify(expected_power,actual_power,irr,bool(w.get("is_day",0)))
    existing=db.scalars(select(Observation).where(Observation.installation_id==x.id,Observation.source=="simulation",Observation.timestamp==simulation_time,Observation.interval_minutes==interval_minutes)).all()
    weather_utc=weather_stamp.astimezone(timezone.utc).replace(tzinfo=None)
    duplicate=next((item for item in existing if item.expected_power_kw is not None and abs(item.expected_power_kw-expected_power)<1e-9 and item.actual_power_kw is not None and abs(item.actual_power_kw-actual_power)<1e-9 and item.scenario_loss_fraction is not None and abs(item.scenario_loss_fraction-scenario_loss)<1e-12 and item.scenario_fault is fault and item.model_version==prediction.model_version and item.prediction_method==prediction.method and item.weather_timestamp==weather_utc and item.irradiance==irr and item.temperature==temp and item.weather_source==weather_source),None)
    if duplicate:
        row=duplicate
        duplicate_run=True
    else:
        row=Observation(timestamp=simulation_time,installation_id=x.id,expected_kwh=expected_energy,actual_kwh=actual_energy,unit="kWh",interval_minutes=interval_minutes,irradiance=irr,temperature=temp,source="simulation",status=status,prediction_method=prediction.method,model_version=prediction.model_version,weather_timestamp=weather_utc,weather_source=weather_source,expected_power_kw=expected_power,actual_power_kw=actual_power,expected_energy_kwh=expected_energy,actual_energy_kwh=actual_energy,measurement_semantics="simulated_power_assumed_constant_over_interval",scenario_loss_fraction=scenario_loss,scenario_fault=fault,requires_unit_review=False)
        db.add(row);db.commit();db.refresh(row)
        duplicate_run=False
    return {"id":row.id,"installation_id":x.id,"timestamp":row.timestamp.isoformat(),"source":"simulation","source_kind":"simulated","prediction_method":row.prediction_method,"pipeline_version":prediction.pipeline_version,"model_version":row.model_version,"weather_timestamp":row.weather_timestamp.isoformat() if row.weather_timestamp else None,"weather_source":row.weather_source,"unit":"kWh","power_unit":"kW","interval_minutes":row.interval_minutes,"dust_loss_pct":round(scenario_loss*100,1),"fault":fault,"expected_kw":round(expected_power,3),"simulated_actual_kw":round(actual_power,3),"expected_energy_kwh":round(expected_energy,3),"simulated_actual_energy_kwh":round(actual_energy,3),"duplicate_run":duplicate_run,"status":row.status,"explanation":explanation}

@app.get("/api/diagnosis")
def diagnosis(db:Session=Depends(get_db)):
    rows=db.scalars(select(Observation).order_by(Observation.timestamp.desc()).limit(24)).all()
    if not rows:return {"status":"insufficient_data","possible_causes":[],"message":"Start a virtual solar simulation to begin monitoring."}
    low=[r for r in rows if r.status in ["underperforming","possible_system_fault"]]
    simulated_fault=any(r.source=="simulation" and r.scenario_fault and r.irradiance>=25 for r in rows)
    sessions={s.id:s.scenario for s in db.scalars(select(SimulationSession).where(SimulationSession.id.in_([r.simulation_session_id for r in rows if r.simulation_session_id]))).all()}
    simulated_dust=any(r.source=="simulation" and sessions.get(r.simulation_session_id)=="dust" and r.scenario_loss_fraction and r.scenario_loss_fraction>0.05 for r in low)
    simulated_shading=any(r.source=="simulation" and sessions.get(r.simulation_session_id)=="shading" for r in low)
    simulated_inverter_loss=any(r.source=="simulation" and sessions.get(r.simulation_session_id)=="inverter_degradation" for r in low)
    measured_zero_fault=any(r.source!="simulation" and r.status=="possible_system_fault" and r.irradiance>=25 for r in rows)
    persistent=len(low)>=3
    lost_kwh=sum(max(0.0,(r.expected_energy_kwh or 0)-(r.actual_energy_kwh or 0)) for r in low)
    weather_reduction=rows[0].irradiance<25 and rows[0].status=="night"
    if simulated_fault: status,causes,message,recommendation="critical_fault",["Simulated inverter failure; zero AC power while irradiance is available."],"The selected virtual failure scenario is active and its output matches the simulated fault.","Inspect the virtual inverter and electrical path in the scenario details."
    elif measured_zero_fault: status,causes,message,recommendation="possible_inverter_issue",["Measured output is near zero under strong irradiance."],"Production suggests a possible inverter or electrical issue; the readings do not confirm a component failure.","Inspect inverter status, AC disconnects, and wiring."
    elif persistent and simulated_inverter_loss: status,causes,message,recommendation="possible_inverter_issue",["Simulated inverter degradation persisted across multiple intervals."],"The inverter degradation scenario reduced actual output while the expected baseline remained unchanged.","Inspect inverter efficiency and temperature in the virtual telemetry."
    elif persistent and simulated_shading: status,causes,message,recommendation="possible_shading",["Simulated shading loss persisted across multiple intervals."],"The partial-shading scenario reduced actual output while expected production stayed weather adjusted.","Review array shading; the scenario applies a fixed illustrative loss."
    elif persistent and simulated_dust: status,causes,message,recommendation="possible_soiling",["Simulated dust loss persisted across multiple intervals."],"The dust scenario reduced simulated output. This is not a physical dust detection.","Consider a cleaning simulation; recovery remains simulated."
    elif persistent: status,causes,message,recommendation="monitoring_required",["Persistent weather-adjusted underperformance; possible shading, soiling, or inverter issue."],"Persistent low production warrants inspection. Weather and production alone cannot confirm dust or identify a failed component.","Inspect for shading, panel cleanliness, and inverter alerts."
    elif weather_reduction: status,causes,message,recommendation="weather_related",["Low irradiance explains low expected production at the latest interval."],"The latest interval has little solar irradiance; the pvlib baseline accounts for that weather condition.","No equipment action is indicated by this interval alone."
    else: status,causes,message,recommendation="normal",[],"No persistent weather-adjusted underperformance in recent records.","Continue monitoring."
    latest=rows[0]
    evidence={"persistence_threshold_intervals":3,"underperforming_intervals_reviewed":len(low),"latest_irradiance_wm2":latest.irradiance,"latest_expected_power_kw":latest.expected_power_kw,"latest_actual_power_kw":latest.actual_power_kw,"latest_prediction_method":latest.prediction_method,"estimated_lost_energy_kwh":round(lost_kwh,3)}
    return {"status":status,"records_reviewed":len(rows),"underperforming_records":len(low),"estimated_lost_energy_kwh":round(lost_kwh,3),"possible_causes":causes,"recommendation":recommendation,"evidence":evidence,"message":message}

@app.post("/api/cleaning-events")
def cleaning(payload:dict,db:Session=Depends(get_db)):
    site=installation(db)
    event=CleaningEvent(installation_id=site.id,notes=str(payload.get("notes","")),source="user");db.add(event);db.commit();return {"id":event.id,"timestamp":event.timestamp.isoformat(),"notes":event.notes,"source":event.source}

@app.get("/api/cleaning-events")
def cleaning_history(db:Session=Depends(get_db)):
    site=installation(db)
    rows=db.scalars(select(CleaningEvent).where(CleaningEvent.installation_id==site.id).order_by(CleaningEvent.timestamp.desc()).limit(100)).all()
    return [{"id":r.id,"timestamp":r.timestamp.isoformat(),"notes":r.notes,"source":r.source,"simulation_session_id":r.simulation_session_id,"estimated_recovery_kw":r.estimated_recovery_kw} for r in rows]

@app.get("/api/datasets")
def datasets(db:Session=Depends(get_db)):
    runs=db.scalars(select(DatasetRun).order_by(DatasetRun.created_at.desc()).limit(20)).all()
    return {"sources":[{"name":"NREL PVDAQ / OEDI Data Lake","status":"available","format":"CSV and Parquet on public S3","url":"https://data.openei.org/submissions/4568","note":"Measured PV data; selective access only."},{"name":"Open-Meteo Historical Weather","status":"available","url":"https://archive-api.open-meteo.com/v1/archive","note":"Historical weather covariates, not production labels."},{"name":"PVGIS","status":"available","url":"https://re.jrc.ec.europa.eu/api/","note":"Independent modeled solar resource; never treated as measured output."}],"runs":[{"source":r.source,"status":r.status,"records":r.records,"message":r.message,"created_at":r.created_at.isoformat()} for r in runs]}

@app.post("/api/datasets/acquire")
def acquire(db:Session=Depends(get_db)):
    from app.services.datasets.pvdaq import acquire_sample
    try:
        raw=os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..","data/raw/pvdaq"))
        discovered,files=acquire_sample(raw)
        message=f"Discovered {len(discovered)} objects; downloaded {len(files)} small metadata files. No generation records have been trained yet."
        run=DatasetRun(source="NREL PVDAQ",status="metadata_acquired" if files else "no_files",records=0,message=message)
        db.add(run);db.commit();return {"status":run.status,"objects_discovered":len(discovered),"files_downloaded":len(files),"records":0,"message":message}
    except Exception as exc:
        run=DatasetRun(source="NREL PVDAQ",status="unavailable",message=str(exc));db.add(run);db.commit()
        raise HTTPException(503,f"PVDAQ acquisition failed; no substitute dataset was used: {exc}")
