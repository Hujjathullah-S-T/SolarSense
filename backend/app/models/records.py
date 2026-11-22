from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, Float, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from app.database.session import Base

class Installation(Base):
    __tablename__ = "installations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), default="My solar installation")
    city: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    administrative_region: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    timezone: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    latitude: Mapped[float] = mapped_column(Float, default=37.7749)
    longitude: Mapped[float] = mapped_column(Float, default=-122.4194)
    capacity_kw: Mapped[float] = mapped_column(Float, default=5.0)
    tilt: Mapped[float] = mapped_column(Float, default=25)
    azimuth: Mapped[float] = mapped_column(Float, default=180)
    inverter_efficiency: Mapped[float] = mapped_column(Float, default=.96)
    losses: Mapped[float] = mapped_column(Float, default=.14)
    panel_count: Mapped[int] = mapped_column(Integer, default=13)
    panel_efficiency: Mapped[float] = mapped_column(Float, default=.20)
    inverter_capacity_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    expected_kwh: Mapped[float] = mapped_column(Float)
    actual_kwh: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(8), default="kWh", nullable=False)
    interval_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prediction_method: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    weather_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    weather_source: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    installation_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    simulation_session_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    raw_measurement_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    raw_measurement_unit: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)
    raw_measurement_components: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    measurement_semantics: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    scenario_loss_fraction: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scenario_fault: Mapped[Optional[bool]] = mapped_column(nullable=True)
    expected_power_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_power_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_energy_kwh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_energy_kwh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    requires_unit_review: Mapped[bool] = mapped_column(default=False, nullable=False)
    requires_interval_review: Mapped[bool] = mapped_column(default=False, nullable=False)
    irradiance: Mapped[float] = mapped_column(Float, default=0)
    temperature: Mapped[float] = mapped_column(Float, default=20)
    source: Mapped[str] = mapped_column(String(30), default="simulation")
    status: Mapped[str] = mapped_column(String(30), default="normal")

class DatasetRun(Base):
    __tablename__ = "dataset_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(30))
    records: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class CleaningEvent(Base):
    __tablename__ = "cleaning_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(30), default="user")
    installation_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    simulation_session_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    estimated_recovery_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

class VirtualDevice(Base):
    __tablename__ = "virtual_devices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    installation_id: Mapped[int] = mapped_column(Integer, index=True)
    device_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    device_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(24), default="online")
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

class SimulationSession(Base):
    __tablename__ = "simulation_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    installation_id: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(24), default="running")
    scenario: Mapped[str] = mapped_column(String(32), default="normal")
    scenario_severity: Mapped[float] = mapped_column(Float, default=0.0)
    start_time: Mapped[datetime] = mapped_column(DateTime)
    cursor_time: Mapped[datetime] = mapped_column(DateTime)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_minutes: Mapped[int] = mapped_column(Integer, default=60)
    interval_minutes: Mapped[int] = mapped_column(Integer, default=15)
    speed: Mapped[int] = mapped_column(Integer, default=1)
    config_json: Mapped[str] = mapped_column(Text, default="{}")

class TelemetryReading(Base):
    __tablename__ = "telemetry_readings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    device_id: Mapped[str] = mapped_column(String(80), index=True)
    installation_id: Mapped[int] = mapped_column(Integer, index=True)
    session_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    source: Mapped[str] = mapped_column(String(32), default="simulated")
    measurement: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16))
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
