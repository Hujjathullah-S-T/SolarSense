from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.models.records import Installation
from app.services.solar.physics import SolarPower, calculate_solar_power


@dataclass(frozen=True)
class PowerPrediction:
    expected_power_kw: float
    method: str
    model_version: Optional[str]
    pipeline_version: str = "pvlib-pvwatts-v1"
    dc_power_kw: float = 0.0
    ghi_wm2: float = 0.0
    dni_wm2: float = 0.0
    dhi_wm2: float = 0.0
    poa_global_wm2: float = 0.0
    solar_zenith_deg: float = 0.0
    solar_azimuth_deg: float = 0.0
    cell_temperature_c: float = 0.0
    irradiance_method: str = ""
    assumptions: tuple[str, ...] = ()


def _prediction(result: SolarPower) -> PowerPrediction:
    return PowerPrediction(
        expected_power_kw=result.ac_power_kw,
        method="pvlib_pvwatts",
        model_version=None,
        dc_power_kw=result.dc_power_kw,
        ghi_wm2=result.ghi_wm2,
        dni_wm2=result.dni_wm2,
        dhi_wm2=result.dhi_wm2,
        poa_global_wm2=result.poa_global_wm2,
        solar_zenith_deg=result.solar_zenith_deg,
        solar_azimuth_deg=result.solar_azimuth_deg,
        cell_temperature_c=result.cell_temperature_c,
        irradiance_method=result.irradiance_method,
        assumptions=result.assumptions,
    )


def predict_installation_power(
    site: Installation,
    irradiance_wm2: Optional[float],
    ambient_temperature_c: Optional[float],
    when: Optional[datetime] = None,
    dni_wm2: Optional[float] = None,
    dhi_wm2: Optional[float] = None,
    wind_speed_kmh: Optional[float] = None,
) -> PowerPrediction:
    """Physics-only expected AC power. Trained model artifacts are not loaded here."""
    capacity_kw = float(site.capacity_kw if site.capacity_kw is not None else 5.0)
    efficiency = float(site.inverter_efficiency if site.inverter_efficiency is not None else 0.96)
    losses = float(site.losses if site.losses is not None else 0.14)
    if not (capacity_kw > 0 and capacity_kw <= 1_000_000):
        raise ValueError("Installation capacity must be greater than 0 and no more than 1,000,000 kWp.")
    if not (0 < efficiency <= 1):
        raise ValueError("Inverter efficiency must be greater than 0 and at most 1.")
    if not (0 <= losses < 1):
        raise ValueError("System losses must be from 0 up to (but not including) 1.")
    inverter_capacity_kw = getattr(site, "inverter_capacity_kw", None)
    if inverter_capacity_kw is not None and not (0 < float(inverter_capacity_kw) <= 1_000_000):
        raise ValueError("Inverter capacity must be greater than 0 and no more than 1,000,000 kW.")
    tilt = float(site.tilt if site.tilt is not None else 25.0)
    azimuth = float(site.azimuth if site.azimuth is not None else 180.0)
    latitude = float(site.latitude if site.latitude is not None else 37.7749)
    longitude = float(site.longitude if site.longitude is not None else -122.4194)
    if not 0 <= tilt <= 90:
        raise ValueError("Panel tilt must be between 0 and 90 degrees.")
    if not 0 <= azimuth <= 360:
        raise ValueError("Panel azimuth must be between 0 and 360 degrees (0=N, 180=S).")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Installation latitude or longitude is invalid.")
    result = calculate_solar_power(
        site, irradiance_wm2, ambient_temperature_c, when,
        dni_wm2=dni_wm2, dhi_wm2=dhi_wm2, wind_speed_kmh=wind_speed_kmh,
    )
    return _prediction(result)
