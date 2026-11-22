"""pvlib-based fixed-array production model used by every prediction workflow.

Equipment assumptions are PVWatts crystalline-silicon defaults: gamma_pdc=-0.0037 1/C,
SAPM open-rack glass/polymer temperature coefficients, 0.20 ground albedo, and
Hay-Davies sky transposition. Installation capacity is the DC STC nameplate in kWp.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pvlib

from app.models.records import Installation

PVWATTS_GAMMA_PDC = -0.0037
GROUND_ALBEDO = 0.20
TEMPERATURE_MODEL = "open_rack_glass_polymer"
TEMPERATURE_COEFFICIENTS = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][TEMPERATURE_MODEL]

def _value(site: Installation, name: str, default: float) -> float:
    value = getattr(site, name, None)
    return float(default if value is None else value)


@dataclass(frozen=True)
class SolarPower:
    ghi_wm2: float
    dni_wm2: float
    dhi_wm2: float
    poa_global_wm2: float
    solar_zenith_deg: float
    solar_azimuth_deg: float
    cell_temperature_c: float
    dc_power_kw: float
    ac_power_kw: float
    irradiance_method: str
    assumptions: tuple[str, ...]


def _finite_nonnegative(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, nonnegative value.")
    return value


def dc_to_ac(site: Installation, dc_power_kw: float, inverter_efficiency: Optional[float] = None) -> float:
    """Convert DC kW through pvlib's PVWatts inverter and enforce its AC nameplate."""
    efficiency = float(inverter_efficiency if inverter_efficiency is not None else _value(site, "inverter_efficiency", 0.96))
    if not 0 < efficiency <= 1:
        raise ValueError("Inverter efficiency must be greater than 0 and at most 1.")
    inverter_ac_kw = float(getattr(site, "inverter_capacity_kw", None) or _value(site, "capacity_kw", 5.0) * efficiency)
    if inverter_ac_kw <= 0:
        raise ValueError("Inverter capacity must be greater than 0 kW.")
    inverter_dc_w = inverter_ac_kw * 1000.0 / efficiency
    ac_w = pvlib.inverter.pvwatts(max(0.0, float(dc_power_kw)) * 1000.0, inverter_dc_w,
                                  eta_inv_nom=efficiency)
    if not math.isfinite(float(ac_w)):
        return 0.0
    return max(0.0, min(float(ac_w) / 1000.0, inverter_ac_kw))


def calculate_solar_power(
    site: Installation,
    ghi_wm2: Optional[float],
    ambient_temperature_c: Optional[float],
    when: Optional[datetime],
    dni_wm2: Optional[float] = None,
    dhi_wm2: Optional[float] = None,
    wind_speed_kmh: Optional[float] = None,
) -> SolarPower:
    """Calculate solar position, POA, cell temperature, DC and AC power.

    GHI and ambient temperature are required weather observations. If either DNI or
    DHI is absent, pvlib's Erbs decomposition is used from measured GHI; GHI is never
    synthesized. Wind is converted from Open-Meteo km/h to the SAPM input m/s.
    """
    ghi = _finite_nonnegative("GHI", ghi_wm2)
    if ghi is None:
        raise ValueError("Open-Meteo GHI is missing; expected power cannot be calculated.")
    if ambient_temperature_c is None or not math.isfinite(float(ambient_temperature_c)):
        raise ValueError("Open-Meteo ambient temperature is missing or invalid.")
    ambient = float(ambient_temperature_c)
    dni = _finite_nonnegative("DNI", dni_wm2)
    dhi = _finite_nonnegative("DHI", dhi_wm2)
    wind = _finite_nonnegative("wind speed", wind_speed_kmh)
    assumptions = []
    if wind is None:
        wind_ms = 0.0
        assumptions.append("Wind speed unavailable; SAPM temperature model uses 0 m/s.")
    else:
        wind_ms = wind / 3.6
    if when is None:
        when = datetime.now(timezone.utc)
    try:
        site_zone = ZoneInfo(site.timezone) if site.timezone else timezone.utc
    except Exception:
        site_zone = timezone.utc
    if when.tzinfo is None:
        when = when.replace(tzinfo=site_zone)
    timestamp = pd.DatetimeIndex([when])
    position = pvlib.solarposition.get_solarposition(timestamp, _value(site, "latitude", 37.7749), _value(site, "longitude", -122.4194))
    zenith = float(position["zenith"].iloc[0])
    azimuth = float(position["azimuth"].iloc[0])

    irradiance_method = "Open-Meteo GHI, DNI and DHI"
    if dni is None and dhi is None:
        erbs = pvlib.irradiance.erbs(ghi, zenith, timestamp[0])
        dni = float(erbs["dni"])
        dhi = float(erbs["dhi"])
        irradiance_method = "Open-Meteo GHI with pvlib Erbs decomposition"
    elif dni is None:
        dni = float(pvlib.irradiance.dni(ghi, dhi, zenith))
        irradiance_method = "Open-Meteo GHI and DHI with pvlib DNI derivation"
    elif dhi is None:
        dhi = max(0.0, ghi - dni * max(0.0, math.cos(math.radians(zenith))))
        irradiance_method = "Open-Meteo GHI and DNI with horizontal-component derivation"
    dni = max(0.0, float(dni))
    dhi = max(0.0, float(dhi))

    if zenith >= 90 or ghi <= 0:
        poa_global = 0.0
    else:
        dni_extra = float(pvlib.irradiance.get_extra_radiation(timestamp).iloc[0])
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=_value(site, "tilt", 25.0), surface_azimuth=_value(site, "azimuth", 180.0),
            solar_zenith=zenith, solar_azimuth=azimuth, dni=dni, ghi=ghi, dhi=dhi,
            dni_extra=dni_extra, albedo=GROUND_ALBEDO, model="haydavies",
        )
        poa_global = max(0.0, float(poa["poa_global"]))

    cell_temp = float(pvlib.temperature.sapm_cell(
        poa_global, ambient, wind_ms, **TEMPERATURE_COEFFICIENTS,
    ))
    capacity_w = _value(site, "capacity_kw", 5.0) * 1000.0
    dc_w = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=poa_global, temp_cell=cell_temp,
        pdc0=capacity_w, gamma_pdc=PVWATTS_GAMMA_PDC,
    )
    dc_w = max(0.0, float(dc_w)) * (1.0 - _value(site, "losses", 0.14))
    dc_kw = min(dc_w / 1000.0, _value(site, "capacity_kw", 5.0))
    ac_kw = dc_to_ac(site, dc_kw)
    return SolarPower(ghi, dni, dhi, poa_global, zenith, azimuth, cell_temp,
                      dc_kw, ac_kw, irradiance_method, tuple(assumptions))
