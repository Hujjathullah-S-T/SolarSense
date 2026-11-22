"""Deterministic, bounded helpers for software-only solar device simulation."""
from datetime import datetime
import math
from app.models.records import Installation
from app.services.solar.physics import dc_to_ac


SCENARIOS = {"normal", "dust", "shading", "inverter_degradation", "inverter_failure", "weather_changes"}
DUST_MAX_LOSS_FRACTION = 0.35
SHADING_LOSS_FRACTION = 0.375
INVERTER_DEGRADATION_FRACTION = 0.175


def scenario_losses(scenario: str, elapsed_minutes: float, duration_minutes: float) -> tuple[float, float, bool]:
    """Return fixed, documented scenario losses without user-adjustable severity."""
    progress = min(1.0, max(0.0, elapsed_minutes / max(1.0, duration_minutes)))
    if scenario == "dust":
        return DUST_MAX_LOSS_FRACTION * progress, 0.0, False
    if scenario == "shading":
        return 0.0, SHADING_LOSS_FRACTION, False
    if scenario == "inverter_failure":
        return 0.0, 0.0, True
    return 0.0, 0.0, False


def telemetry_values(site: Installation, prediction, ambient_c: float,
                     scenario: str, elapsed_minutes: float, duration_minutes: float) -> dict[str, tuple[float, str]]:
    if not all(math.isfinite(float(x)) for x in (site.capacity_kw, prediction.expected_power_kw, ambient_c)):
        raise ValueError("Telemetry inputs must be finite numbers.")
    soiling, shading, failed = scenario_losses(scenario, elapsed_minutes, duration_minutes)
    baseline_efficiency = float(site.inverter_efficiency if site.inverter_efficiency is not None else 0.96)
    degraded_efficiency = baseline_efficiency * (1 - (INVERTER_DEGRADATION_FRACTION if scenario == "inverter_degradation" else 0.0))
    actual_dc_kw = max(0.0, prediction.dc_power_kw * (1 - soiling) * (1 - shading))
    actual_kw = dc_to_ac(site, actual_dc_kw, degraded_efficiency)
    if failed:
        actual_kw = 0.0
    return {
        "irradiance": (prediction.ghi_wm2, "W/m2"),
        "dni": (prediction.dni_wm2, "W/m2"),
        "dhi": (prediction.dhi_wm2, "W/m2"),
        "poa_irradiance": (prediction.poa_global_wm2, "W/m2"),
        "solar_zenith": (prediction.solar_zenith_deg, "deg"),
        "solar_azimuth": (prediction.solar_azimuth_deg, "deg"),
        "ambient_temperature": (ambient_c, "C"),
        "panel_temperature": (prediction.cell_temperature_c, "C"),
        "dc_power": (actual_dc_kw, "kW"),
        "ac_power": (actual_kw, "kW"),
        "inverter_efficiency": (degraded_efficiency, "ratio"),
        "soiling_loss": (soiling, "ratio"),
        "shading_loss": (shading, "ratio"),
        "expected_dc_power": (prediction.dc_power_kw, "kW"),
        "expected_power": (prediction.expected_power_kw, "kW"),
    }

