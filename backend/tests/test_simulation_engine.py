import pytest
from datetime import datetime, timezone

from app.models.records import Installation
from app.services.prediction.engine import predict_installation_power
from app.services.production.units import power_to_energy_kwh
from app.services.simulation.engine import scenario_losses, telemetry_values


def test_dust_accumulates_gradually_and_clean_sky_is_not_a_dust_signal():
    early = scenario_losses("dust", 30, 120)
    late = scenario_losses("dust", 120, 120)
    cloudy_normal = scenario_losses("normal", 120, 120)
    assert early[0] == pytest.approx(0.0875)
    assert late[0] == pytest.approx(0.35)
    assert cloudy_normal == (0.0, 0.0, False)


def test_scenario_outputs_are_bounded_and_faults_zero_ac_output():
    site = Installation(capacity_kw=10, timezone="UTC", latitude=11, longitude=76)
    prediction = predict_installation_power(site, 800, 25, datetime(2026, 11, 17, 10, tzinfo=timezone.utc), dni_wm2=700, dhi_wm2=100, wind_speed_kmh=5)
    normal = telemetry_values(site, prediction, 25, "normal", 30, 120)
    shaded = telemetry_values(site, prediction, 25, "shading", 30, 120)
    degraded = telemetry_values(site, prediction, 25, "inverter_degradation", 30, 120)
    failed = telemetry_values(site, prediction, 25, "inverter_failure", 30, 120)
    assert 0 <= normal["ac_power"][0] <= 10 * 0.96
    assert shaded["ac_power"][0] < normal["ac_power"][0]
    assert degraded["ac_power"][0] < normal["ac_power"][0]
    assert failed["ac_power"][0] == 0
    assert failed["dc_power"][0] > 0


def test_interval_energy_uses_power_and_selected_sampling_interval():
    assert power_to_energy_kwh(4, 15) == pytest.approx(1)
    assert power_to_energy_kwh(4, 30) == pytest.approx(2)
