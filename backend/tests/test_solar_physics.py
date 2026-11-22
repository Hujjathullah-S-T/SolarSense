from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.models.records import Installation
from app.services.prediction.engine import predict_installation_power
from app.services.solar.physics import calculate_solar_power


def site(**kwargs):
    values = {"capacity_kw": 8, "latitude": 11, "longitude": 76, "timezone": "Asia/Kolkata",
              "tilt": 25, "azimuth": 180, "inverter_efficiency": .96, "inverter_capacity_kw": 7.5,
              "losses": .14}
    values.update(kwargs)
    return Installation(**values)


def test_pvlib_calculates_position_poa_dc_and_ac():
    when = datetime(2026, 11, 17, 12, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = calculate_solar_power(site(), 850, 28, when, 800, 100, 8)
    assert 0 <= result.solar_zenith_deg < 90
    assert 0 <= result.solar_azimuth_deg <= 360
    assert result.poa_global_wm2 > 0
    assert 0 < result.dc_power_kw <= 8
    assert 0 < result.ac_power_kw <= 7.5


def test_panel_orientation_changes_plane_of_array_irradiance():
    when = datetime(2026, 11, 17, 9, tzinfo=ZoneInfo("Asia/Kolkata"))
    weather = (850, 28, when, 800, 100, 8)
    south = calculate_solar_power(site(azimuth=180), *weather)
    east = calculate_solar_power(site(azimuth=90), *weather)
    assert south.poa_global_wm2 != pytest.approx(east.poa_global_wm2)


def test_cell_temperature_reduces_dc_power_at_hotter_ambient_temperature():
    when = datetime(2026, 11, 17, 12, tzinfo=ZoneInfo("Asia/Kolkata"))
    weather = (900, when, 850, 50, 5)
    cool = calculate_solar_power(site(), weather[0], 20, weather[1], weather[2], weather[3], weather[4])
    hot = calculate_solar_power(site(), weather[0], 45, weather[1], weather[2], weather[3], weather[4])
    assert hot.cell_temperature_c > cool.cell_temperature_c
    assert hot.dc_power_kw < cool.dc_power_kw


def test_missing_dni_and_dhi_are_derived_from_real_ghi_with_erbs():
    when = datetime(2026, 11, 17, 12, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = calculate_solar_power(site(), 700, 25, when, wind_speed_kmh=4)
    assert result.ghi_wm2 == 700
    assert result.dni_wm2 >= 0
    assert result.dhi_wm2 >= 0
    assert result.irradiance_method == "Open-Meteo GHI with pvlib Erbs decomposition"


def test_night_produces_zero_dc_and_ac_without_fabricated_irradiance():
    when = datetime(2026, 11, 17, 2, tzinfo=ZoneInfo("Asia/Kolkata"))
    result = calculate_solar_power(site(), 0, 22, when, 0, 0, 3)
    assert result.poa_global_wm2 == 0
    assert result.dc_power_kw == 0
    assert result.ac_power_kw == 0


def test_weather_missing_ghi_is_explicit_error():
    when = datetime(2026, 11, 17, 12, tzinfo=ZoneInfo("Asia/Kolkata"))
    with pytest.raises(ValueError, match="GHI is missing"):
        predict_installation_power(site(), None, 25, when)
