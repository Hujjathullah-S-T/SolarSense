# Physics-based prediction model

SolarSense uses `pvlib` as the runtime prediction engine. Saved Random Forest artifacts and model training remain available for evaluation, but they are not loaded by the prediction service and are not required to start a simulation.

## Inputs and weather alignment

The prediction service receives the installation latitude, longitude, timezone, timestamp, DC nameplate capacity, tilt, azimuth, inverter efficiency and capacity, and system-loss fraction. It receives Open-Meteo GHI, ambient temperature, and wind speed where available, plus DNI and DHI when returned by the selected weather API.

Historical simulation uses the nearest hourly record for its requested timestamp from the Open-Meteo archive or its recent forecast endpoint. Current-style simulation uses the current Open-Meteo observation. Open-Meteo solar radiation values are hourly means. The original weather timestamp and source are retained with the simulated record.

GHI and ambient temperature are required. Missing GHI is an explicit prediction error; the service does not replace it with zero or a guessed value. If both DNI and DHI are supplied, they are used as received. If exactly one is missing, pvlib derives the missing horizontal component. If both are missing, pvlib's Erbs decomposition estimates both from the observed GHI, solar zenith and timestamp. The API reports which decomposition was used. Missing wind is handled with a 0 m/s input to SAPM and included as an explicit assumption in the prediction response.

## Calculation sequence

1. **Solar position:** `pvlib.solarposition.get_solarposition` calculates geometric zenith and azimuth for the site and timestamp.
2. **Plane-of-array irradiance:** `pvlib.irradiance.get_total_irradiance` transposes GHI, DNI and DHI onto the configured fixed array using the Hay-Davies sky-diffuse model. Ground albedo is 0.20.
3. **Cell temperature:** `pvlib.temperature.sapm_cell` uses the open-rack glass/polymer SAPM coefficients (`a = -3.56`, `b = -0.075`, `deltaT = 3`), ambient temperature, plane-of-array irradiance, and wind speed.
4. **DC power:** `pvlib.pvsystem.pvwatts_dc` uses DC nameplate capacity as `pdc0`, effective irradiance as plane-of-array irradiance, a reference temperature of 25 °C, and the default crystalline-silicon power coefficient `gamma_pdc = -0.0037 / °C`. Configured system losses are applied to DC output.
5. **AC power:** `pvlib.inverter.pvwatts` converts DC power using configured nominal inverter efficiency. The output is capped at configured inverter AC nameplate; if none is configured, it is derived as DC capacity × nominal inverter efficiency.
6. **Interval energy:** expected or simulated average AC power is converted once using `energy_kWh = power_kW × interval_minutes / 60`.

The core equations are:

```text
POA_global = Hay-Davies( GHI, DNI, DHI, solar_zenith, solar_azimuth,
                         panel_tilt, panel_azimuth, extraterrestrial_DNI, albedo )
T_module   = POA_global × exp(a + b × wind_speed_m/s) + ambient_temperature
T_cell     = T_module + (POA_global / 1000) × deltaT
Pdc_raw    = Pdc0 × (POA_global / 1000) × [1 + gamma_pdc × (T_cell - 25°C)]
Pdc        = max(0, Pdc_raw) × (1 - configured_system_losses)
Pac        = pvlib PVWatts inverter(Pdc, nominal_efficiency, inverter_AC_nameplate)
E_interval = Pac × interval_minutes / 60
```

`pvlib.pvsystem.pvwatts_dc` implements the DC equation and clamps negative output to zero in SolarSense. `pvlib.inverter.pvwatts` implements the nonlinear inverter conversion and its nameplate clipping. The app reports those powers in kW and energy in kWh.

The fixed-array azimuth convention follows pvlib: 0° north, 90° east, 180° south, and 270° west. DC capacity is the module array STC rating in kWp; panel efficiency is installation metadata and is not applied a second time to an already-rated kWp capacity.

## Scenario handling

The physics model first calculates the expected weather-adjusted DC and AC baseline. Dust, shading, and degradation are applied only to the simulated actual path. Dust accumulates linearly to a fixed 35% maximum loss over a session and resets when simulated cleaning occurs. Shading applies a fixed 37.5% DC loss. Inverter degradation applies a fixed 17.5% relative reduction to configured conversion efficiency. Inverter failure sets actual AC output to zero while retaining baseline expected output. These deterministic scenario settings are illustrative simulation assumptions, not site measurements, and do not change the expected baseline.

## Assumptions and limitations

- The generic PVWatts crystalline-silicon temperature coefficient and SAPM open-rack glass/polymer thermal coefficients are used because the installation does not identify an exact module make/model or mounting specification.
- A fixed 0.20 ground albedo is assumed.
- Angle-of-incidence, spectral, mismatch, wiring, degradation, and detailed module electrical effects are not modeled separately. The configured system-loss fraction is a combined allowance for losses outside the modeled PVWatts DC and inverter stages.
- Open-Meteo weather is hourly resolution. A shorter simulation interval uses the weather record nearest that interval timestamp; it does not imply minute-resolution weather.
- PVWatts outputs are estimates for demonstration and planning; they are not a certified plant forecast or an inverter measurement.

The model uses pvlib's documented [Hay-Davies transposition](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.irradiance.get_total_irradiance.html), [SAPM cell-temperature model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.temperature.sapm_cell.html), [PVWatts DC model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html), and [PVWatts inverter model](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.inverter.pvwatts.html). Open-Meteo definitions for [forecast weather variables](https://open-meteo.com/en/docs) and [historical radiation variables](https://open-meteo.com/en/docs/historical-weather-api) define the weather inputs and time resolution.
