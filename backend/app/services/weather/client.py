import httpx
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

async def forecast(lat: float, lon: float):
    params = {"latitude":lat,"longitude":lon,"current":"temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,precipitation,is_day,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,direct_radiation","hourly":"temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m,precipitation,sunshine_duration,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,direct_radiation","forecast_days":2,"timezone":"auto"}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        return response.json()

async def historical_weather(lat: float, lon: float, timestamp: datetime, timezone_name: str):
    try:
        zone = ZoneInfo(timezone_name)
    except Exception:
        zone = timezone.utc
    local_time = timestamp.astimezone(zone) if timestamp.tzinfo else timestamp.replace(tzinfo=zone)
    now = datetime.now(zone)
    is_recent = now - timedelta(days=7) <= local_time <= now + timedelta(days=2)
    if is_recent:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {"latitude": lat, "longitude": lon, "hourly": "temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,relative_humidity_2m,cloud_cover,wind_speed_10m,precipitation", "timezone": timezone_name or "auto", "past_days": 7, "forecast_days": 2}
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {"latitude": lat, "longitude": lon, "start_date": local_time.date().isoformat(), "end_date": local_time.date().isoformat(), "hourly": "temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,relative_humidity_2m,cloud_cover,wind_speed_10m,precipitation", "timezone": timezone_name or "auto"}

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise ValueError("No hourly weather records are available for this timestamp.")
    target = local_time.replace(tzinfo=None)
    parsed_times = [datetime.fromisoformat(value) for value in times]
    index = min(range(len(parsed_times)), key=lambda i: abs((parsed_times[i] - target).total_seconds()))
    if abs((parsed_times[index] - target).total_seconds()) > 30 * 60:
        raise ValueError("No weather interval is close enough to the entered production time.")
    irradiance = hourly.get("shortwave_radiation", [])[index]
    temperature = hourly.get("temperature_2m", [])[index]
    if irradiance is None or temperature is None:
        raise ValueError("Weather data is missing for the selected production interval.")
    def value(name):
        values = hourly.get(name, [])
        return values[index] if index < len(values) else None
    return {"timestamp": times[index], "irradiance": float(irradiance), "ghi": float(irradiance), "dni": value("direct_normal_irradiance"), "dhi": value("diffuse_radiation"), "temperature": float(temperature), "humidity": value("relative_humidity_2m"), "cloud_cover": value("cloud_cover"), "wind_speed": value("wind_speed_10m"), "precipitation": value("precipitation"), "timezone": payload.get("timezone") or timezone_name, "source": "Open-Meteo forecast hourly" if is_recent else "Open-Meteo archive hourly"}
