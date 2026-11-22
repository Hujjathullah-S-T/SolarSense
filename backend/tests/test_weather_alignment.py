from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import asyncio

from app.services.weather import client as weather_client


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        date = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
        return {"timezone": "Asia/Kolkata", "hourly": {
            "time": [f"{date}T09:00", f"{date}T10:00", f"{date}T11:00"],
            "shortwave_radiation": [100, 500, 900],
            "direct_normal_irradiance": [90, 400, 700],
            "diffuse_radiation": [10, 100, 200],
            "temperature_2m": [20, 25, 30],
            "relative_humidity_2m": [40, 50, 60],
            "cloud_cover": [80, 40, 10],
            "wind_speed_10m": [3, 5, 7],
            "precipitation": [0, 0.2, 0],
        }}


class FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def get(self, *args, **kwargs):
        return FakeResponse()


def test_historical_weather_uses_nearest_hour_in_installation_timezone(monkeypatch):
    monkeypatch.setattr(weather_client.httpx, "AsyncClient", FakeAsyncClient)
    local_reading_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(hour=10, minute=35, second=0, microsecond=0)
    result = asyncio.run(weather_client.historical_weather(11.0, 76.0, local_reading_time, "Asia/Kolkata"))
    assert result["timestamp"].endswith("T11:00")
    assert result["irradiance"] == 900
    assert result["ghi"] == 900
    assert result["dni"] == 700
    assert result["dhi"] == 200
    assert result["temperature"] == 30
    assert result["humidity"] == 60
    assert result["cloud_cover"] == 10
    assert result["wind_speed"] == 7
    assert result["precipitation"] == 0
    assert result["timezone"] == "Asia/Kolkata"
