import time
from typing import Any, Optional

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CACHE_TTL_SECONDS = 6 * 60 * 60
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_location_cache: dict[int, tuple[float, dict[str, Any]]] = {}


async def search_locations(query: str) -> list[dict[str, Any]]:
    normalized = " ".join(query.split()).casefold()
    if len(normalized) < 2:
        return []

    now = time.monotonic()
    cached = _cache.get(normalized)
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            GEOCODING_URL,
            params={"name": query.strip(), "count": 10, "language": "en", "format": "json"},
        )
        response.raise_for_status()
        payload = response.json()

    locations = []
    for result in payload.get("results") or []:
        if not isinstance(result.get("id"), int):
            continue
        item = {
            "id": result["id"],
            "city": result.get("name", ""),
            "administrative_region": result.get("admin1") or result.get("admin2") or "",
            "country": result.get("country", ""),
            "latitude": float(result["latitude"]),
            "longitude": float(result["longitude"]),
            "timezone": result.get("timezone") or "",
            "population": result.get("population"),
        }
        locations.append(item)
        _location_cache[item["id"]] = (now, item)

    _cache[normalized] = (now, locations)
    if len(_cache) > 256:
        expired = [key for key, (created, _) in _cache.items() if now - created >= CACHE_TTL_SECONDS]
        for key in expired:
            _cache.pop(key, None)
    if len(_location_cache) > 1024:
        expired = [key for key, (created, _) in _location_cache.items() if now - created >= CACHE_TTL_SECONDS]
        for key in expired:
            _location_cache.pop(key, None)
    return locations


async def resolve_location(location_id: int, city_hint: str = "") -> Optional[dict[str, Any]]:
    cached = _location_cache.get(location_id)
    now = time.monotonic()
    if cached and now - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    if city_hint.strip():
        for item in await search_locations(city_hint):
            if item["id"] == location_id:
                return item
    return None
