# Verification

Run backend checks from `backend` with `PYTHONPATH=. .venv/bin/pytest -q` and frontend production compilation from `frontend` with `npm run build`. Location tests cover city results, duplicate names, persistence, weather coordinate routing, invalid queries and geocoding failure handling. Live API and database checks depend on installed dependencies and network availability. No external dataset row count or model metric is asserted unless an acquisition run actually produces it.
