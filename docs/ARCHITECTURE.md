# Architecture

```mermaid
flowchart TB
  UI[React + TypeScript] --> API[FastAPI REST]
  API --> SQL[(SQLAlchemy: SQLite / PostgreSQL)]
  API --> WX[Open-Meteo]
  API --> PV[pvlib PVWatts physics engine]
  API --> S3[Anonymous OEDI PVDAQ]
  API --> MON[Monitoring and diagnosis]
  UI --> CTRL[Simulation control center]
  CTRL --> SIM[Timestamped virtual solar plant]
  SIM --> GW[Software telemetry gateway]
  GW --> API
  SIM --> PRED[Shared prediction pipeline]
  API --> DEVICE[Virtual panels, inverter, environment]
```

Weather and PVDAQ are external dependencies. Each simulation session stores its configuration, scenario, cursor, and status. Generated device telemetry retains a device ID, installation ID, source, timestamp, measurement, value, unit, and validity flag. Simulated observations keep interval power and energy in separate columns and link back to the session. CSV/manual observations remain separately tagged as measured input. The simulation gateway uses REST; no physical devices or broker are required.
