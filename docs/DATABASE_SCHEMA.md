# Database schema

- `installations`: city, administrative region, country, timezone, internally stored coordinates, name, rated capacity and model parameters. Existing tables receive nullable location metadata columns at startup so legacy coordinates remain intact.
- `observations`: timestamp, expected and actual values, irradiance, temperature, provenance, classification.
- `dataset_runs`: source, acquisition outcome, record count, message, timestamp.
- `cleaning_events`: timestamp, user note and origin.

Current tables are initialized by SQLAlchemy metadata at startup. Alembic revision files have not yet been added.
