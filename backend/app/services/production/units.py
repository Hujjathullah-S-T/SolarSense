def interval_hours(interval_minutes: int) -> float:
    if not 1 <= int(interval_minutes) <= 1440:
        raise ValueError("Interval duration must be between 1 and 1440 minutes.")
    return int(interval_minutes) / 60.0


def power_to_energy_kwh(power_kw: float, interval_minutes: int) -> float:
    """Convert interval-average power into interval energy exactly once."""
    return float(power_kw) * interval_hours(interval_minutes)


def energy_to_average_power_kw(energy_kwh: float, interval_minutes: int) -> float:
    """Convert measured interval energy into average power for comparison."""
    return float(energy_kwh) / interval_hours(interval_minutes)
