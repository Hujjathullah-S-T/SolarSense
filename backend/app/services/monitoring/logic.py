def classify(expected: float, actual: float, irradiance: float, daylight: bool=True):
    if not daylight or irradiance < 25: return "night", "No daylight production expected."
    if expected <= .05: return "normal", "Expected production is negligible for current conditions."
    ratio = actual / expected
    if actual < .02 and irradiance > 300: return "possible_system_fault", "Production is near zero under strong irradiance; inspect inverter, AC disconnect, and wiring."
    if ratio < .55: return "underperforming", "Production is materially below weather-adjusted expectation. Check shading, inverter alerts, and panel cleanliness."
    if ratio < .8: return "watch", "Production is below expectation; continue monitoring before taking action."
    return "normal", "Production is within the expected weather-adjusted range."
