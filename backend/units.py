"""Unit conversion for the /solve endpoint.

CoolProp works in SI base units (Pa, K, J/kg, J/kg/K). The frontend sends
values in whatever unit system is selected, so every value is converted to SI
on the way in and back to display units on the way out.

    SI value = display value * scale + offset

`system` is either a preset name ("SI", "EN") or, from the current frontend,
a dict of the units the user picked: { key: [scale, offset, label] }.
"""

_FACTORS = {
    "SI": {
        "P": (1000.0, 0.0),            # kPa
        "T": (1.0, 273.15),            # degC
        "h": (1000.0, 0.0),            # kJ/kg
        "s": (1000.0, 0.0),            # kJ/kg.K
        "m": (1.0, 0.0),               # kg/s
        "x": (1.0, 0.0),               # quality, dimensionless
        "V": (1.0, 0.0),               # m/s
        "vol": (1.0, 0.0),             # m^3/s
        "dT": (1.0, 0.0),              # temperature difference, degC
        "z": (1.0, 0.0),               # m
        "v": (1.0, 0.0),               # m^3/kg
        "mass": (1.0, 0.0),            # kg
        "E": (1000.0, 0.0),            # kJ
        "bore": (0.001, 0.0),          # mm
        "disp": (1e-6, 0.0),           # cm^3
        "A": (1.0, 0.0),               # m^2 (flow area)
        "w": (1.0, 0.0),               # humidity ratio, kg/kg dry air
        "RH": (0.01, 0.0),             # relative humidity, %
    },
    "EN": {
        "P": (6894.757293168, 0.0),    # psi (absolute, psia)
        "T": (5.0 / 9.0, 459.67 * 5.0 / 9.0),  # degF
        "h": (2326.0, 0.0),            # Btu/lbm (IT)
        "s": (4186.8, 0.0),            # Btu/lbm.R
        "m": (0.45359237, 0.0),        # lbm/s
        "x": (1.0, 0.0),
        "V": (0.3048, 0.0),            # ft/s
        "vol": (0.028316846592, 0.0),  # ft^3/s
        "dT": (5.0 / 9.0, 0.0),        # temperature difference, degF
        "z": (0.3048, 0.0),            # ft
        "v": (0.028316846592 / 0.45359237, 0.0),  # ft^3/lbm
        "mass": (0.45359237, 0.0),     # lbm
        "E": (1055.05585262, 0.0),     # Btu
        "bore": (0.0254, 0.0),         # in
        "disp": (0.0254 ** 3, 0.0),    # in^3
        "A": (0.09290304, 0.0),        # ft^2
        "w": (1.0, 0.0),
        "RH": (0.01, 0.0),
    },
}

LABELS = {
    "SI": {"P": "kPa", "T": "°C", "h": "kJ/kg", "s": "kJ/kg·K", "m": "kg/s", "x": "-", "V": "m/s",
           "vol": "m³/s", "dT": "°C", "z": "m", "v": "m³/kg", "mass": "kg", "E": "kJ", "bore": "mm", "disp": "cm³",
           "A": "m²"},
    "EN": {"P": "psi", "T": "°F", "h": "Btu/lbm", "s": "Btu/lbm·R", "m": "lbm/s", "x": "-", "V": "ft/s",
           "vol": "ft³/s", "dT": "°F", "z": "ft", "v": "ft³/lbm", "mass": "lbm", "E": "Btu", "bore": "in", "disp": "in³",
           "A": "ft²"},
}

# Power unit = (enthalpy unit) x (mass-flow unit): kW in SI, Btu/s in English.
POWER_LABELS = {"SI": "kW", "EN": "Btu/s"}


def _factor(system, key):
    if isinstance(system, dict):
        if key not in system and key in ("V", "z", "v", "mass"):
            return 1.0, 0.0                 # older frontends sent no length unit: SI
        if key not in system and key == "E":
            return 1000.0, 0.0              # kJ
        if key not in system and key == "bore":
            return 0.001, 0.0               # mm
        if key not in system and key == "disp":
            return 1e-6, 0.0                # cm³
        if key not in system and key == "w":
            return 1.0, 0.0                 # humidity ratio, kg water per kg dry air
        if key not in system and key == "RH":
            return 0.01, 0.0                # relative humidity, typed in %
        if key not in system and key == "A":
            return _factor(system, "z")[0] ** 2, 0.0   # flow area: the length unit squared
        try:
            scale, offset = system[key][:2]
            return float(scale), float(offset)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"No unit given for '{key}'")
    try:
        return _FACTORS[system][key]
    except KeyError:
        raise ValueError(f"Unknown unit system '{system}' or property '{key}'")


def to_si(system, key, value):
    scale, offset = _factor(system, key)
    return value * scale + offset


def from_si(system, key, value):
    scale, offset = _factor(system, key)
    return (value - offset) / scale


def label(system, key):
    if isinstance(system, dict):
        if key == "A" and key not in system:
            return f"{label(system, 'z') or 'm'}²"
        entry = system.get(key)
        return str(entry[2]) if entry and len(entry) > 2 else ""
    if key == "power":
        return POWER_LABELS.get(system, "")
    return LABELS.get(system, {}).get(key, "")


def power_factor(system):
    """Watts per display power unit (kW, MW, Btu/s, ...)."""
    if isinstance(system, dict) and "power" in system:
        return _factor(system, "power")[0]
    return _factor(system, "h")[0] * _factor(system, "m")[0]