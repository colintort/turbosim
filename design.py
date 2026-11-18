"""Design studio for /design: size the parts of a cycle one device at a time.

The parts share one design basis (the refrigerant, the capacity and the
operating temperatures). From it the ideal-with-losses vapor-compression
cycle is worked out first:

    1  compressor inlet     P_evap, T_evap + superheat
    2  compressor outlet    P_cond, from the isentropic efficiency
    3  condenser outlet     P_cond, T_cond - subcooling
    4  evaporator inlet     P_evap, h4 = h3 (throttling)

Each device is then sized for its duty in that cycle:

  evaporator / condenser  Heat exchanger split into zones (superheat and
                          boiling; desuperheat, condensing and subcooling).
                          Each zone is a counterflow LMTD problem
                          UA = Q / LMTD, with U from the inside and outside
                          film coefficients. The area gives the tube length.
                          "Fewest circuits" picks the fewest parallel tube
                          circuits whose refrigerant pressure drop costs no
                          more than the allowed saturation temperature drop.
  compressor              Reciprocating: the volumetric efficiency
                          eta_v = 1 + c - c (P2/P1)^(1/n) gives the displacement,
                          then the bore and stroke at the chosen speed.
  expansion               Capillary tube: adiabatic homogeneous two-phase flow
                          (Stoecker's step method) gives the length that drops
                          the liquid from P_cond to P_evap, and checks choking.

Everything is computed in SI and converted to the units the page shows.
"""
import math

from CoolProp.CoolProp import PropsSI

import convection as cv


class DesignError(Exception):
    pass


# Display units: key -> (label, scale, offset), SI value = shown * scale + offset.
UNITS = {
    "SI": {
        "T": ("°C", 1.0, 273.15), "dT": ("K", 1.0, 0.0), "P": ("kPa", 1e3, 0.0),
        "Q": ("kW", 1e3, 0.0), "W": ("kW", 1e3, 0.0), "m": ("kg/s", 1.0, 0.0),
        "L": ("m", 1.0, 0.0), "d": ("mm", 1e-3, 0.0), "A": ("m²", 1.0, 0.0),
        "U": ("W/m²·K", 1.0, 0.0), "UA": ("kW/K", 1e3, 0.0), "k": ("W/m·K", 1.0, 0.0),
        "vol": ("m³/h", 1 / 3600, 0.0), "V": ("m/s", 1.0, 0.0), "G": ("kg/m²·s", 1.0, 0.0),
        "h": ("kJ/kg", 1e3, 0.0), "s": ("kJ/kg·K", 1e3, 0.0), "cyl": ("cm³", 1e-6, 0.0),
        "dp": ("kPa", 1e3, 0.0), "R": ("m²·K/W", 1.0, 0.0),
    },
    "EN": {
        "T": ("°F", 5 / 9, 459.67 * 5 / 9), "dT": ("°F", 5 / 9, 0.0), "P": ("psia", 6894.757293168, 0.0),
        "Q": ("tons", 3516.8528, 0.0), "W": ("hp", 745.69987, 0.0), "m": ("lbm/min", 0.45359237 / 60, 0.0),
        "L": ("ft", 0.3048, 0.0), "d": ("in", 0.0254, 0.0), "A": ("ft²", 0.09290304, 0.0),
        "U": ("Btu/h·ft²·°F", 5.678263, 0.0), "UA": ("Btu/h·°F", 0.5275279, 0.0),
        "k": ("Btu/h·ft·°F", 1.730735, 0.0),
        "vol": ("cfm", 4.719474e-4, 0.0), "V": ("ft/s", 0.3048, 0.0), "G": ("lbm/ft²·s", 4.882428, 0.0),
        "h": ("Btu/lbm", 2326.0, 0.0), "s": ("Btu/lbm·R", 4186.8, 0.0), "cyl": ("in³", 0.0254 ** 3, 0.0),
        "dp": ("psi", 6894.757293168, 0.0), "R": ("h·ft²·°F/Btu", 0.17611, 0.0),
    },
}

REFRIGERANTS = ["R134a", "R1234yf", "R410A", "R32", "Propane", "Ammonia", "CO2"]

# Secondary (air or water) side of a heat exchanger.
SECONDARY = {
    "air": {"fluid": "Air", "P": 101325.0, "label": "Air"},
    "water": {"fluid": "Water", "P": 300e3, "label": "Water"},
}

# Exchanger types: what flows outside the tubes, and the finned-surface factor
# (fin efficiency x outside-to-bare-tube area ratio). The refrigerant is
# always inside the tubes. Film coefficients come from convection.py.
HX_TYPES = {
    "finCoil": {"label": "Finned-tube air coil", "secondary": "air", "fin": 15.0},
    "shellTube": {"label": "Shell-and-tube (water)", "secondary": "water", "fin": 1.0},
    "tubeInTube": {"label": "Tube-in-tube (water)", "secondary": "water", "fin": 1.0},
}
# Fouling factors R″_f, m²·K/W (Penoncello, Thermal Energy Systems, Table 5.2)
FOULING = {"refrigerant": 0.000176, "water": 0.000352, "air": 0.000176, "steam": 0.000088,
           "flueGas": 0.000088, "feedwater": 0.000088}


# What sets the velocity outside the tubes, for each exchanger type (mm, m/s):
# air velocity over a coil; shell diameter, baffle spacing, pitch ratio and
# layout of a shell-and-tube; the outer tube's inside diameter of a tube-in-tube.
# Blank = the SI value in HX_GEOMETRY_SI.
HX_GEOMETRY = {"Vair": "", "Ds": "", "B": "", "pitch": "", "layout": "triangular", "Da": ""}
HX_GEOMETRY_SI = {"Vair": 2.5, "Ds": 0.1, "B": 0.04, "pitch": 1.25, "Da": 0.0158}


def default_inputs():
    """The starting design, in SI units as shown (°C, kPa, kW, mm...).
    Close to Çengel's Example 11-1 refrigerator, made a little more real."""
    return {
        "system": "SI",
        "basis": {"fluid": "R134a", "QL": "7", "Tevap": "-10", "Tcond": "40", "superheat": "5", "subcool": "5"},
        "evaporator": {"type": "finCoil", "Tin": "5", "Tout": "-2", "Do": "9.52", "Di": "8.0", "kw": "385",
                       "fin": "", "circuits": "", "dTsatMax": "1", **HX_GEOMETRY},
        "condenser": {"type": "finCoil", "Tin": "30", "Tout": "36", "Do": "9.52", "Di": "8.0", "kw": "385",
                      "fin": "", "circuits": "", "dTsatMax": "1", **HX_GEOMETRY},
        "compressor": {"eta": "0.75", "etaMech": "0.9", "clearance": "0.04", "n": "", "rpm": "1750",
                       "cylinders": "2", "boreStroke": "1.0"},
        "expansion": {"type": "txv", "Di": "1.6", "tubes": "", "Cd": "0.6"},
    }


# ----------------------------------------------------------------- helpers

def _units(system):
    return UNITS.get(system, UNITS["SI"])


def _num(text, what, *, positive=False, nonneg=False, blank=None):
    s = "" if text is None else str(text).strip()
    if s == "":
        if blank is not None:
            return blank
        raise DesignError(f"{what}: enter a value.")
    try:
        x = float(s)
    except ValueError:
        raise DesignError(f"{what}: '{s}' is not a number.")
    if positive and not x > 0:
        raise DesignError(f"{what} must be greater than zero.")
    if nonneg and x < 0:
        raise DesignError(f"{what} can't be negative.")
    return x


class _Conv:
    def __init__(self, system):
        self.u = _units(system)

    def si(self, key, x):
        _, a, b = self.u[key]
        return x * a + b

    def show(self, key, x):
        _, a, b = self.u[key]
        return (x - b) / a

    def label(self, key):
        return self.u[key][0]

    def row(self, label, key, x, note=""):
        """One result line: label, shown value and unit."""
        return {"label": label, "value": None if x is None else self.show(key, x),
                "unit": self.u[key][0] if key else "", **({"note": note} if note else {})}

    def get(self, d, k, key, what, **kw):
        blank = kw.pop("blank", None)
        if blank is not None and str(d.get(k, "")).strip() == "":
            return blank      # an SI default
        return self.si(key, _num(d.get(k), what, **kw))


def _plain(label, x, unit="", note=""):
    return {"label": label, "value": x, "unit": unit, **({"note": note} if note else {})}


def _props(fluid, out, a, x, b, y):
    try:
        return PropsSI(out, a, x, b, y, fluid)
    except ValueError as e:
        raise DesignError(f"{fluid}: property lookup failed ({str(e).split(':')[0]}).")


# ------------------------------------------------------------------ cycle

def cycle_basis(basis, conv, eta_s):
    fluid = basis.get("fluid") or "R134a"
    QL = conv.get(basis, "QL", "Q", "Cooling capacity", positive=True)
    Te = conv.get(basis, "Tevap", "T", "Evaporating temperature")
    Tc = conv.get(basis, "Tcond", "T", "Condensing temperature")
    sh = conv.get(basis, "superheat", "dT", "Superheat", nonneg=True)
    sc = conv.get(basis, "subcool", "dT", "Subcooling", nonneg=True)
    if Tc <= Te + 5:
        raise DesignError("The condensing temperature must be well above the evaporating temperature.")
    Tcrit = _props(fluid, "Tcrit", "T", 300, "P", 1e5)
    if Tc >= Tcrit - 1:
        raise DesignError(f"{fluid} has no condensation at {conv.show('T', Tc):.4g} {conv.label('T')}: "
                          f"its critical temperature is {conv.show('T', Tcrit):.4g} {conv.label('T')}.")

    Pe = _props(fluid, "P", "T", Te, "Q", 1)
    Pc = _props(fluid, "P", "T", Tc, "Q", 0)
    hg_e = _props(fluid, "H", "P", Pe, "Q", 1)
    if sh > 0:
        T1 = Te + sh
        h1 = _props(fluid, "H", "P", Pe, "T", T1)
    else:
        T1, h1 = Te, hg_e
    s1 = _props(fluid, "S", "P", Pe, "H", h1)
    v1 = 1 / _props(fluid, "D", "P", Pe, "H", h1)
    h2s = _props(fluid, "H", "P", Pc, "S", s1)
    h2 = h1 + (h2s - h1) / eta_s
    T2 = _props(fluid, "T", "P", Pc, "H", h2)
    hf_c = _props(fluid, "H", "P", Pc, "Q", 0)
    hg_c = _props(fluid, "H", "P", Pc, "Q", 1)
    Tsat_c_dew = _props(fluid, "T", "P", Pc, "Q", 1)
    if sc > 0:
        T3 = Tc - sc
        h3 = _props(fluid, "H", "P", Pc, "T", T3)
    else:
        T3, h3 = Tc, hf_c
    h4 = h3
    T4 = _props(fluid, "T", "P", Pe, "H", h4)
    x4 = _props(fluid, "Q", "P", Pe, "H", h4)
    if h1 - h4 <= 0:
        raise DesignError("This cycle produces no cooling; check the temperatures.")
    m = QL / (h1 - h4)
    W = m * (h2 - h1)
    QH = m * (h2 - h3)
    states = {
        1: {"P": Pe, "T": T1, "h": h1, "s": s1, "v": v1},
        2: {"P": Pc, "T": T2, "h": h2},
        3: {"P": Pc, "T": T3, "h": h3},
        4: {"P": Pe, "T": T4, "h": h4, "x": x4},
    }
    return {"fluid": fluid, "QL": QL, "QH": QH, "W": W, "m": m, "COP": QL / W, "Pe": Pe, "Pc": Pc,
            "Te": Te, "Tc": Tc, "Tc_dew": Tsat_c_dew, "sh": sh, "sc": sc, "hg_e": hg_e, "hf_c": hf_c,
            "hg_c": hg_c, "h2s": h2s, "states": states}


def _basis_out(cyc, conv):
    rows = [
        conv.row("Refrigerant mass flow ṁ", "m", cyc["m"]),
        conv.row("Evaporating pressure", "P", cyc["Pe"]),
        conv.row("Condensing pressure", "P", cyc["Pc"]),
        conv.row("Compressor power Ẇ", "W", cyc["W"]),
        conv.row("Heat rejected Q̇_H", "Q", cyc["QH"]),
        _plain("COP_R", cyc["COP"]),
        conv.row("Compressor discharge temperature", "T", cyc["states"][2]["T"]),
        _plain("Quality entering the evaporator", cyc["states"][4]["x"]),
    ]
    states = [{"state": k, "P": conv.show("P", s["P"]), "T": conv.show("T", s["T"]),
               "h": conv.show("h", s["h"])} for k, s in cyc["states"].items()]
    return {"rows": rows, "states": states,
            "units": {k: conv.label(k) for k in ("P", "T", "h")}}


# ---------------------------------------------------------- heat exchanger

def _secondary_cp(kind, T):
    sec = SECONDARY[kind]
    return _props(sec["fluid"], "C", "T", T, "P", sec["P"])


def _mix_mu(fluid, P, x):
    """Homogeneous two-phase viscosity, McAdams: 1/μ = x/μ_g + (1-x)/μ_f."""
    mu_f = _props(fluid, "V", "P", P, "Q", 0)
    mu_g = _props(fluid, "V", "P", P, "Q", 1)
    return 1 / (x / mu_g + (1 - x) / mu_f)


def _friction(Re):
    return 64 / Re if Re < 2300 else 0.316 * Re ** -0.25


def _zone_dp(fluid, P, zone, Lc, G, Di):
    """Frictional pressure drop of one zone along one circuit."""
    if zone["phase"] == "two":
        x = 0.5 * (zone["x_in"] + zone["x_out"])
        v = x / _props(fluid, "D", "P", P, "Q", 1) + (1 - x) / _props(fluid, "D", "P", P, "Q", 0)
        mu = _mix_mu(fluid, P, x)
    else:
        h = 0.5 * (zone["h_in"] + zone["h_out"])
        v = 1 / _props(fluid, "D", "P", P, "H", h)
        mu = _props(fluid, "V", "P", P, "H", h)
    Re = G * Di / mu
    return _friction(Re) * (Lc / Di) * G * G * v / 2


def design_hx(which, cyc, d, conv):
    """Evaporator or condenser: zones in counterflow with the secondary stream."""
    evap = which == "evaporator"
    fluid = cyc["fluid"]
    kind = d.get("type") or "finCoil"
    if kind not in HX_TYPES:
        raise DesignError(f"Unknown heat exchanger type '{kind}'.")
    hx = HX_TYPES[kind]
    sec = hx["secondary"]
    Tin = conv.get(d, "Tin", "T", f"{SECONDARY[sec]['label']} inlet temperature")
    Tout = conv.get(d, "Tout", "T", f"{SECONDARY[sec]['label']} outlet temperature")
    Do = conv.get(d, "Do", "d", "Tube outside diameter", positive=True)
    Di = conv.get(d, "Di", "d", "Tube inside diameter", positive=True)
    if Di >= Do:
        raise DesignError("The tube inside diameter must be smaller than the outside diameter.")
    kw = conv.get(d, "kw", "k", "Tube wall conductivity", positive=True)
    fin = _num(d.get("fin"), "Finned surface factor", positive=True, blank=hx["fin"])
    geo = HX_GEOMETRY_SI
    Vair = conv.get(d, "Vair", "V", "Air velocity over the tubes", positive=True, blank=geo["Vair"])
    Ds = conv.get(d, "Ds", "d", "Shell inside diameter", positive=True, blank=geo["Ds"])
    Bs = conv.get(d, "B", "d", "Baffle spacing", positive=True, blank=geo["B"])
    pitch = _num(d.get("pitch"), "Tube pitch ratio", positive=True, blank=geo["pitch"])
    Da = conv.get(d, "Da", "d", "Outer tube inside diameter", positive=True, blank=geo["Da"])
    if kind == "shellTube" and pitch <= 1.05:
        raise DesignError("The tube pitch ratio (pitch / tube diameter) must be above 1, usually 1.25 to 1.5.")
    if kind == "tubeInTube" and Da <= Do:
        raise DesignError("The outer tube must be larger inside than the inner tube is outside.")
    dTsat_max = conv.get(d, "dTsatMax", "dT", "Allowed saturation temperature drop", positive=True, blank=1.0)
    # Fouling factors, Penoncello Table 5.2: refrigerant liquid inside; on the
    # outside city water, or air (taken like a clean gas, between steam and
    # compressed air).
    Rfi = conv.get(d, "Rfi", "R", "Inside fouling factor", nonneg=True, blank=FOULING["refrigerant"])
    Rfo = conv.get(d, "Rfo", "R", "Outside fouling factor", nonneg=True, blank=FOULING[sec])

    m = cyc["m"]
    st = cyc["states"]
    P = cyc["Pe"] if evap else cyc["Pc"]
    # Zones in the refrigerant's direction of flow.
    if evap:
        x4 = st[4]["x"]
        zones = [
            {"name": "Boiling", "phase": "two", "h_in": st[4]["h"], "h_out": cyc["hg_e"], "x_in": x4, "x_out": 1.0,
             "T_in": st[4]["T"], "T_out": _props(fluid, "T", "P", P, "Q", 1)},
            {"name": "Superheating", "phase": "one", "h_in": cyc["hg_e"], "h_out": st[1]["h"],
             "T_in": _props(fluid, "T", "P", P, "Q", 1), "T_out": st[1]["T"]},
        ]
    else:
        zones = [
            {"name": "Desuperheating", "phase": "one", "h_in": st[2]["h"], "h_out": cyc["hg_c"],
             "T_in": st[2]["T"], "T_out": cyc["Tc_dew"]},
            {"name": "Condensing", "phase": "two", "h_in": cyc["hg_c"], "h_out": cyc["hf_c"], "x_in": 1.0,
             "x_out": 0.0, "T_in": cyc["Tc_dew"], "T_out": _props(fluid, "T", "P", P, "Q", 0)},
            {"name": "Subcooling", "phase": "one", "h_in": cyc["hf_c"], "h_out": st[3]["h"],
             "T_in": _props(fluid, "T", "P", P, "Q", 0), "T_out": st[3]["T"]},
        ]
    zones = [z for z in zones if abs(z["h_out"] - z["h_in"]) * m > 1e-6]
    Q = sum(abs(z["h_out"] - z["h_in"]) for z in zones) * m

    label = SECONDARY[sec]["label"]
    if evap and not Tout < Tin:
        raise DesignError(f"The {label.lower()} must leave the evaporator colder than it enters.")
    if not evap and not Tout > Tin:
        raise DesignError(f"The {label.lower()} must leave the condenser warmer than it enters.")
    cp = _secondary_cp(sec, 0.5 * (Tin + Tout))
    C = Q / abs(Tout - Tin)             # secondary capacity rate ṁ c_p, W/K
    m_sec = C / cp
    rho_sec = _props(SECONDARY[sec]["fluid"], "D", "T", Tin, "P", SECONDARY[sec]["P"])

    # Walk along the secondary stream, which meets the refrigerant's last zone first.
    s = 1.0 if not evap else -1.0     # +1 when the refrigerant is the hot side
    Tsec = Tin
    out_zones = []
    UA_tot = 0.0
    A_tot = 0.0
    for z in reversed(zones):
        Qz = abs(z["h_out"] - z["h_in"]) * m
        Tsec_out = Tsec + s * Qz / C
        dTa = s * (z["T_out"] - Tsec)
        dTb = s * (z["T_in"] - Tsec_out)
        if dTa <= 0 or dTb <= 0:
            raise DesignError(
                f"{which.capitalize()} {z['name'].lower()} zone: the temperatures cross "
                f"(refrigerant {conv.show('T', min(z['T_in'], z['T_out'])):.3g} to "
                f"{conv.show('T', max(z['T_in'], z['T_out'])):.3g} {conv.label('T')}, {label.lower()} "
                f"{conv.show('T', Tsec):.3g} to {conv.show('T', Tsec_out):.3g} {conv.label('T')}). "
                f"{'Lower the evaporating temperature or the air/water outlet temperature.' if evap else 'Raise the condensing temperature or lower the air/water outlet temperature.'}")
        lmtd = dTa if abs(dTa - dTb) < 1e-9 else (dTa - dTb) / math.log(dTa / dTb)
        UA = Qz / lmtd
        out_zones.append({"name": z["name"], "Q": Qz, "LMTD": lmtd, "UA": UA,
                          "Tsec_in": Tsec, "Tsec_out": Tsec_out, "z": z})
        UA_tot += UA
        Tsec = Tsec_out
    out_zones.reverse()       # back to the refrigerant's order
    pinch = min(min(abs(zz["z"]["T_out"] - zz["Tsec_in"]), abs(zz["z"]["T_in"] - zz["Tsec_out"]))
                for zz in out_zones)

    # --- film coefficients from Nusselt correlations (convection.py) ---
    sat = _sat_props(fluid, P)
    sec_fluid, sec_P = SECONDARY[sec]["fluid"], SECONDARY[sec]["P"]
    flow_area = math.pi * Di * Di / 4
    rr = cv.ROUGHNESS / Di
    for zz in out_zones:
        z = zz["z"]
        T_sec = 0.5 * (zz["Tsec_in"] + zz["Tsec_out"])
        T_ref = 0.5 * (z["T_in"] + z["T_out"])
        # air: properties at the film temperature (Eq. 5.21), the wall taken near the refrigerant
        zz["sec"] = _transport(sec_fluid, "T", 0.5 * (T_sec + T_ref) if sec == "air" else T_sec, sec_P)
        if z["phase"] == "one":
            zz["ref"] = _transport(fluid, "H", 0.5 * (z["h_in"] + z["h_out"]), P)

    def outside(zz, n):
        """(Re, Nu, h) on the air or water side of a zone."""
        pr = zz["sec"]
        if kind == "finCoil":                       # cross flow over the tubes, Eq. 5.20
            Re = pr["rho"] * Vair * Do / pr["mu"]
            Nu = cv.nu_cross_cylinder(Re, pr["Pr"])
            return Re, Nu, Nu * pr["k"] / Do
        if kind == "shellTube":                     # baffled shell, Eqs. 5.115 to 5.120
            PT = pitch * Do
            De = cv.shell_equivalent_diameter(PT, Do, d.get("layout") != "square")
            V = m_sec / (pr["rho"] * Ds * (PT - Do) * Bs / PT)
            Re = pr["rho"] * V * De / pr["mu"]
            Nu = cv.nu_shell(Re, pr["Pr"])
            return Re, Nu, Nu * pr["k"] / De
        De = (Da * Da - Do * Do) / Do               # annulus, equivalent diameter Eq. 5.81
        V = m_sec / n / (pr["rho"] * math.pi / 4 * (Da * Da - Do * Do))
        Re = pr["rho"] * V * De / pr["mu"]
        Nu = cv.nu_tube(Re, pr["Pr"])
        return Re, Nu, Nu * pr["k"] / De

    def size(n):
        """Every zone's h, U and area with n parallel refrigerant circuits."""
        G = m / (n * flow_area)
        zs = []
        for zz in out_zones:
            z = zz["z"]
            Re_o, Nu_o, h_o = outside(zz, n)
            wall = Do * math.log(Do / Di) / (2 * kw)

            def U_of(h_i):
                # Penoncello Eq. 5.33, on the outside tube area, fins folded into h_o
                return 1 / (1 / (h_o * fin) + Rfo + (Do / Di) * (1 / h_i + Rfi) + wall)
            if z["phase"] == "one":
                pr = zz["ref"]
                Re_i = G * Di / pr["mu"]
                Nu_i = cv.nu_tube(Re_i, pr["Pr"], rr)
                h_i = Nu_i * pr["k"] / Di
            else:
                Re_i = G * Di / sat["mu_l"]         # liquid-only Reynolds number
                if evap:
                    h_i = 2000.0
                    for _ in range(8):              # the heat flux sets the boiling number
                        q = U_of(h_i) * zz["LMTD"] * Do / Di
                        h_i = cv.h_boiling_tube(G, Di, z["x_in"], z["x_out"], q, sat)
                else:
                    h_i = cv.h_condensing_tube(G, Di, z["x_in"], z["x_out"], sat)
                Nu_i = h_i * Di / sat["k_l"]
            U = U_of(h_i)
            zs.append({**zz, "U": U, "A": zz["UA"] / U, "Re_i": Re_i, "Nu_i": Nu_i, "h_i": h_i,
                       "Re_o": Re_o, "Nu_o": Nu_o, "h_o": h_o})
        A_tot = sum(z["A"] for z in zs)
        L_tot = A_tot / (math.pi * Do)
        Lc = L_tot / n
        dp = sum(_zone_dp(fluid, P, z["z"], Lc * z["A"] / A_tot, G, Di) for z in zs)
        P_low = max(P - dp, 0.2 * P)
        dT = abs(_props(fluid, "T", "P", P, "Q", 1) - _props(fluid, "T", "P", P_low, "Q", 1))
        return {"zones": zs, "A": A_tot, "L": L_tot, "Lc": Lc, "G": G, "dp": dp, "dT": dT}

    # Circuits: the fewest parallel paths whose friction drop stays within the
    # allowed saturation temperature drop (or the number typed in).
    typed = str(d.get("circuits", "")).strip()
    if typed:
        n = int(round(_num(typed, "Number of circuits", positive=True)))
        auto = False
        sized = size(n)
    else:
        n = 1
        sized = size(n)
        while n < 400 and sized["dT"] > dTsat_max:
            n += 1
            sized = size(n)
        auto = True
    out_zones = sized["zones"]
    A_tot, L_tot, Lc, G, dp, dTsat = (sized[k] for k in ("A", "L", "Lc", "G", "dp", "dT"))

    warnings = []
    for zz in out_zones:
        if zz["z"]["phase"] == "one" and zz["Re_i"] < 3000:
            warnings.append(f"{zz['name']} zone: the refrigerant flow is laminar or transitional "
                            f"(Re = {zz['Re_i']:.0f}), so its film coefficient is poor. Fewer circuits raise it.")
    Re_o = out_zones[0]["Re_o"]
    if kind == "shellTube" and not 2000 <= Re_o <= 1e6:
        warnings.append(f"Shell-side Re = {Re_o:.0f} is outside the 2000 to 1 000 000 range of the McAdams "
                        "correlation; change the shell diameter or baffle spacing.")
    if kind == "tubeInTube" and Re_o < 3000:
        warnings.append(f"The water in the annulus is laminar or transitional (Re = {Re_o:.0f}); "
                        "a smaller outer tube speeds it up.")
    if dTsat > dTsat_max:
        warnings.append(f"The refrigerant pressure drop costs {conv.show('dT', dTsat):.2g} {conv.label('dT')} of "
                        f"saturation temperature, more than the {conv.show('dT', dTsat_max):.2g} allowed. "
                        "Use more circuits.")
    if pinch < 2:
        warnings.append("The closest approach is under 2 K, which makes for a very large exchanger. "
                        "Widen the temperature difference if you can.")
    if Lc < 1:
        warnings.append("Each circuit is under 1 m long; fewer circuits or smaller tubes would suit better.")

    vol_sec = m_sec / rho_sec
    rows = [
        conv.row("Heat duty Q̇", "Q", Q),
        conv.row("Overall UA", "UA", UA_tot),
        conv.row("Tube outside area", "A", A_tot),
        conv.row("Total tube length", "L", L_tot),
        _plain("Parallel circuits", n, "", "fewest that meet the pressure-drop limit" if auto else "as typed"),
        conv.row("Length of each circuit", "L", Lc),
        conv.row("Refrigerant mass flux in the tubes", "G", G),
        conv.row("Refrigerant pressure drop", "dp", dp),
        conv.row("Saturation temperature lost to it", "dT", dTsat),
        conv.row("Closest approach (pinch)", "dT", pinch),
        conv.row(f"{label} mass flow", "m", m_sec),
        conv.row(f"{label} volume flow", "vol", vol_sec),
    ]
    zone_rows = [{"name": zz["name"], "Q": conv.show("Q", zz["Q"]), "LMTD": conv.show("dT", zz["LMTD"]),
                  "U": conv.show("U", zz["U"]), "A": conv.show("A", zz["A"]),
                  "Tsec_in": conv.show("T", zz["Tsec_in"]), "Tsec_out": conv.show("T", zz["Tsec_out"]),
                  **_film_out(zz, conv)}
                 for zz in out_zones]
    inside_tp = "Gungor–Winterton boiling" if evap else "Shah condensation"
    outside_by = {"finCoil": "Churchill–Bernstein cross flow (Eq. 5.20)",
                  "shellTube": "McAdams shell side (Eq. 5.115)",
                  "tubeInTube": "Gnielinski on the annulus equivalent diameter (Eqs. 5.26, 5.81)"}[kind]
    return {"rows": rows, "zones": zone_rows, "warnings": warnings,
            "zoneUnits": {k: conv.label(k) for k in ("Q", "dT", "U", "A", "T")},
            "film": {"inside": f"Refrigerant in the tubes: Gnielinski (Eq. 5.26) for vapor and liquid, {inside_tp} "
                               "for the two-phase zone (Nu there is on the liquid's k).",
                     "outside": f"{label} outside: {outside_by}."},
            "defaults": {"fin": hx["fin"], "Rfi": conv.show("R", FOULING["refrigerant"]),
                         "Rfo": conv.show("R", FOULING[sec]),
                         **{k: conv.show(u, HX_GEOMETRY_SI[k]) for k, u in (("Vair", "V"), ("Ds", "d"), ("B", "d"),
                                                                            ("Da", "d"))},
                         "pitch": HX_GEOMETRY_SI["pitch"]},
            "summary": {"A": A_tot, "L": L_tot, "circuits": n, "UA": UA_tot}}


def _film_out(z, conv):
    """A zone's Re, Nu and h on both sides, for the zone table."""
    return {"Re_i": z["Re_i"], "Nu_i": z["Nu_i"], "h_i": conv.show("U", z["h_i"]),
            "Re_o": z["Re_o"], "Nu_o": z["Nu_o"], "h_o": conv.show("U", z["h_o"])}


def _transport(fluid, by, value, P):
    """ρ, μ, k, Pr and c_p of a single-phase fluid at pressure P and a
    temperature (by="T") or enthalpy (by="H")."""
    get = lambda out: _props(fluid, out, by, value, "P", P)  # noqa: E731
    return {"rho": get("D"), "mu": get("V"), "k": get("L"), "Pr": get("Prandtl"), "cp": get("C")}


def _sat_props(fluid, P):
    """Saturated liquid and vapor properties at P, for the two-phase correlations."""
    liq = lambda out: _props(fluid, out, "P", P, "Q", 0)  # noqa: E731
    vap = lambda out: _props(fluid, out, "P", P, "Q", 1)  # noqa: E731
    return {"rho_l": liq("D"), "rho_v": vap("D"), "mu_l": liq("V"), "k_l": liq("L"), "Pr_l": liq("Prandtl"),
            "cp_l": liq("C"), "hfg": vap("H") - liq("H"), "p_r": P / _props(fluid, "Pcrit", "T", 300, "P", 1e5)}


# --------------------------------------------------------------- compressor

def design_compressor(cyc, d, conv):
    fluid = cyc["fluid"]
    eta = _efficiency(d.get("eta"), "Isentropic efficiency")
    eta_m = _efficiency(d.get("etaMech"), "Mechanical and motor efficiency")
    c = _num(d.get("clearance"), "Clearance ratio", nonneg=True)
    if c >= 0.3:
        raise DesignError("The clearance ratio is the clearance volume over the swept volume, usually 0.02 to 0.1.")
    rpm = _num(d.get("rpm"), "Speed", positive=True)
    ncyl = int(round(_num(d.get("cylinders"), "Number of cylinders", positive=True)))
    bs = _num(d.get("boreStroke"), "Stroke-to-bore ratio", positive=True)
    st = cyc["states"]
    rp = cyc["Pc"] / cyc["Pe"]
    v1 = st[1]["v"]
    v2s = 1 / _props(fluid, "D", "P", cyc["Pc"], "H", cyc["h2s"])
    n_auto = math.log(rp) / math.log(v1 / v2s)       # isentropic exponent of the real gas
    n = _num(d.get("n"), "Polytropic exponent n", positive=True, blank=n_auto)
    eta_v = 1 + c - c * rp ** (1 / n)
    if eta_v <= 0.05:
        raise DesignError(f"At a pressure ratio of {rp:.3g} the clearance gas re-expands to fill the whole "
                          "cylinder; this compressor pumps nothing. Use less clearance or two stages.")
    Vin = cyc["m"] * v1                 # suction volume flow, m³/s
    Vd = Vin / eta_v                    # displacement
    Vs = Vd / (ncyl * rpm / 60)         # swept volume of one cylinder per stroke
    bore = (4 * Vs / (math.pi * bs)) ** (1 / 3)
    stroke = bs * bore
    piston = 2 * stroke * rpm / 60
    W_shaft = cyc["W"] / eta_m
    warnings = []
    if rp > 10:
        warnings.append(f"A pressure ratio of {rp:.3g} is high for one stage; consider two-stage compression.")
    if st[2]["T"] > 273.15 + 120:
        warnings.append("Discharge above 120 °C can break down the oil; reduce superheat or the pressure ratio.")
    if piston > 5:
        warnings.append("Mean piston speed is above 5 m/s; slow it down or add cylinders.")
    rows = [
        _plain("Pressure ratio P₂/P₁", rp),
        _plain("Polytropic exponent n", n, "", "from the refrigerant's isentropic path" if str(d.get("n", "")).strip() == "" else "as typed"),
        _plain("Volumetric efficiency η_v", eta_v),
        conv.row("Suction volume flow", "vol", Vin),
        conv.row("Required displacement", "vol", Vd),
        conv.row("Swept volume per cylinder", "cyl", Vs),
        conv.row("Bore", "d", bore),
        conv.row("Stroke", "d", stroke),
        conv.row("Mean piston speed", "V", piston),
        conv.row("Power to the gas", "W", cyc["W"]),
        conv.row("Shaft (motor) power", "W", W_shaft),
        conv.row("Discharge temperature", "T", st[2]["T"]),
    ]
    return {"rows": rows, "warnings": warnings,
            "summary": {"eta": eta, "bore": bore, "stroke": stroke, "Vd": Vd}}


def _efficiency(text, what):
    s = "" if text is None else str(text).strip()
    pct = s.endswith("%")
    x = _num(s.rstrip("%"), what, positive=True)
    if pct or x > 1:
        x /= 100
    if not 0 < x <= 1:
        raise DesignError(f"{what} must be between 0 and 1 (or 0 to 100 %).")
    return x


# ---------------------------------------------------------------- capillary

def design_capillary(cyc, d, conv):
    """Length of capillary tube from the condenser outlet to P_evap.

    Steps down in pressure. In each step energy (h + V²/2 const, V = G v)
    gives the state, and momentum with wall friction gives the length:
        dL = 2 D (-dP - G dV) / (f G² v_mean)
    When dL turns negative the flow has choked (sonic) at that pressure.
    """
    if (d.get("type") or "capillary") == "txv":
        return design_txv(cyc, d, conv)
    Di = conv.get(d, "Di", "d", "Capillary bore", positive=True)
    typed = str(d.get("tubes", "")).strip()
    if typed:
        return _capillary(cyc, Di, int(round(_num(typed, "Tubes in parallel", positive=True))), conv, False)
    # Blank: the fewest tubes in parallel that run without choking. More tubes
    # means less flow per tube, so choking stops past one count: bisect for it.
    lo, hi = 0, 64
    best = _capillary(cyc, Di, hi, conv, True)
    if best["summary"]["choked"]:
        return best
    while hi - lo > 1:
        mid = (lo + hi) // 2
        r = _capillary(cyc, Di, mid, conv, True)
        if r["summary"]["choked"]:
            lo = mid
        else:
            hi, best = mid, r
    return best


def design_txv(cyc, d, conv):
    """Thermostatic expansion valve port: ṁ = C_d A √(2 ρ ΔP) for the liquid
    entering, sized so the valve is about 75 % open at the design point."""
    Cd = _num(d.get("Cd"), "Discharge coefficient C_d", positive=True, blank=0.6)
    fluid = cyc["fluid"]
    rho = _props(fluid, "D", "P", cyc["Pc"], "H", cyc["states"][3]["h"])
    dP = cyc["Pc"] - cyc["Pe"]
    A = cyc["m"] / (Cd * math.sqrt(2 * rho * dP))
    A_full = A / 0.75
    D_port = math.sqrt(4 * A_full / math.pi)
    rows = [
        _plain("Valve capacity (rated)", conv.show("Q", cyc["QL"] / 0.75), conv.label("Q"),
               "pick a valve rated at least this for this refrigerant"),
        conv.row("Pressure drop across the valve", "dp", dP),
        conv.row("Port diameter, fully open", "d", D_port),
        conv.row("Superheat the bulb holds", "dT", cyc["sh"]),
    ]
    warnings = []
    if cyc["sh"] < 2:
        warnings.append("A TXV needs some superheat to sense (usually 4 to 8 K); set the superheat in the basis.")
    return {"rows": rows, "warnings": warnings, "summary": {"D": D_port}}


def _capillary(cyc, Di, ntube, conv, auto):
    fluid = cyc["fluid"]
    G = cyc["m"] / ntube / (math.pi * Di * Di / 4)
    st = cyc["states"]
    P0, h0 = cyc["Pc"], st[3]["h"]
    v0 = 1 / _props(fluid, "D", "P", P0, "H", h0)
    H = h0 + (G * v0) ** 2 / 2       # stagnation enthalpy
    Pend = cyc["Pe"]

    def state(P):
        # solve h + (G v(P,h))²/2 = H for h by fixed-point iteration
        h = H
        for _ in range(40):
            v = 1 / _props(fluid, "D", "P", P, "H", h)
            h_new = H - (G * v) ** 2 / 2
            if abs(h_new - h) < 1e-3:
                break
            h = h_new
        v = 1 / _props(fluid, "D", "P", P, "H", h)
        x = _props(fluid, "Q", "P", P, "H", h)
        if 0 <= x <= 1:
            mu = _mix_mu(fluid, P, x)
        else:
            mu = _props(fluid, "V", "P", P, "H", h)
        return h, v, mu, x

    steps = 60
    L = 0.0
    P_prev = P0
    _, v_prev, mu_prev, _ = state(P0)
    choked_at = None
    for i in range(1, steps + 1):
        P = P0 + (Pend - P0) * i / steps
        _, v, mu, x = state(P)
        vm, mum = 0.5 * (v + v_prev), 0.5 * (mu + mu_prev)
        f = _friction(G * Di / mum)
        dL = 2 * Di * ((P_prev - P) - G * G * (v - v_prev)) / (f * G * G * vm)
        if dL <= 0:
            choked_at = P_prev
            break
        L += dL
        P_prev, v_prev, mu_prev = P, v, mu
    warnings = []
    Vout = G * v_prev
    if choked_at is not None:
        warnings.append(f"The flow chokes at {conv.show('P', choked_at):.4g} {conv.label('P')}, above the "
                        "evaporating pressure, so the length shown is the most this bore can use; the last of "
                        "the pressure drop happens as a jet at the exit. A smaller bore or fewer tubes gives a "
                        "longer, steadier tube.")
    if L < 0.5:
        warnings.append("Under 0.5 m is too short to make reliably; use a smaller bore or fewer tubes.")
    if L > 6:
        warnings.append("Over 6 m is unwieldy; a smaller bore gives a shorter tube.")
    if cyc["QL"] > 5e3 * ntube:
        warnings.append("Capillary tubes suit small systems (a few kW per tube); a thermostatic expansion valve "
                        "is the usual choice at this capacity.")
    rows = [
        conv.row("Capillary length", "L", L),
        _plain("Tubes in parallel", ntube, "", "fewest that don't choke" if auto else "as typed"),
        conv.row("Mass flux in each tube", "G", G),
        conv.row("Exit velocity", "V", Vout),
        conv.row("Inlet pressure", "P", P0),
        conv.row("Outlet pressure", "P", choked_at or Pend),
    ]
    return {"rows": rows, "warnings": warnings, "summary": {"L": L, "choked": choked_at is not None}}


# ------------------------------------------------------------------- entry

def _merged(payload):
    base = default_inputs()
    for k, v in (payload or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def design_refrigeration(payload):
    """Sizes every part. A part that can't be sized reports its own error;
    the others still come back."""
    d = _merged(payload)
    conv = _Conv(d.get("system", "SI"))
    out = {"errors": [], "units": {k: v[0] for k, v in _units(d.get("system", "SI")).items()}}
    try:
        eta = _efficiency(d["compressor"].get("eta"), "Isentropic efficiency")
        cyc = cycle_basis(d["basis"], conv, eta)
    except DesignError as e:
        out["errors"].append(str(e))
        return out
    out["basis"] = _basis_out(cyc, conv)
    parts = {
        "evaporator": lambda: design_hx("evaporator", cyc, d["evaporator"], conv),
        "condenser": lambda: design_hx("condenser", cyc, d["condenser"], conv),
        "compressor": lambda: design_compressor(cyc, d["compressor"], conv),
        "expansion": lambda: design_capillary(cyc, d["expansion"], conv),
    }
    for k, fn in parts.items():
        try:
            out[k] = fn()
        except DesignError as e:
            out[k] = {"error": str(e)}
    # What the assembled cycle needs, in SI display units (°C, kPa, kg/s).
    st = cyc["states"]
    out["assemble"] = {
        "fluid": cyc["fluid"],
        "Pe_kPa": cyc["Pe"] / 1e3, "Pc_kPa": cyc["Pc"] / 1e3, "m": cyc["m"],
        "T1_C": st[1]["T"] - 273.15, "T3_C": st[3]["T"] - 273.15,
        "superheat": cyc["sh"] > 0, "subcool": cyc["sc"] > 0, "eta": eta,
        "COP": cyc["COP"], "QL_kW": cyc["QL"] / 1e3,
    }
    return out


def design(payload):
    if (payload or {}).get("cycle") in ("rankine", "brayton", "turbojet"):
        from power_design import design_power
        return design_power(payload)
    try:
        return design_refrigeration(payload)
    except DesignError as e:
        return {"errors": [str(e)]}
