"""Design studio, power and propulsion cycles: Rankine, Brayton and turbojet.

Each cycle is worked out from its basis first (the textbook cycle with the
pump, compressor and turbine efficiencies of the parts), then every part is
sized for its duty in it:

  pump          centrifugal: head, specific speed Ω = ω√Q / (gH)^¾, the
                number of stages that keeps Ω in the efficient range, and
                the impeller tip diameter from the head coefficient ψ = gH/U².
  compressor    axial: stage count from the stage loading ψ = Δh₀/U², the
                annulus from the axial velocity, blade heights and speed.
  turbine       axial: stage count from the stage loading, blade heights from
                the flow coefficient φ = cₓ/U, last-stage checks.
  boiler        counterflow zones (economizer, evaporator, superheater) against
                the furnace gas, UA = Q̇ / LMTD per zone, and the fuel burned.
  condenser     steam condensing against cooling water: area, tube count,
                water velocity.
  combustor     fuel flow and fuel–air ratio from the heat added, and the liner
                volume for a residence time.
  diffuser      intake capture area and the ram pressure rise.
  nozzle        throat and exit areas, choking, and the thrust.

Everything is in SI inside and converted with design._Conv for display.
"""
import math

import convection as cv
from design import (DesignError, FOULING, _Conv, _efficiency, _film_out, _num, _plain, _props, _sat_props,
                    _transport, UNITS)

# Extra display units for these cycles (added to design.UNITS).
UNITS["SI"].update({"Pw": ("MW", 1e6, 0.0), "F": ("kN", 1e3, 0.0), "Vdot": ("m³/s", 1.0, 0.0),
                    "LHV": ("kJ/kg", 1e3, 0.0), "TSFC": ("g/(kN·s)", 1e-6, 0.0), "H": ("m", 1.0, 0.0),
                    "Vol": ("m³", 1.0, 0.0), "mf": ("kg/s", 1.0, 0.0), "mbig": ("kg/s", 1.0, 0.0),
                    "Pk": ("kPa", 1e3, 0.0)})
UNITS["EN"].update({"Pw": ("MW", 1e6, 0.0), "F": ("lbf", 4.4482216, 0.0), "Vdot": ("ft³/s", 0.028316847, 0.0),
                    "LHV": ("Btu/lbm", 2326.0, 0.0), "TSFC": ("lbm/(lbf·h)", 0.45359237 / 4.4482216 / 3600, 0.0),
                    "H": ("ft", 0.3048, 0.0), "Vol": ("ft³", 0.028316847, 0.0), "mf": ("lbm/s", 0.45359237, 0.0),
                    "mbig": ("lbm/s", 0.45359237, 0.0), "Pk": ("psia", 6894.757293168, 0.0)})

G0 = 9.80665
AIR = "Air"
WATER = "Water"
CP_GAS = 1150.0         # furnace gas c_p, J/kg·K
F_STOICH = 0.068        # stoichiometric fuel–air ratio of a kerosene-like fuel


def defaults():
    """Starting designs, in SI as shown. Close to Çengel's Examples 10-2,
    9-6 and 9-10, with real efficiencies."""
    return {
        "rankine": {
            "basis": {"Wnet": "100", "Pboil": "15000", "T3": "600", "Pcond": "10"},
            "pump": {"eta": "0.85", "rpm": "6000", "stages": "", "psi": "0.5", "z": "30", "hL": "0.5", "S": "3"},
            "turbine": {"eta": "0.87", "rpm": "3600", "U": "300", "psi": "2.0", "phi": "0.6", "flows": "2"},
            "boiler": {"Tgin": "1300", "Tgout": "350", "Vgas": "10", "circuits": "400",
                       "Do": "50.8", "Di": "42.8", "kw": "40", "LHV": "50000", "etaB": "0.9"},
            "condenser": {"Tin": "20", "Tout": "30", "Do": "25.4", "Di": "22.9", "kw": "110", "Ltube": "12"},
        },
        "brayton": {
            "basis": {"Wnet": "10", "P1": "100", "T1": "15", "rp": "12", "T3": "1300"},
            "compressor": {"eta": "0.86", "U": "350", "psi": "0.35", "cx": "150", "hubTip": "0.5"},
            "combustor": {"LHV": "50000", "etaB": "0.99", "tau": "0.008"},
            "turbine": {"eta": "0.89", "rpm": "", "U": "400", "psi": "1.5", "phi": "0.6"},
        },
        "turbojet": {
            "basis": {"thrust": "50", "T0": "-40", "P0": "26.5", "V0": "250", "rp": "12", "T4": "1300"},
            "diffuser": {"eta": "0.95"},
            "compressor": {"eta": "0.85", "U": "350", "psi": "0.35", "cx": "150", "hubTip": "0.5"},
            "combustor": {"LHV": "43000", "etaB": "0.98", "tau": "0.005"},
            "turbine": {"eta": "0.88", "rpm": "", "U": "400", "psi": "1.5", "phi": "0.6"},
            "nozzle": {"eta": "0.97"},
        },
    }


def _merged(cycle, payload):
    base = defaults()[cycle]
    for k, v in (payload or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        elif k in ("system",):
            base[k] = v
    return base


def _blank(d, k):
    return str(d.get(k, "")).strip() == ""


# ------------------------------------------------------------------ parts

def _turbine(fluid, m, inlet, outlet, h_out_s, d, conv, rpm_from=None, steam=False):
    """Axial turbine: stages from the loading, blade heights from the flow
    coefficient at inlet and exit. inlet/outlet: dicts with P, h."""
    psi = _num(d.get("psi"), "Stage loading ψ", positive=True)
    phi = _num(d.get("phi"), "Flow coefficient φ", positive=True)
    dh = inlet["h"] - outlet["h"]
    dh_s = inlet["h"] - h_out_s
    U = conv.get(d, "U", "V", "Mean blade speed", positive=True)
    if rpm_from is not None and _blank(d, "rpm"):
        rpm = rpm_from[1]          # on the compressor's shaft
        note = "same shaft as the compressor"
    else:
        rpm = _num(d.get("rpm"), "Speed", positive=True)
        note = ""
    Dm = 2 * U / (rpm * 2 * math.pi / 60)
    n = max(1, math.ceil(dh / (psi * U * U) - 1e-9))
    cx = phi * U

    flows = int(round(_num(d.get("flows"), "Exhaust flows", positive=True, blank=1)))

    def height(P, h, k=1):
        v = 1 / _props(fluid, "D", "P", P, "H", h)
        return m * v / k / (math.pi * Dm * cx), m * v
    h_in, V_in = height(inlet["P"], inlet["h"])
    h_out, V_out = height(outlet["P"], outlet["h"], flows)
    warnings = []
    if h_out > 0.4 * Dm:
        k = math.ceil(h_out * flows / (0.4 * Dm))
        warnings.append(f"The last blades would be {h_out / Dm:.2g} of the mean diameter long. Split the exhaust into "
                        f"{k} parallel flows or raise the blade speed.")
    if h_in < 0.01:
        warnings.append("First-stage blades come out under 10 mm, which is very lossy. Real steam turbines feed the "
                        "first stages through only part of the circumference (partial admission) or use a smaller "
                        "high-pressure section.")
    if U > 450:
        warnings.append("A mean blade speed over 450 m/s is beyond what blade materials take.")
    rows = [
        conv.row("Enthalpy drop (actual)", "h", dh),
        conv.row("Isentropic enthalpy drop", "h", dh_s),
        _plain("Stages", n, "", f"each takes ψU² = {conv.show('h', psi * U * U):.3g} {conv.label('h')}"),
        _plain("Speed", rpm, "rpm", note),
        conv.row("Mean blade speed U", "V", U),
        conv.row("Mean diameter", "L", Dm),
        conv.row("Axial velocity cₓ = φU", "V", cx),
        conv.row("First-stage blade height", "L", h_in),
        conv.row("Last-stage blade height", "L", h_out, f"exhaust split into {flows} flows" if flows > 1 else ""),
        conv.row("Exhaust volume flow", "Vdot", V_out),
        conv.row("Power", "Pw", m * dh),
    ]
    return {"rows": rows, "warnings": warnings, "summary": {"stages": n, "Dm": Dm, "rpm": rpm}}


def _axial_compressor(fluid, m, inlet, outlet, d, conv):
    """Axial compressor: stages from ψ = Δh₀/U², annulus from cₓ and the
    hub–tip ratio at the inlet. Returns (result, (U_mean, rpm))."""
    U_tip = conv.get(d, "U", "V", "Tip speed", positive=True)
    psi = _num(d.get("psi"), "Stage loading ψ", positive=True)
    cx = conv.get(d, "cx", "V", "Axial velocity", positive=True)
    nu = _num(d.get("hubTip"), "Hub–tip ratio", positive=True)
    if nu >= 0.95:
        raise DesignError("The hub–tip ratio must be below 1 (0.4 to 0.6 at a compressor inlet).")
    v1 = 1 / _props(fluid, "D", "P", inlet["P"], "H", inlet["h"])
    A1 = m * v1 / cx
    Dt = math.sqrt(4 * A1 / (math.pi * (1 - nu * nu)))
    rpm = U_tip / (Dt / 2) * 60 / (2 * math.pi)
    Dm = Dt * (1 + nu) / 2
    Um = U_tip * (1 + nu) / 2
    dh = outlet["h"] - inlet["h"]
    n = max(1, math.ceil(dh / (psi * Um * Um) - 1e-9))
    v2 = 1 / _props(fluid, "D", "P", outlet["P"], "H", outlet["h"])
    A2 = m * v2 / cx
    h_last = A2 / (math.pi * Dm)
    rp = outlet["P"] / inlet["P"]
    warnings = []
    if h_last < 0.012:
        warnings.append("The last blades come out under 12 mm; tip clearance losses get large. "
                        "Finish with a centrifugal stage or use a smaller mean diameter.")
    if U_tip > 500:
        warnings.append("A tip speed over 500 m/s makes the first stage strongly supersonic.")
    rows = [
        _plain("Pressure ratio", rp),
        _plain("Stages", n, "", f"stage pressure ratio about {rp ** (1 / n):.3g}"),
        _plain("Speed", rpm, "rpm"),
        conv.row("Inlet tip diameter", "L", Dt),
        conv.row("Inlet blade height", "L", (Dt - nu * Dt) / 2),
        conv.row("Last blade height (same mean diameter)", "L", h_last),
        conv.row("Inlet volume flow", "Vdot", m * v1),
        conv.row("Power", "Pw", m * dh),
    ]
    return {"rows": rows, "warnings": warnings, "summary": {"stages": n, "Dt": Dt}}, (Um, rpm)


def _combustor(m, Q, T_out, P_out, d, conv, fluid):
    LHV = conv.get(d, "LHV", "LHV", "Fuel heating value", positive=True)
    etaB = _efficiency(d.get("etaB"), "Combustion efficiency")
    tau = _num(d.get("tau"), "Residence time", positive=True)
    mf = Q / (LHV * etaB)
    f = mf / m
    v_out = 1 / _props(fluid, "D", "P", P_out, "T", T_out)
    V = m * v_out * tau
    warnings = []
    phi = f / F_STOICH
    if phi > 1:
        warnings.append("More fuel than the air can burn (φ > 1): lower the turbine inlet temperature.")
    rows = [
        conv.row("Heat added", "Pw", Q),
        conv.row("Fuel flow", "mf", mf),
        _plain("Fuel–air ratio", f),
        _plain("Overall equivalence ratio φ", phi, "", "the rest of the air dilutes and cools the gas"),
        conv.row("Liner volume for the residence time", "Vol", V),
    ]
    return {"rows": rows, "warnings": warnings, "summary": {"mf": mf}}


def _lmtd(dTa, dTb):
    return dTa if abs(dTa - dTb) < 1e-9 else (dTa - dTb) / math.log(dTa / dTb)


def _zones(zones, hot_working, T_sec_in, C, what, conv):
    """Counterflow zones. zones: in the working fluid's flow order, each
    {name, Q, T_in, T_out, U}. The other stream (capacity C) meets the last
    zone first. Returns the zones with LMTD, UA, A and the stream temperatures."""
    s = 1.0 if hot_working else -1.0
    T = T_sec_in
    out = []
    for z in reversed(zones):
        T_next = T + s * z["Q"] / C
        dTa = s * (z["T_out"] - T)
        dTb = s * (z["T_in"] - T_next)
        if dTa <= 0 or dTb <= 0:
            raise DesignError(f"{what}, {z['name'].lower()} zone: the temperatures cross "
                              f"(gas or water {conv.show('T', T):.4g} to {conv.show('T', T_next):.4g} {conv.label('T')}, "
                              f"working fluid {conv.show('T', z['T_in']):.4g} to {conv.show('T', z['T_out']):.4g} "
                              f"{conv.label('T')}). Change the inlet or outlet temperatures.")
        L = _lmtd(dTa, dTb)
        out.append({**z, "LMTD": L, "UA": z["Q"] / L, "A": z["Q"] / L / z["U"], "Tsec_in": T, "Tsec_out": T_next,
                    "pinch": min(dTa, dTb)})
        T = T_next
    out.reverse()
    return out


def _zone_table(zs, conv):
    return [{"name": z["name"], "Q": conv.show("Pw", z["Q"]), "LMTD": conv.show("dT", z["LMTD"]),
             "U": conv.show("U", z["U"]), "A": conv.show("A", z["A"]),
             "Tsec_in": conv.show("T", z["Tsec_in"]), "Tsec_out": conv.show("T", z["Tsec_out"]),
             **(_film_out(z, conv) if "h_i" in z else {})} for z in zs]


def _zone_units(conv):
    return {"Q": conv.label("Pw"), "dT": conv.label("dT"), "U": conv.label("U"), "A": conv.label("A"),
            "T": conv.label("T")}


# ---------------------------------------------------------------- Rankine

def rankine(payload):
    d = _merged("rankine", payload)
    conv = _Conv(payload.get("system", "SI"))
    b = d["basis"]
    Wnet = conv.get(b, "Wnet", "Pw", "Net power", positive=True)
    Pb = conv.get(b, "Pboil", "P", "Boiler pressure", positive=True)
    T3 = conv.get(b, "T3", "T", "Turbine inlet temperature")
    Pc = conv.get(b, "Pcond", "P", "Condenser pressure", positive=True)
    etaP = _efficiency(d["pump"].get("eta"), "Pump efficiency")
    etaT = _efficiency(d["turbine"].get("eta"), "Turbine efficiency")
    if Pc >= Pb:
        raise DesignError("The boiler pressure must be above the condenser pressure.")
    Pcrit = _props(WATER, "Pcrit", "T", 300, "P", 1e5)
    if Pb < Pcrit and T3 <= _props(WATER, "T", "P", Pb, "Q", 1):
        raise DesignError("The turbine inlet temperature must be above the boiling temperature at the boiler pressure.")
    h1 = _props(WATER, "H", "P", Pc, "Q", 0)
    s1 = _props(WATER, "S", "P", Pc, "Q", 0)
    T1 = _props(WATER, "T", "P", Pc, "Q", 0)
    h2s = _props(WATER, "H", "P", Pb, "S", s1)
    h2 = h1 + (h2s - h1) / etaP
    T2 = _props(WATER, "T", "P", Pb, "H", h2)
    h3 = _props(WATER, "H", "P", Pb, "T", T3)
    s3 = _props(WATER, "S", "P", Pb, "T", T3)
    h4s = _props(WATER, "H", "P", Pc, "S", s3)
    h4 = h3 - etaT * (h3 - h4s)
    x4 = _props(WATER, "Q", "P", Pc, "H", h4)
    T4 = _props(WATER, "T", "P", Pc, "H", h4)
    wnet = (h3 - h4) - (h2 - h1)
    m = Wnet / wnet
    Qin, Qout = m * (h3 - h2), m * (h4 - h1)
    st = {1: (Pc, T1, h1), 2: (Pb, T2, h2), 3: (Pb, T3, h3), 4: (Pc, T4, h4)}
    out = {"errors": [], "cycle": "rankine"}
    out["basis"] = {
        "rows": [conv.row("Steam flow ṁ", "mbig", m), conv.row("Heat in Q̇_in", "Pw", Qin),
                 conv.row("Heat rejected Q̇_out", "Pw", Qout), _plain("Thermal efficiency η_th", Wnet / Qin),
                 _plain("Back work ratio", (h2 - h1) / (h3 - h4)),
                 _plain("Turbine exit quality x₄", x4 if 0 <= x4 <= 1 else None, "", "superheated" if x4 > 1 or x4 < 0 else "")],
        "states": [{"state": k, "P": conv.show("P", s[0]), "T": conv.show("T", s[1]), "h": conv.show("h", s[2])}
                   for k, s in st.items()],
        "units": {k: conv.label(k) for k in ("P", "T", "h")},
    }

    def part(name, fn):
        try:
            out[name] = fn()
        except DesignError as e:
            out[name] = {"error": str(e)}

    part("pump", lambda: _pump(m, Pc, Pb, h1, h2, d["pump"], conv))
    part("turbine", lambda: _rankine_turbine(m, Pb, h3, Pc, h4, h4s, x4, d["turbine"], conv))
    part("boiler", lambda: _boiler(m, Pb, Pcrit, T2, h2, T3, h3, d["boiler"], conv))
    part("condenser", lambda: _condenser(m, Pc, h4, h1, T4, d["condenser"], conv))
    out["assemble"] = {"cycle": "rankine", "fluid": WATER, "P1_kPa": Pc / 1e3, "P2_kPa": Pb / 1e3, "T3_C": T3 - 273.15, "m": m,
                       "etaP": etaP, "etaT": etaT, "eta_th": Wnet / Qin, "W_MW": Wnet / 1e6}
    return out


def _pump(m, P1, P2, h1, h2, d, conv):
    rho = _props(WATER, "D", "P", P1, "Q", 0)
    Q = m / rho
    H = (P2 - P1) / (rho * G0)
    rpm = _num(d.get("rpm"), "Speed", positive=True)
    psi = _num(d.get("psi"), "Head coefficient ψ", positive=True)
    w = rpm * 2 * math.pi / 60

    def omega(n):
        return w * math.sqrt(Q) / (G0 * H / n) ** 0.75
    if _blank(d, "stages"):
        n = 1
        while omega(n) < 0.6 and n < 30:
            n += 1
        note = "fewest that give Ω ≥ 0.6"
    else:
        n = int(round(_num(d.get("stages"), "Stages", positive=True)))
        note = "as typed"
    Om = omega(n)
    U2 = math.sqrt(G0 * H / n / psi)
    D2 = 2 * U2 / w
    # Cavitation (Penoncello 4.7.6): the condensate is saturated, so the NPSH
    # available is the hotwell height above the pump less the suction losses.
    z = conv.get(d, "z", "H", "Hotwell height above the pump", nonneg=True)
    hL = conv.get(d, "hL", "H", "Suction line loss", nonneg=True, blank=0.0)
    S = _num(d.get("S"), "Suction specific speed", positive=True, blank=3.0)
    npsha = z - hL
    npshr = (w * math.sqrt(Q) / S) ** (4 / 3) / G0      # from S = ω√Q / (g NPSHr)^¾ (Dick, ch. 8)
    warnings = []
    if npsha < 1.1 * npshr:
        warnings.append(f"The pump would cavitate: NPSH available {conv.show('H', npsha):.3g} {conv.label('H')} is "
                        f"under 1.1 × the {conv.show('H', npshr):.3g} {conv.label('H')} required. Raise the hotwell, "
                        "slow the first stage, or add a low-speed booster pump.")
    if Om < 0.3:
        warnings.append("Specific speed is very low (Ω < 0.3): efficiency suffers. Add stages or raise the speed.")
    if Om > 2.5:
        warnings.append("Specific speed is high (Ω > 2.5): a mixed-flow or axial pump suits this duty better.")
    rows = [
        conv.row("Head", "H", H), conv.row("Volume flow", "Vdot", Q),
        _plain("Stages", n, "", note), _plain("Specific speed Ω per stage", Om, "", "centrifugal pumps run best near 0.8 to 1.2"),
        conv.row("Impeller tip speed", "V", U2), conv.row("Impeller diameter", "d", D2),
        conv.row("NPSH available", "H", npsha), conv.row("NPSH required (estimate)", "H", npshr,
                                                          f"suction specific speed S = {S:g}"),
        conv.row("Power", "Pw", m * (h2 - h1)),
    ]
    return {"rows": rows, "warnings": warnings, "summary": {"stages": n, "D2": D2}}


def _rankine_turbine(m, Pb, h3, Pc, h4, h4s, x4, d, conv):
    r = _turbine(WATER, m, {"P": Pb, "h": h3}, {"P": Pc, "h": h4}, h4s, d, conv, steam=True)
    if 0 <= x4 < 0.88:
        r["warnings"].append(f"Exit quality {x4:.3f} is wet enough to erode the last blades (keep x ≥ 0.88): "
                             "raise the turbine inlet temperature, lower the boiler pressure, or add reheat.")
    return r


def _boiler(m, Pb, Pcrit, T2, h2, T3, h3, d, conv):
    Tgin = conv.get(d, "Tgin", "T", "Furnace gas inlet temperature")
    Tgout = conv.get(d, "Tgout", "T", "Stack gas temperature")
    Do = conv.get(d, "Do", "d", "Tube outside diameter", positive=True)
    Di = conv.get(d, "Di", "d", "Tube inside diameter", positive=True, blank=Do * 0.84)
    kw = conv.get(d, "kw", "k", "Tube wall conductivity", positive=True, blank=40.0)
    Vg = conv.get(d, "Vgas", "V", "Gas velocity over the tubes", positive=True, blank=10.0)
    ntube = int(round(_num(d.get("circuits"), "Parallel water tubes", positive=True, blank=400)))
    LHV = conv.get(d, "LHV", "LHV", "Fuel heating value", positive=True)
    etaB = _efficiency(d.get("etaB"), "Boiler efficiency")
    if Tgout >= Tgin:
        raise DesignError("The gas must leave the boiler colder than it enters.")
    if Di >= Do:
        raise DesignError("The tube inside diameter must be smaller than the outside diameter.")
    Rfi = conv.get(d, "Rfi", "R", "Water-side fouling", nonneg=True, blank=FOULING["feedwater"])
    Rfo = conv.get(d, "Rfo", "R", "Gas-side fouling", nonneg=True, blank=FOULING["flueGas"])
    if Pb < Pcrit:
        Ts = _props(WATER, "T", "P", Pb, "Q", 0)
        hf, hg = _props(WATER, "H", "P", Pb, "Q", 0), _props(WATER, "H", "P", Pb, "Q", 1)
        zones = [{"name": "Economizer", "Q": m * (hf - h2), "T_in": T2, "T_out": Ts, "h_mean": 0.5 * (h2 + hf)},
                 {"name": "Evaporator", "Q": m * (hg - hf), "T_in": Ts, "T_out": Ts, "boil": True},
                 {"name": "Superheater", "Q": m * (h3 - hg), "T_in": Ts, "T_out": T3, "h_mean": 0.5 * (hg + h3)}]
    else:
        zones = [{"name": "Heating (supercritical)", "Q": m * (h3 - h2), "T_in": T2, "T_out": T3,
                  "h_mean": 0.5 * (h2 + h3)}]
    zones = [{**z, "U": 1.0} for z in zones if z["Q"] > 1e-6]
    Q = sum(z["Q"] for z in zones)
    C = Q / (Tgin - Tgout)
    zs = _zones(zones, False, Tgin, C, "Boiler", conv)      # temperatures and LMTD first
    # Film coefficients: water inside the tubes (Gnielinski, Eq. 5.26; Gungor-Winterton
    # where it boils), furnace gas in cross flow outside (Churchill-Bernstein, Eq. 5.20,
    # with air properties at the film temperature). Radiation from the flame is left out.
    G = m / (ntube * math.pi * Di * Di / 4)
    wall = Do * math.log(Do / Di) / (2 * kw)
    sat = _sat_props(WATER, Pb) if Pb < Pcrit else None
    for z in zs:
        Tg = 0.5 * (z["Tsec_in"] + z["Tsec_out"])
        gas = _transport(AIR, "T", 0.5 * (Tg + 0.5 * (z["T_in"] + z["T_out"])), 101325.0)
        z["Re_o"] = gas["rho"] * Vg * Do / gas["mu"]
        z["Nu_o"] = cv.nu_cross_cylinder(z["Re_o"], gas["Pr"])
        z["h_o"] = z["Nu_o"] * gas["k"] / Do

        def U_of(h_i, z=z):
            return 1 / (1 / z["h_o"] + Rfo + (Do / Di) * (1 / h_i + Rfi) + wall)     # Eq. 5.33
        if z.get("boil"):
            z["Re_i"] = G * Di / sat["mu_l"]
            h_i = 10000.0
            for _ in range(8):
                h_i = cv.h_boiling_tube(G, Di, 0.0, 1.0, U_of(h_i) * z["LMTD"] * Do / Di, sat)
            z["Nu_i"] = h_i * Di / sat["k_l"]
        else:
            w = _transport(WATER, "H", z["h_mean"], Pb)
            z["Re_i"] = G * Di / w["mu"]
            z["Nu_i"] = cv.nu_tube(z["Re_i"], w["Pr"], cv.ROUGHNESS * 30 / Di)   # commercial steel
            h_i = z["Nu_i"] * w["k"] / Di
        z["h_i"] = h_i
        z["U"] = U_of(h_i)
        z["A"] = z["UA"] / z["U"]
    A = sum(z["A"] for z in zs)
    mf = Q / (LHV * etaB)
    pinch = min(z["pinch"] for z in zs)
    warnings = []
    if pinch < 20:
        warnings.append(f"The gas comes within {conv.show('dT', pinch):.3g} {conv.label('dT')} of the steam; boilers "
                        "are usually kept 20 K or more apart. Raise the stack temperature or the gas inlet temperature.")
    rows = [
        conv.row("Heat duty", "Pw", Q), conv.row("Total tube area", "A", A),
        conv.row("Total tube length", "L", A / (math.pi * Do)),
        _plain("Parallel water tubes", ntube), conv.row("Length of each tube", "L", A / (math.pi * Do) / ntube),
        conv.row("Water mass flux in the tubes", "G", G),
        conv.row("Furnace gas flow", "mbig", C / CP_GAS), conv.row("Fuel burned", "mf", mf),
        conv.row("Closest approach (pinch)", "dT", pinch),
    ]
    return {"rows": rows, "warnings": warnings, "zones": _zone_table(zs, conv), "zoneUnits": _zone_units(conv),
            "film": {"inside": "Water in the tubes: Gnielinski (Eq. 5.26) for liquid and steam, Gungor–Winterton "
                               "where it boils (Nu there is on the liquid's k).",
                     "outside": "Furnace gas outside: Churchill–Bernstein cross flow (Eq. 5.20), convection only; "
                                "flame radiation would shrink the real evaporator."},
            "summary": {"A": A}}


def _condenser(m, Pc, h4, h1, T4, d, conv):
    Tin = conv.get(d, "Tin", "T", "Cooling water inlet temperature")
    Tout = conv.get(d, "Tout", "T", "Cooling water outlet temperature")
    Do = conv.get(d, "Do", "d", "Tube outside diameter", positive=True)
    Di = conv.get(d, "Di", "d", "Tube inside diameter", positive=True)
    Lt = conv.get(d, "Ltube", "L", "Tube length", positive=True)
    kw = conv.get(d, "kw", "k", "Tube wall conductivity", positive=True, blank=110.0)
    if Di >= Do:
        raise DesignError("The tube inside diameter must be smaller than the outside diameter.")
    if Tout <= Tin:
        raise DesignError("The cooling water must leave warmer than it enters.")
    # clean U plus fouling: cooling water inside the tubes, steam outside (Eq. 5.33)
    Rfi = conv.get(d, "Rfi", "R", "Water-side fouling", nonneg=True, blank=FOULING["water"])
    Rfo = conv.get(d, "Rfo", "R", "Steam-side fouling", nonneg=True, blank=FOULING["steam"])
    Ts = _props(WATER, "T", "P", Pc, "Q", 1)
    hg = _props(WATER, "H", "P", Pc, "Q", 1)
    zones = []
    if h4 > hg:
        zones.append({"name": "Desuperheating", "Q": m * (h4 - hg), "T_in": T4, "T_out": Ts, "U": 1.0, "dry": True})
    zones.append({"name": "Condensing", "Q": m * (min(h4, hg) - h1), "T_in": Ts, "T_out": Ts, "U": 1.0})
    Q = sum(z["Q"] for z in zones)
    C = Q / (Tout - Tin)
    zs = _zones(zones, True, Tin, C, "Condenser", conv)     # temperatures and LMTD first
    # Film coefficients: cooling water in the tubes (Gnielinski, Eq. 5.26), steam
    # condensing on the tube bank (Nusselt film theory, N tubes in a column). The
    # water velocity and N depend on the tube count, which depends on the area, so iterate.
    w = _transport(WATER, "T", 0.5 * (Tin + Tout), 2e5)
    mw = Q / (w["cp"] * (Tout - Tin))
    sat = _sat_props(WATER, Pc)
    wall = Do * math.log(Do / Di) / (2 * kw)
    ntubes, dT_film = 2000, 5.0
    for _ in range(60):
        vel = mw / (w["rho"] * ntubes * math.pi * Di * Di / 4)
        Re_i = w["rho"] * vel * Di / w["mu"]
        Nu_i = cv.nu_tube(Re_i, w["Pr"], cv.ROUGHNESS / Di)
        h_i = Nu_i * w["k"] / Di
        rows_N = max(1.0, math.sqrt(ntubes))
        h_o = cv.h_condensing_bank(Do, dT_film, rows_N, sat)
        U = 1 / (1 / h_o + Rfo + (Do / Di) * (1 / h_i + Rfi) + wall)       # Eq. 5.33
        for z in zs:
            z["U"] = U * (0.3 if z.get("dry") else 1.0)      # dry steam first: a much poorer film
            z["A"] = z["UA"] / z["U"]
            z.update({"Re_i": Re_i, "Nu_i": Nu_i, "h_i": h_i, "Re_o": None,
                      "Nu_o": h_o * Do / sat["k_l"], "h_o": h_o})
        A = sum(z["A"] for z in zs)
        new_n = max(1, math.ceil(A / (math.pi * Do) / Lt))
        new_dT = max(0.05, U * zs[-1]["LMTD"] / h_o)
        done = new_n == ntubes and abs(new_dT - dT_film) < 1e-4
        ntubes, dT_film = new_n, 0.5 * (dT_film + new_dT)
        if done:
            break
    warnings = []
    if Re_i < 3000:
        warnings.append(f"The cooling water is laminar or transitional (Re = {Re_i:.0f}); use longer tubes.")
    if vel < 1:
        warnings.append(f"Water velocity {conv.show('V', vel):.2g} {conv.label('V')} is low and fouls quickly: "
                        "use longer tubes (fewer of them) or two passes.")
    if vel > 2.5:
        warnings.append("Water velocity over 2.5 m/s erodes copper-alloy tubes: use shorter tubes (more of them).")
    rows = [
        conv.row("Heat duty", "Pw", Q), conv.row("Tube area", "A", A),
        _plain("Tubes (one pass)", ntubes, "", f"each {conv.show('L', Lt):.3g} {conv.label('L')} long"),
        conv.row("Cooling water flow", "mbig", mw), conv.row("Water velocity in the tubes", "V", vel),
        conv.row("Saturation temperature", "T", Ts),
    ]
    return {"rows": rows, "warnings": warnings, "zones": _zone_table(zs, conv), "zoneUnits": _zone_units(conv),
            "film": {"inside": "Cooling water in the tubes: Gnielinski (Eq. 5.26).",
                     "outside": f"Steam outside: Nusselt film condensation on a bank about {round(math.sqrt(ntubes))} "
                                "tubes deep (Nu is on the condensate's k)."},
            "summary": {"A": A, "tubes": ntubes}}


# ---------------------------------------------------------------- Brayton

def brayton(payload):
    d = _merged("brayton", payload)
    conv = _Conv(payload.get("system", "SI"))
    b = d["basis"]
    Wnet = conv.get(b, "Wnet", "Pw", "Net power", positive=True)
    P1 = conv.get(b, "P1", "P", "Inlet pressure", positive=True)
    T1 = conv.get(b, "T1", "T", "Inlet temperature")
    rp = _num(b.get("rp"), "Pressure ratio", positive=True)
    T3 = conv.get(b, "T3", "T", "Turbine inlet temperature")
    etaC = _efficiency(d["compressor"].get("eta"), "Compressor efficiency")
    etaT = _efficiency(d["turbine"].get("eta"), "Turbine efficiency")
    if rp <= 1:
        raise DesignError("The pressure ratio must be above 1.")
    P2 = P1 * rp
    h1, s1 = _props(AIR, "H", "P", P1, "T", T1), _props(AIR, "S", "P", P1, "T", T1)
    h2s = _props(AIR, "H", "P", P2, "S", s1)
    h2 = h1 + (h2s - h1) / etaC
    T2 = _props(AIR, "T", "P", P2, "H", h2)
    if T3 <= T2:
        raise DesignError("The turbine inlet temperature must be above the compressor exit temperature.")
    h3, s3 = _props(AIR, "H", "P", P2, "T", T3), _props(AIR, "S", "P", P2, "T", T3)
    h4s = _props(AIR, "H", "P", P1, "S", s3)
    h4 = h3 - etaT * (h3 - h4s)
    T4 = _props(AIR, "T", "P", P1, "H", h4)
    wnet = (h3 - h4) - (h2 - h1)
    if wnet <= 0:
        raise DesignError("This engine makes no net work: raise the turbine inlet temperature or the efficiencies.")
    m = Wnet / wnet
    Qin = m * (h3 - h2)
    st = {1: (P1, T1, h1), 2: (P2, T2, h2), 3: (P2, T3, h3), 4: (P1, T4, h4)}
    out = {"errors": [], "cycle": "brayton"}
    out["basis"] = {
        "rows": [conv.row("Air flow ṁ", "mbig", m), conv.row("Heat in Q̇_in", "Pw", Qin),
                 _plain("Thermal efficiency η_th", Wnet / Qin), _plain("Back work ratio", (h2 - h1) / (h3 - h4)),
                 conv.row("Exhaust temperature", "T", T4)],
        "states": [{"state": k, "P": conv.show("P", s[0]), "T": conv.show("T", s[1]), "h": conv.show("h", s[2])}
                   for k, s in st.items()],
        "units": {k: conv.label(k) for k in ("P", "T", "h")},
    }
    shaft = {}

    def comp():
        r, shaft["s"] = _axial_compressor(AIR, m, {"P": P1, "h": h1}, {"P": P2, "h": h2}, d["compressor"], conv)
        return r

    def part(name, fn):
        try:
            out[name] = fn()
        except DesignError as e:
            out[name] = {"error": str(e)}
    part("compressor", comp)
    part("combustor", lambda: _combustor(m, Qin, T3, P2, d["combustor"], conv, AIR))
    part("turbine", lambda: _turbine(AIR, m, {"P": P2, "h": h3}, {"P": P1, "h": h4}, h4s, d["turbine"], conv,
                                     rpm_from=shaft.get("s")))
    out["assemble"] = {"cycle": "brayton", "fluid": AIR, "P1_kPa": P1 / 1e3, "T1_C": T1 - 273.15, "rp": rp, "T3_C": T3 - 273.15, "m": m,
                       "etaC": etaC, "etaT": etaT, "eta_th": Wnet / Qin, "W_MW": Wnet / 1e6}
    return out


# --------------------------------------------------------------- turbojet

def turbojet(payload):
    d = _merged("turbojet", payload)
    conv = _Conv(payload.get("system", "SI"))
    b = d["basis"]
    F = conv.get(b, "thrust", "F", "Thrust wanted", positive=True)
    T0 = conv.get(b, "T0", "T", "Ambient temperature")
    P0 = conv.get(b, "P0", "P", "Ambient pressure", positive=True)
    V0 = conv.get(b, "V0", "V", "Flight speed", nonneg=True)
    rp = _num(b.get("rp"), "Compressor pressure ratio", positive=True)
    T4 = conv.get(b, "T4", "T", "Turbine inlet temperature")
    etaD = _efficiency(d["diffuser"].get("eta"), "Diffuser efficiency")
    etaC = _efficiency(d["compressor"].get("eta"), "Compressor efficiency")
    etaT = _efficiency(d["turbine"].get("eta"), "Turbine efficiency")
    etaN = _efficiency(d["nozzle"].get("eta"), "Nozzle efficiency")
    if rp <= 1:
        raise DesignError("The pressure ratio must be above 1.")
    h0, s0 = _props(AIR, "H", "P", P0, "T", T0), _props(AIR, "S", "P", P0, "T", T0)
    h1 = h0 + V0 * V0 / 2                                 # diffuser exit, V₁ ≈ 0
    h1s = h0 + etaD * V0 * V0 / 2                         # isentropic to the same pressure
    P1 = _props(AIR, "P", "H", h1s, "S", s0)
    T1 = _props(AIR, "T", "P", P1, "H", h1)
    s1 = _props(AIR, "S", "P", P1, "H", h1)
    P2 = P1 * rp
    h2s = _props(AIR, "H", "P", P2, "S", s1)
    h2 = h1 + (h2s - h1) / etaC
    T2 = _props(AIR, "T", "P", P2, "H", h2)
    if T4 <= T2:
        raise DesignError("The turbine inlet temperature must be above the compressor exit temperature.")
    h3, s3 = _props(AIR, "H", "P", P2, "T", T4), _props(AIR, "S", "P", P2, "T", T4)
    h4 = h3 - (h2 - h1)                                   # turbine drives the compressor
    h4s = h3 - (h3 - h4) / etaT
    P4 = _props(AIR, "P", "H", h4s, "S", s3)
    if P4 <= P0:
        raise DesignError("The turbine uses up all the pressure driving the compressor; nothing is left for the "
                          "nozzle. Raise the turbine inlet temperature or the efficiencies.")
    T4t = _props(AIR, "T", "P", P4, "H", h4)
    s4 = _props(AIR, "S", "P", P4, "H", h4)
    h5s = _props(AIR, "H", "P", P0, "S", s4)
    h5 = h4 - etaN * (h4 - h5s)
    V5 = math.sqrt(2 * (h4 - h5))
    T5 = _props(AIR, "T", "P", P0, "H", h5)
    if V5 <= V0:
        raise DesignError("The jet leaves slower than the flight speed, so there is no thrust.")
    m = F / (V5 - V0)
    Q = m * (h3 - h2)
    st = {"1 (diffuser exit)": (P1, T1, h1), "2 (compressor exit)": (P2, T2, h2), "3 (turbine inlet)": (P2, T4, h3),
          "4 (turbine exit)": (P4, T4t, h4), "5 (nozzle exit)": (P0, T5, h5)}
    out = {"errors": [], "cycle": "turbojet"}
    comb = None
    try:
        comb = _combustor(m, Q, T4, P2, d["combustor"], conv, AIR)
    except DesignError as e:
        out["combustor"] = {"error": str(e)}
    mf = comb["summary"]["mf"] if comb else None
    eta_p = 2 * V0 / (V5 + V0)
    out["basis"] = {
        "rows": [conv.row("Air flow ṁ", "mbig", m), conv.row("Jet velocity", "V", V5),
                 conv.row("Propulsive power F·V₀", "Pw", F * V0), _plain("Propulsive efficiency", eta_p),
                 *([conv.row("Thrust-specific fuel use (TSFC)", "TSFC", mf / F),
                    _plain("Overall efficiency", F * V0 / (mf * conv.si("LHV", _num(d["combustor"].get("LHV"), "LHV"))))]
                   if mf else [])],
        "states": [{"state": k, "P": conv.show("P", s[0]), "T": conv.show("T", s[1]), "h": conv.show("h", s[2])}
                   for k, s in st.items()],
        "units": {k: conv.label(k) for k in ("P", "T", "h")},
    }
    shaft = {}

    def comp():
        r, shaft["s"] = _axial_compressor(AIR, m, {"P": P1, "h": h1}, {"P": P2, "h": h2}, d["compressor"], conv)
        return r

    def part(name, fn):
        try:
            out[name] = fn()
        except DesignError as e:
            out[name] = {"error": str(e)}
    part("diffuser", lambda: _diffuser(m, P0, T0, V0, P1, conv))
    part("compressor", comp)
    if comb:
        out["combustor"] = comb
    part("turbine", lambda: _turbine(AIR, m, {"P": P2, "h": h3}, {"P": P4, "h": h4}, h4s, d["turbine"], conv,
                                     rpm_from=shaft.get("s")))
    part("nozzle", lambda: _nozzle(m, P4, h4, s4, P0, etaN, V5, V0, F, conv))
    out["assemble"] = {"cycle": "turbojet", "fluid": AIR, "P0_kPa": P0 / 1e3, "T0_C": T0 - 273.15, "V0": V0, "m": m, "rp": rp,
                       "T4_C": T4 - 273.15, "etaD": etaD, "etaC": etaC, "etaT": etaT, "etaN": etaN,
                       "F_kN": F / 1e3}
    return out


def _diffuser(m, P0, T0, V0, P1, conv):
    rho0 = _props(AIR, "D", "P", P0, "T", T0)
    rows = [conv.row("Capture area (free stream)", "A", m / (rho0 * V0) if V0 > 0 else None),
            conv.row("Ram pressure rise", "dp", P1 - P0), _plain("Ram pressure ratio", P1 / P0)]
    warnings = [] if V0 > 0 else ["At zero flight speed there is no ram rise; the capture area is undefined."]
    return {"rows": rows, "warnings": warnings, "summary": {}}


def _nozzle(m, P4, h4, s4, P0, etaN, V5, V0, F, conv):
    """Throat where ρV peaks along the expansion; C-D nozzle if it is above P0."""
    best = (0.0, None)
    n = 200
    for i in range(1, n + 1):
        P = P4 + (P0 - P4) * i / n
        hs = _props(AIR, "H", "P", P, "S", s4)
        h = h4 - etaN * (h4 - hs)
        G = _props(AIR, "D", "P", P, "H", h) * math.sqrt(2 * (h4 - h))
        if G > best[0]:
            best = (G, P)
    Gmax, Pt = best
    choked = Pt is not None and Pt > P0 * 1.0001
    h5s = _props(AIR, "H", "P", P0, "S", s4)
    h5 = h4 - etaN * (h4 - h5s)
    rho5 = _props(AIR, "D", "P", P0, "H", h5)
    Ae = m / (rho5 * V5)
    rows = [_plain("Nozzle pressure ratio P₄/P₀", P4 / P0),
            _plain("Type", "converging-diverging" if choked else "converging", "",
                   "the flow chokes, so it must diverge after the throat to expand fully" if choked else "")]
    if choked:
        At = m / Gmax
        rows += [conv.row("Throat area", "A", At), conv.row("Throat pressure", "P", Pt),
                 conv.row("Exit area (fully expanded)", "A", Ae), _plain("Exit-to-throat area ratio", Ae / At)]
    else:
        rows += [conv.row("Exit area", "A", Ae)]
    rows += [conv.row("Jet velocity", "V", V5), conv.row("Thrust F = ṁ(V₅ − V₀)", "F", F)]
    return {"rows": rows, "warnings": [], "summary": {"Ae": Ae}}


def design_power(payload):
    fn = {"rankine": rankine, "brayton": brayton, "turbojet": turbojet}[payload["cycle"]]
    try:
        return fn(payload)
    except DesignError as e:
        return {"errors": [str(e)], "cycle": payload["cycle"]}
