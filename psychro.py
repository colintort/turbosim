"""Psychrometrics: humid air as a stream fluid, and the air-conditioning
processes that work on it.

A state whose fluid is HUMID_AIR is a mixture of dry air and water vapor at
pressure P. Everything is per kg of DRY air, as in the psychrometric chart:
    m    dry-air mass flow (kg/s)
    h    enthalpy, J per kg of dry air (0 for dry air at 0 °C)
    s    entropy, J per kg of dry air per K
    w    humidity ratio ω, kg of water vapor per kg of dry air
    RH   relative humidity φ, 0 to 1
P and any two of T, h, w, RH fix the state (CoolProp's HAPropsSI).

The AC device (type "ac") is one air stream in and out, plus a water stream for
the modes that add or remove water. One mode for every way a state can move on
the psychrometric chart (T across, ω up):
    heating          heater: heat only, ω₂ = ω₁                      (right)
    sensibleCooling  dry coil above the dew point: ω₂ = ω₁           (left)
    cooling          cooling coil: cooled and dehumidified; condensate
                     drains at "water" (ω₁ − ω₂ per kg air)        (down-left)
    coolReheat       dehumidifier: cooled to the dew point of the outlet
                     air, condensate drained, then reheated           (down)
    desiccant        desiccant wheel: water held by the desiccant, the
                     heat it gives off warms the air (q = 0: h₂ = h₁) (down-right)
    steam            steam humidifier: steam injected, Q = 0          (up)
    humidifying      heater + humidifier: heat and water or steam   (up-right)
    evaporative      water sprayed in with no heat (Q = 0)            (up-left)
Adiabatic mixing of two airstreams is the mixing chamber (mix_moisture below).
Per kg of dry air, with water through the port in either direction:
    h₂ = h₁ + (ω₂ − ω₁) h_water + q,       ṁ_water = ṁ_a |ω₂ − ω₁|
"""

HUMID_AIR = "Humid air"
HUMID_FIELDS = ("w", "RH")
# Pairs (with P) handed to HAPropsSI, in the order tried. CoolProp names.
_PAIRS = [("T", "RH"), ("T", "w"), ("T", "h"), ("h", "w"), ("h", "RH"), ("w", "RH")]
_HA = {"T": "T", "h": "H", "w": "W", "RH": "R", "s": "S"}
AC_MODES = ("heating", "sensibleCooling", "cooling", "coolReheat", "desiccant", "steam", "humidifying", "evaporative")
WATER_OUT = ("cooling", "coolReheat")              # condensate leaves the air at "water"
WATER_IN = ("humidifying", "evaporative", "steam")  # water or steam enters at "water"
SAME_W = ("heating", "sensibleCooling")             # ω₂ = ω₁
NO_HEAT = ("evaporative", "steam")                  # Q = 0 always
REF_MODES = ("heating", "sensibleCooling", "cooling")   # can carry a refrigerant side
COOLERS = ("sensibleCooling", "cooling", "coolReheat")
TON = 12000 * 1055.05585262 / 3600                  # W per ton of refrigeration
# Names shown on the chart next to each process line.
AC_LABELS = {
    "heating": "Heater", "sensibleCooling": "Sensible cooler", "cooling": "Cooling coil",
    "coolReheat": "Dehumidifier", "desiccant": "Desiccant", "steam": "Steam humidifier",
    "humidifying": "Heat + humidify", "evaporative": "Evaporative cooler",
}


def is_humid(solver, n):
    return solver._fluid(n) == HUMID_AIR


def _need_ha(solver, where):
    if solver.ha_props is None:
        from system_solver import SolveError
        raise SolveError(f"{where}: humid air needs CoolProp's humid-air functions, which aren't available.")
    return solver.ha_props


def _t_from_w_rh(ha, P, w, rh):
    """T at which humidity ratio w has relative humidity rh (bisection)."""
    lo, hi = 200.0, 420.0

    def f(t):
        try:
            return ha("R", "T", t, "P", P, "W", w) - rh
        except (ValueError, RuntimeError):
            return 1.0          # below the dew point: more than saturated

    if f(lo) * f(hi) > 0:
        raise ValueError(f"no temperature gives φ = {rh * 100:.4g}% with ω = {w:.4g}")
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def resolve(solver, n):
    """The state pass for a humid-air state: P plus two of T, h, w, RH give
    the rest. Returns True if anything new was found."""
    from system_solver import SolveError, _close
    v = solver._v(n)
    if "RH" in v and not (-1e-9 <= v["RH"] <= 1 + 1e-9):
        raise SolveError(f"S{n}: relative humidity {v['RH'] * 100:.4g}% must be between 0 and 100%.")
    if "w" in v and v["w"] < 0:
        raise SolveError(f"S{n}: the humidity ratio ω can't be negative.")
    if "P" not in v:
        return False
    pairs = [p for p in _PAIRS if p[0] in v and p[1] in v]
    if not pairs:
        return False
    signature = tuple(sorted(v))
    if solver.verified.get(n) == signature:
        return False
    ha = _need_ha(solver, f"S{n}")
    k1, k2 = pairs[0]
    P = v["P"]
    try:
        if (k1, k2) == ("w", "RH"):
            T = _t_from_w_rh(ha, P, v["w"], v["RH"])
            k1, k2, a, b = "T", "w", T, v["w"]
        else:
            a, b = v[k1], v[k2]
        found = {}
        for k in ("T", "h", "w", "RH", "s"):
            if k in (k1, k2):
                continue
            found[k] = ha(_HA[k], _HA[k1], a, "P", P, _HA[k2], b)
        if k1 == "T" and "T" not in v:
            found["T"] = a
    except (ValueError, RuntimeError) as e:
        raise SolveError(f"S{n}: humid air at these values is out of range ({e}).")
    changed = False
    for k, value in found.items():
        if k in v:
            if not _close(v[k], value):
                raise SolveError(
                    f"S{n}: {k} = {solver._fmt(k, v[k])} doesn't match the other values given "
                    f"(they give {solver._fmt(k, value)}). The state is over-specified.")
        else:
            v[k] = value
            changed = True
    solver.verified[n] = tuple(sorted(v))
    return changed


def ac(solver, node_id, node, name, ports):
    """Air-conditioning process (see the module docstring)."""
    from system_solver import SolveError, _number
    from units import to_si, power_factor
    mode = (node.get("config") or {}).get("mode", "heating")
    if mode not in AC_MODES:
        raise SolveError(f"{name}: unknown air-conditioning process '{mode}'.")
    label = AC_LABELS[mode].lower()
    asm = node.get("assumptions") or {}
    params = node.get("params") or {}
    a, b = ports["in"], ports["out"]
    unused = set(node.get("unused") or [])
    has_port = mode in WATER_OUT or mode in WATER_IN
    wat = ports.get("water") if has_port and "water" not in unused else None
    for s in (a, b):
        if not is_humid(solver, s):
            raise SolveError(f"{name}: the air stream (S{a} to S{b}) must be Humid air. Pick it in the state table's Fluid column.")
    if has_port and wat is None:
        raise SolveError(f"{name}: connect the water port ({'condensate out' if mode in WATER_OUT else 'water in'}).")

    va, vb = solver._v(a), solver._v(b)
    changed = solver._propagate(a, b, "m", name)                  # dry air in = dry air out
    if asm.get("isobaric"):
        changed |= solver._propagate(a, b, "P", name)
    if mode in SAME_W:
        changed |= solver._propagate(a, b, "w", name)
    vw = solver._v(wat) if wat is not None else {}
    if mode == "cooling" and asm.get("saturatedOut"):
        changed |= _coil_outlet(solver, name, a, b, va, vb)
    if mode == "cooling" and asm.get("condensateAtOutletT", True) and "T" in vb:
        changed |= solver._set(wat, "T", vb["T"], name)
        changed |= solver._set(wat, "x", 0.0, name)
    if mode == "coolReheat" and asm.get("condensateAtCoilT", True) and "w" in vb and "P" in vb:
        changed |= solver._set(wat, "T", _dew_point(solver, vb["P"], vb["w"], name), name)
        changed |= solver._set(wat, "x", 0.0, name)
    _check_direction(solver, name, mode, label, va, vb)

    # water: ṁ_w = ṁ_a |ω₂ − ω₁|
    if wat is not None:
        sign = -1.0 if mode in WATER_OUT else 1.0                  # out of the air / into it
        ma, mw = va.get("m"), vw.get("m")
        w1, w2 = va.get("w"), vb.get("w")
        if ma is not None and w1 is not None and w2 is not None:
            m_water = sign * ma * (w2 - w1)
            if m_water < -1e-9 * max(ma, 1.0):
                what = "loses" if mode in WATER_IN else "gains"
                raise SolveError(f"{name}: the air {what} moisture (ω {w1:.4g} to {w2:.4g}), "
                                 f"which a {label} can't do.")
            changed |= solver._set(wat, "m", max(m_water, 0.0), name)
        elif ma is not None and mw is not None and w1 is not None:
            changed |= solver._set(b, "w", w1 + sign * mw / ma, name)
        elif ma is not None and mw is not None and w2 is not None:
            changed |= solver._set(a, "w", w2 - sign * mw / ma, name)
        elif mw is not None and w1 is not None and w2 is not None and abs(w2 - w1) > 1e-12:
            changed |= solver._set(a, "m", mw / (sign * (w2 - w1)), name)

    # The other side of the coil (refrigerant, chilled or hot water): one
    # stream through, which takes up exactly the heat the air gives off.
    ri, ro = _ref_ports(node, ports, mode)
    q_ref = None                                                  # W into the refrigerant
    if ri is not None:
        for s in (ri, ro):
            if is_humid(solver, s):
                raise SolveError(f"{name}: the refrigerant side (S{ri} to S{ro}) can't be Humid air. "
                                 f"Pick its fluid in the state table's Fluid column.")
        changed |= solver._propagate(ri, ro, "m", name)
        if asm.get("refIsobaric"):
            changed |= solver._propagate(ri, ro, "P", name)
        if asm.get("refSatVaporOut"):
            changed |= solver._set(ro, "x", 1.0, name)
        if asm.get("refSatLiquidOut"):
            changed |= solver._set(ro, "x", 0.0, name)
        vi, vo = solver._v(ri), solver._v(ro)
        if all(k in vi for k in ("m", "h")) and "h" in vo:
            q_ref = vi["m"] * (vo["h"] - vi["h"])

    # energy per kg of dry air: h₂ = h₁ + (ω₂ − ω₁) h_w + q
    q = None
    if mode in NO_HEAT or (mode == "desiccant" and asm.get("adiabatic", True)):
        q = 0.0
    else:
        q_kg = _number(params.get("q"), f"{name} q")
        q_dot = _number(params.get("Qdot"), f"{name} Q̇")
        if q_kg is None and q_dot is None and q_ref is not None:
            # the refrigerant side is known: it sets the heat the air loses
            if va.get("m"):
                q = -q_ref / va["m"]
            elif vb.get("h") is not None and va.get("h") is not None:
                dw = _water_term(va, vb, vw, wat)
                if dw is not None and abs(vb["h"] - va["h"] - dw) > 1e-9:
                    changed |= solver._set(a, "m", -q_ref / (vb["h"] - va["h"] - dw), name)
        if q_kg is not None:
            q = to_si(solver.system, "h", q_kg)
        elif q_dot is not None and va.get("m"):
            q = q_dot * power_factor(solver.system) / va["m"]
        elif q_dot is not None and vb.get("h") is not None and va.get("h") is not None:
            dw = _water_term(va, vb, vw, wat)
            if dw is not None and abs(vb["h"] - va["h"] - dw) > 1e-9:
                changed |= solver._set(a, "m", q_dot * power_factor(solver.system) / (vb["h"] - va["h"] - dw), name)
    dw = _water_term(va, vb, vw, wat)
    if q is not None and dw is not None:
        changed |= solver._linear([(a, -1), (b, 1)], "h", name, "energy", -dw - q)
    elif q is not None and wat is not None and "h" in vw and "h" in va and "w" in va and "w" not in vb:
        changed |= _close_outlet(solver, name, va, vb, b, vw["h"], q)

    # air side known: its heat goes into the refrigerant, ṁ_r (h_out − h_in) = −Q̇_air
    if ri is not None and q_ref is None:
        dw = _water_term(va, vb, vw, wat)
        if dw is not None and all(k in va for k in ("m", "h")) and "h" in vb:
            q_air = va["m"] * (vb["h"] - va["h"] - dw)
            changed |= solver._cv(name, [(ri, 1), (ro, -1)], [[ri, ro]], -q_air)
    return changed


def _ref_ports(node, ports, mode):
    """(inlet, outlet) states of the coil's refrigerant side, or (None, None)."""
    if not (node.get("config") or {}).get("refrigerant") or mode not in REF_MODES:
        return None, None
    ri, ro = ports.get("refIn"), ports.get("refOut")
    if ri is None or ro is None:
        return None, None
    return ri, ro


def _coil_outlet(solver, name, a, b, va, vb):
    """Cooling coil, "air leaves saturated once it condenses": with the outlet
    T below the inlet's dew point the air leaves at φ₂ = 100 %, so the outlet
    T alone fixes it; above the dew point nothing condenses and ω₂ = ω₁."""
    if "T" not in vb or "w" not in va or "P" not in va:
        return False
    if "w" in vb and "RH" in vb:
        return False            # the outlet is fixed already; resolve() checks it
    t_dp = _dew_point(solver, va["P"], va["w"], name)
    if vb["T"] < t_dp - 1e-6:
        return solver._set(b, "RH", 1.0, f"{name} (φ = 100 % once the air condenses)")
    return solver._propagate(a, b, "w", name)


def _dew_point(solver, P, w, name):
    """Dew-point temperature of air with humidity ratio w at pressure P."""
    from system_solver import SolveError
    ha = _need_ha(solver, name)
    try:
        return ha("D", "P", P, "W", w, "T", 300.0)   # T doesn't change the dew point
    except (ValueError, RuntimeError) as e:
        raise SolveError(f"{name}: no dew point for ω = {w:.4g} ({e}).")


def _check_direction(solver, name, mode, label, va, vb):
    """Each process moves one way on the chart; say which device does the
    move asked for when the states point the other way."""
    from system_solver import SolveError
    tol = 1e-6
    if "T" in va and "T" in vb:
        dT = vb["T"] - va["T"]
        if mode == "heating" and dT < -tol:
            raise SolveError(f"{name}: a heater can't cool the air (T falls {-dT:.3g} K). Use a Sensible cooler or Cooling coil.")
        if mode == "sensibleCooling" and dT > tol:
            raise SolveError(f"{name}: a sensible cooler can't warm the air (T rises {dT:.3g} K). Use a Heater.")
    if "w" in va and "w" in vb:
        dw = vb["w"] - va["w"]
        if mode == "desiccant" and dw > tol * 1e-3:
            raise SolveError(f"{name}: a desiccant only takes water out of the air (ω rises {va['w']:.4g} to {vb['w']:.4g}).")
    if mode == "sensibleCooling" and "T" in vb and "w" in va and "P" in va:
        t_dp = _dew_point(solver, va["P"], va["w"], name)
        if vb["T"] < t_dp - 1e-3:
            raise SolveError(
                f"{name}: {solver._fmt('T', vb['T'])} is below the air's dew point ({solver._fmt('T', t_dp)}), "
                f"so water would condense. Use a Cooling coil, which drains the condensate.")


def _close_outlet(solver, name, va, vb, b, h_w, q):
    """Outlet ω when only one of T or φ is given there (an evaporative cooler
    to 30 % φ, say): ω₂ with h₂ = h₁ + (ω₂ − ω₁) h_w + q lands on it."""
    from system_solver import SolveError
    target = next((k for k in ("RH", "T") if k in vb), None)
    if target is None or "P" not in vb:
        return False
    ha, P = solver.ha_props, vb["P"]

    def miss(w2):
        h2 = va["h"] + (w2 - va["w"]) * h_w + q
        return ha(_HA[target], "H", h2, "P", P, "W", w2) - vb[target]

    # Walk ω₂ up from a little below ω₁ until the outlet value is passed;
    # past saturation CoolProp has no answer, so the walk stops there.
    w = max(va["w"] * 0.5, 1e-6)
    lo = f_lo = hi = None
    for _ in range(200):
        try:
            f = miss(w)
        except (ValueError, RuntimeError):
            break
        if f_lo is not None and f_lo * f <= 0:
            hi = w
            break
        lo, f_lo = w, f
        w *= 1.05
    if hi is None:
        raise SolveError(f"{name}: no amount of water reaches the outlet {('φ' if target == 'RH' else 'T')} given.")
    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = miss(mid)
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return solver._set(b, "w", (lo + hi) / 2, name)


def _water_term(va, vb, vw, wat):
    """(ω₂ − ω₁) h_water per kg of dry air, 0 with no water port, None while unknown."""
    if wat is None:
        return 0.0
    if "w" not in va or "w" not in vb or "h" not in vw:
        return None
    return (vb["w"] - va["w"]) * vw["h"]


def ac_report(solver, node, ports):
    """q per kg of dry air, Q̇ and the water flow, in display units."""
    from system_solver import _round
    from units import from_si, power_factor
    mode = (node.get("config") or {}).get("mode", "heating")
    va, vb = solver.values.get(ports["in"], {}), solver.values.get(ports["out"], {})
    unused = set(node.get("unused") or [])
    has_port = mode in WATER_OUT or mode in WATER_IN
    wat = ports.get("water") if has_port and "water" not in unused else None
    vw = solver.values.get(wat, {}) if wat is not None else {}
    r = {}
    ma = va.get("m")
    sys_ = solver.system
    rate = lambda q: _round(q * ma / power_factor(sys_))  # noqa: E731
    dw = _water_term(va, vb, vw, wat)
    if "h" in va and "h" in vb and dw is not None:
        q = vb["h"] - va["h"] - dw
        r["q"] = _round(from_si(sys_, "h", q))
        if ma is not None:
            r["Qdot"] = rate(q)
        # Capacity, and the split into sensible (the T change at the inlet's ω)
        # and latent (the moisture change) heat.
        ha = solver.ha_props
        if ma is not None:
            if mode in COOLERS and q < 0:
                r["capacity"] = rate(-q)
                r["tons"] = _round(-q * ma / TON)
            elif mode in ("heating", "humidifying") and q > 0:
                r["capacity"] = rate(q)
        if ha is not None and mode not in NO_HEAT and all(k in vb for k in ("T", "P")) and "w" in va:
            try:
                q_s = ha("H", "T", vb["T"], "P", vb["P"], "W", va["w"]) - va["h"]
                q_l = q - q_s
                if ma is not None:
                    r["Qsens"], r["Qlat"] = rate(q_s), rate(q_l)
                if abs(q) > 1e-9:
                    r["SHR"] = _round(q_s / q)
            except (ValueError, RuntimeError):
                pass
    if solver.ha_props is not None and all(k in va for k in ("T", "P", "w")):
        try:
            ha = solver.ha_props
            if ma is not None:
                r["Vin"] = _round(from_si(sys_, "vol", ma * ha("V", "T", va["T"], "P", va["P"], "W", va["w"])))
            r["Tdp"] = _round(from_si(sys_, "T", ha("D", "T", va["T"], "P", va["P"], "W", va["w"])))
            r["Twb"] = _round(from_si(sys_, "T", ha("B", "T", va["T"], "P", va["P"], "W", va["w"])))
        except (ValueError, RuntimeError):
            pass
    ri, ro = _ref_ports(node, ports, mode)
    if ri is not None:
        vi, vo = solver.values.get(ri, {}), solver.values.get(ro, {})
        if all(k in vi for k in ("m", "h")) and "h" in vo:
            r["Qref"] = _round(vi["m"] * (vo["h"] - vi["h"]) / power_factor(sys_))
            r["mRef"] = _round(from_si(sys_, "m", vi["m"]))
    if vw.get("m") is not None:
        r["mWater"] = _round(from_si(sys_, "m", vw["m"]))
    elif mode == "desiccant" and ma is not None and "w" in va and "w" in vb:
        r["mWater"] = _round(from_si(sys_, "m", ma * (va["w"] - vb["w"])))
    # Dehumidifier: the coil cools the air to its dew point (saturated at ω₂),
    # then the reheater warms it to the outlet state.
    if mode == "coolReheat" and all(k in vb for k in ("P", "w", "h")) and "h" in va and "w" in va \
            and solver.ha_props is not None:
        try:
            t_c = solver.ha_props("D", "P", vb["P"], "W", vb["w"], "T", 300.0)
            h_c = solver.ha_props("H", "T", t_c, "P", vb["P"], "W", vb["w"])
        except (ValueError, RuntimeError):
            h_c = t_c = None
        if h_c is not None:
            h_cond = vw.get("h", 0.0)
            q_cool = h_c - va["h"] - (vb["w"] - va["w"]) * h_cond
            q_heat = vb["h"] - h_c
            r["Tcoil"] = _round(from_si(sys_, "T", t_c))
            r["qCool"] = _round(from_si(sys_, "h", q_cool))
            r["qReheat"] = _round(from_si(sys_, "h", q_heat))
            if ma is not None:
                r["QdotCool"] = rate(q_cool)
                r["QdotReheat"] = rate(q_heat)
    return r


def mix_moisture(solver, name, ins, out):
    """Water balance of humid streams mixing: Σ ṁ ω (in) = ṁ ω (out)."""
    masses = [solver._v(s).get("m") for s in ins + [out]]
    if any(m is None for m in masses):
        return False
    terms = [(s, m) for s, m in zip(ins, masses)] + [(out, -masses[-1])]
    return solver._linear(terms, "w", name, "moisture")


def chart(solver, from_si):
    """Psychrometric chart data in display units: T on x, ω on y, with the
    saturation curve as the 'dome', the humid states and the process lines."""
    hum = [n for n in solver.values if is_humid(solver, n)]
    pts = {n: solver.values[n] for n in hum if "T" in solver.values[n] and "w" in solver.values[n]}
    if not pts or solver.ha_props is None:
        return None
    P = next((v["P"] for v in pts.values() if "P" in v), 101325.0)
    sys_ = solver.system
    # A textbook-style window (0 to 45 °C, ω up to 0.03), widened to fit the states.
    tlo = min([273.15] + [v["T"] - 3 for v in pts.values()])
    thi = max([318.15] + [v["T"] + 3 for v in pts.values()])
    whi = max([0.03] + [v["w"] * 1.15 + 0.001 for v in pts.values()])
    sat, grid = _chart_lines(solver.ha_props, sys_, round(P, 1), round(tlo, 2), round(thi, 2), round(whi, 4))
    states = {str(n): [from_si(sys_, "T", v["T"]), v["w"]] for n, v in pts.items()}
    paths = []
    for node in solver.nodes.values():
        p = node.get("ports") or {}
        t = node.get("type")
        if t in ("ac", "mixer", "splitter", "coolingTower"):
            ins = [int(v) for k, v in p.items() if k in ("in", "airIn") or (k.startswith("in") and k[2:].isdigit())]
            outs = [int(v) for k, v in p.items() if k in ("out", "airOut") or (k.startswith("out") and k[3:].isdigit())]
            for i in ins:
                for o in outs:
                    if str(i) in states and str(o) in states:
                        kind, label = "", {"mixer": "Mixing", "splitter": "", "coolingTower": "Tower air"}.get(t, "")
                        if t == "ac":
                            mode = (node.get("config") or {}).get("mode")
                            label = AC_LABELS.get(mode, "")
                            q = (solver.values.get(o, {}).get("h", 0) - solver.values.get(i, {}).get("h", 0))
                            if mode not in NO_HEAT and abs(q) > 1:
                                kind = "Qin" if q > 0 else "Qout"
                        points = [states[str(i)], states[str(o)]]
                        # Coolers that condense water run along the saturation
                        # curve once they reach the dew point (a straight line
                        # would cut through the fog region left of the curve).
                        ac_mode = (node.get("config") or {}).get("mode")
                        if t == "ac" and ac_mode in ("coolReheat", "cooling", "sensibleCooling"):
                            points = _cool_reheat_path(solver, solver.values[i], solver.values[o], from_si) or points
                        paths.append({"from": i, "to": o, "kind": kind, "label": label, "device": node.get("name", ""),
                                      "points": points})
    return {"fluids": {HUMID_AIR: {
        "dome": sat, "states": states, "paths": paths, "grid": grid,
        "frame": {"x": [from_si(sys_, "T", tlo), from_si(sys_, "T", thi)], "y": [0, whi]},
        "P": from_si(sys_, "P", P),
    }}}


def _cool_reheat_path(solver, va, vb, from_si):
    """The dehumidifier as drawn in textbooks: cool at constant ω to the dew
    point, down the saturation curve to the coil temperature, then reheat at
    constant ω to the outlet. None when it can't be worked out."""
    if "P" not in vb or solver.ha_props is None:
        return None
    ha, P, sys_ = solver.ha_props, vb["P"], solver.system
    try:
        if vb["w"] >= va["w"] - 1e-9:
            return None         # nothing condensed: the straight line is the process
        t1 = ha("D", "P", P, "W", va["w"], "T", 300.0)
        t2 = ha("D", "P", P, "W", vb["w"], "T", 300.0)
        pts = [[from_si(sys_, "T", va["T"]), va["w"]]]
        for k in range(9):
            t = t1 + (t2 - t1) * k / 8
            pts.append([from_si(sys_, "T", t), ha("W", "T", t, "P", P, "R", 1.0)])
        pts.append([from_si(sys_, "T", vb["T"]), vb["w"]])
        return pts
    except (ValueError, RuntimeError, KeyError):
        return None


_LINE_CACHE = {}


def _chart_lines(ha, sys_, P, tlo, thi, whi):
    """Saturation curve plus the chart's background lines, in display units:
    relative humidity 10 to 90 %, wet-bulb temperature every 5 degrees and
    enthalpy every 10 kJ/kg (5 Btu/lbm). Each line: {kind, label, points}."""
    from units import from_si, to_si
    import json
    key = (P, tlo, thi, whi, json.dumps(sys_, sort_keys=True, default=str))   # custom units come as a dict
    if key in _LINE_CACHE:
        return _LINE_CACHE[key]

    def T(t):
        return from_si(sys_, "T", t)

    n = 48
    ts = [tlo + (thi - tlo) * i / n for i in range(n + 1)]

    def curve(rh):
        out = []
        for t in ts:
            try:
                w = ha("W", "T", t, "P", P, "R", rh)
            except (ValueError, RuntimeError):
                continue
            out.append([T(t), w])
            if w > whi:
                break
        return out

    sat = curve(1.0)
    grid = [{"kind": "rh", "label": f"{k * 10}%", "points": curve(k / 10)} for k in range(1, 10)]

    # wet bulb: from the saturation curve down to dry air
    deg = 5 if from_si(sys_, "T", 373.15) - from_si(sys_, "T", 273.15) < 150 else 10   # °C or °F step
    t0 = from_si(sys_, "T", tlo)
    first = (int(t0 // deg) + 1) * deg
    t_disp = first
    while True:
        t_k = to_si(sys_, "T", t_disp)
        if t_k > thi:
            break
        try:
            ws = ha("W", "T", t_k, "P", P, "R", 1.0)
            pts = []
            for f in (1.0, 0.75, 0.5, 0.25, 0.0):
                w = max(ws * f, 1e-6)
                pts.append([T(ha("T", "B", t_k, "P", P, "W", w)), w])
            grid.append({"kind": "wb", "label": f"{t_disp:g}°", "points": pts})
        except (ValueError, RuntimeError):
            pass
        t_disp += deg

    # enthalpy: nearly straight, so three points from saturation to dry air
    step_disp = 10 if abs(from_si(sys_, "h", 10000.0) - 10) < 0.5 else 5
    step = to_si(sys_, "h", step_disp) - to_si(sys_, "h", 0)
    try:
        h_lo = ha("H", "T", tlo, "P", P, "W", 1e-6)
        h_hi = ha("H", "T", thi, "P", P, "R", 1.0)
    except (ValueError, RuntimeError):
        h_lo = h_hi = 0
    zero = to_si(sys_, "h", 0)
    k = int((h_lo - zero) // step) + 1
    while zero + k * step < h_hi and k < 200:
        h = zero + k * step
        try:
            t_sat = ha("T", "H", h, "P", P, "R", 1.0)
            w_sat = ha("W", "H", h, "P", P, "R", 1.0)
            pts = []
            for f in (1.0, 0.5, 0.0):
                w = max(w_sat * f, 1e-6)
                pts.append([T(ha("T", "H", h, "P", P, "W", w) if f < 1 else t_sat), w])
            grid.append({"kind": "h", "label": f"{round(from_si(sys_, 'h', h)):g}", "points": pts})
        except (ValueError, RuntimeError):
            pass
        k += 1

    if len(_LINE_CACHE) > 32:
        _LINE_CACHE.clear()
    _LINE_CACHE[key] = (sat, grid)
    return sat, grid
