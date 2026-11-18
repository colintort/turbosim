"""Jet propulsion: thrust and efficiencies from a solved open system, plus the
two pieces of physics that let velocity be an unknown in more problems.

1. propulsion_summary(payload, result)
   Runs after solve_system, only when detect_jet() finds a jet engine: a
   nozzle exhausting out of the diagram with heat added upstream of it. Treats every stream that enters the diagram from
   outside as air taken in at the flight speed, and every stream that leaves
   it as a jet:

       F    = sum(m_e V_e) - sum(m_i) V_flight + sum((P_e - P_amb) A_e)
       Ẇ_P  = F V_flight                     propulsive power
       η_P  = Ẇ_P / Q̇_in                    Çengel's propulsive efficiency
       η_th = ΔKĖ / Q̇_in                    jet thermal efficiency
       η_F  = Ẇ_P / ΔKĖ                     Froude efficiency (η_P = η_th η_F)

   With no settings the flight speed is the velocity of the inlet state and
   the ambient pressure is its pressure, which is how textbook problems give
   them. A_e = m_e v_e / V_e, so a nozzle that does not expand to ambient
   pressure gets its pressure thrust without any extra input.

2. Rules for the device pass (nozzle, diffuser):
   continuity()            m = A V / v, with A typed per port: solves V or m
   nozzle_outlet_h()       η_N on stagnation enthalpy, so it holds when V1 > 0
   diffuser_outlet_P()     η_D sets the pressure recovered, not the enthalpy

All math is in SI. Values in `result` are display units and are converted here.
"""

import math

from units import to_si, from_si, power_factor, label
from combustion import products_props, fuel_flows

N_PER_LBF = 4.4482216152605

# Port names that are outlets. Everything else is an inlet.
_OUT_PORTS = {"vapor", "liquid", "bleed"}
# Device types whose positive Q̇ is heat added to the engine (burner, heater).
# A one-stream heat exchanger (the app's boiler / heater) counts too; see _heats.
_HEAT_IN = {"boiler", "combustor", "heater"}


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _is_out(port):
    return port in _OUT_PORTS or "out" in port.lower() or port.startswith("ext")


def _ports(node):
    ports = node.get("ports")
    if ports:
        return {k: int(v) for k, v in ports.items() if v is not None}
    return {"in": int(node["inState"]), "out": int(node["outState"])}


def _heats(node):
    """Heat added from outside: a burner, or a one-stream heat exchanger (a
    two-stream one only moves heat between the engine's own streams)."""
    t = node.get("type")
    if t == "hx":
        return not any(k.startswith("b_") for k in _ports(node))
    return t in _HEAT_IN


def _in_out(node):
    """(inlet, outlet) states of a one-inlet, one-outlet device: "in"/"out",
    or a one-stream heat exchanger's a_in1/a_out1."""
    p = _ports(node)
    return p.get("in", p.get("a_in1")), p.get("out", p.get("a_out1"))


def _english(system):
    """Thrust in lbf when flows are in pounds, N otherwise."""
    if isinstance(system, dict):
        return "lb" in label(system, "m")
    return system == "EN"


def boundary_streams(nodes):
    """(inlets, exits): state numbers that cross the diagram's boundary.
    A state some device produces and another consumes is internal. An
    atmosphere device marks the boundary itself, so it is left out."""
    produced, consumed = set(), set()
    for node in nodes.values():
        if node.get("type") == "atmosphere":
            continue
        unused = set(node.get("unused") or [])
        for port, n in _ports(node).items():
            if port in unused:
                continue
            (produced if _is_out(port) else consumed).add(n)
    return sorted(consumed - produced), sorted(produced - consumed)


def _upstream_finder(nodes):
    """upstream(node_id) -> ids of every device the flow into that device has
    passed through, itself included."""
    producers = {}                       # state -> ids of devices that put flow into it
    for node_id, node in nodes.items():
        if node.get("type") == "atmosphere":
            continue
        unused = set(node.get("unused") or [])
        for port, n in _ports(node).items():
            if port not in unused and _is_out(port):
                producers.setdefault(n, []).append(node_id)

    def upstream(start):
        seen, stack = set(), [start]
        while stack:
            d = stack.pop()
            if d in seen:
                continue
            seen.add(d)
            node = nodes[d]
            unused = set(node.get("unused") or [])
            for port, n in _ports(node).items():
                if port not in unused and not _is_out(port):
                    stack.extend(producers.get(n, []))
        return seen

    return upstream


def jet_layout(nodes):
    """What kind of jet engine the diagram is, or None.

    A jet is a nozzle that exhausts out of the diagram with heat added somewhere
    upstream of it. That leaves out a plain nozzle or diffuser problem (no heat
    input) and any closed loop (nothing leaves). The kind is only a label:
      turbofan   a compressor and turbine, and more than one jet or a splitter
      turbojet   a compressor and a turbine upstream of the nozzle
      ramjet     heat input but no compressor
      jet engine anything else that matches (e.g. a compressor with no turbine)
    An afterburner is heat added after a turbine and before a jet nozzle; it
    puts "afterburning" in front of the kind.

    Returns {kind, jets: [nozzle id], afterburners: [(heater id, nozzle id)]}.
    """
    inlets, exits = boundary_streams(nodes)
    if not inlets or not exits:
        return None
    upstream = _upstream_finder(nodes)
    type_of = lambda d: nodes[d].get("type")  # noqa: E731

    exits = set(exits)
    jets, types, afterburners = [], set(), []
    for node_id, node in nodes.items():
        if node.get("type") != "nozzle" or _ports(node).get("out") not in exits:
            continue
        up = upstream(node_id)
        if not any(_heats(nodes[d]) for d in up):
            continue
        jets.append(node_id)
        types |= {type_of(d) for d in up}
        for d in up:
            if _heats(nodes[d]) and any(
                type_of(t) in ("turbine", "turbineBleed") for t in upstream(d) - {d}
            ):
                afterburners.append((d, node_id))
    if not jets:
        return None

    all_types = {n.get("type") for n in nodes.values()}
    if "compressor" in types and types & {"turbine", "turbineBleed"}:
        if len([n for n in nodes.values() if n.get("type") == "nozzle"]) > 1 or "splitter" in all_types:
            kind = "turbofan"
        else:
            kind = "turbojet"
    elif "compressor" not in types:
        kind = "ramjet"
    else:
        kind = "jet engine"
    if afterburners:
        kind = "afterburning " + kind
    return {"kind": kind, "jets": jets, "afterburners": sorted(afterburners)}


def detect_jet(nodes):
    """Just the kind from jet_layout, or None."""
    layout = jet_layout(nodes)
    return layout and layout["kind"]


def thrust_units(system):
    return ("lbf", "lbf·s/lbm") if _english(system) else ("N", "N·s/kg")


def _thrust_from_si(system, newtons):
    return newtons / N_PER_LBF if _english(system) else newtons


def _specific_volume(props, fluid, P, h):
    return 1.0 / props("D", "P", P, "H", h, fluid)


def _speed_of_sound(props, fluid, P, h):
    try:
        return props("A", "P", P, "H", h, fluid)
    except Exception:
        return None   # two-phase: CoolProp has no single speed of sound


def propulsion_summary(payload, result, props=None):
    """Thrust, propulsive power and efficiencies, or None when the diagram is
    not a jet engine (see detect_jet).

    payload.propulsion or payload.cycle.propulsion (optional, display units):
        flightSpeed  blank = V of the inlet state
        Pamb         blank = P of the inlet state
        LHV          blank = 42,800 kJ/kg (jet fuel), for TSFC
    """
    nodes = payload.get("nodes") or {}
    layout = jet_layout(nodes)
    if layout is None:
        return None
    kind = layout["kind"]
    if props is None:
        from CoolProp.CoolProp import PropsSI
        from ideal_gas import with_ideal_air
        props = with_ideal_air(PropsSI)

    system = payload.get("unitFactors") or payload.get("unitSystem", "SI")
    fluid = payload.get("fluid", "Water")
    fluids = {int(k): v for k, v in (payload.get("stateFluids") or {}).items() if v}
    # Combustion products (from a Combustor) are the fluid of every state downstream of it.
    props, burnt = products_props(props, result)
    fluids.update(burnt)
    settings =payload.get("propulsion") or (payload.get("cycle") or {}).get("propulsion") or {}
    states = result.get("states") or {}
    devices = result.get("devices") or {}
    notes = []

    def si(n, key):
        x = _num((states.get(str(n)) or {}).get(key))
        return None if x is None else to_si(system, key, x)

    inlets, exits = boundary_streams(nodes)

    # Flight speed and ambient pressure: typed, or read off the first inlet.
    V0 = _num(settings.get("flightSpeed"))
    V0 = to_si(system, "V", V0) if V0 is not None else si(inlets[0], "V")
    P0 = _num(settings.get("Pamb"))
    P0 = to_si(system, "P", P0) if P0 is not None else si(inlets[0], "P")
    if V0 is None:
        return {"ready": False, "kind": kind, "need": "the flight speed (V at the inlet state, or type it here)"}

    m_in = [si(n, "m") for n in inlets]
    jets = [{"state": n, "m": si(n, "m"), "V": si(n, "V"), "P": si(n, "P"), "h": si(n, "h"),
             "fluid": fluids.get(n, fluid)} for n in exits]
    if any(j["V"] is None for j in jets):
        missing = ", ".join(f"S{j['state']}" for j in jets if j["V"] is None)
        return {"ready": False, "kind": kind, "need": f"the exit velocity at {missing}"}

    # Without mass flows, a single-stream engine still has per-kg results.
    per_kg = any(m is None for m in m_in + [j["m"] for j in jets])
    if per_kg:
        if len(inlets) != 1 or len(exits) != 1:
            return {"ready": False, "kind": kind, "need": "the mass flow of every stream in and out"}
        m_in = [1.0]
        # per kg of air in; a combustor adds its fuel to the jet
        frac = ((result.get("products") or {}).get(str(jets[0]["state"])) or {}).get("fuelFrac", 0.0)
        jets[0]["m"] = 1.0 / (1.0 - frac)
        notes.append("No mass flow given, so results are per unit mass of air.")
    m_air = sum(m_in)

    F, exits_out, mach_max = _thrust(jets, m_air, V0, P0, props, system)
    if any(abs(e.get("dP", 0)) > 0 for e in exits_out) and not per_kg:
        notes.append("A jet leaves at a different pressure than ambient, so F includes "
                     "pressure thrust (P_e − P_amb)A_e.")
    if mach_max is not None and mach_max > 1.0 + 1e-3:
        notes.append(
            f"The jet leaves at Mach {mach_max:.2f}. A plain converging nozzle chokes at Mach 1, "
            f"so this needs a converging–diverging nozzle."
        )
    for e in exits_out:
        e.pop("dP", None)

    W_P = F * V0                                                     # W
    dKE = sum(j["m"] * j["V"] ** 2 for j in jets) / 2 - m_air * V0 ** 2 / 2

    # Heat added, from the solved devices (display power units -> W).
    pf = power_factor(system)
    q_by_node = {}
    for node_id, node in nodes.items():
        q = _num((devices.get(node_id) or {}).get("Qdot"))
        if _heats(node) and q is not None and q > 0:
            q_by_node[node_id] = q * pf
    q_in = sum(q_by_node.values())

    # Fuel: Q̇_in / LHV (complete combustion; the fuel's own mass is not added
    # to the gas, as in the air-standard model).
    lhv = _num(settings.get("LHV"))
    lhv = to_si(system, "h", lhv) if lhv is not None else LHV_JET_FUEL
    # With Combustors the fuel flow is known exactly, and LHV comes from the fuel picked.
    fuel = fuel_flows(nodes, devices, system)

    thrust_unit, spec_unit = thrust_units(system)
    out = {
        "ready": True,
        "kind": kind,
        "perKg": per_kg,
        "units": {"F": thrust_unit, "Fspec": spec_unit, "A": label(system, "A"),
                  "TSFC": "lbm/(lbf·h)" if _english(system) else "kg/(N·h)"},
        "flightSpeed": from_si(system, "V", V0),
        "Pamb": None if P0 is None else from_si(system, "P", P0),
        "LHV": None if fuel else from_si(system, "h", lhv),
        "fuelFromCombustor": bool(fuel),
        "F": None if per_kg else _thrust_from_si(system, F),
        "Fspec": _spec_from_si(system, F / m_air),
        "WdotP": None if per_kg else W_P / pf,
        "wP": from_si(system, "h", W_P / m_air),       # propulsive work per unit mass
        "exits": exits_out,
        "notes": notes,
    }
    if q_in > 0 and not per_kg:
        out["QdotIn"] = q_in / pf
        out["etaP"] = W_P / q_in
        out["etaTh"] = dKE / q_in
        if F > 0:
            m_fuel = sum(fuel.values()) if fuel else q_in / lhv
            out["mFuel"] = from_si(system, "m", m_fuel)
            out["TSFC"] = _tsfc(system, m_fuel, F)
    if dKE > 0 and V0 > 0:
        out["etaFroude"] = W_P / dKE

    # Afterburner off: the same engine with the gas going straight from the
    # afterburner inlet into the nozzle, at the same exit pressure and the same
    # nozzle efficiency (backed out of the solved nozzle).
    if layout["afterburners"]:
        frac = {int(n): p["fuelFrac"] for n, p in (result.get("products") or {}).items()}
        dry = _dry_jets(layout["afterburners"], nodes, jets, si, props, notes, fluids, frac)
        if dry is not None:
            dry_jets, ab_nodes = dry
            F_dry, dry_exits, _ = _thrust(dry_jets, m_air, V0, P0, props, system)
            q_dry = q_in - sum(q_by_node.get(d, 0.0) for d in ab_nodes)
            ab = {
                "Fspec": _spec_from_si(system, F_dry / m_air),
                "Vexit": [from_si(system, "V", j["V"]) for j in dry_jets],
                "gain": F / F_dry - 1 if F_dry > 0 else None,
            }
            if not per_kg:
                ab["F"] = _thrust_from_si(system, F_dry)
                ab["QdotAB"] = (q_in - q_dry) / pf
                if q_dry > 0 and F_dry > 0:
                    m_dry = (sum(m for d, m in fuel.items() if d not in ab_nodes) if fuel else q_dry / lhv)
                    ab["TSFC"] = _tsfc(system, m_dry, F_dry)
                    ab["etaP"] = F_dry * V0 / q_dry
            out["dry"] = ab
    return out


LHV_JET_FUEL = 42_800e3      # J/kg, kerosene-type jet fuel (Çengel's value)


def _spec_from_si(system, n_s_per_kg):
    return n_s_per_kg / N_PER_LBF * 0.45359237 if _english(system) else n_s_per_kg


def _tsfc(system, m_fuel, F):
    """Fuel mass per unit thrust per hour: kg/(N·h) or lbm/(lbf·h)."""
    per_hour = m_fuel * 3600 / F
    return per_hour / 0.45359237 * N_PER_LBF if _english(system) else per_hour


def _thrust(jets, m_air, V0, P0, props, system):
    """F = sum(m_e V_e) − m_air V0 + sum((P_e − P0) A_e), with A_e = m_e v_e / V_e.
    Returns (F in N, a row per jet with A and Mach, the highest exit Mach)."""
    F = sum(j["m"] * j["V"] for j in jets) - m_air * V0
    rows, mach_max = [], None
    for j in jets:
        row = {"state": j["state"]}
        P, h, V = j["P"], j["h"], j["V"]
        if P is not None and h is not None:
            try:
                v = _specific_volume(props, j["fluid"], P, h)
            except Exception:
                v = None
            if v is not None and V > 0:
                A = j["m"] * v / V                         # m² (per kg/s when per kg)
                row["A"] = from_si(system, "A", A)
                if P0 is not None and not math.isclose(P, P0, rel_tol=1e-3):
                    F += (P - P0) * A
                    row["dP"] = P - P0
            a = _speed_of_sound(props, j["fluid"], P, h)
            if a:
                row["Mach"] = V / a
                mach_max = max(mach_max or 0.0, V / a)
        rows.append(row)
    return F, rows, mach_max


def _dry_jets(afterburners, nodes, jets, si, props, notes, fluids=None, fuel_frac=None):
    """The jets with every afterburner off, or None if that can't be worked out.
    Needs each afterburner to feed its nozzle directly."""
    by_state = {j["state"]: dict(j) for j in jets}
    ab_nodes = []
    for ab_id, nz_id in afterburners:
        a, b = _in_out(nodes[ab_id])
        nz = _ports(nodes[nz_id])
        e = nz.get("out")
        if nz.get("in") != b or e not in by_state:
            notes.append("Thrust without the afterburner needs the afterburner to feed the nozzle directly.")
            return None
        h_a, P_a, V_a = si(a, "h"), si(a, "P"), si(a, "V") or 0.0
        h_b, V_b = si(b, "h"), si(b, "V") or 0.0
        jet = by_state[e]
        if None in (h_a, P_a, h_b, jet["P"], jet["h"]):
            return None
        f = jet["fluid"]
        # Afterburner off: the nozzle gets the gas from the afterburner inlet,
        # and without the afterburner's fuel.
        if fluids and a in fluids:
            f = jet["fluid"] = fluids[a]
        if fuel_frac:
            jet["m"] *= (1 - fuel_frac.get(e, 0.0)) / (1 - fuel_frac.get(a, 0.0))
        try:
            # Nozzle efficiency of the solved (afterburner on) nozzle.
            h0b = h_b + V_b ** 2 / 2
            h_es = props("H", "P", jet["P"], "S", props("S", "P", si(b, "P") or P_a, "H", h_b, f), f)
            eta = (h0b - jet["h"]) / (h0b - h_es) if h0b > h_es else 1.0
            # Same nozzle fed from the afterburner inlet.
            h_es_dry = props("H", "P", jet["P"], "S", props("S", "P", P_a, "H", h_a, f), f)
            h_e = nozzle_outlet_h(h_a, V_a, h_es_dry, eta)
        except Exception:
            return None
        h0a = h_a + V_a ** 2 / 2
        if h0a <= h_e:
            return None
        jet["h"], jet["V"] = h_e, math.sqrt(2 * (h0a - h_e))
        ab_nodes.append(ab_id)
    return list(by_state.values()), ab_nodes


# ---------- device-pass rules (nozzle, diffuser) ----------

def continuity(m, V, v, A):
    """m = A V / v. Given A and v, returns (m, V) with the missing one filled
    in, or None if both are missing. Raises ValueError if both are given and
    disagree with A."""
    if m is None and V is None:
        return None
    if m is None:
        return A * V / v, V
    if V is None:
        return m, m * v / A
    if not math.isclose(m, A * V / v, rel_tol=1e-4, abs_tol=1e-6):
        raise ValueError(f"ṁ = {m:.6g} but A·V/v = {A * V / v:.6g}")
    return m, V


def nozzle_outlet_h(h1, V1, h2s, eta):
    """Nozzle efficiency η_N = V2² / V2s², with both jets starting from the same
    stagnation enthalpy h01 = h1 + V1²/2. Reduces to h1 − η(h1 − h2s) when V1 = 0."""
    h01 = h1 + V1 ** 2 / 2
    return h01 - eta * (h01 - h2s)


def diffuser_outlet_P(h1, V1, s1, h2, V2, eta, props, fluid):
    """Diffuser efficiency η_D = (h02s − h1) / (h02 − h1): the share of the
    inlet kinetic energy that turns into pressure. The energy balance already
    fixes h2, so η_D sets P2 (it cannot also set h2, which is what the
    compressor formula would do).

    h02s is on the inlet isentrope at the outlet stagnation pressure P02."""
    h02 = h1 + V1 ** 2 / 2
    h02s = h1 + eta * (h02 - h1)
    P02 = props("P", "H", h02s, "S", s1, fluid)
    if not V2:
        return P02
    s2 = props("S", "H", h02, "P", P02, fluid)       # stagnation and static share s
    return props("P", "H", h2, "S", s2, fluid)
