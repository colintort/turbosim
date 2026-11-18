"""Exergy: where a solved open system loses its ability to do work.

exergy_summary(payload, result) runs after solve_system, like the propulsion
summary, and reads the solved states from `result` (display units). Nothing in
the solver changes.

Dead state: T0, P0 from the Exergy panel (payload["cycle"]["exergy"]). Blank
means the ambient T and P of an Atmosphere device when the diagram has one,
else 25 °C and 101.325 kPa.

    flow exergy   ψ = (h − h0) − T0 (s − s0) + V²/2 + g z
    destroyed     Ẋ_dest = T0 Ṡ_gen,  Ṡ_gen = Σ ṁ s (out) − Σ ṁ s (in) − Σ Q̇ / T_b

Heat crosses a device boundary at T_b. Heat added comes from the "heat source"
temperature (the furnace for a power cycle, the cold space for a
refrigerator); heat removed goes to the "heat sink" temperature (blank = T0).
With no source temperature the heat-adding devices are left out, since their
loss depends on how hot the source is.

Every device's heat and work come from its energy balance:
turbines, compressors and pumps are adiabatic (Ẇ = −Δ(ṁe)), everything else
has no work (Q̇ = Δ(ṁe)). The cooling tower is left out because its air
streams are not diagram states.

With no mass flow anywhere (a per-kg textbook problem) every flow is taken as
1 and the results are per kg of flow; devices that split or join flows need
real flows then and are left out.

Second-law efficiencies:
  turbine            Ẇ_out / (Ẋ_in − Ẋ_out)
  compressor, pump   (Ẋ_out − Ẋ_in) / Ẇ_in
  two-stream hx      exergy gained by the cold stream / exergy given by the hot one
  power cycle        Ẇ_net / exergy supplied (heat from the source + streams in)
  refrigerator       exergy of the cooling, Q̇_L (T0/T_L − 1), / Ẇ_in  (= COP_R / COP_R,rev)
  heat pump          exergy of the heating, Q̇_H (1 − T0/T_H), / Ẇ_in (= COP_HP / COP_HP,rev)

All math is in SI. Values in `result` are display units and are converted here.
"""

import math

from units import to_si, from_si, power_factor, label
from combustion import FUELS, DEFAULT_FUEL, is_combustor, products_props

G = 9.80665

# Chemical exergy of each fuel as a multiple of its LHV (Szargut; Kotas;
# Çengel's tables): what a reversible process could get out of 1 kg of it.
FUEL_EXERGY_RATIO = {
    "Jet fuel (Jet-A)": 1.068,
    "Natural gas (methane)": 1.04,
    "Propane": 1.078,
    "Gasoline (octane)": 1.067,
    "Diesel (dodecane)": 1.068,
    "Hydrogen": 0.984,
    "Ethanol": 1.098,
    "Methanol": 1.125,
}
T0_DEFAULT, P0_DEFAULT = 298.15, 101325.0

WORK_OUT = ("turbine",)
WORK_IN = ("compressor", "pump")
SKIPPED = {"coolingTower": "air streams aren't diagram states"}


def _num(v):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ports(node):
    ports = node.get("ports")
    if ports:
        return {k: int(v) for k, v in ports.items() if v is not None}
    return {"in": int(node["inState"]), "out": int(node["outState"])}


def is_outlet(port):
    """out, out1, a_out1, b_out2, ext1 (turbine extraction), vapor, liquid."""
    p = port.lower()
    return "out" in p or p.startswith("ext") or p in ("vapor", "liquid", "bleed")


def _sides(node):
    """(inlets, outlets) state numbers of a device, leaving out unused ports."""
    unused = set(node.get("unused") or [])
    ins, outs = [], []
    for port, n in _ports(node).items():
        if port in unused:
            continue
        (outs if is_outlet(port) else ins).append(n)
    return ins, outs


def _settings(payload):
    s = (payload.get("cycle") or {}).get("exergy") or payload.get("exergy") or {}
    return s if isinstance(s, dict) else {}


def dead_state(payload):
    """(T0, P0) in SI, and where they came from ("typed", "atmosphere", "default")."""
    system = payload.get("unitFactors") or payload.get("unitSystem", "SI")
    s = _settings(payload)
    T0, P0 = _num(s.get("T0")), _num(s.get("P0"))
    T0 = None if T0 is None else to_si(system, "T", T0)
    P0 = None if P0 is None else to_si(system, "P", P0)
    source = "typed" if T0 is not None or P0 is not None else "default"

    # an Atmosphere intake (it has T0) first, then any Atmosphere
    atm = sorted(
        (n for n in (payload.get("nodes") or {}).values() if n.get("type") == "atmosphere"),
        key=lambda n: _num((n.get("params") or {}).get("T0")) is None,
    )
    for node in atm:
        params = node.get("params") or {}
        if T0 is None and _num(params.get("T0")) is not None:
            T0, source = to_si(system, "T", _num(params["T0"])), "atmosphere"
        if P0 is None and _num(params.get("P0")) is not None:
            P0, source = to_si(system, "P", _num(params["P0"])), "atmosphere"
    return (T0 if T0 is not None else T0_DEFAULT), (P0 if P0 is not None else P0_DEFAULT), source


def _state_si(system, row):
    """{P, T, h, s, m, V, z} in SI for one reported state; missing keys left out."""
    out = {}
    for key in ("P", "T", "h", "s", "m", "V", "z"):
        x = _num(row.get(key))
        if x is not None:
            out[key] = to_si(system, key, x)
    return out


def exergy_summary(payload, result, props=None):
    """The exergy panel's numbers in display units, or None in closed mode."""
    if payload.get("mode") == "closed":
        return None
    if props is None:
        from CoolProp.CoolProp import PropsSI
        from ideal_gas import with_ideal_air
        props = with_ideal_air(PropsSI)

    system = payload.get("unitFactors") or payload.get("unitSystem", "SI")
    nodes = payload.get("nodes") or {}
    T0, P0, source = dead_state(payload)
    s = _settings(payload)
    t_src = _num(s.get("Tsource"))
    t_snk = _num(s.get("Tsink"))
    t_src = None if t_src is None else to_si(system, "T", t_src)
    t_snk = T0 if t_snk is None else to_si(system, "T", t_snk)

    out = {
        "ready": False,
        "need": "",
        "deadState": {
            "T0": float(f"{from_si(system, 'T', T0):.6g}"),
            "P0": float(f"{from_si(system, 'P', P0):.6g}"),
            "from": source,
        },
        "devices": [],
        "lost": [],
        "states": {},
        "notes": [],
    }
    if result.get("errors"):
        out["need"] = "a solve with no errors"
        return out
    if not nodes:
        out["need"] = "a diagram with devices"
        return out
    for name, T in (("dead state T0", T0), ("heat source T", t_src), ("heat sink T", t_snk)):
        if T is not None and T <= 0:
            out["need"] = f"a {name} above absolute zero"
            return out
    if P0 <= 0:
        out["need"] = "a positive dead-state pressure"
        return out

    states = {int(k): _state_si(system, row) for k, row in (result.get("states") or {}).items()}
    per_kg = not any("m" in v for v in states.values())
    # States after a combustor carry its products: (1 + f) kg per kg of air.
    burnt_frac = {int(n): p.get("fuelFrac", 0.0) for n, p in (result.get("products") or {}).items()}
    if per_kg:
        for n, v in states.items():
            v["m"] = 1.0 / (1.0 - burnt_frac.get(n, 0.0))

    fluids = {int(k): f for k, f in (payload.get("stateFluids") or {}).items() if f}
    # Combustion products are their own fluid, and their ψ is taken against
    # the products at the dead state (thermomechanical exergy).
    props, burnt = products_props(props, result)
    fluids.update(burnt)
    devices_out = result.get("devices") or {}
    default_fluid = payload.get("fluid", "Water")
    dead = {}  # fluid -> (h0, s0)

    def psi(n):
        """Flow exergy at state n (J/kg), or None while h or s is unknown."""
        v = states.get(n, {})
        if "h" not in v or "s" not in v:
            return None
        fluid = fluids.get(n, default_fluid)
        if fluid not in dead:
            try:
                dead[fluid] = (props("H", "T", T0, "P", P0, fluid), props("S", "T", T0, "P", P0, fluid))
            except Exception as e:
                raise ValueError(f"{fluid} has no property data at the dead state ({e})")
        h0, s0 = dead[fluid]
        return (v["h"] - h0) - T0 * (v["s"] - s0) + v.get("V", 0.0) ** 2 / 2 + G * v.get("z", 0.0)

    try:
        for n in sorted(states):
            p = psi(n)
            if p is not None:
                out["states"][str(n)] = float(f"{from_si(system, 'h', p):.6g}")
    except ValueError as e:
        out["need"] = f"a dead state every fluid can exist at: {e}"
        return out

    pf = 1.0 if per_kg else power_factor(system)
    to_disp = (lambda x: from_si(system, "h", x)) if per_kg else (lambda x: x / pf)
    out["perKg"] = per_kg
    out["units"] = {"X": label(system, "h") if per_kg else label(system, "power"), "psi": label(system, "h")}

    w_out = w_in = 0.0
    q_in = q_out = 0.0         # heat added / removed by devices (W, positive)
    x_q_in = 0.0               # exergy of the heat added, when t_src is known
    x_fuel_in = 0.0            # chemical exergy of the fuel burned in combustors
    missing_heat = []          # heat-adding devices left out (no source T)
    rows, flagged = [], []
    total = 0.0

    for node_id, node in nodes.items():
        t = node.get("type")
        name = node.get("name") or t
        if t == "atmosphere":
            continue
        row = {"id": node_id, "name": name, "type": t, "Xdest": None, "etaII": None, "note": ""}
        rows.append(row)
        if t in SKIPPED:
            row["note"] = f"not included: {SKIPPED[t]}"
            continue
        ins, outs = _sides(node)
        if not ins or not outs:
            row["note"] = "needs an inlet and an outlet"
            continue
        if per_kg and (len(ins) > 1 or len(outs) > 1):
            row["note"] = "needs mass flows (it splits or joins streams)"
            continue
        every = ins + outs
        if any(k not in states.get(n, {}) for n in every for k in ("h", "s", "m")):
            row["note"] = "needs every port solved"
            continue

        def e(n):
            v = states[n]
            return v["h"] + v.get("V", 0.0) ** 2 / 2 + G * v.get("z", 0.0)

        m = {n: states[n]["m"] for n in every}
        energy = sum(m[n] * e(n) for n in outs) - sum(m[n] * e(n) for n in ins)
        scale = sum(m[n] * (abs(states[n]["h"]) + abs(e(n) - states[n]["h"])) for n in every) or 1.0
        tiny = 1e-6 * scale
        entropy = sum(m[n] * states[n]["s"] for n in outs) - sum(m[n] * states[n]["s"] for n in ins)
        x_in = sum(m[n] * psi(n) for n in ins)
        x_out = sum(m[n] * psi(n) for n in outs)

        asm = node.get("assumptions") or {}
        if is_combustor(node):
            # Fuel in, no heat or work across the boundary: Ẋ_dest = Ẋ_in + ṁ_f ex_fuel − Ẋ_out.
            r = devices_out.get(node_id) or {}
            if per_kg:
                mf = _num(r.get("far"))
                mf = None if mf is None else mf * m[ins[0]]
            else:
                mf = _num(r.get("mf"))
                mf = None if mf is None else to_si(system, "m", mf)
            if mf is None:
                row["note"] = "needs the fuel flow (AF, ṁ_f or the exit T)"
                continue
            fuel_name = (node.get("params") or {}).get("fuel") or DEFAULT_FUEL
            fuel = FUELS.get(fuel_name, FUELS[DEFAULT_FUEL])
            x_fuel = mf * fuel["LHV"] * 1e3 * FUEL_EXERGY_RATIO.get(fuel_name, 1.06)
            x_fuel_in += x_fuel
            q_in += mf * fuel["LHV"] * 1e3
            x_dest = x_in + x_fuel - x_out
            row["Xdest"] = to_disp(x_dest)
            row["etaII"] = (x_out - x_in) / x_fuel if x_fuel > 0 else None
            if x_dest < 0:
                flagged.append(name)
                row["note"] = "breaks the second law as given (Ṡ_gen < 0)"
            total += x_dest
            continue
        W = Q = 0.0
        t_b = None
        dr = devices_out.get(node_id) or {}
        if t in WORK_OUT and not asm.get("adiabatic") and _num(dr.get("Qdot")) is not None \
                and _num(dr.get("Wdot")) is not None and not per_kg:
            # a turbine losing heat (Penoncello Ex. 3.11): Ẇ as given, Q̇ = Δ(ṁe) + Ẇ,
            # lost through its casing at T_b (blank: the heat sink T)
            W = _num(dr["Wdot"]) * power_factor(system)
            Q = energy + W
            tb_txt = _num((node.get("params") or {}).get("Tb"))
            t_b = None if tb_txt is None else to_si(system, "T", tb_txt)
        elif t in WORK_OUT + WORK_IN:
            W = -energy                       # work produced (negative: work put in)
        elif not (asm.get("adiabatic") or asm.get("noLoss")) and abs(energy) > tiny:
            Q = energy                        # heat added (negative: removed)

        if W > 0:
            w_out += W
        else:
            w_in -= W

        s_gen = entropy
        if Q > 0:
            q_in += Q
            if t_src is None:
                missing_heat.append(name)
                row["note"] = "set the heat source T to include it"
                continue
            s_gen -= Q / t_src
            x_q_in += Q * (1 - T0 / t_src)
        elif Q < 0 and t_b is not None:
            # heat leaves at the casing temperature: its exergy Q (1 - T0/T_b)
            # is lost to the surroundings, not destroyed inside the device
            q_out -= Q
            s_gen -= Q / t_b
            row["Xheat"] = to_disp(-Q * (1 - T0 / t_b))
        elif Q < 0:
            q_out -= Q
            s_gen -= Q / t_snk

        x_dest = T0 * s_gen
        if abs(x_dest) < T0 * 1e-6 * sum(m[n] * (abs(states[n]["s"]) + 1.0) for n in every):
            x_dest = 0.0                      # isentropic within rounding
        if x_dest < 0:
            flagged.append(name)
            row["note"] = "breaks the second law as given (Ṡ_gen < 0)"
            if Q > 0:
                row["note"] += ": the heat source must be hotter than the fluid it heats"
            elif Q < 0:
                row["note"] += ": the heat sink must be colder than the fluid it cools"
        row["Xdest"] = to_disp(x_dest)
        total += x_dest

        if t in WORK_OUT and x_in - x_out > tiny:
            row["etaII"] = W / (x_in - x_out)
        elif t in WORK_IN and W < -tiny:
            row["etaII"] = (x_out - x_in) / -W
        elif t == "hx" and any(k.startswith("b_") for k in _ports(node)):
            row["etaII"] = _hx_eta(node, states, psi, tiny)

    # Streams that cross the diagram's boundary. An Atmosphere device's port
    # is the surroundings: its outlet is air drawn in, its inlet is exhaust.
    produced, consumed, exhausts = set(), set(), []
    for node in nodes.values():
        ins, outs = _sides(node)
        if node.get("type") == "atmosphere":
            exhausts += ins
            continue
        consumed.update(ins)
        produced.update(outs)
    x_flow_in = 0.0
    for n in sorted(consumed - produced):
        if psi(n) is not None and "m" in states.get(n, {}):
            x_flow_in += states[n]["m"] * psi(n)
    for n in sorted(set(exhausts)):
        if psi(n) is not None and "m" in states.get(n, {}):
            out["lost"].append({"label": f"Exhaust S{n} to the surroundings", "X": to_disp(states[n]["m"] * psi(n))})
    if q_out > 0 and t_snk > T0 * (1 + 1e-9):
        out["lost"].append({"label": "Heat rejected to the sink", "X": to_disp(q_out * (1 - T0 / t_snk))})

    done = [r for r in rows if r["Xdest"] is not None]
    for r in done:
        r["share"] = (r["Xdest"] / to_disp(total)) if total > 0 else 0.0
    rows.sort(key=lambda r: (r["Xdest"] is None, -(r["Xdest"] or 0.0)))
    out["devices"] = rows
    out["total"] = to_disp(total)
    out["ready"] = bool(done)
    if not done:
        out["need"] = "solved devices (every port with h, s and ṁ)"

    # whole-cycle second-law efficiency
    w_net = w_out - w_in
    purpose = (payload.get("cycle") or {}).get("purpose")
    reversed_cycle = w_net < 0 and q_in > 0 and q_out > 0
    if w_net > 0 and q_in > 0 and not missing_heat and (t_src is not None or x_fuel_in > 0):
        supplied = x_q_in + x_fuel_in + x_flow_in
        if supplied > 0:
            out["cycle"] = {"kind": "power", "etaII": w_net / supplied, "Xsupplied": to_disp(supplied)}
    elif reversed_cycle and purpose == "heating":
        if t_snk > T0 * (1 + 1e-9):
            out["cycle"] = {"kind": "heat pump", "etaII": q_out * (1 - T0 / t_snk) / -w_net}
        else:
            out["notes"].append("For the heat pump's second-law efficiency, set the heat sink T to the warm space.")
    elif reversed_cycle:
        if t_src is not None and t_src < T0 * (1 - 1e-9):
            out["cycle"] = {"kind": "refrigerator", "etaII": q_in * (T0 / t_src - 1) / -w_net}
        else:
            out["notes"].append("For the refrigerator's second-law efficiency, set the heat source T to the cold space.")
    if "cycle" in out:
        out["cycle"]["Wnet"] = to_disp(w_net)

    if missing_heat:
        out["notes"].append(
            f"{', '.join(missing_heat)} {'adds' if len(missing_heat) == 1 else 'add'} heat. Set the heat source "
            "temperature (the furnace or flame for a power cycle, the cold space for a refrigerator) to include "
            f"{'it' if len(missing_heat) == 1 else 'them'}."
        )
    if flagged:
        out["notes"].append(
            f"{', '.join(flagged)}: the states given make entropy decrease, which no real device can do."
        )
    if per_kg:
        out["notes"].append("No mass flow is given, so the results are per kg of flow.")

    for r in rows:
        for k in ("Xdest", "share", "etaII"):
            if r.get(k) is not None:
                r[k] = float(f"{r[k]:.6g}")
    for item in out["lost"]:
        item["X"] = float(f"{item['X']:.6g}")
    out["total"] = float(f"{out['total']:.6g}")
    return out


def _hx_eta(node, states, psi, tiny):
    """Two-stream heat exchanger: exergy gained by the cold stream over exergy
    given up by the hot one."""
    ports = _ports(node)
    unused = set(node.get("unused") or [])
    gain = {}
    for side in ("a", "b"):
        x = 0.0
        for port, n in ports.items():
            if port in unused or not port.startswith(f"{side}_"):
                continue
            sign = 1 if is_outlet(port) else -1
            x += sign * states[n]["m"] * psi(n)
        gain[side] = x
    hot, cold = (gain["a"], gain["b"]) if gain["a"] < gain["b"] else (gain["b"], gain["a"])
    if hot < -tiny and cold > 0:
        return cold / -hot
    return None
