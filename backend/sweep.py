"""Off-design sweep for /sweep: run the cycle on the canvas over a range of one
input and report how it performs at each point.

The input swept can be
    {"kind": "state", "state": n, "prop": "T" | "P" | "m" | "RH" | ...}
        a value typed in the state table (an ambient temperature, a flow);
    {"kind": "param", "node": id, "key": "Qdot" | "eta" | "rp" | "T0" | ...}
        a device input (heat added, an efficiency, an Atmosphere's T₀);
    {"kind": "netPower"}
        the net power wanted in the cycle summary: the cycle's load.
Each point is one full solve_system call, so it is the same cycle the user
built, at that one input changed. The outputs are the cycle-summary numbers
(η_th, Ẇ_net, Q̇_in, Q̇_out, COP) plus, if asked, one property of one state.
"""

import math

from pressure_optimizer import cycle_totals, metric_value
from system_solver import solve_system

MAX_POINTS = 41


class SweepError(Exception):
    pass


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _with_value(payload, target, value):
    """A copy of payload with the swept input set to value (display units)."""
    out = dict(payload)
    text = repr(float(value))
    kind = target.get("kind")
    if kind == "state":
        states = {k: dict(s) for k, s in (payload.get("states") or {}).items()}
        states.setdefault(str(target["state"]), {})[target["prop"]] = text
        out["states"] = states
    elif kind == "param":
        nodes = {k: dict(n) for k, n in (payload.get("nodes") or {}).items()}
        node = nodes.get(target["node"])
        if node is None:
            raise SweepError("The device to sweep is no longer on the canvas.")
        node["params"] = dict(node.get("params") or {}, **{target["key"]: text})
        if target["key"] == "eta":            # a typed η replaces "η = 100%"
            node["assumptions"] = dict(node.get("assumptions") or {}, eta100=False, isentropic=False)
        nodes[target["node"]] = node
        out["nodes"] = nodes
    elif kind == "netPower":
        out["cycle"] = dict(payload.get("cycle") or {}, netPower=text)
    else:
        raise SweepError("Pick what to sweep.")
    return out


def run_sweep(payload, spec):
    try:
        return _sweep(payload, spec)
    except SweepError as e:
        return {"errors": [str(e)]}


def _sweep(payload, spec):
    target = spec.get("target") or {}
    lo, hi = _num(spec.get("from")), _num(spec.get("to"))
    steps = int(_num(spec.get("steps")) or 11)
    if lo is None or hi is None or lo == hi:
        raise SweepError("Give the range to sweep: two different numbers.")
    steps = max(2, min(steps, MAX_POINTS))
    watch = spec.get("watch")          # {"state": n, "prop": "T"} or None

    points = []
    tear = None
    for i in range(steps):
        x = lo + (hi - lo) * i / (steps - 1)
        p = _with_value(payload, target, x)
        if tear:
            p = dict(p, tearHint=tear)         # a loop of flows starts where the last point closed
        try:
            r = solve_system(p, lean=True)     # no charts per point (1.10.12): far faster
            tear = r.get("tear") or tear
        except Exception as e:         # CoolProp can throw from deep inside a solve
            r = {"errors": [str(e)], "states": {}, "devices": {}}
        point = {"x": x, "ok": not r.get("errors"), "error": (r.get("errors") or [""])[0]}
        totals = None if r.get("errors") else cycle_totals(p, r)
        if totals:
            point.update(Wnet=totals["Wnet"], Qin=totals["Qin"], Qout=totals["Qout"], perMass=totals["perMass"])
            for key in ("eta", "COP_R", "COP_HP"):
                v = metric_value(key, totals)
                if v is not None:
                    point[key] = v
        if watch and not r.get("errors"):
            point["watch"] = _num(((r.get("states") or {}).get(str(watch.get("state"))) or {}).get(watch.get("prop")))
        points.append(point)
    if not any(pt["ok"] for pt in points):
        raise SweepError("The cycle doesn't solve anywhere in this range: " + points[0]["error"])
    return {"points": points, "errors": []}
