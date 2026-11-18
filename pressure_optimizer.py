"""Pressure optimizer for /optimize.

Finds the pressure(s) that maximize the cycle's figure of merit (η_th, Ẇ_net,
COP_R or COP_HP) by solving the KKT conditions of

    maximize   M(P)
    subject to Pmin <= P <= Pmax
               g_j(P) >= 0          (the user's limits, e.g. x4 >= 0.88)

with SciPy's SLSQP, a sequential quadratic programming method that drives the
gradient of the Lagrangian  L = M + sum(λ_j g_j)  to zero. Each M(P) and g_j(P)
is one full solve of the user's diagram by solve_system(), so the optimizer
sees exactly the cycle on the canvas. Gradients are central differences of
those solves.

The pressure curve returned for the plot is only a picture and a starting
guess: the reported optimum comes from the KKT conditions, not from the grid,
so it is not limited by the grid spacing the way an EES parametric table is.

Pressures are optimized in ln(P), scaled to [0, 1] over the user's range.
That keeps a 10 kPa to 10 MPa range well conditioned.

The multiplier λ_j of an active limit is reported in the metric's units per
unit of the limit: it says how much the optimum would gain if that limit were
relaxed by one unit. That is the Lagrange multiplier's physical meaning.
"""

import math

import contextlib
import itertools
import threading

import numpy as np
from scipy.optimize import minimize

import system_solver
from system_solver import solve_system

# Devices whose shaft work counts toward Ẇ_net. Heat crosses the cycle
# boundary at one-stream heat exchangers (boilers, condensers, heaters); heat
# moved between the two streams of a two-stream exchanger is internal.
WORK_OUT = ("turbine",)
WORK_IN = ("compressor", "pump")

METRICS = {
    "eta": {"label": "η_th", "unit": "%"},
    "Wnet": {"label": "Ẇ_net", "unit": "power"},
    "COP_R": {"label": "COP_R", "unit": ""},
    "COP_HP": {"label": "COP_HP", "unit": ""},
}

CURVE_POINTS = 41
FD_STEP = 1e-5          # central-difference step in scaled ln P
ACTIVE_TOL = 1e-6       # a limit within this (relative) is active


class OptimizeError(Exception):
    pass


_full_precision_lock = threading.Lock()
_full_precision_depth = 0


@contextlib.contextmanager
def full_precision():
    """solve_system rounds its report to 7 significant digits for display.
    Central differences need the unrounded values, so the optimizer turns the
    rounding off while it runs."""
    global _full_precision_depth
    with _full_precision_lock:
        if _full_precision_depth == 0:
            full_precision.saved = system_solver._round
            system_solver._round = float
        _full_precision_depth += 1
    try:
        yield
    finally:
        with _full_precision_lock:
            _full_precision_depth -= 1
            if _full_precision_depth == 0:
                system_solver._round = full_precision.saved


# ---------------------------------------------------------------- helpers
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ports(node):
    ports = node.get("ports")
    if ports:
        return {k: int(v) for k, v in ports.items() if v is not None and v != ""}
    return {"in": int(node["inState"]), "out": int(node["outState"])}


def _with_prefix(ports, prefix):
    return [n for k, n in ports.items() if k.startswith(prefix)]


def _one_stream_hx(node):
    return node.get("type") == "hx" and not any(k.startswith("b_") for k in _ports(node))


def pressure_groups(payload):
    """States that must share one pressure because a device says so (P₁ = P₂,
    a flash chamber, the outlets of one heat exchanger stream...). Varying one
    state's P has to move its whole group, or a P the user typed elsewhere in
    the group would conflict with the new value."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def tie(states):
        states = [s for s in states if s is not None]
        for s in states:
            find(s)
        for s in states[1:]:
            parent[find(s)] = find(states[0])

    for node in (payload.get("nodes") or {}).values():
        t = node.get("type")
        asm = node.get("assumptions") or {}
        ports = _ports(node)
        if t == "hx":
            for p in ("a", "b"):
                tie(_with_prefix(ports, f"{p}_out"))         # one stream's outlets share a state
                if asm.get(f"{p}Isobaric"):
                    tie(_with_prefix(ports, f"{p}_"))
        elif t == "turbine":
            exts = _with_prefix(ports, "ext")
            if exts and asm.get("sameOutlet"):
                tie(exts + [ports.get("out")])
        elif t == "flash":
            tie(list(ports.values()))
        elif t in ("mixer", "splitter") and asm.get("isobaric"):
            tie(list(ports.values()))
        elif asm.get("isobaric"):
            tie([ports.get("in"), ports.get("out")])

    groups = {}
    for s in parent:
        groups.setdefault(find(s), set()).add(s)
    return {s: sorted(g) for g in groups.values() for s in g}


def cycle_totals(payload, result):
    """Ẇ_out, Ẇ_in, Q̇_in, Q̇_out of the solved cycle, in display units: power
    when every counted device has its flow, per unit mass otherwise. Counts
    what CycleSummary.js counts: turbines, compressors, pumps and one-stream
    heat exchangers. Returns None if any counted device is unsolved."""
    states = result.get("states") or {}
    devices = result.get("devices") or {}
    counted = []
    for node_id, node in (payload.get("nodes") or {}).items():
        t = node.get("type")
        if t in WORK_OUT + WORK_IN:
            counted.append((node_id, node, "W"))
        elif _one_stream_hx(node) and not (node.get("assumptions") or {}).get("adiabatic"):
            counted.append((node_id, node, "Q"))
    if not counted:
        return None

    def h(n):
        return _num((states.get(str(n)) or {}).get("h"))

    per_mass = any(_num((devices.get(i) or {}).get(f"{k}dot")) is None for i, _, k in counted)
    w_out = w_in = q_in = q_out = 0.0
    for node_id, node, kind in counted:
        r = devices.get(node_id) or {}
        if kind == "W":
            w = _num(r.get("w" if per_mass else "Wdot"))
            if w is None:
                return None
            if node["type"] in WORK_OUT:
                w_out += w
            else:
                w_in += w
            continue
        if per_mass:
            ports = _ports(node)
            ins, outs = _with_prefix(ports, "a_in"), _with_prefix(ports, "a_out")
            if len(ins) != 1 or not outs or h(ins[0]) is None or h(outs[0]) is None:
                return None
            q = h(outs[0]) - h(ins[0])
        else:
            q = _num(r.get("Qdot"))
        if q >= 0:
            q_in += q
        else:
            q_out += -q
    return {"Wout": w_out, "Win": w_in, "Wnet": w_out - w_in, "Qin": q_in, "Qout": q_out,
            "perMass": per_mass}


def metric_value(key, totals):
    if totals is None:
        return None
    wnet = totals["Wnet"]
    if key == "Wnet":
        return wnet
    if key == "eta":
        return 100.0 * wnet / totals["Qin"] if totals["Qin"] > 0 else None
    if key == "COP_R":
        return totals["Qin"] / -wnet if wnet < 0 else None
    if key == "COP_HP":
        return totals["Qout"] / -wnet if wnet < 0 else None
    raise OptimizeError(f"Unknown objective '{key}'.")


def auto_metric(payload, totals):
    """η_th for a power cycle, COP for a refrigerator or heat pump. The
    Refrigerator / Heat pump switch in the cycle summary (cycle.purpose)
    picks which COP."""
    if totals is None:
        raise OptimizeError("Solve the cycle first: the starting point has unsolved devices.")
    if totals["Wnet"] > 0:
        return "eta" if totals["Qin"] > 0 else "Wnet"
    return "COP_HP" if (payload.get("cycle") or {}).get("purpose") == "heating" else "COP_R"


def state_value(states, n, prop):
    """A state property for a limit. Quality is extended smoothly outside the
    dome, (h - h_f) / (h_g - h_f), so a limit like x4 >= 0.88 has a gradient
    whether the turbine exit is wet or superheated."""
    row = states.get(str(n)) or {}
    v = _num(row.get(prop))
    if v is not None or prop != "x":
        return v
    return _num(row.get("x_ext"))


# ---------------------------------------------------------------- the problem
class _Problem:
    def __init__(self, payload, spec):
        self.base = payload
        self.groups = pressure_groups(payload)
        self.vars = []
        for v in spec.get("variables") or []:
            n = int(v["state"])
            lo, hi = _num(v.get("min")), _num(v.get("max"))
            if lo is None or hi is None or not (0 < lo < hi):
                raise OptimizeError(f"P{n}: the range needs 0 < min < max.")
            self.vars.append({"state": n, "group": self.groups.get(n, [n]), "lo": lo, "hi": hi})
        if not self.vars:
            raise OptimizeError("Pick a pressure to vary.")
        seen = {}
        for v in self.vars:
            for n in v["group"]:
                if n in seen:
                    raise OptimizeError(
                        f"P{v['state']} and P{seen[n]} are tied together (P₁ = P₂), so they can't be varied separately."
                    )
                seen[n] = v["state"]
        self.limits = []
        for c in spec.get("constraints") or []:
            value = _num(c.get("value"))
            if value is None or c.get("op") not in (">=", "<="):
                raise OptimizeError("Each limit needs a state, a property, ≥ or ≤, and a number.")
            self.limits.append({"state": int(c["state"]), "prop": c["prop"], "op": c["op"], "value": value})
        self.cache = {}
        self.evals = 0
        self.tear = None        # the flow that closed a loop of unknown flows last time (warm start)

    # scaled u in [0, 1]  <->  pressure in display units
    def to_p(self, u):
        return [math.exp(math.log(v["lo"]) + ui * (math.log(v["hi"]) - math.log(v["lo"])))
                for v, ui in zip(self.vars, u)]

    def to_u(self, p):
        return [(math.log(pi) - math.log(v["lo"])) / (math.log(v["hi"]) - math.log(v["lo"]))
                for v, pi in zip(self.vars, p)]

    def payload_at(self, u):
        payload = dict(self.base)
        states = {k: dict(s) for k, s in (self.base.get("states") or {}).items()}
        for v, p in zip(self.vars, self.to_p(u)):
            for n in v["group"]:
                row = states.setdefault(str(n), {})
                # move the typed P of every state in the group, and the chosen one
                if n == v["state"] or str(row.get("P", "")).strip() != "":
                    row["P"] = repr(p)
        payload["states"] = states
        if self.tear:
            payload["tearHint"] = self.tear     # start the loop's iteration where the last point ended
        return payload

    def solve_at(self, u):
        key = tuple(round(float(x), 13) for x in u)
        if key in self.cache:
            return self.cache[key]
        payload = self.payload_at(u)
        self.evals += 1
        try:
            result = solve_system(payload, lean=True)   # no charts: they were ~90 % of each solve
        except Exception as e:      # CoolProp can throw from deep inside a solve
            result = {"errors": [str(e)], "states": {}}
        if not result.get("errors"):
            _add_extended_quality(payload, result)
        if result.get("tear"):
            self.tear = result["tear"]
        self.cache[key] = result
        return result

    def metric(self, key, u):
        r = self.solve_at(u)
        if r.get("errors"):
            return None
        return metric_value(key, cycle_totals(self.base, r))

    def limit_g(self, c, u):
        """g >= 0 when the limit holds, in the limit's own units."""
        r = self.solve_at(u)
        if r.get("errors"):
            return None
        v = state_value(r.get("states") or {}, c["state"], c["prop"])
        if v is None:
            return None
        return v - c["value"] if c["op"] == ">=" else c["value"] - v


def _add_extended_quality(payload, result):
    """Adds x_ext to states that have P and h, for smooth quality limits."""
    from CoolProp.CoolProp import PropsSI
    from units import to_si
    system = payload.get("unitFactors") or payload.get("unitSystem", "SI")
    fluids = payload.get("stateFluids") or {}
    for n, row in (result.get("states") or {}).items():
        P, h = _num(row.get("P")), _num(row.get("h"))
        if P is None or h is None:
            continue
        fluid = fluids.get(n) or fluids.get(int(n)) or payload.get("fluid", "Water")
        try:
            Psi, hsi = to_si(system, "P", P), to_si(system, "h", h)
            hf = PropsSI("H", "P", Psi, "Q", 0, fluid)
            hg = PropsSI("H", "P", Psi, "Q", 1, fluid)
            row["x_ext"] = (hsi - hf) / (hg - hf)
        except Exception:
            pass        # supercritical or no saturation data: no quality


def _meets_limits(prob, u):
    """For the pictures: a point sitting on a limit (as the optimum does) counts
    as meeting it, despite the solver's last-digit noise."""
    for c in prob.limits:
        g = prob.limit_g(c, u)
        if g is None or g < -1e-5 * max(1.0, abs(c["value"])):
            return False
    return True


def _as_list(p):
    return p if isinstance(p, list) else [p]


def _grad(fun, u, lo=0.0, hi=1.0):
    """Central differences, one-sided at a bound."""
    u = np.asarray(u, float)
    g = np.zeros_like(u)
    for i in range(len(u)):
        a, b = u.copy(), u.copy()
        a[i] = max(lo, u[i] - FD_STEP)
        b[i] = min(hi, u[i] + FD_STEP)
        fa, fb = fun(a), fun(b)
        if fa is None or fb is None:
            return None
        g[i] = (fb - fa) / (b[i] - a[i])
    return g


MAP_POINTS = 15         # per axis for the two-pressure contour map
SLICE_POINTS = 21       # per curve for three or more pressures


def _goal_map(prob, key):
    """M on a grid over both pressures, plus each limit's g, so the frontend
    can draw contours, shade where a limit breaks and draw the g = 0 line."""
    us = np.linspace(0.0, 1.0, MAP_POINTS)
    value, g = [], [[] for _ in prob.limits]
    for uy in us:
        row, grow = [], [[] for _ in prob.limits]
        for ux in us:
            row.append(prob.metric(key, [ux, uy]))
            for j, c in enumerate(prob.limits):
                grow[j].append(prob.limit_g(c, [ux, uy]))
        value.append(row)
        for j in range(len(prob.limits)):
            g[j].append(grow[j])
    return {
        "kind": "map",
        "x": [prob.to_p([u, 0.0])[0] for u in us],
        "y": [prob.to_p([0.0, u])[1] for u in us],
        "value": value,             # value[iy][ix], None where the cycle does not solve
        "limits": g,                # limits[j][iy][ix] = g_j, >= 0 where limit j holds
    }


def _slices(prob, key, u_opt):
    """One curve per pressure through the optimum, the others held there."""
    out = []
    for i, v in enumerate(prob.vars):
        pts = []
        for t in np.linspace(0.0, 1.0, SLICE_POINTS):
            u = list(u_opt)
            u[i] = t
            m = prob.metric(key, u)
            ok = m is not None and _meets_limits(prob, u)
            pts.append({"P": prob.to_p(u)[i], "value": m, "feasible": bool(ok)})
        out.append({"state": v["state"], "curve": pts})
    return {"kind": "slices", "slices": out}


def _check_free_goal(payload, key):
    """Ẇ_net can't be maximized while "Ẇ_net wanted" is typed in the cycle
    summary: the solver picks the mass flow that makes exactly that power at
    every pressure, so the goal is flat and its "curve" is rounding noise."""
    from units import label
    target = _num((payload.get("cycle") or {}).get("netPower"))
    if key == "Wnet" and target is not None:
        unit = label(payload.get("unitFactors") or payload.get("unitSystem", "SI"), "power")
        raise OptimizeError(
            f"Ẇ_net can't be maximized here: \"Ẇ_net wanted\" in the cycle summary holds it at {target:g} {unit}, "
            "so the solver changes the mass flow to make exactly that at every pressure. "
            "Maximize η_th instead, which is what changes with this pressure, or clear Ẇ_net wanted."
        )


def _check_goal_moves(key, curve):
    """The goal has to change across the range to have an optimum. A flat one
    (to within the solver's last digits) means something else fixes it."""
    vals = [c["value"] for c in curve if c.get("value") is not None]
    if len(vals) < 3:
        return
    spread, size = max(vals) - min(vals), max(abs(v) for v in vals) or 1.0
    if spread <= 1e-7 * size:
        label = METRICS[key]["label"]
        raise OptimizeError(
            f"{label} stays at {vals[0]:.6g} across this whole range, so there is nothing to optimize: "
            f"something else in the diagram fixes it (a typed value, a fixed heat or power). "
            f"Try another goal, or free the value that pins it."
        )


# ---------------------------------------------------------------- entry point
def optimize_pressure(payload, spec):
    """spec = {
        variables:   [{state, min, max}]               display units
        objective:   "auto" | "eta" | "Wnet" | "COP_R" | "COP_HP"
        constraints: [{state, prop, op: ">=" | "<=", value}]
        curve:       true to sample M(P) for the plot (one variable only)
    }"""
    try:
        with full_precision():
            return _optimize(payload, spec)
    except OptimizeError as e:
        return {"errors": [str(e)]}


def _optimize(payload, spec):
    prob = _Problem(payload, spec)
    nv = len(prob.vars)

    # starting point: the pressures on the canvas, clamped into the range
    start = solve_system(payload, lean=True)
    if start.get("errors"):
        raise OptimizeError("Solve the cycle first: " + start["errors"][0])
    prob.tear = start.get("tear")
    p0 = []
    for v in prob.vars:
        p = _num((start["states"].get(str(v["state"])) or {}).get("P"))
        p0.append(min(max(p if p else math.sqrt(v["lo"] * v["hi"]), v["lo"]), v["hi"]))

    key = spec.get("objective") or "auto"
    if key == "auto":
        key = auto_metric(payload, cycle_totals(payload, start))
    if key not in METRICS:
        raise OptimizeError(f"Unknown objective '{key}'.")
    _check_free_goal(payload, key)

    # ---- the curve: a picture, and a safe starting guess
    curve = []
    if nv == 1 and spec.get("curve", True):
        for i in range(CURVE_POINTS):
            u = [i / (CURVE_POINTS - 1)]
            m = prob.metric(key, u)
            feas = m is not None and _meets_limits(prob, u)
            curve.append({"P": prob.to_p(u)[0], "value": m, "feasible": bool(feas)})

    # Several pressures: no curve to start from, so a coarse grid supplies a
    # feasible start (and a guard against a poor local optimum). The answer
    # still comes from the KKT solve below, not from the grid.
    if nv > 1:
        k = 5 if nv == 2 else 4 if nv == 3 else 3
        for u in itertools.product(*[np.linspace(0.0, 1.0, k)] * nv):
            u = list(u)
            m = prob.metric(key, u)
            ok = m is not None and _meets_limits(prob, u)
            curve.append({"u": u, "value": m, "feasible": bool(ok)})
        for c in curve:
            c["P"] = prob.to_p(c.pop("u"))
        if not any(c["feasible"] for c in curve):
            curve = []

    u0 = prob.to_u(p0)
    if prob.metric(key, u0) is None or any((prob.limit_g(c, u0) or -1) < 0 for c in prob.limits):
        good = [c for c in curve if c["feasible"] and c["value"] is not None]
        if not good:
            raise OptimizeError(
                "No pressure in this range gives a solvable cycle that meets every limit. "
                "Widen the range or loosen a limit."
            )
        u0 = prob.to_u(_as_list(max(good, key=lambda c: c["value"])["P"]))
    elif curve:
        good = [c for c in curve if c["feasible"] and c["value"] is not None]
        best = max(good, key=lambda c: c["value"]) if good else None
        if best and best["value"] > prob.metric(key, u0):
            u0 = prob.to_u(_as_list(best["P"]))
    _check_goal_moves(key, curve)
    if nv > 1:
        curve = []      # the grid was only a start; nothing to plot for several pressures

    # Normalize M so SLSQP's tolerances mean the same thing for % and kW.
    scale = abs(prob.metric(key, u0)) or 1.0
    BAD = 1e6

    def f(u):
        m = prob.metric(key, u)
        return BAD if m is None else -m / scale

    def fgrad(u):
        g = _grad(lambda x: prob.metric(key, x), u)
        return np.zeros(nv) if g is None else -g / scale

    cons = []
    for c in prob.limits:
        s = abs(c["value"]) or 1.0
        cons.append({
            "type": "ineq",
            "fun": (lambda u, c=c, s=s: (prob.limit_g(c, u) if prob.limit_g(c, u) is not None else -BAD) / s),
            "jac": (lambda u, c=c, s=s: (lambda g: np.zeros(nv) if g is None else g / s)(
                _grad(lambda x: prob.limit_g(c, x), u))),
        })

    res = minimize(f, np.array(u0, float), jac=fgrad, bounds=[(0.0, 1.0)] * nv,
                   constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 200})
    u_opt = np.clip(res.x, 0.0, 1.0)
    m_opt = prob.metric(key, u_opt)
    if m_opt is None:
        raise OptimizeError("The optimizer stepped into a pressure the cycle cannot solve. Narrow the range.")

    # ---- KKT report: gradient, active limits and their multipliers
    p_opt = prob.to_p(u_opt)
    grad_u = _grad(lambda x: prob.metric(key, x), u_opt)
    dudp = np.array([1.0 / (p * (math.log(v["hi"]) - math.log(v["lo"]))) for v, p in zip(prob.vars, p_opt)])
    active, a_grads = [], []
    for c in prob.limits:
        g = prob.limit_g(c, u_opt)
        on = g is not None and abs(g) <= ACTIVE_TOL * max(1.0, abs(c["value"])) * 1e3
        active.append(on)
        if on:
            a_grads.append(_grad(lambda x, c=c: prob.limit_g(c, x), u_opt))
    at_bound = [("min" if u <= 1e-9 else "max" if u >= 1 - 1e-9 else None) for u in u_opt]

    # ∇M + Σ λ_j ∇g_j = 0 on the free variables (bounds carry their own multipliers)
    lambdas = []
    free = [i for i, b in enumerate(at_bound) if b is None]
    if a_grads and grad_u is not None and free:
        A = np.array([[g[i] for i in free] for g in a_grads]).T
        lam, *_ = np.linalg.lstsq(A, -grad_u[free], rcond=None)
        lambdas = [float(x) for x in lam]
    limits_out, k = [], 0
    for c, on in zip(prob.limits, active):
        item = dict(c, active=bool(on))
        if on:
            item["lambda"] = lambdas[k] if k < len(lambdas) else None
            k += 1
        limits_out.append(item)

    free_grad = None
    if grad_u is not None and free and not a_grads:
        free_grad = float(np.max(np.abs(grad_u[free] * dudp[free])))

    # ---- pictures for several pressures (the optimum above does not use them)
    if nv == 2 and spec.get("curve", True):
        curve = _goal_map(prob, key)
    elif nv > 2 and spec.get("curve", True):
        curve = _slices(prob, key, u_opt)

    final = prob.solve_at(u_opt)
    try:
        # the optimum once more in full, for its cycle on the h-s chart
        full = solve_system(prob.payload_at(u_opt))
        final = dict(final, hs=full.get("hs"))
    except Exception:
        pass
    objective ={"key": key, **METRICS[key]}
    if objective["unit"] == "power":
        totals = cycle_totals(payload, final)
        objective["unit"] = "h" if totals and totals["perMass"] else "power"
    return {
        "objective": objective,
        "optimum": {
            "pressures": [{"state": v["state"], "group": v["group"], "P": p, "bound": b}
                          for v, p, b in zip(prob.vars, p_opt, at_bound)],
            "value": m_opt,
            "start_value": prob.metric(key, prob.to_u(p0)),
            "states": final.get("states"),
            "devices": final.get("devices"),
            "hs": final.get("hs"),          # the optimum cycle on the Mollier chart
        },
        "kkt": {
            "dM_dP": free_grad,         # ≈ 0 at an interior optimum with no active limit
            "limits": limits_out,       # λ: gain in M per unit the limit is relaxed
            "converged": bool(res.success),
            "message": res.message,
            "iterations": int(res.nit),
            "solves": prob.evals,
        },
        "curve": curve,
        "units": start.get("units"),
        "errors": [],
    }
