"""Optimization mode: pipe diameters for a run of pipe segments in series.

Darcy-Weisbach head loss in segment i with diameter D_i:

    h_i = f (L_i / D_i) V_i² / (2 g),  V_i = 4 Q_i / (π D_i²)
        = K_i / D_i^5,                  K_i = 8 f Q_i² L_i / (π² g)

Two problems, each a Lagrange-multiplier problem like the MATLAB original:

  minLoss   least total head loss Σ K_i / D_i^5 for a fixed amount of pipe
            wall area π Σ L_i D_i = A (the material budget)
  minArea   least pipe wall area for an allowed total head loss h_max

They are solved with scipy's SLSQP (sequential quadratic programming), which
handles the equality constraint and keeps every diameter positive, so a
negative diameter can't come out. For these two problems there is also a
closed form, D_i ∝ (K_i / L_i)^(1/6), used as the starting guess and as a
check on the answer.
"""
import math

# (length unit, flow unit -> base flow, g) per unit system. Base units:
# EN: ft, ft³/s, ft² ; SI: m, m³/s, m².
SYSTEMS = {
    "EN": {"g": 32.174, "len": "ft", "area": "ft²", "dia": ("in", 12.0), "vel": "ft/s",
           "flow": {"cfm": 1 / 60, "ft³/s": 1.0, "gpm": 0.13368055556 / 60}},
    "SI": {"g": 9.80665, "len": "m", "area": "m²", "dia": ("mm", 1000.0), "vel": "m/s",
           "flow": {"m³/s": 1.0, "L/s": 1e-3, "m³/h": 1 / 3600}},
}


class OptimizeError(Exception):
    pass


def _num(x, what, positive=True):
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        raise OptimizeError(f"{what}: '{x}' is not a number.")
    if positive and not v > 0:
        raise OptimizeError(f"{what} must be greater than zero.")
    return v


def _closed_form(K, L, goal, budget):
    """Exact optimum: D_i = c (K_i / L_i)^(1/6), with c set by the constraint."""
    shape = [(k / l) ** (1 / 6) for k, l in zip(K, L)]
    if goal == "minLoss":   # π Σ L c s_i = A
        c = budget / (math.pi * sum(l * s for l, s in zip(L, shape)))
    else:                   # Σ K / (c s_i)^5 = h_max
        c = (sum(k / s ** 5 for k, s in zip(K, shape)) / budget) ** (1 / 5)
    return [c * s for s in shape]


def optimize_pipes(payload):
    try:
        return _optimize(payload)
    except OptimizeError as e:
        return {"errors": [str(e)]}


def _optimize(payload):
    import numpy as np
    from scipy.optimize import minimize

    sys_ = SYSTEMS.get(payload.get("system", "EN"), SYSTEMS["EN"])
    flow_unit = payload.get("flowUnit") or next(iter(sys_["flow"]))
    if flow_unit not in sys_["flow"]:
        raise OptimizeError(f"Unknown flow unit '{flow_unit}'.")
    to_base = sys_["flow"][flow_unit]
    f = _num(payload.get("f"), "Friction factor f")
    g = sys_["g"]
    goal = payload.get("goal", "minLoss")
    if goal not in ("minLoss", "minArea"):
        raise OptimizeError(f"Unknown goal '{goal}'.")
    budget = _num(payload.get("budget"), "Pipe wall area A" if goal == "minLoss" else "Allowed head loss")

    segs = payload.get("segments") or []
    if not segs:
        raise OptimizeError("Add at least one pipe segment.")
    L = [_num(s.get("L"), f"Segment {i + 1} length") for i, s in enumerate(segs)]
    Q = [_num(s.get("Q"), f"Segment {i + 1} flow") * to_base for i, s in enumerate(segs)]
    K = np.array([8 * f * q ** 2 * l / (math.pi ** 2 * g) for q, l in zip(Q, L)])
    La = np.array(L)

    exact = np.array(_closed_form(K, L, goal, budget))
    scale = exact.mean()   # solve in D / scale so the numbers are near 1

    def loss(D):
        return float(np.sum(K / D ** 5))

    def area(D):
        return float(math.pi * np.sum(La * D))

    if goal == "minLoss":
        obj = lambda y: loss(y * scale) / loss(exact)  # noqa: E731
        jac = lambda y: -5 * K * (y * scale) ** -6 * scale / loss(exact)  # noqa: E731
        con = {"type": "eq", "fun": lambda y: area(y * scale) / budget - 1,
               "jac": lambda y: math.pi * La * scale / budget}
    else:
        obj = lambda y: area(y * scale) / area(exact)  # noqa: E731
        jac = lambda y: math.pi * La * scale / area(exact)  # noqa: E731
        con = {"type": "eq", "fun": lambda y: loss(y * scale) / budget - 1,
               "jac": lambda y: -5 * K * (y * scale) ** -6 * scale / budget}

    # Start away from the answer (all diameters equal), as fsolve did from
    # x_init, so the optimizer does real work; the closed form checks it.
    y0 = np.full(len(L), 1.0)
    res = minimize(obj, y0, jac=jac, constraints=[con], method="SLSQP",
                   bounds=[(1e-6, None)] * len(L), options={"ftol": 1e-12, "maxiter": 500})
    D = res.x * scale
    if not res.success or np.max(np.abs(D - exact) / exact) > 1e-4:
        raise OptimizeError(f"The optimizer did not converge ({res.message}).")

    # Lagrange multiplier, in the MATLAB sign convention
    # (lag = Σ K D^-5 - λ (π Σ L D - A)): ∂lag/∂D_i = 0 gives the same λ for every i.
    lam = float(np.mean(-5 * K * D ** -6 / (math.pi * La)))
    dia_unit, dia_factor = sys_["dia"]
    out = []
    for i in range(len(L)):
        v = 4 * Q[i] / (math.pi * D[i] ** 2)
        out.append({"D": float(D[i]), "Dsmall": float(D[i] * dia_factor), "V": float(v), "h": float(K[i] / D[i] ** 5),
                    "K": float(K[i])})
    return {
        "errors": [],
        "segments": out,
        "totalLoss": loss(D),
        "area": area(D),
        "lambda": lam if goal == "minLoss" else 1 / lam,
        "iterations": int(res.nit),
        "units": {"len": sys_["len"], "area": sys_["area"], "dia": dia_unit, "vel": sys_["vel"]},
    }
