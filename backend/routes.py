"""What each TurboSim route does, as plain functions.

main.py (the desktop app's web server) and the browser version's worker
(web/pyworker.js, which runs these files in Pyodide) both call these, so the
two give the same answers from one source. Nothing here imports FastAPI.
"""


def solve(payload):
    """Whole-system solve. Payload comes straight from the React app:
    { fluid, unitSystem, states, nodes, edges }
    Returns { states, devices, units, errors, messages, propulsion, exergy }."""
    from system_solver import solve_system
    result = solve_system(payload)
    if payload.get("mode") != "closed":
        from propulsion import propulsion_summary
        from exergy import exergy_summary
        try:
            result["propulsion"] = propulsion_summary(payload, result)
        except Exception as e:          # never let the summary break a solve
            result["propulsion"] = {"ready": False, "need": f"a solvable jet ({e})"}
        try:
            result["exergy"] = exergy_summary(payload, result)
        except Exception as e:          # never let the summary break a solve
            result["exergy"] = {"ready": False, "need": f"a solvable cycle ({e})",
                                "devices": [], "lost": [], "states": {}, "notes": []}
    return result


def optimize(payload):
    """Optimization mode: pipe diameters (see optimize.py)."""
    from optimize import optimize_pipes
    return optimize_pipes(payload)


def design(payload):
    """Design studio (see design.py)."""
    from design import design as run_design
    return run_design(payload)


def optimize_pressure(payload):
    """Pressure optimizer: the /solve payload plus `optimize`
    { variables, objective, constraints }."""
    from pressure_optimizer import optimize_pressure as run
    spec = payload.pop("optimize", None) or {}
    return run(payload, spec)


def sweep(payload):
    """Off-design sweep: the /solve payload plus `sweep`
    { target, from, to, steps, watch }. Returns { points, errors }."""
    from sweep import run_sweep
    spec = payload.pop("sweep", None) or {}
    return run_sweep(payload, spec)


# route name -> function (the worker looks them up here)
JSON_ROUTES = {
    "solve": solve,
    "optimize": optimize,
    "design": design,
    "optimize-pressure": optimize_pressure,
    "sweep": sweep,
}
