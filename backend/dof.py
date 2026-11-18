"""Degrees of freedom: knowns against unknowns for a steady-flow diagram (1.10.4).

count(payload) adds up, without solving anything,

  unknowns  3 per state (two properties fix a pure fluid's state, plus ṁ),
            plus V at the ends of a nozzle or diffuser
  knowns    what is typed into the state table (P, T, h, s, x, ṁ, V)
            + the independent mass balances (a closed loop has one fewer than
              it has devices: its flow can be anything)
            + one per equation a device's inputs and checked assumptions add
              (η or s₁ = s₂ fixes an outlet, P₁ = P₂, h₁ = h₂, Q̇ = 0, a given
              Q̇ or q, an effectiveness, a pressure ratio, ...)

missing = unknowns - knowns. Above zero the cycle needs that many more values
(one of them may only be a mass flow, which a per-kg answer can do without);
below zero something is given twice. The count is structural, the way a
textbook counts equations, so it can't see a value that happens to repeat
another one. The solver's own checks catch those.

Humid air, cooling towers, air-conditioning processes and combustors have
balances of their own (moisture, products) the count doesn't cover yet; for
them it reports supported = False and why.
"""

import re

INTENSIVE = ("P", "T", "h", "s", "x")
NAMES = {"P": "P", "T": "T", "h": "h", "s": "s", "x": "x", "m": "ṁ", "V": "V"}


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    t = str(v).strip().rstrip("%")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _given(params, key):
    return str((params or {}).get(key, "")).strip() != ""


def ports_of(node):
    """{port: state} without the optional ports nothing is connected to."""
    ports = node.get("ports")
    if not ports:
        ports = {"in": node.get("inState"), "out": node.get("outState")}
    unused = set(node.get("unused") or [])
    out = {}
    for k, v in ports.items():
        if v is None or v == "" or k in unused:
            continue
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def numbered(ports, prefix):
    found = []
    for k, v in ports.items():
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", k)
        if m:
            found.append((int(m.group(1)), v))
    return [v for _, v in sorted(found)]


def typed(payload, n):
    """The properties typed at state n, in table order."""
    cells = (payload.get("states") or {}).get(str(n)) or (payload.get("states") or {}).get(n) or {}
    return [k for k in INTENSIVE + ("m", "V") if _num(cells.get(k)) is not None]


def fluid_of(payload, n):
    sf = payload.get("stateFluids") or {}
    return sf.get(str(n)) or sf.get(n) or payload.get("fluid", "Water")


def _rank(rows, ncols):
    """Rank of a small dense matrix by Gaussian elimination."""
    m = [list(r) for r in rows]
    rank, col = 0, 0
    while rank < len(m) and col < ncols:
        piv = max(range(rank, len(m)), key=lambda i: abs(m[i][col]))
        if abs(m[piv][col]) < 1e-9:
            col += 1
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and abs(m[i][col]) > 0:
                f = m[i][col] / m[rank][col]
                m[i] = [a - f * b for a, b in zip(m[i], m[rank])]
        rank += 1
        col += 1
    return rank


def outlet_spec(node):
    """What a device's inputs fix at its outlet, as (short phrase, the outlet
    properties it sets), or None. Used to name the culprit when an outlet is
    given twice."""
    t = node.get("type")
    asm = node.get("assumptions") or {}
    params = node.get("params") or {}
    if t in ("turbine", "compressor", "pump", "nozzle", "diffuser"):
        if asm.get("isentropic"):
            return ("s₁ = s₂", ("h", "s", "T", "x"))
        if (asm.get("eta100") or _given(params, "eta")) and asm.get("adiabatic"):
            return ("η", ("h", "s", "T", "x"))
    if t == "valve" and asm.get("isenthalpic"):
        return ("h₁ = h₂", ("h", "s", "T", "x"))
    if t == "hx" and not any(k.startswith("b_") for k in ports_of(node)):
        if asm.get("adiabatic"):
            return ("Q̇ = 0", ("h", "s", "T", "x"))
        if _given(params, "q"):
            return ("q", ("h", "s", "T", "x"))
        if _given(params, "Qdot"):
            return ("Q̇", ("h", "s", "T", "x", "m"))
    if t == "hx" and (_given(params, "Qdot") or _given(params, "eps") or _given(params, "UA")):
        which = "Q̇" if _given(params, "Qdot") else "ε" if _given(params, "eps") else "UA"
        return (which, ("h", "s", "T", "x", "m"))
    return None


def count(payload):
    nodes = payload.get("nodes") or {}
    states = set()
    for node in nodes.values():
        states.update(ports_of(node).values())
    for k in (payload.get("states") or {}):
        try:
            states.add(int(k))
        except (TypeError, ValueError):
            pass
    states = sorted(states)

    # ---- what the count doesn't cover yet
    skip = []
    for node in nodes.values():
        t = node.get("type")
        cfg = node.get("config") or {}
        if t == "hx" and cfg.get("combustor"):
            skip.append("combustors")
        elif t == "coolingTower":
            skip.append("cooling towers")
        elif t == "ac":
            skip.append("air-conditioning processes")
    if any(fluid_of(payload, n) == "Humid air" for n in states):
        skip.append("humid air")
    if skip:
        seen = list(dict.fromkeys(skip))
        return {"supported": False,
                "reason": f"The knowns-vs-unknowns count doesn't cover {', '.join(seen)} yet."}

    # nozzles and diffusers solve for velocities; elsewhere a blank V is zero
    v_open = set()
    for node in nodes.values():
        if node.get("type") in ("nozzle", "diffuser"):
            p = ports_of(node)
            v_open.update(s for s in (p.get("in"), p.get("out")) if s is not None)

    # ---- typed values
    per_state = {}
    typed_int = typed_v = 0
    typed_m = []
    for n in states:
        props = typed(payload, n)
        if "x" in props and fluid_of(payload, n).lower().startswith(("air", "nitrogen", "co2", "helium", "argon")):
            props.remove("x")
        ints = [k for k in props if k in INTENSIVE]
        typed_int += len(ints)
        if "V" in props and n in v_open:
            typed_v += 1
        if "m" in props:
            typed_m.append(n)
        per_state[str(n)] = {"typed": [NAMES[k] for k in props if k != "V" or n in v_open],
                             "intensive": len(ints)}

    # ---- mass balances, as linear rows over every state's ṁ
    idx = {n: i for i, n in enumerate(states)}
    rows = []

    def row(terms):
        r = [0.0] * len(states)
        for s, c in terms:
            if s in idx:
                r[idx[s]] += c
        if any(abs(c) > 0 for c in r):
            rows.append(r)

    # ---- device equations
    devices = {}
    relations = 0
    for node_id, node in nodes.items():
        t = node.get("type")
        name = node.get("name") or t
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        p = ports_of(node)
        gives = []

        def add(k, text):
            if k > 0:
                gives.append({"n": k, "text": text})

        if t in ("compressor", "pump", "nozzle", "diffuser", "valve") and "in" in p and "out" in p:
            a, b = p["in"], p["out"]
            row([(a, 1), (b, -1)])
            if t == "valve":
                add(1 if asm.get("isenthalpic") else 0, "h₁ = h₂")
            else:
                spec = outlet_spec(node)
                add(1 if spec else 0, spec[0] if spec else "")
                if t == "compressor":
                    add(1 if _given(params, "rp") else 0, "pressure ratio")
                if t == "nozzle":
                    add(1 if _given(params, "Pb") else 0, "back pressure")
                if t in ("nozzle", "diffuser"):
                    add(1 if asm.get("adiabatic") else 0, "energy balance (Q̇ = 0)")
                    for key, neg in (("V1", "negV1"), ("V2", "negV2")):
                        add(1 if asm.get(neg) or _given(params, key) else 0, f"{key} given" if not asm.get(neg) else f"{key} ≈ 0")
                    for key in ("A1", "A2"):
                        add(1 if _given(params, key) else 0, f"{key} (ṁ = A V / v)")
        elif t == "turbine" and "in" in p and "out" in p:
            a = p["in"]
            exts = numbered(p, "ext")
            outlets = exts + [p["out"]]
            row([(a, 1)] + [(o, -1) for o in outlets])
            for i, e in enumerate(exts, 1):
                y = _num(params.get(f"y{i}"))
                if y is not None:
                    row([(e, 1), (a, -y)])
            staged = exts and not asm.get("sameOutlet")
            spec = outlet_spec(node)
            if staged:
                add(len(outlets) if spec else 0, f"{spec[0]} on each stage" if spec else "")
            else:
                add(2 * len(exts), "every outlet at the exhaust state")
                add(1 if spec else 0, spec[0] if spec else "")
            if asm.get("drivesCompressor"):
                add(1, "Ẇ turbine = Ẇ compressor")
            elif not asm.get("adiabatic") and len(outlets) == 1 and _given(params, "Wdot") and _given(params, "Qdot"):
                add(1, "Ẇ and Q̇ given")
        elif t == "hx":
            two = any(k.startswith("b_") for k in p)
            for side in ("a", "b"):
                ins, outs = numbered(p, f"{side}_in"), numbered(p, f"{side}_out")
                if not ins and not outs:
                    continue
                label = f"stream {side.upper()}: " if two else ""
                add(2 * max(0, len(outs) - 1), f"{label}outlets at one state")
                if asm.get(f"{side}Isobaric"):
                    add(len(ins) + min(len(outs), 1) - 1, f"{label}P in = P out")
                if asm.get(f"{side}SatLiquidOut") or asm.get(f"{side}SatVaporOut"):
                    add(1, f"{label}saturated outlet")
                if ins and outs:
                    row([(s, 1) for s in ins] + [(s, -1) for s in outs])
            if not two:
                if asm.get("adiabatic"):
                    add(1, "energy balance (Q̇ = 0)")
                else:
                    add(1 if _given(params, "q") else 0, "q given")
                    add(1 if _given(params, "Qdot") else 0, "Q̇ given")
            else:
                rate = _given(params, "Qdot") or _given(params, "eps") or _given(params, "UA")
                if rate:
                    which = "Q̇" if _given(params, "Qdot") else "ε" if _given(params, "eps") else "UA"
                    add(2, f"energy balances with {which} given")
                elif asm.get("noLoss"):
                    add(1, "energy balance (no heat loss)")
                add(1 if asm.get("idealOutlet") else 0, "cold outlet at the hot outlet T")
        elif t == "mixer":
            ins, c = numbered(p, "in"), p.get("out")
            if c is not None:
                row([(s, 1) for s in ins] + [(c, -1)])
                if asm.get("isobaric"):
                    add(len(ins), "P in = P out")
                if asm.get("adiabatic") or _given(params, "Qdot"):
                    add(1, "energy balance" + (" (Q̇ = 0)" if asm.get("adiabatic") else ", Q̇ given"))
        elif t == "splitter":
            a, outs = p.get("in"), numbered(p, "out")
            if a is not None:
                row([(a, 1)] + [(o, -1) for o in outs])
                for i, o in enumerate(outs, 1):
                    y = _num(params.get(f"y{i}"))
                    if y is not None:
                        row([(o, 1), (a, -y)])
                per = min(2, (1 if asm.get("isobaric") else 0) + (2 if asm.get("sameState") else 0))
                add(per * len(outs), "outlets at the inlet state" if asm.get("sameState") else "P in = P out")
        elif t == "flash":
            a, v, l = p.get("in"), p.get("vapor"), p.get("liquid")
            if None not in (a, v, l):
                row([(a, 1), (v, -1), (l, -1)])
                add(4, "both outlets at the inlet P, x = 1 and x = 0")
                add(1, "vapor flow = x · inlet flow")
        elif t == "atmosphere":
            add(1 if _given(params, "P0") else 0, "ambient P")
            if "out" in p:
                add(1 if _given(params, "T0") else 0, "ambient T")

        k = sum(g["n"] for g in gives)
        relations += k
        devices[node_id] = {"name": name, "n": k, "gives": [g["text"] for g in gives]}

    mass_rank = _rank(rows, len(states))
    unit_rows = []
    for n in typed_m:
        r = [0.0] * len(states)
        r[idx[n]] = 1.0
        unit_rows.append(r)
    mass_known = _rank(rows + unit_rows, len(states))

    # states tied together by mass balances alone (one stream / loop):
    # union states that appear together in a two-term row with equal and
    # opposite coefficients
    parent = {n: n for n in states}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        nz = [(states[i], c) for i, c in enumerate(r) if abs(c) > 0]
        if len(nz) == 2 and abs(nz[0][1] + nz[1][1]) < 1e-12:
            parent[find(nz[0][0])] = find(nz[1][0])
    streams = {}
    for n in states:
        streams.setdefault(find(n), []).append(n)
    repeated = []          # ṁ typed more than once on one stream
    for group in streams.values():
        hits = [n for n in group if n in typed_m]
        if len(hits) > 1:
            repeated.append(hits)

    # pressures tied together by P in = P out: typed twice on one isobar
    pp = {n: n for n in states}

    def pfind(x):
        while pp[x] != x:
            pp[x] = pp[pp[x]]
            x = pp[x]
        return x

    def tie(group):
        group = [s for s in group if s in pp]
        for s in group[1:]:
            pp[pfind(s)] = pfind(group[0])

    for node in nodes.values():
        t, asm, p = node.get("type"), node.get("assumptions") or {}, ports_of(node)
        if t == "hx":
            for side in ("a", "b"):
                ins, outs = numbered(p, f"{side}_in"), numbered(p, f"{side}_out")
                if asm.get(f"{side}Isobaric"):
                    tie(ins + outs)
                elif len(outs) > 1:
                    tie(outs)
        elif t == "mixer" and asm.get("isobaric"):
            tie(numbered(p, "in") + [p.get("out")])
        elif t == "splitter" and (asm.get("isobaric") or asm.get("sameState")):
            tie([p.get("in")] + numbered(p, "out"))
        elif t == "flash":
            tie([p.get("in"), p.get("vapor"), p.get("liquid")])
        elif t == "turbine" and asm.get("sameOutlet"):
            tie(numbered(p, "ext") + [p.get("out")])
    isobars = {}
    for n in states:
        isobars.setdefault(pfind(n), []).append(n)
    repeated_p = []
    for group in isobars.values():
        hits = [n for n in group if "P" in typed(payload, n)]
        if len(hits) > 1:
            repeated_p.append(hits)

    # separate streams (connected by any mass balance), for "a flow per loop"
    cc = {n: n for n in states}

    def cfind(x):
        while cc[x] != x:
            cc[x] = cc[cc[x]]
            x = cc[x]
        return x

    for r in rows:
        nz = [states[i] for i, c in enumerate(r) if abs(c) > 0]
        for s in nz[1:]:
            cc[cfind(s)] = cfind(nz[0])
    loops = {}
    for n in states:
        loops.setdefault(cfind(n), []).append(n)

    net = 1 if _num((payload.get("cycle") or {}).get("netPower")) is not None else 0

    # A loop with no ṁ typed and no rate input (a Q̇, UA, flow area, net
    # power) can carry any flow: its results are per kg. That slack is one
    # unknown per loop, kept apart so it doesn't hide a property given twice.
    rate_states = set()
    for node in nodes.values():
        t, asm, params, p = node.get("type"), node.get("assumptions") or {}, node.get("params") or {}, ports_of(node)
        rate = False
        if t == "hx":
            rate = (_given(params, "Qdot") and not asm.get("adiabatic")) or _given(params, "UA")
        elif t == "mixer":
            rate = _given(params, "Qdot") and not asm.get("adiabatic")
        elif t == "turbine":
            rate = not asm.get("adiabatic") and _given(params, "Wdot") and _given(params, "Qdot")
        elif t in ("nozzle", "diffuser"):
            rate = _given(params, "A1") or _given(params, "A2")
        if rate:
            rate_states.update(p.values())
    loop_list = list(loops.values())
    free = [lp for lp in loop_list if not any(s in typed_m or s in rate_states for s in lp)]
    flow_free = max(0, len(free) - net)

    unknowns = 3 * len(states) + len(v_open)
    knowns = typed_int + typed_v + mass_known + relations + net
    return {
        "supported": True,
        "unknowns": unknowns,
        "knowns": knowns,
        "missing": unknowns - knowns,
        "flowFree": flow_free,                         # loops whose flow nothing sets (per kg)
        "stateMissing": unknowns - knowns - flow_free,  # > 0 too few, < 0 too many
        "parts": {
            "states": len(states),
            "velocities": len(v_open),
            "typed": typed_int + typed_v + len(typed_m),
            "typedProps": typed_int + typed_v,
            "typedFlows": len(typed_m),
            "massBalances": mass_rank,
            "flowsUsed": mass_known - mass_rank,
            "relations": relations,
            "netPower": net,
        },
        "states": per_state,
        "devices": devices,
        "repeatedFlows": repeated,
        "repeatedPressures": repeated_p,
        "loops": loop_list,
        "freeLoops": free,
        "typedFlowStates": typed_m,
    }
