"""Solver hints: when a cycle doesn't solve, say exactly where.

diagnose(payload, solver, errors) looks at what the solve left behind and
returns
    { "states":  { "3": {"kind": "missing" | "conflict", "text": ...} },
      "devices": { node id: text },
      "messages": [ plain sentences, most useful first ] }

"missing": the state ended with fewer than two of P, T, h, s, x, so nothing
fixes it yet. The text says what it does have and what would close it, from
the devices it connects to. "conflict": an error names the state (a value the
equations contradict, or an over-specified state). A device is flagged when an
error starts with its name, or when it touches a state that is still missing.
"""

import re

PROPS = ("P", "T", "h", "s", "x")

# What usually closes a state at each kind of device, in the app's words.
DEVICE_FIX = {
    "turbine": "η or s₁ = s₂",
    "compressor": "η, s₁ = s₂ or the pressure ratio",
    "pump": "η or s₁ = s₂",
    "hx": "Q̇, q per kg, or the outlet T",
    "valve": "h₁ = h₂",
    "nozzle": "a velocity, or η",
    "diffuser": "a velocity, or η",
    "mixer": "Q̇ = 0 and the other streams' flows",
    "splitter": "same T and h on every outlet",
    "flash": "the inlet quality",
    "atmosphere": "the ambient P and T",
    "ac": "Q̇ or q per kg of dry air",
}


def _already_set(node):
    """True when the device already has what DEVICE_FIX would suggest, so
    suggesting it again wouldn't help."""
    t = node.get("type")
    asm = node.get("assumptions") or {}
    params = node.get("params") or {}
    given = lambda k: str(params.get(k, "")).strip() != ""  # noqa: E731
    if t in ("turbine", "compressor", "pump"):
        return bool(asm.get("isentropic") or asm.get("eta100") or given("eta"))
    if t == "hx":
        return bool(asm.get("adiabatic") or given("Qdot") or given("q"))
    if t == "valve":
        return bool(asm.get("isenthalpic"))
    return False


def _ports(node):
    ports = node.get("ports")
    if ports:
        return {k: int(v) for k, v in ports.items() if v is not None and v != ""}
    return {"in": int(node["inState"]), "out": int(node["outState"])}


def _list(items, word="and"):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {word} {items[-1]}"


def diagnose(payload, solver=None, errors=()):
    nodes = payload.get("nodes") or {}
    touching = {}                 # state -> [(node id, node)]
    for node_id, node in nodes.items():
        try:
            ports = _ports(node)
        except (KeyError, TypeError, ValueError):
            continue
        unused = set(node.get("unused") or [])
        for port, n in ports.items():
            if port not in unused:
                touching.setdefault(n, []).append((node_id, node))

    states, devices, messages = {}, {}, []

    # ---- errors: the states and devices they name
    names = sorted(((n.get("name") or "", i) for i, n in nodes.items()), key=lambda x: -len(x[0]))
    for err in errors:
        for m in re.finditer(r"\bS(\d+)\b", err):
            states.setdefault(m.group(1), {"kind": "conflict", "text": err})
        for name, node_id in names:
            if name and err.startswith(name):
                devices.setdefault(node_id, err)
                break

    # ---- states nothing fixes yet (only meaningful after a clean run)
    if solver is not None and not errors:
        missing = []
        for n in sorted(solver.values):
            v = solver.values[n]
            humid = solver._fluid(n) == "Humid air"
            props = ("T", "h", "w", "RH") if humid else PROPS     # humid air: P plus two of these
            known = [k for k in props if k in v]
            if len(known) >= 2 and (not humid or "P" in v):
                continue
            missing.append(n)
            if humid and "P" not in v:
                known = known[:1]
            names = {"w": "ω", "RH": "φ"}
            have = f"has only {names.get(known[0], known[0])}" if known else "has nothing yet"
            fixes = []
            for node_id, node in touching.get(n, []):
                fix = DEVICE_FIX.get(node.get("type"))
                if node.get("type") == "ac" and (node.get("config") or {}).get("mode") in ("evaporative", "steam", "desiccant"):
                    fix = "the outlet φ, T or ω (no heat is added)"
                label = node.get("name") or node.get("type")
                if fix and not _already_set(node):
                    fixes.append(f"{label}: {fix}")
                devices.setdefault(node_id, f"Waiting on S{n}, which {have}.")
            others = [names.get(k, k) for k in props if k not in known]
            need = "one more" if known else "two"
            text = f"S{n} {have}. Type {need} of {_list(others, 'or')} there"
            if humid and "P" not in v:
                text = f"S{n} has no P. Humid air needs P plus two of T, h, ω or φ"
            if fixes:
                text += f", or set {'; '.join(fixes)}"
            states[str(n)] = {"kind": "missing", "text": text + "."}
        if missing:
            ids = [f"S{n}" for n in missing]
            first = states[str(missing[0])]["text"]
            messages.append(f"Not enough is given yet: {_list(ids)} {'is' if len(ids) == 1 else 'are'} not fixed. {first}")
            messages += [states[str(n)]["text"] for n in missing[1:]]

    return {"states": states, "devices": devices, "messages": messages}


# ====================================================================== 1.10.4
# explain(): the reason a cycle doesn't solve, in a student's words, with the
# state or device to look at. Each item is
#   {"kind": "error" | "missing" | "info", "text": ..., "states": [n], "devices": [node id]}
# "error" and "missing" items lead the status bar; "info" items (a value given
# twice that happens to agree, results per kg) only show in the Degrees of
# freedom window.

WORD = {"T": "temperature", "h": "enthalpy", "s": "entropy", "x": "quality", "P": "pressure", "m": "mass flow"}
SYMBOL = {"m": "ṁ"}
NUM = r"(-?[\d.]+(?:e[-+]?\d+)?\s*[^\s,()]*)"


def _plural(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def _outlets(node):
    from dof import ports_of
    return [s for k, s in ports_of(node).items() if k == "out" or k.startswith(("ext", "a_out", "b_out", "vapor", "liquid"))]


def _count_sentence(dof):
    if not dof or not dof.get("supported"):
        return ""
    k, u, free = dof["knowns"], dof["unknowns"], dof.get("flowFree", 0)
    text = f"{k} knowns for {u} unknowns"
    if free:
        text += (f"; {'one' if free == 1 else free} of the unknowns {'is a mass flow' if free == 1 else 'are mass flows'}, "
                 f"which per-kg results can do without")
    return text


def explain(payload, solver, errors, dof):
    from dof import outlet_spec, typed, ports_of, INTENSIVE
    nodes = payload.get("nodes") or {}
    by_name = sorted(((n.get("name") or n.get("type") or "", i) for i, n in nodes.items()), key=lambda x: -len(x[0]))
    items = []

    def node_named(text):
        for name, node_id in by_name:
            if name and (text == name or text.startswith(name + " ") or text.startswith(name + ":")):
                return node_id
        return None

    def typed_int(n):
        return [k for k in typed(payload, n) if k in INTENSIVE]

    def props_list(ks):
        return _list([SYMBOL.get(k, k) for k in ks])

    def touching(n):
        return [i for i, nd in nodes.items() if n in ports_of(nd).values()]

    def double_fix(n, key=None):
        """A device whose inputs fix outlet S{n} while S{n} also has a typed
        value they set: (node id, sentence) or None."""
        ty = [k for k in typed_int(n) if k != "P"]
        for node_id in touching(n):
            node = nodes[node_id]
            spec = outlet_spec(node)
            if not spec or n not in _outlets(node):
                continue
            hits = [k for k in ty if k in spec[1]] or ([key] if key in spec[1] and key in typed(payload, n) else [])
            if not hits:
                continue
            name = node.get("name") or node.get("type")
            what = WORD.get(hits[0], hits[0])
            return node_id, (f"{name}: give either {spec[0]} or the exit {what} (S{n}), not both. "
                             f"Each one fixes S{n} on its own, and they don't agree.")
        return None

    def repeated_flow(around):
        """An item for ṁ typed more than once on one stream, if any of the
        states `around` is on it."""
        for group in (dof or {}).get("repeatedFlows") or []:
            if set(group) & set(around) or not around:
                return {"kind": "error", "states": group, "devices": [],
                        "text": f"ṁ is typed at {_list([f'S{s}' for s in group])}, which are one stream (mass in = "
                                f"mass out through every device between them), and the values differ. Type ṁ at "
                                f"only one of them."}
        return None

    # ---------------------------------------------------------------- errors
    for err in errors[:1]:
        m = re.match(r"S(\d+): (\w+) = .* is inconsistent with (\w+) and (\w+)", err)
        if m:
            n = int(m.group(1))
            ty = typed_int(n)
            if len(ty) > 2:
                items.append({"kind": "error", "states": [n], "devices": [],
                              "text": f"S{n} has {len(ty)} properties given ({props_list(ty)}), but only 2 are "
                                      f"needed to fix a state, and these don't agree. Remove one."})
                break
            fix = double_fix(n)
            if fix:
                items.append({"kind": "error", "states": [n], "devices": [fix[0]], "text": fix[1]})
                break
            items.append({"kind": "error", "states": [n], "devices": touching(n),
                          "text": f"S{n} is fixed twice and the two don't agree: the {WORD.get(m.group(2), m.group(2))} "
                                  f"that reaches it differs from what {m.group(3)} and {m.group(4)} give. One value "
                                  f"feeding S{n} (typed there or at a device next to it) is extra."})
            break

        m = re.match(r"(.+?): S(\d+) (\w+) = " + NUM + r" conflicts with the value the equations require \((.+)\)\.", err)
        if m:
            where, n, key, have, need = m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5)
            node_id = node_named(where)
            name = nodes[node_id].get("name") if node_id else where
            node = nodes.get(node_id) or {}
            params = node.get("params") or {}
            if key == "P" and str(params.get("rp", "")).strip():
                items.append({"kind": "error", "states": [n], "devices": [node_id],
                              "text": f"{name}: give either the pressure ratio or the pressure at S{n}, not both. "
                                      f"The ratio needs {need}, but S{n} has {have}."})
                break
            if key == "m":
                rep = repeated_flow([n] + (sorted(ports_of(node).values()) if node else []))
                if rep:
                    items.append(rep)
                    break
                items.append({"kind": "error", "states": [n], "devices": [node_id] if node_id else [],
                              "text": f"S{n}: ṁ = {have}, but {name}'s balances need {need}. Too many flows are "
                                      f"given: type ṁ at one state per stream and let the solver find the rest."})
                break
            fix = double_fix(n, key)
            if fix:
                items.append({"kind": "error", "states": [n], "devices": [fix[0]], "text": fix[1]})
                break
            if key in typed(payload, n):
                items.append({"kind": "error", "states": [n], "devices": [node_id] if node_id else [],
                              "text": f"S{n}: the {WORD.get(key, key)} typed there ({have}) and what {name} needs "
                                      f"({need}) disagree. Remove the typed value, or the {name} input that sets it."})
                break
            items.append({"kind": "error", "states": [n], "devices": [node_id] if node_id else [],
                          "text": f"S{n} gets its {WORD.get(key, key)} from two places that disagree: {name} needs "
                                  f"{need}, while the rest of the cycle gives {have}. One input along the way is extra."})
            break

        m = re.match(r"(.+?): (\w+) is " + NUM + r" at S(\d+) but " + NUM + r" at S(\d+), and the assumptions require them equal", err)
        if m:
            where, key, xa, a, xb, b = m.group(1), m.group(2), m.group(3), int(m.group(4)), m.group(5), int(m.group(6))
            node_id = node_named(where)
            name = nodes[node_id].get("name") if node_id else where
            devs = [node_id] if node_id else []
            fix = (double_fix(b, key) or double_fix(a, key)) if key in ("s", "h") else None
            if fix:
                items.append({"kind": "error", "states": [a, b], "devices": [fix[0]], "text": fix[1]})
                break
            if key == "P":
                text = (f"{name} has P in = P out checked, but S{a} and S{b} have different pressures ({xa} and {xb}). "
                        f"Give the pressure at only one of them, or uncheck P in = P out.")
            elif key == "m" and repeated_flow([a, b]):
                items.append(repeated_flow([a, b]))
                break
            elif key == "m":
                text = (f"S{a} and S{b} are the same stream through {name}, so their ṁ must match, but they are "
                        f"{xa} and {xb}. Type ṁ at only one of them.")
            elif key == "s":
                text = (f"{name} has s₁ = s₂ checked, but S{a} and S{b} have different entropies. For a real device "
                        f"uncheck s₁ = s₂ and give η, or remove the extra value at S{b}.")
            elif key == "h":
                text = (f"{name} has h₁ = h₂ checked, but S{a} and S{b} have different enthalpies. "
                        f"Remove the extra value at S{b}, or uncheck h₁ = h₂.")
            else:
                text = (f"{name} needs the same {WORD.get(key, key)} at S{a} and S{b}, but they are {xa} and {xb}. "
                        f"Give it at only one of them.")
            items.append({"kind": "error", "states": [a, b], "devices": devs, "text": text})
            break

        m = re.match(r"(.+?): the (.+?) balance does not close", err) or re.match(r"(.+?): the energy balance (.+?) does not close", err)
        if m:
            node_id = node_named(m.group(1))
            name = nodes[node_id].get("name") if node_id else m.group(1)
            ports = ports_of(nodes[node_id]) if node_id else {}
            around = sorted(set(ports.values()))
            if "mass" in m.group(2) or m.group(2) in ("split", "vapor fraction", "liquid fraction") or m.group(2).startswith("extraction"):
                rep = repeated_flow(around)
                if rep:
                    items.append(rep)
                    break
                flows = [s for s in around if "m" in typed(payload, s)]
                text = (f"{name}: the flows in and out don't add up. ṁ is typed at {_list([f'S{s}' for s in flows]) or 'too many states'}, "
                        f"more than its mass balance allows. Leave all but one of them blank.")
                items.append({"kind": "error", "states": flows, "devices": devs_of(node_id), "text": text})
            else:
                text = (f"{name}: every state around it is already fixed, so its energy balance has nothing left to "
                        f"find, and it doesn't close. One value around {name} is extra: a typed T or h at "
                        f"{_list([f'S{s}' for s in around])}, or its Q̇.")
                items.append({"kind": "error", "states": around, "devices": devs_of(node_id), "text": text})
            break

    if errors and dof and dof.get("supported") and dof["stateMissing"] < 0:
        extra = -dof["stateMissing"]
        items.append({"kind": "count", "states": [], "devices": [],
                      "text": f"Count: {_count_sentence(dof)}, so {_plural(extra, 'value')} more than the cycle needs "
                              f"{'is' if extra == 1 else 'are'} given."})

    # ---------------------------------------------------------- nothing wrong, but not enough
    flows_known = solver is not None and any("m" in v for v in solver.values.values())
    unfixed = []
    if solver is not None and not errors:
        for n, v in solver.values.items():
            humid = solver._fluid(n) == "Humid air"
            if humid:
                if "P" not in v or sum(k in v for k in ("T", "h", "w", "RH")) < 2:
                    unfixed.append(n)
            elif sum(k in v for k in PROPS) < 2:
                unfixed.append(n)

    if unfixed and dof and dof.get("supported"):
        need = dof["stateMissing"]
        if need > 0:
            text = f"Needs {_plural(need, 'more value')} to fix every state ({_count_sentence(dof)})."
        elif dof.get("flowFree"):
            text = (f"Needs a mass flow: the states left wait on an energy balance that needs ṁ "
                    f"({dof['knowns']} knowns for {dof['unknowns']} unknowns).")
        else:
            text = (f"The count balances ({_count_sentence(dof)}), but the solver works one step at a time and "
                    f"can't untangle this one. Typing one of the values the highlighted states are waiting for "
                    f"usually unlocks it.")
        items.append({"kind": "missing", "states": sorted(unfixed), "devices": [], "text": text})

    # a balance that waits on a mass flow
    if unfixed and solver is not None:
        for node_id, node in nodes.items():
            ports = ports_of(node)
            t = node.get("type")
            two = t == "hx" and any(k.startswith("b_") for k in ports)
            params = node.get("params") or {}
            q_given = t == "hx" and str(params.get("Qdot", "")).strip() != ""
            if not (two or t == "mixer" or q_given):
                continue
            around = set(ports.values())
            if not (around & set(unfixed)):
                continue
            no_m = sorted(s for s in around if "m" not in solver.values.get(s, {}))
            if not no_m:
                continue
            name = node.get("name") or t
            items.append({"kind": "missing", "states": no_m, "devices": [node_id],
                          "text": f"Missing: a mass flow. {name} can't use its energy balance until ṁ is known at "
                                  f"{_list([f'S{s}' for s in no_m])}. Type ṁ at one state of each stream "
                                  f"(or the split fraction)."})
            break

    # ---------------------------------------------------------------- info
    if dof and dof.get("supported"):
        for n, st in dof["states"].items():
            if st["intensive"] > 2 and not errors:
                items.append({"kind": "info", "states": [int(n)], "devices": [],
                              "text": f"S{n} has {st['intensive']} properties given; 2 fix a state, so "
                                      f"{st['intensive'] - 2} {'is' if st['intensive'] == 3 else 'are'} extra. "
                                      f"They agree here, so it still solves."})
        for group in dof["repeatedPressures"]:
            if not errors:
                items.append({"kind": "info", "states": group, "devices": [],
                              "text": f"P is typed at {_list([f'S{s}' for s in group])}, but P in = P out ties them "
                                      f"together, so only one is needed. They agree here."})
        for group in dof["repeatedFlows"]:
            if not errors:
                items.append({"kind": "info", "states": group, "devices": [],
                              "text": f"ṁ is typed at {_list([f'S{s}' for s in group])}, which are one stream, "
                                      f"so only one is needed. They agree here."})
    if solver is not None and not errors and not unfixed and not flows_known:
        items.append({"kind": "info", "states": [], "devices": [],
                      "text": "No mass flow is given anywhere, so the results are per kg. Type ṁ at one state "
                              "of each loop, or give a Q̇ or the net power, to get rates (kW)."})

    for node_id, node in nodes.items():
        t = node.get("type")
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        if t in ("turbine", "compressor", "pump", "nozzle", "diffuser") \
                and (str(params.get("eta", "")).strip() or asm.get("eta100")) \
                and not asm.get("adiabatic") and not asm.get("isentropic"):
            name = node.get("name") or t
            items.append({"kind": "warning", "states": [], "devices": [node_id],
                          "text": f"{name}: η is typed but Q̇ = 0 is unchecked, so η isn't used. "
                                  f"Check Q̇ = 0 (the isentropic efficiency assumes an adiabatic device)."})
    if any(i["text"].startswith("Missing: a mass flow") for i in items):
        items = [i for i in items if not i["text"].startswith("Needs a mass flow")]
    return items


def devs_of(node_id):
    return [node_id] if node_id else []
