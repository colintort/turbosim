"""Closed-system solver: a fixed mass of gas in a piston-cylinder going through
processes, for the Otto, Diesel, dual, Stirling, Ericsson and Atkinson cycles.

States carry P, T, v (specific volume), u, h, s and x (two-phase only). Any
two give the rest. Each process connects an initial state ("in") to a final
state ("out") and follows a path set by `config.path`:

  isochoric    v1 = v2                 isobaric     P1 = P2
  isothermal   T1 = T2                 isentropic   s1 = s2
  adiabatic    s2 from an isentropic efficiency η (blank η: solved for)
  polytropic   P1 v1^n = P2 v2^n       (blank n: solved for)

Parameters (all optional):
  compression   r = v1/v2, rp = P2/P1, n, η
  expansion     r = v2/v1, rp = P1/P2, n, η
  heatAddition  q (heat added), rc = v2/v1 (cutoff ratio), rp = P2/P1
  heatRejection q (heat removed)

A heat process with the `regenerated` assumption trades its heat with a
regenerator, so it doesn't count toward the heat in or out of the cycle.

Energy per kg of gas, with w the boundary work done by the gas:
  q - w = u2 - u1,   w = integral of P dv along the path.
Like the open solver this works in SI, and only typed values are given.
"""
import math

from ideal_gas import IDEAL_AIR, humid_air, with_ideal_air

from units import to_si, from_si, label, power_factor
from system_solver import SolveError, _close, _number, _efficiency, _round, _ports, _dome, MAX_PASSES

PROPS = ("P", "T", "v", "u", "h", "s")
FIELDS = PROPS + ("x",)
CP_KEY = {"P": "P", "T": "T", "v": "D", "u": "U", "h": "H", "s": "S", "x": "Q"}
# Unit keys: u shares the energy-per-mass unit of h.
UNIT_KEY = {"u": "h"}

# Pairs to try, best first. Quality pairs pin saturated states; some pairs
# CoolProp can't invert (T-h, T-u, ...) or that fail for a fluid fall through
# to the next one.
PAIR_PRIORITY = [
    ("P", "x"), ("T", "x"), ("v", "x"), ("u", "x"), ("h", "x"), ("s", "x"),
    ("P", "v"), ("T", "v"), ("P", "T"), ("P", "h"), ("P", "s"), ("P", "u"),
    ("v", "u"), ("v", "s"), ("v", "h"), ("T", "s"), ("h", "s"), ("u", "s"),
    ("T", "u"), ("T", "h"),
]

PATHS = ("isochoric", "isobaric", "isothermal", "isentropic", "adiabatic", "polytropic")
DEFAULT_PATH = {"compression": "isentropic", "expansion": "isentropic",
                "heatAddition": "isochoric", "heatRejection": "isochoric"}
HEAT_TYPES = ("heatAddition", "heatRejection")
SAMPLES = 32


def _to_cp(key, value):
    return 1.0 / value if key == "v" else value


def _from_cp(key, value):
    return 1.0 / value if key == "v" else value


def evaluate(props, fluid, known, want):
    """{key: SI value} for each key in want, from the first pair of `known`
    that the property library can solve. Quality is left out of single-phase
    states. Raises the first pair's error when none works."""
    first = None
    for k1, k2 in PAIR_PRIORITY:
        if k1 not in known or k2 not in known:
            continue
        try:
            out = {}
            for k in want:
                if k in (k1, k2):
                    out[k] = known[k]
                    continue
                try:
                    val = _from_cp(k, props(CP_KEY[k], CP_KEY[k1], _to_cp(k1, known[k1]),
                                            CP_KEY[k2], _to_cp(k2, known[k2]), fluid))
                except Exception:
                    if k == "x":
                        continue
                    raise
                if k == "x" and not (-1e-9 <= val <= 1 + 1e-9):
                    continue
                if not math.isfinite(val):
                    raise ValueError(f"{k} is not finite")
                out[k] = val
            return (k1, k2), out
        except Exception as e:
            if first is None:
                first = ((k1, k2), e)
    if first is None:
        return None, None
    raise _PairError(*first)


class _PairError(Exception):
    def __init__(self, pair, error):
        super().__init__(str(error))
        self.pair, self.error = pair, error


def _process_kind(node):
    """What crosses the boundary in this process: "Win", "Wout", "Qin", "Qout" or ""
    (the regenerator's heat stays inside the engine)."""
    if (node.get("assumptions") or {}).get("regenerated"):
        return ""
    return {"compression": "Win", "expansion": "Wout", "heatAddition": "Qin",
            "heatRejection": "Qout"}.get(node.get("type"), "")


class _ClosedSolver:
    def __init__(self, payload, props):
        self.props = props
        self.fluid = payload.get("fluid", "Air")
        self.system = payload.get("unitFactors") or payload.get("unitSystem", "SI")
        self.nodes = payload.get("nodes") or {}
        self.values = {}
        self.verified = {}
        self.dev = {}        # node id -> solved η or n
        self.state_fluids = {int(k): v for k, v in (payload.get("stateFluids") or {}).items() if v}
        for key, cells in (payload.get("states") or {}).items():
            n = int(key)
            self.values[n] = {}
            for f in FIELDS:
                x = _number(cells.get(f), f"S{n} {f}")
                if x is not None:
                    self.values[n][f] = to_si(self.system, UNIT_KEY.get(f, f), x)
        cycle = payload.get("cycle") or {}
        m = _number(cycle.get("mass"), "Mass")
        self.mass = to_si(self.system, "mass", m) if m is not None else None
        if self.mass is not None and self.mass <= 0:
            raise SolveError("Mass: must be greater than zero.")
        self.engine = self._engine(cycle.get("engine") or {})
        self.gas_info = {}   # humid air: omega, k

    def _engine(self, e):
        """Cylinder geometry (SI) from bore, stroke and compression ratio, plus
        speed for the power. Displacement Vd = (π/4)·bore²·stroke, clearance
        volume V_TDC = Vd / (r − 1), V_BDC = r·V_TDC. Older files gave the
        piston's distance from the head at TDC and BDC instead; those still work."""
        def num(key, label, unit=None):
            x = _number(e.get(key), label)
            if x is not None and x <= 0:
                raise SolveError(f"{label} must be greater than zero.")
            return to_si(self.system, unit, x) if x is not None and unit else x
        bore = num("bore", "Bore", "bore")
        stroke = num("stroke", "Stroke", "bore")
        r = num("r", "Compression ratio")
        if stroke is None and r is None:
            tdc = num("tdc", "Top dead centre", "bore")
            bdc = num("bdc", "Bottom dead centre", "bore")
            if tdc is not None and bdc is not None:
                if bdc <= tdc:
                    raise SolveError("Bottom dead centre must be farther from the head than top dead centre.")
                stroke, r = bdc - tdc, bdc / tdc
        if bore is None or stroke is None or r is None:
            return {}
        if r <= 1:
            raise SolveError("Compression ratio must be greater than 1.")
        vd = math.pi * bore ** 2 / 4 * stroke
        out = {"Vtdc": vd / (r - 1), "Vbdc": vd * r / (r - 1), "r": r}
        rpm = num("rpm", "Engine speed")
        if rpm is not None:
            out["cycles_per_s"] = rpm / 60 / ((num("strokes", "Strokes per cycle") or 4) / 2)
            out["cylinders"] = num("cylinders", "Cylinders") or 1
        return out

    def _bdc_state(self):
        """The state at the start of compression (bottom dead centre)."""
        for node in self.nodes.values():
            if node.get("type") == "compression":
                return _ports(node).get("in")
        return None

    # ---------- helpers ----------
    def _v(self, n):
        return self.values.setdefault(n, {})

    def _fluid(self, n):
        return self.state_fluids.get(n, self.fluid)

    def _fmt(self, f, si):
        k = UNIT_KEY.get(f, f)
        return f"{from_si(self.system, k, si):.6g} {label(self.system, k)}"

    def _set(self, n, key, value, where):
        v = self._v(n)
        if key in v:
            if not _close(v[key], value):
                raise SolveError(
                    f"{where}: S{n} {key} = {self._fmt(key, v[key])} conflicts with "
                    f"the value the process requires ({self._fmt(key, value)})."
                )
            return False
        v[key] = value
        return True

    def _same(self, a, b, key, where):
        va, vb = self._v(a), self._v(b)
        if key in va and key in vb:
            if not _close(va[key], vb[key]):
                raise SolveError(
                    f"{where}: {key} is {self._fmt(key, va[key])} at S{a} but "
                    f"{self._fmt(key, vb[key])} at S{b}, and the process keeps it constant."
                )
            return False
        if key in va:
            vb[key] = va[key]
            return True
        if key in vb:
            va[key] = vb[key]
            return True
        return False

    def _ratio(self, num, den, key, r, where):
        """num[key] = r * den[key]."""
        vn, vd = self._v(num), self._v(den)
        if key in vd:
            return self._set(num, key, r * vd[key], where)
        if key in vn:
            vd[key] = vn[key] / r
            return True
        return False

    def _offset(self, b, a, key, d, where):
        """b[key] = a[key] + d."""
        va, vb = self._v(a), self._v(b)
        if key in va:
            return self._set(b, key, va[key] + d, where)
        if key in vb:
            va[key] = vb[key] - d
            return True
        return False

    # ---------- states ----------
    def _resolve_state(self, n):
        v = self._v(n)
        if "x" in v and not (-1e-9 <= v["x"] <= 1 + 1e-9):
            raise SolveError(f"S{n}: quality x = {v['x']:g} must be between 0 and 1.")
        for k in ("v", "P"):
            if k in v and v[k] <= 0:
                raise SolveError(f"S{n}: {k} must be greater than zero.")
        signature = tuple(sorted(v))
        if self.verified.get(n) == signature:
            return False
        try:
            pair, found = evaluate(self.props, self._fluid(n), v, FIELDS)
        except _PairError as e:
            given = ", ".join(f"{q} = {self._fmt(q, v[q])}" for q in e.pair)
            raise SolveError(
                f"S{n}: could not find a {self._fluid(n)} state with {given}. Check the values and their "
                f"units (the property data may not reach this far)."
            )
        if pair is None:
            return False
        changed = False
        for k, value in found.items():
            if k in v:
                if not _close(v[k], value):
                    raise SolveError(
                        f"S{n}: {k} = {self._fmt(k, v[k])} is inconsistent with {pair[0]} and {pair[1]} "
                        f"(they give {self._fmt(k, value)}). The state is over-specified."
                    )
            else:
                v[k] = value
                changed = True
        self.verified[n] = tuple(sorted(v))
        return changed

    def _at(self, n, key, **known):
        """One property of state n's fluid from two others (SI)."""
        _, out = evaluate(self.props, self._fluid(n), known, (key,))
        return out[key]

    # ---------- processes ----------
    @staticmethod
    def path_of(node):
        t = node.get("type")
        p = (node.get("config") or {}).get("path") or DEFAULT_PATH.get(t)
        return p if p in PATHS else DEFAULT_PATH.get(t, "isentropic")

    def _param(self, node, key, name, positive=True):
        x = _number((node.get("params") or {}).get(key), f"{name} {key}")
        if x is not None and positive and x <= 0:
            raise SolveError(f"{name}: {key} must be greater than zero.")
        return x

    def _apply(self, node_id, node):
        t = node.get("type")
        if t not in DEFAULT_PATH:
            return False
        name = node.get("name") or t
        ports = _ports(node)
        a, b = ports.get("in"), ports.get("out")
        if a is None or b is None:
            return False
        path = self.path_of(node)
        va, vb = self._v(a), self._v(b)
        changed = False

        # ratios
        r = self._param(node, "r", name)
        if t == "compression" and self.engine:
            if r is not None and not _close(r, self.engine["r"]):
                raise SolveError(f"{name}: r = {r:g}, but the cylinder's compression ratio is {self.engine['r']:.6g}.")
            r = self.engine["r"]
        if r is not None and t == "compression":
            changed |= self._ratio(a, b, "v", r, name)
        if r is not None and t == "expansion":
            changed |= self._ratio(b, a, "v", r, name)
        rc = self._param(node, "rc", name)
        if rc is not None and t == "heatAddition":
            changed |= self._ratio(b, a, "v", rc, name)
        rp = self._param(node, "rp", name)
        if rp is not None:
            if t == "expansion":
                changed |= self._ratio(a, b, "P", rp, name)
            else:
                changed |= self._ratio(b, a, "P", rp, name)

        # the path
        if path == "isochoric":
            changed |= self._same(a, b, "v", name)
        elif path == "isobaric":
            changed |= self._same(a, b, "P", name)
        elif path == "isothermal":
            changed |= self._same(a, b, "T", name)
        elif path == "isentropic":
            changed |= self._same(a, b, "s", name)
        elif path == "adiabatic":
            changed |= self._adiabatic(node_id, node, name, a, b, t == "compression")
        elif path == "polytropic":
            changed |= self._polytropic(node_id, node, name, a, b)

        # heat, from the first law along the path
        q = self._param(node, "q", name, positive=False) if t in HEAT_TYPES else None
        if q is not None:
            q = to_si(self.system, "h", q)
            if t == "heatRejection":
                q = -q
            if path == "isochoric":
                changed |= self._offset(b, a, "u", q, name)
            elif path == "isobaric":
                changed |= self._offset(b, a, "h", q, name)
            elif path == "isothermal" and va.get("T", vb.get("T")) is not None:
                changed |= self._offset(b, a, "s", q / va.get("T", vb.get("T")), name)
            elif path in ("isentropic", "adiabatic") and abs(q) > 0:
                raise SolveError(f"{name}: an {path} process has no heat, so q must be blank or 0.")
        return changed

    def _adiabatic(self, node_id, node, name, a, b, compression):
        eta = _efficiency((node.get("params") or {}).get("eta"), name)
        va, vb = self._v(a), self._v(b)
        if "s" not in va or "u" not in va:
            return False
        if "v" in vb:
            u2s = self._at(a, "u", v=vb["v"], s=va["s"])
        elif "P" in vb:
            u2s = self._at(a, "u", P=vb["P"], s=va["s"])
        else:
            return False
        if eta is None:
            if "u" in vb and not _close(vb["u"], va["u"]):
                found = (u2s - va["u"]) / (vb["u"] - va["u"]) if compression else (va["u"] - vb["u"]) / (va["u"] - u2s)
                self.dev.setdefault(node_id, {})["eta"] = found
            return False
        u2 = va["u"] + (u2s - va["u"]) / eta if compression else va["u"] - eta * (va["u"] - u2s)
        return self._set(b, "u", u2, name)

    def _polytropic(self, node_id, node, name, a, b):
        n = self._param(node, "n", name)
        va, vb = self._v(a), self._v(b)
        have = [k in d for d in (va, vb) for k in ("P", "v")]
        if n is None:
            if all(have) and not _close(va["v"], vb["v"]):
                self.dev.setdefault(node_id, {})["n"] = math.log(vb["P"] / va["P"]) / math.log(va["v"] / vb["v"])
            return False
        if have.count(False) != 1:
            if all(have):
                lhs, rhs = va["P"] * va["v"] ** n, vb["P"] * vb["v"] ** n
                if not _close(lhs, rhs):
                    raise SolveError(f"{name}: P v^n is {lhs:.6g} at S{a} but {rhs:.6g} at S{b} (n = {n:g}).")
            return False
        if "P" not in vb:
            vb["P"] = va["P"] * (va["v"] / vb["v"]) ** n
        elif "v" not in vb:
            vb["v"] = va["v"] * (va["P"] / vb["P"]) ** (1 / n)
        elif "P" not in va:
            va["P"] = vb["P"] * (vb["v"] / va["v"]) ** n
        else:
            va["v"] = vb["v"] * (vb["P"] / va["P"]) ** (1 / n)
        return True

    # ---------- run ----------
    def run(self):
        try:
            for _ in range(MAX_PASSES):
                changed = False
                for n in list(self.values):
                    changed |= self._resolve_state(n)
                for node_id, node in self.nodes.items():
                    changed |= self._apply(node_id, node)
                if not changed:
                    break
        except SolveError as e:
            return [str(e)]
        if self.engine:
            v1 = self.values.get(self._bdc_state(), {}).get("v")
            if v1:
                self.mass = self.engine["Vbdc"] / v1    # the gas filling the cylinder at BDC
        return []

    # ---------- results ----------
    def work_heat(self, node):
        """(w by the gas, q into the gas) per kg in SI, or None."""
        ports = _ports(node)
        va, vb = self.values.get(ports.get("in"), {}), self.values.get(ports.get("out"), {})
        if not all(k in d for d in (va, vb) for k in ("P", "v", "u")):
            return None
        path = self.path_of(node)
        du = vb["u"] - va["u"]
        if path == "isochoric":
            w = 0.0
        elif path == "isobaric":
            w = va["P"] * (vb["v"] - va["v"])
        elif path in ("isentropic", "adiabatic"):
            w = -du
        elif path == "isothermal":
            if "s" not in va or "s" not in vb:
                return None
            w = va["T"] * (vb["s"] - va["s"]) - du
        else:   # polytropic: n from the end states
            if _close(va["v"], vb["v"]):
                w = 0.0
            else:
                n = math.log(vb["P"] / va["P"]) / math.log(va["v"] / vb["v"])
                if abs(n - 1) < 1e-9:
                    w = va["P"] * va["v"] * math.log(vb["v"] / va["v"])
                else:
                    w = (vb["P"] * vb["v"] - va["P"] * va["v"]) / (1 - n)
        return w, du + w

    def _energy(self, si_per_kg):
        """(per kg in display units, total for the mass in display units or None)."""
        per = _round(from_si(self.system, "h", si_per_kg))
        tot = _round(from_si(self.system, "E", si_per_kg * self.mass)) if self.mass is not None else None
        return per, tot

    def report_states(self):
        return {
            str(n): {f: (_round(from_si(self.system, UNIT_KEY.get(f, f), v[f])) if f in v else "") for f in FIELDS}
            for n, v in sorted(self.values.items())
        }

    def report_devices(self):
        out = {}
        for node_id, node in self.nodes.items():
            t = node.get("type")
            r = {}
            wq = self.work_heat(node)
            if wq is not None:
                w, q = wq
                shown_w = -w if t == "compression" else w     # compression: work in
                shown_q = -q if t == "heatRejection" else q    # rejection: heat out
                r["w"], r["W"] = self._energy(shown_w)
                r["q"], r["Q"] = self._energy(shown_q)
                r = {k: x for k, x in r.items() if x is not None}
            for k in ("eta", "n"):
                x = self.dev.get(node_id, {}).get(k)
                if x is not None:
                    r[k] = _round(x)
            if r:
                out[node_id] = r
        return out

    def report_cycle(self):
        """Cycle totals in display units, once every process is known."""
        procs = [n for n in self.nodes.values() if n.get("type") in DEFAULT_PATH]
        if not procs:
            return {}
        w_out = w_in = q_in = q_out = 0.0
        for node in procs:
            wq = self.work_heat(node)
            if wq is None:
                return {}
            w, q = wq
            if w > 0:
                w_out += w
            else:
                w_in -= w
            if (node.get("assumptions") or {}).get("regenerated"):
                continue
            if q > 0:
                q_in += q
            else:
                q_out -= q
        vs = [v["v"] for v in self.values.values() if "v" in v]
        w_net = w_out - w_in
        c = {}
        c["wnet"], c["Wnet"] = self._energy(w_net)
        c["win"], c["Win"] = self._energy(w_in)
        c["wout"], c["Wout"] = self._energy(w_out)
        c["qin"], c["Qin"] = self._energy(q_in)
        c["qout"], c["Qout"] = self._energy(q_out)
        if q_in > 0:
            c["eta"] = _round(w_net / q_in)
        if w_out > 0:
            c["rbw"] = _round(w_in / w_out)
        if vs and min(vs) > 0:
            c["r"] = _round(max(vs) / min(vs))
            if max(vs) > min(vs):
                c["mep"] = _round(from_si(self.system, "P", w_net / (max(vs) - min(vs))))
        e = self.engine
        if e:
            c["Vd"] = _round(from_si(self.system, "disp", e["Vbdc"] - e["Vtdc"]))
            c["Vtdc"] = _round(from_si(self.system, "disp", e["Vtdc"]))
            c["Vbdc"] = _round(from_si(self.system, "disp", e["Vbdc"]))
            if self.mass is not None:
                c["mass"] = _round(from_si(self.system, "mass", self.mass))
                if "cycles_per_s" in e:
                    c["power"] = _round(w_net * self.mass * e["cycles_per_s"] * e["cylinders"] / power_factor(self.system))
        c.update(self.gas_info)
        return {k: x for k, x in c.items() if x is not None}

    def report_messages(self):
        msgs = []
        for n in sorted(self.values):
            known = sum(1 for k in FIELDS if k in self.values[n])
            if known < 2:
                msgs.append(f"S{n}: needs {2 - known} more of P, T, v, u, h, s, x.")
        return msgs

    # ---------- diagrams ----------
    def _path_points(self, node):
        """[(v, P, T, s, h)] along the process (SI)."""
        ports = _ports(node)
        a, b = ports.get("in"), ports.get("out")
        va, vb = self.values.get(a, {}), self.values.get(b, {})
        if not all(k in d for d in (va, vb) for k in ("P", "v", "T", "s", "h")):
            return None
        fluid = self._fluid(a)
        path = self.path_of(node)
        ends = [tuple(d[k] for k in ("v", "P", "T", "s", "h")) for d in (va, vb)]
        pts = []
        try:
            for i in range(SAMPLES + 1):
                f = i / SAMPLES
                if path == "isochoric" or (path != "isobaric" and _close(va["v"], vb["v"])):
                    known = {"v": va["v"], "u": va["u"] + (vb["u"] - va["u"]) * f}
                elif path == "isobaric":
                    known = {"P": va["P"], "s": va["s"] + (vb["s"] - va["s"]) * f}
                else:
                    v = va["v"] * (vb["v"] / va["v"]) ** f
                    if path == "isothermal":
                        known = {"T": va["T"], "v": v}
                    elif path == "isentropic":
                        known = {"s": va["s"], "v": v}
                    else:   # polytropic or adiabatic: P v^n with n from the ends
                        n = math.log(vb["P"] / va["P"]) / math.log(va["v"] / vb["v"])
                        known = {"P": va["P"] * (va["v"] / v) ** n, "v": v}
                _, o = evaluate(self.props, fluid, known, ("v", "P", "T", "s", "h"))
                pts.append((o["v"], o["P"], o["T"], o["s"], o["h"]))
        except Exception:
            pts = []
        return pts or ends

    def diagrams(self):
        """{ts, pv, ph}: per fluid, the dome, the states and the process paths
        in display units. T-s points are [s, T]; P-v [v, P]; P-h [h, P]."""
        sys_ = self.system
        ts_pt = lambda v, P, T, s, h: [from_si(sys_, "s", s), from_si(sys_, "T", T)]  # noqa: E731
        pv_pt = lambda v, P, T, s, h: [from_si(sys_, "v", v), from_si(sys_, "P", P)]  # noqa: E731
        ph_pt = lambda v, P, T, s, h: [from_si(sys_, "h", h), from_si(sys_, "P", P)]  # noqa: E731
        ts, pv, ph = {}, {}, {}

        def entry(fluid):
            if fluid not in ts:
                dome = _dome(self.props, fluid)
                ts[fluid] = {"dome": [[from_si(sys_, "s", s), from_si(sys_, "T", T)] for s, T, *_ in dome],
                             "states": {}, "paths": []}
                pv[fluid] = {"dome": self._pv_dome(fluid), "states": {}, "paths": []}
                ph[fluid] = {"dome": [[from_si(sys_, "h", h), from_si(sys_, "P", P)] for _, _, h, P in dome],
                             "states": {}, "paths": []}
            return ts[fluid], pv[fluid], ph[fluid]

        for n, v in self.values.items():
            if all(k in v for k in ("P", "v", "T", "s", "h")):
                t_, p_, h_ = entry(self._fluid(n))
                pt = (v["v"], v["P"], v["T"], v["s"], v["h"])
                t_["states"][str(n)] = ts_pt(*pt)
                p_["states"][str(n)] = pv_pt(*pt)
                h_["states"][str(n)] = ph_pt(*pt)
        for node in self.nodes.values():
            pts = self._path_points(node) if node.get("type") in DEFAULT_PATH else None
            if not pts:
                continue
            ports = _ports(node)
            t_, p_, h_ = entry(self._fluid(ports["in"]))
            ends = {"from": ports["in"], "to": ports["out"], "kind": _process_kind(node)}
            t_["paths"].append({**ends, "points": [ts_pt(*p) for p in pts]})
            p_["paths"].append({**ends, "points": [pv_pt(*p) for p in pts]})
            h_["paths"].append({**ends, "points": [ph_pt(*p) for p in pts]})
        return {"fluids": ts}, {"fluids": pv}, {"fluids": ph}

    def _pv_dome(self, fluid):
        try:
            t_lo = self.props("Ttriple", "", 0, "", 0, fluid)
            t_crit = self.props("Tcrit", "", 0, "", 0, fluid)
        except Exception:
            return []
        t_lo = max(t_lo, t_crit * 0.4) + 0.01
        liq, vap = [], []
        for i in range(61):
            f = 1 - (1 - i / 60) ** 2
            t = t_lo + (t_crit - t_lo) * f * 0.9999
            try:
                p = self.props("P", "T", t, "Q", 0, fluid)
                liq.append([1 / self.props("D", "T", t, "Q", 0, fluid), p])
                vap.append([1 / self.props("D", "T", t, "Q", 1, fluid), p])
            except Exception:
                continue
        return [[from_si(self.system, "v", v), from_si(self.system, "P", p)] for v, p in liq + vap[::-1]]


def _humid_props(payload, props):
    """With a relative humidity given, the ideal-gas air carries water vapor:
    omega from the humidity at the first state's P and T, then cp and R of the
    mixture. Returns (props, {omega, k})."""
    rh = _number((payload.get("cycle") or {}).get("RH"), "Relative humidity")
    if not rh:
        return props, {}
    if not 0 < rh <= 100:
        raise SolveError("Relative humidity must be between 0 and 100 %.")
    if payload.get("fluid") != IDEAL_AIR:
        raise SolveError(f"Humidity works with the gas '{IDEAL_AIR}'. Pick it, or leave the humidity blank.")
    system = payload.get("unitFactors") or payload.get("unitSystem", "SI")
    first = None
    for node in (payload.get("nodes") or {}).values():
        if node.get("type") == "compression":
            first = _ports(node).get("in")
    cells = (payload.get("states") or {}).get(str(first)) or {}
    P, T = _number(cells.get("P"), "P"), _number(cells.get("T"), "T")
    if P is None or T is None:
        raise SolveError(f"Humidity needs the pressure and temperature of S{first}, where the air comes in.")
    P, T = to_si(system, "P", P), to_si(system, "T", T)
    base = getattr(props, "base", props)
    try:
        p_sat = base("P", "T", T, "Q", 0, "Water")
    except Exception:
        raise SolveError(f"Humidity: no saturation pressure for water at S{first}'s temperature.")
    pv = rh / 100 * p_sat
    if pv >= P:
        raise SolveError("Humidity: the vapor pressure would exceed the total pressure.")
    omega = 0.622 * pv / (P - pv)
    gas = humid_air(omega)
    return with_ideal_air(base, gas), {"omega": _round(omega), "k": _round(gas.k)}


def solve_closed(payload, props):
    try:
        props, gas_info = _humid_props(payload, props)
        solver = _ClosedSolver(payload, props)
        solver.gas_info = gas_info
        errors = solver.run()
    except SolveError as e:
        return {"mode": "closed", "states": {}, "devices": {}, "cycle": {}, "errors": [str(e)], "messages": []}
    ts, pv, ph = solver.diagrams()
    return {
        "mode": "closed",
        "states": solver.report_states(),
        "devices": solver.report_devices(),
        "cycle": {} if errors else solver.report_cycle(),
        "units": {"power": label(solver.system, "power"), "E": label(solver.system, "E")},
        "errors": errors,
        "messages": [] if errors else solver.report_messages(),
        "ts": ts,
        "pv": pv,
        "ph": ph,
    }
