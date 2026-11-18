"""System solver for /solve.

Approach: repeated passes until nothing new can be determined.

  1. State pass:  any state with two known properties among P, T, h, s, x gets
                  the rest from CoolProp. Extra known values are checked for
                  consistency (over-specified or conflicting input -> error).
  2. Device pass: each device applies the equations implied by its type and its
                  checked assumptions, e.g. s1 = s2 copies entropy across a
                  device, P1 = P2 copies pressure, mass and energy balances fix
                  unknown flows and enthalpies, an isentropic efficiency gives
                  the outlet enthalpy.

Only values the user typed are ever treated as given. Computed values never
overwrite them; a mismatch is reported as an error instead.

Devices are described by named ports ("in", "out", "in1", "vapor", ...). The
frontend sends `ports: {port name: state number}` for every device.

Three devices have a variable number of ports, numbered from 1:
  turbine    in, out (main exhaust), ext1, ext2, ... (extractions)
  splitter   in, out1, out2, ...
  hx         a_in1.., a_out1.. (stream A) and, for a two-stream exchanger,
             b_in1.., b_out1.. (stream B). Ports of one stream mix; heat
             crosses between streams (or to the surroundings with one stream).

Kinetic and potential energy: a state may carry a velocity V and an
elevation z. Every energy balance uses e = h + V²/2 + g z, with a blank V or z
counting as zero, except at nozzles and diffusers, where a missing V is solved
for.

The solver works entirely in SI. Property calls go through `props`
(CoolProp.PropsSI by default) so the logic can be tested without CoolProp.
"""

import math
import re

from units import to_si, from_si, power_factor, label
from propulsion import continuity, nozzle_outlet_h, diffuser_outlet_P
from combustion import Combustion, is_combustor
from hints import diagnose, explain
import psychro
from psychro import HUMID_FIELDS

PROPS = ("P", "T", "h", "s")          # always solved once two properties are known
FIELDS = PROPS + ("x", "m")           # x = quality (two-phase only), m = mass flow
MOTION = ("V", "z")                    # velocity, elevation (kinetic and potential energy)
G = 9.80665
INTENSIVE = PROPS + ("x",)
CP_KEY = {"P": "P", "T": "T", "h": "H", "s": "S", "x": "Q"}

# Which pair to hand to CoolProp when more than two properties are known.
# Quality pairs come first: x is the only way to pin down a saturated state
# (P and T are not independent there), and a given x is then never re-derived.
PAIR_PRIORITY = [
    ("P", "x"), ("T", "x"), ("h", "x"), ("s", "x"),
    ("P", "h"), ("P", "s"), ("P", "T"), ("T", "h"), ("T", "s"), ("h", "s"),
]

# One inlet, one outlet, same mass flow through.
SINGLE_STREAM = ("compressor", "pump", "nozzle", "diffuser", "valve")

REL_TOL = 1e-4   # loose on purpose: users type rounded textbook values
ABS_TOL = 1e-3
MAX_PASSES = 100


class SolveError(Exception):
    pass


class Mismatch(SolveError):
    """A balance or equality that doesn't close, with its signed relative
    residual, so a loop of unknown flows can be iterated (see _tear_flows)."""

    def __init__(self, message, residual):
        super().__init__(message)
        self.residual = residual


def _rel(required, existing):
    return (required - existing) / max(abs(required), abs(existing), 1e-12)


def _close(a, b):
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _number(value, label):
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        raise SolveError(f"{label}: '{value}' is not a number.")


def _efficiency(value, name):
    """Accepts 0.85 or 85%. Anything outside (0, 1] is rejected."""
    text = "" if value is None else str(value).strip()
    if text == "":
        return None
    is_percent = text.endswith("%")
    x = _number(text.rstrip("%"), f"{name} η")
    if x is None:
        return None
    if is_percent:
        x /= 100.0
    if not 0 < x <= 1:
        raise SolveError(
            f"{name}: η = {text} is outside (0, 1]. Enter a fraction such as 0.85, or 85%."
        )
    return x


def _round(x):
    return float(f"{x:.7g}")


def _ports(node):
    """{port name: state number}. Older payloads only had inState / outState."""
    ports = node.get("ports")
    if ports:
        return {k: int(v) for k, v in ports.items() if v is not None}
    return {"in": int(node["inState"]), "out": int(node["outState"])}


def _numbered(ports, prefix):
    """States at prefix1, prefix2, ... in port order."""
    found = []
    for k, v in ports.items():
        m = re.fullmatch(re.escape(prefix) + r"(\d+)", k)
        if m:
            found.append((int(m.group(1)), v))
    return [v for _, v in sorted(found)]


class _Solver:
    def __init__(self, payload, props, ha_props=None):
        self.props = props
        self.ha_props = ha_props
        self.fluid = payload.get("fluid", "Water")
        self.system = payload.get("unitFactors") or payload.get("unitSystem", "SI")
        self.nodes = payload.get("nodes") or {}
        self.values = {}     # state number -> {property: SI value}
        self.verified = {}   # state number -> which properties were known when last checked
        self.pairs = {}      # state number -> the property pair that fixed it (Show my work)
        self.dev = {}        # node id -> device-level values reported back (eta)
        self.ct = {}         # node id -> cooling tower values (SI): ma, mMakeup, Vair, Q, Twb
        self.air = {}        # node id -> cooling tower air states, computed once
        # Each stream can carry its own fluid; `fluid` is the fallback.
        self.state_fluids = {
            int(k): v for k, v in (payload.get("stateFluids") or {}).items() if v
        }

        for key, cells in (payload.get("states") or {}).items():
            n = int(key)
            self.values[n] = {}
            for f in FIELDS + MOTION + HUMID_FIELDS:
                x = _number(cells.get(f), f"S{n} {f}")
                if x is not None:
                    self.values[n][f] = to_si(self.system, f, x)

        # States whose velocity a nozzle or diffuser may still solve for: until
        # it does, energy balances that touch them wait.
        self.v_open = set()
        for node in self.nodes.values():
            if node.get("type") in ("nozzle", "diffuser"):
                ports = _ports(node)
                self.v_open.update((ports["in"], ports["out"]))

        # Combustors: their products become the fluid of every state downstream.
        self.comb = Combustion(self)

    # ---------- helpers ----------
    def _v(self, n):
        return self.values.setdefault(n, {})

    def _fluid(self, n):
        comb = getattr(self, "comb", None)
        if comb is not None and n in comb.fluid:
            return comb.fluid[n]
        return self.state_fluids.get(n, self.fluid)

    def _fmt(self, f, si):
        return f"{from_si(self.system, f, si):.6g} {label(self.system, f)}"

    def _extra(self, n):
        """Kinetic plus potential energy per kg at state n (J/kg). A blank V or
        z counts as zero, except a velocity a nozzle or diffuser has yet to
        solve for: then None, and the balance waits."""
        v = self._v(n)
        if "V" not in v and n in self.v_open:
            return None
        return v.get("V", 0.0) ** 2 / 2 + G * v.get("z", 0.0)

    def _e(self, n):
        """h + V²/2 + g z, or None while h (or a pending velocity) is unknown."""
        h, x = self._v(n).get("h"), self._extra(n)
        return None if h is None or x is None else h + x

    def _set(self, n, key, value, where):
        """Set a missing value, or check an existing one. True if newly set."""
        v = self._v(n)
        if key in v:
            if not _close(v[key], value):
                raise Mismatch(
                    f"{where}: S{n} {key} = {self._fmt(key, v[key])} conflicts with "
                    f"the value the equations require ({self._fmt(key, value)}).",
                    _rel(value, v[key]),
                )
            return False
        v[key] = value
        return True

    def _propagate(self, a, b, key, where):
        """Apply the equation key_a = key_b."""
        va, vb = self._v(a), self._v(b)
        if key in va and key in vb:
            if not _close(va[key], vb[key]):
                raise Mismatch(
                    f"{where}: {key} is {self._fmt(key, va[key])} at S{a} but "
                    f"{self._fmt(key, vb[key])} at S{b}, and the assumptions require them equal.",
                    _rel(va[key], vb[key]),
                )
            return False
        if key in va:
            vb[key] = va[key]
            return True
        if key in vb:
            va[key] = vb[key]
            return True
        return False

    def _linear(self, terms, key, where, label=None, const=0.0):
        """Apply sum(coef * key[state]) + const = 0.
        Solves for the one unknown if exactly one value is missing, and checks
        the balance if none is."""
        unknown = [(s, c) for s, c in terms if key not in self._v(s)]
        if not unknown:
            parts = [c * self._v(s)[key] for s, c in terms] + [const]
            residual = sum(parts)
            scale = max(abs(p) for p in parts) or 1.0
            if abs(residual) > REL_TOL * scale + ABS_TOL:
                raise Mismatch(f"{where}: the {label or key} balance does not close (residual {residual:.4g}).",
                               residual / scale)
            return False
        if len(unknown) == 1:
            s, c = unknown[0]
            if c == 0:
                return False
            known = sum(cc * self._v(ss)[key] for ss, cc in terms if key in self._v(ss))
            self._v(s)[key] = -(known + const) / c
            return True
        return False

    def _mass_system(self, eqs, where):
        """Each eq is ({state: coef}, const) meaning sum(coef * m[state]) + const = 0.
        Solves for up to two unknown flows when the equations determine them."""
        unknown = sorted({s for eq, _ in eqs for s in eq if "m" not in self._v(s)})
        if not unknown or len(unknown) > 2:
            return False

        rows, rhs = [], []
        for eq, const in eqs:
            row = [eq.get(s, 0.0) for s in unknown]
            if all(abs(c) < 1e-12 for c in row):
                continue
            known = sum(c * self._v(s)["m"] for s, c in eq.items() if s not in unknown)
            rows.append(row)
            rhs.append(-(known + const))

        if len(unknown) == 1:
            if not rows:
                return False
            self._v(unknown[0])["m"] = rhs[0] / rows[0][0]
            return True

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (a, b), (c, d) = rows[i], rows[j]
                det = a * d - b * c
                if abs(det) > 1e-9 * max(1.0, abs(a * d), abs(b * c)):
                    self._v(unknown[0])["m"] = (rhs[i] * d - b * rhs[j]) / det
                    self._v(unknown[1])["m"] = (a * rhs[j] - c * rhs[i]) / det
                    return True
        return False

    def _cv(self, name, signed, groups, q=0.0):
        """A control volume with no work crossing its boundary and heat q (W)
        added to it. signed = [(state, +1 inflow / -1 outflow)]; groups = lists
        of states whose flows must balance among themselves (one list per
        separate stream)."""
        sign = dict(signed)
        changed = False
        for g in groups:
            changed |= self._linear([(s, sign[s]) for s in g], "m", name, "mass")

        extras = {s: self._extra(s) for s, _ in signed}
        if any(x is None for x in extras.values()):
            return changed
        masses = {s: self._v(s).get("m") for s, _ in signed}
        if all(m is not None for m in masses.values()):
            ke_pe = sum(sg * masses[s] * extras[s] for s, sg in signed)
            changed |= self._linear([(s, sg * masses[s]) for s, sg in signed], "h", name, "energy", q + ke_pe)
        elif all(self._v(s).get("h") is not None for s, _ in signed):
            eqs = [({s: sign[s] for s in g}, 0.0) for g in groups]
            eqs.append(({s: sg * (self._v(s)["h"] + extras[s]) for s, sg in signed}, q))
            changed |= self._mass_system(eqs, name)
        return changed

    # ---------- state pass ----------
    def _evaluate(self, n, v, pair):
        """The other properties of state n from the pair, {key: SI value}.
        Quality is left out of single-phase states."""
        k1, k2 = pair
        found = {}
        for k in PROPS + ("x",):
            if k in pair:
                continue
            try:
                value = self.props(CP_KEY[k], CP_KEY[k1], v[k1], CP_KEY[k2], v[k2], self._fluid(n))
            except Exception as e:
                if k == "x":
                    continue      # single-phase: quality is not defined
                if "out of range" in str(e) or "unable to solve" in str(e):
                    given = ", ".join(
                        f"{q} = {from_si(self.system, q, v[q]):.6g} {label(self.system, q)}" for q in pair
                    )
                    hint = (
                        "This usually means the heat added is too large for the mass flow "
                        "(or a value was entered in the wrong unit)."
                        if "h" in pair else
                        "Check that the fluid can exist there: below its triple-point pressure "
                        "there is no liquid, for instance."
                    )
                    raise SolveError(
                        f"S{n}: {given} is outside the range {self._fluid(n)}'s property data covers. {hint}"
                    )
                raise SolveError(f"S{n}: could not evaluate {self._fluid(n)} from {k1} and {k2} ({e}).")
            if k == "x" and not (-1e-9 <= value <= 1 + 1e-9):
                continue          # single-phase
            found[k] = value
        return found

    def _resolve_state(self, n):
        if self.comb.waiting(n):
            return False          # its fluid (combustion products) isn't known yet
        if psychro.is_humid(self, n):
            return psychro.resolve(self, n)
        v = self._v(n)

        if "x" in v and not (-1e-9 <= v["x"] <= 1 + 1e-9):
            raise SolveError(f"S{n}: quality x = {v['x']:g} must be between 0 and 1.")

        pairs = [p for p in PAIR_PRIORITY if p[0] in v and p[1] in v]
        if not pairs:
            return False

        signature = tuple(sorted(v))
        if self.verified.get(n) == signature:
            return False          # nothing new known since the last check

        # Some pairs can't be inverted for some fluids (T-h of an ideal gas,
        # say); then the next known pair is tried, and only if every pair
        # fails is the first one's error reported.
        first_error = None
        for pair in pairs:
            try:
                found = self._evaluate(n, v, pair)
                break
            except SolveError as e:
                first_error = first_error or e
        else:
            raise first_error

        k1, k2 = pair
        self.pairs.setdefault(n, pair)
        changed = False
        for k, value in found.items():
            if k in v:
                if not _close(v[k], value):
                    raise SolveError(
                        f"S{n}: {k} = {self._fmt(k, v[k])} is inconsistent with {k1} and {k2} "
                        f"(they give {self._fmt(k, value)}). The state is over-specified."
                    )
            else:
                v[k] = value
                changed = True

        self.verified[n] = tuple(sorted(v))
        return changed

    # ---------- device pass ----------
    def _apply_device(self, node_id, node):
        t = node.get("type")
        name = node.get("name") or t
        ports = _ports(node)
        changed = False

        if self.comb.active:
            if is_combustor(node):
                return self.comb.combustor(node_id, node, name, ports)
            if t == "mixer" and self.comb.mixes_products(node):
                return self.comb.mixer(node, name, ports)
            if self.comb.blocks(node):
                return False      # waits for the combustion products to reach it

        if t in SINGLE_STREAM:
            a, b = ports["in"], ports["out"]
            changed |= self._propagate(a, b, "m", name)   # mass in = mass out

            if t == "valve":
                if (node.get("assumptions") or {}).get("isenthalpic"):
                    changed |= self._propagate(a, b, "h", name)
            else:
                if t == "compressor":
                    changed |= self._pressure_ratio(node, name, a, b)
                changed |= self._turbomachine(node_id, node, name, a, b, compressor=(t != "nozzle"))
                if t == "nozzle":
                    changed |= self._back_pressure(node_id, node, name, a, b)
                if t in ("nozzle", "diffuser"):
                    changed |= self._kinetic(node_id, node, name, a, b)
                    changed |= self._flow_areas(node, name, a, b)
        elif t == "turbine":
            changed |= self._turbine(node_id, node, name, ports)
        elif t == "hx":
            changed |= self._hx(node, name, ports)
        elif t == "mixer":
            changed |= self._mixer(node, name, ports)
        elif t == "splitter":
            changed |= self._splitter(node, name, ports)
        elif t == "flash":
            changed |= self._flash(name, ports)
        elif t == "coolingTower":
            changed |= self._cooling_tower(node_id, node, name, ports)
        elif t == "atmosphere":
            changed |= self._atmosphere(node, name, ports)
        elif t == "ac":
            changed |= psychro.ac(self, node_id, node, name, ports)
        else:
            raise SolveError(f"{name}: unknown device type '{t}'.")
        return changed

    def _turbine(self, node_id, node, name, ports):
        """Turbine with any number of outlets: the main exhaust `out` plus
        extractions ext1, ext2, ...

        With "same outlet state" checked (the default) every outlet leaves at
        the exhaust's T and P, so the turbine is one expansion whose flow is
        split. Unchecked, the outlets are taken in order ext1, ext2, ..., out
        as the ends of successive stages, each with the same efficiency."""
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        a, out = ports["in"], ports["out"]
        exts = _numbered(ports, "ext")
        outlets = exts + [out]

        changed = self._linear([(a, 1)] + [(o, -1) for o in outlets], "m", name, "mass")
        for i, e in enumerate(exts, 1):
            y = _number(params.get(f"y{i}"), f"{name} extraction {i} fraction")
            if y is not None:
                if not 0 <= y <= 1:
                    raise SolveError(f"{name}: extraction {i} fraction {y:g} must be between 0 and 1.")
                changed |= self._linear([(e, 1), (a, -y)], "m", name, f"extraction {i}")

        staged = exts and not asm.get("sameOutlet")
        if staged:
            chain = [a] + exts + [out]
            stages = list(zip(chain, chain[1:]))
        else:
            for e in exts:
                for key in INTENSIVE:
                    changed |= self._propagate(out, e, key, name)
            stages = [(a, out)]

        for x, y in stages:
            changed |= self._turbomachine(node_id if len(stages) == 1 else None, node, name, x, y, compressor=False)
        if asm.get("drivesCompressor"):
            changed |= self._shaft(name, a, outlets)
        changed |= self._heat_loss(node, name, a, outlets)
        return changed

    def _heat_loss(self, node, name, a, outlets):
        """A turbine that is not adiabatic (Penoncello Ex. 3.2): with its power
        Ẇ and heat Q̇ (added; negative = lost) both typed, the energy balance
        m_in e_in + Q̇ = Σ m_out e_out + Ẇ fixes the exhaust enthalpy. With only
        Ẇ typed, the heat is reported once the states are known."""
        asm = node.get("assumptions") or {}
        if asm.get("adiabatic") or len(outlets) > 1:
            return False
        params = node.get("params") or {}
        w = _number(params.get("Wdot"), f"{name} Ẇ")
        q = _number(params.get("Qdot"), f"{name} Q̇")
        if w is None or q is None:
            return False
        pf = power_factor(self.system)
        w, q = w * pf, q * pf
        out = outlets[0]
        e_in, x_out = self._e(a), self._extra(out)
        m = self._v(a).get("m") or self._v(out).get("m")
        if e_in is None or x_out is None or not m:
            return False
        return self._set(out, "h", e_in + (q - w) / m - x_out, name)

    def _shaft(self, name, a, outlets):
        """The turbine drives the compressors and pumps on its shaft, as in a
        jet engine: Ẇ turbine = Ẇ compressors. Fixes the turbine outlet's
        energy. With every flow known the powers are matched; without flows,
        the same flow is taken through the turbine and the compressors (per kg)."""
        w_in, per_kg = 0.0, 0.0
        flows_known = True
        for node in self.nodes.values():
            if node.get("type") not in WORK_IN:
                continue
            p = _ports(node)
            e1, e2 = self._e(p["in"]), self._e(p["out"])
            if e1 is None or e2 is None:
                return False
            m = self._v(p["in"]).get("m")
            per_kg += e2 - e1
            if m is None:
                flows_known = False
            else:
                w_in += m * (e2 - e1)
        if per_kg == 0.0:
            raise SolveError(f"{name}: 'drives the compressor' is checked, but there is no compressor or pump to drive.")
        if len(outlets) > 1:
            raise SolveError(f"{name}: 'drives the compressor' works for a turbine with one outlet.")
        out = outlets[0]
        e_in, x_out = self._e(a), self._extra(out)
        if e_in is None or x_out is None:
            return False
        m_in, m_out = self._v(a).get("m"), self._v(out).get("m")
        if flows_known and m_in is not None and m_out:
            h_out = (m_in * e_in - w_in) / m_out - x_out
        else:
            h_out = e_in - per_kg - x_out
        return self._set(out, "h", h_out, name)

    def _pressure_ratio(self, node, name, a, b):
        """Compressor pressure ratio r = P2 / P1, when given."""
        r = _number((node.get("params") or {}).get("rp"), f"{name} pressure ratio")
        if r is None:
            return False
        if r <= 0:
            raise SolveError(f"{name}: the pressure ratio must be positive.")
        pa, pb = self._v(a).get("P"), self._v(b).get("P")
        if pa is not None:
            return self._set(b, "P", pa * r, name)
        if pb is not None:
            return self._set(a, "P", pb / r, name)
        return False

    def _turbomachine(self, node_id, node, name, a, b, compressor):
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        eta = 1.0 if asm.get("eta100") else _efficiency(params.get("eta"), name)

        changed = False
        if eta is None and not asm.get("isentropic") and asm.get("adiabatic") and node_id is not None:
            self._solve_efficiency(node_id, a, b, compressor, node.get("type"))

        if asm.get("isentropic"):
            if eta is not None and not _close(eta, 1.0):
                raise SolveError(
                    f"{name}: 's₁ = s₂' conflicts with η = {eta:g}. "
                    f"Uncheck 's₁ = s₂' to model a real device."
                )
            changed |= self._propagate(a, b, "s", name)
        elif eta is not None and asm.get("adiabatic"):
            changed |= self._outlet_from_efficiency(name, a, b, eta, compressor, node.get("type"))
        return changed

    def _isentropic_h(self, a, b):
        va, vb = self._v(a), self._v(b)
        if "s" not in va or "P" not in vb:
            return None
        return self.props("H", "P", vb["P"], "S", va["s"], self._fluid(a))

    def _solve_efficiency(self, node_id, a, b, compressor, t=None):
        """No efficiency given and both ends known: report the efficiency
        the states imply. Turbine and nozzle: (h1 - h2)/(h1 - h2s);
        compressor, pump and diffuser: (h2s - h1)/(h2 - h1)."""
        va, vb = self._v(a), self._v(b)
        if "h" not in va or "h" not in vb or "eta" in self.dev.get(node_id, {}):
            return
        try:
            h2s = self._isentropic_h(a, b)
        except Exception:
            return
        if h2s is None:
            return
        h1, h2 = va["h"], vb["h"]
        if t == "nozzle" and "V" in va:
            h1 += va["V"] ** 2 / 2        # a nozzle with an inlet velocity: η_N = V₂² / V₂s²
        num, den = (h2s - h1, h2 - h1) if compressor else (h1 - h2, h1 - h2s)
        if abs(den) > 1e-6 * max(abs(h1), 1.0):
            self.dev.setdefault(node_id, {})["eta"] = num / den

    def _outlet_from_efficiency(self, name, a, b, eta, compressor, t=None):
        va, vb = self._v(a), self._v(b)
        if t == "diffuser":
            # η_D sets the pressure recovered; h2 comes from the energy balance
            if all(k in va for k in ("h", "s", "V")) and "h" in vb and "V" in vb:
                try:
                    p2 = diffuser_outlet_P(va["h"], va["V"], va["s"], vb["h"], vb["V"], eta,
                                           self.props, self._fluid(a))
                except Exception as e:
                    raise SolveError(f"{name}: could not find the outlet pressure from the efficiency ({e}).")
                return self._set(b, "P", p2, name)
            return False
        # A nozzle's efficiency is on the inlet's stagnation enthalpy h1 + V1²/2
        # once V1 is known (the static formula only holds for V1 = 0).
        v1 = va.get("V") if t == "nozzle" else None
        if "h" in va and "s" in va and "h" in vb and "P" not in vb:
            # outlet energy known (a jet engine turbine driving its compressor),
            # pressure not: find the pressure whose isentropic outlet matches
            h1, h2 = va["h"], vb["h"]
            if v1 is not None:
                h1 += v1 ** 2 / 2
            h2s = h1 + eta * (h2 - h1) if compressor else h1 - (h1 - h2) / eta
            try:
                p2 = self.props("P", "H", h2s, "S", va["s"], self._fluid(a))
            except Exception as e:
                raise SolveError(f"{name}: could not find the outlet pressure from the efficiency ({e}).")
            return self._set(b, "P", p2, name)
        if "h" not in va or "s" not in va or "P" not in vb:
            return False
        try:
            h2s = self.props("H", "P", vb["P"], "S", va["s"], self._fluid(a))
        except Exception as e:
            raise SolveError(f"{name}: could not evaluate the isentropic outlet state ({e}).")
        h1 = va["h"]
        if v1 is not None:
            return self._set(b, "h", nozzle_outlet_h(h1, v1, h2s, eta), name)
        h2 = h1 + (h2s - h1) / eta if compressor else h1 - eta * (h1 - h2s)
        return self._set(b, "h", h2, name)

    def _back_pressure(self, node_id, node, name, a, b):
        """Converging nozzle with a back pressure Pb (Moran Sec. 9.13): the exit
        pressure is Pb, unless Pb is below the critical pressure p*, where the
        mass flux rho V along the expansion peaks. Then the nozzle chokes: the
        exit is sonic at p* and lowering Pb further changes nothing inside.
        p* is found from the fluid's own properties (0.528 p0 for air, k = 1.4),
        along the isentrope, or along the real path when an efficiency is given."""
        params = node.get("params") or {}
        pb = _number(params.get("Pb"), f"{name} back pressure")
        if pb is None:
            return False
        pb = to_si(self.system, "P", pb)
        if pb <= 0:
            raise SolveError(f"{name}: the back pressure must be positive.")
        if "Pstar" in self.dev.get(node_id, {}):
            return False
        va = self._v(a)
        if "h" not in va or "s" not in va or "P" not in va:
            return False
        asm = node.get("assumptions") or {}
        eta = 1.0 if asm.get("eta100") or asm.get("isentropic") else (_efficiency(params.get("eta"), name) or 1.0)
        fl, s1 = self._fluid(a), va["s"]

        def exit_state(V1):
            """(p*, P2, h2, rho2, p0) for an inlet velocity V1."""
            h0 = va["h"] + V1 ** 2 / 2

            def path(p):
                hs = self.props("H", "P", p, "S", s1, fl)
                return h0 - eta * (h0 - hs)

            def flux(p):
                h = path(p)
                if h >= h0:
                    return 0.0
                return self.props("D", "P", p, "H", h, fl) * math.sqrt(2 * (h0 - h))

            p0 = self.props("P", "H", h0, "S", s1, fl)
            lo, hi = 0.05 * p0, p0 * 0.999
            g = (math.sqrt(5) - 1) / 2
            x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
            f1, f2 = flux(x1), flux(x2)
            for _ in range(80):
                if f1 > f2:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - g * (hi - lo)
                    f1 = flux(x1)
                else:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + g * (hi - lo)
                    f2 = flux(x2)
            p_star = 0.5 * (lo + hi)
            p2 = max(pb, p_star)
            h2 = path(p2)
            return p_star, p2, h2, self.props("D", "P", p2, "H", h2, fl), p0, h0

        try:
            if "V" not in va:
                # Inlet velocity unknown (a nozzle at the end of a pipe, Dick
                # Ex. 4.8.2): with both areas typed, iterate V1 = m / (rho1 A1)
                # with m = rho2 A2 V2 from the exit the back pressure allows.
                A1 = _number(params.get("A1"), f"{name} A1")
                A2 = _number(params.get("A2"), f"{name} A2")
                if A1 is None or A2 is None or asm.get("negV1"):
                    return False
                A1, A2 = to_si(self.system, "A", A1), to_si(self.system, "A", A2)
                rho1 = self.props("D", "P", va["P"], "H", va["h"], fl)
                V1 = 0.0
                for _ in range(200):
                    _, _, h2, rho2, _, h0 = exit_state(V1)
                    V1_new = rho2 * A2 * math.sqrt(max(2 * (h0 - h2), 0.0)) / (rho1 * A1)
                    if abs(V1_new - V1) < 1e-9 * max(V1_new, 1.0):
                        V1 = V1_new
                        break
                    V1 = 0.5 * (V1 + V1_new)
                self._set(a, "V", V1, name)
            p_star, p2, _, _, p0, _ = exit_state(va["V"])
        except SolveError:
            raise
        except Exception as e:
            raise SolveError(f"{name}: could not find the critical (choking) pressure ({e}).")
        if pb > p0:
            raise SolveError(f"{name}: the back pressure is above the inlet stagnation pressure, so nothing flows out.")
        choked = pb < p_star
        self.dev.setdefault(node_id, {}).update(Pstar=p_star, choked=choked)
        return self._set(b, "P", p2, f"{name} (back pressure)")

    def _flow_areas(self, node, name, a, b):
        """ṁ = A V / v at each port whose flow area (A1, A2) is typed: solves
        the velocity or the mass flow there."""
        params = node.get("params") or {}
        changed = False
        for n, key in ((a, "A1"), (b, "A2")):
            A = _number(params.get(key), f"{name} {key}")
            if A is None:
                continue
            if A <= 0:
                raise SolveError(f"{name}: {key} must be positive.")
            v = self._v(n)
            if "P" not in v or "h" not in v:
                continue
            spec_vol = 1.0 / self.props("D", "P", v["P"], "H", v["h"], self._fluid(n))
            try:
                got = continuity(v.get("m"), v.get("V"), spec_vol, to_si(self.system, "A", A))
            except ValueError as e:
                raise SolveError(f"{name}: at S{n}, {e}. Check {key}, ṁ and V.")
            if got:
                changed |= self._set(n, "m", got[0], name)
                changed |= self._set(n, "V", got[1], name)
        return changed

    def _kinetic(self, node_id, node, name, a, b):
        """Nozzle / diffuser energy balance: h1 + V1²/2 + g z1 = h2 + V2²/2 + g z2.
        Velocities are state values (the V column); older diagrams gave them
        as device inputs V1 and V2, which still work."""
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        changed = False
        for key, state, negligible in (("V1", a, "negV1"), ("V2", b, "negV2")):
            if asm.get(negligible):
                changed |= self._set(state, "V", 0.0, f"{name}: {key} ≈ 0")
            else:
                v = _number(params.get(key), f"{name} {key}")
                if v is not None:
                    changed |= self._set(state, "V", to_si(self.system, "V", v), name)
        if not asm.get("adiabatic"):
            return changed

        va, vb = self._v(a), self._v(b)
        h1, h2, V1, V2 = va.get("h"), vb.get("h"), va.get("V"), vb.get("V")
        pe = G * (va.get("z", 0.0) - vb.get("z", 0.0))     # g (z1 - z2)
        missing = [k for k, val in (("h1", h1), ("h2", h2), ("V1", V1), ("V2", V2)) if val is None]

        if not missing:
            if not _close(h1 + V1 ** 2 / 2 + pe, h2 + V2 ** 2 / 2):
                raise SolveError(f"{name}: the energy balance h₁ + V₁²/2 = h₂ + V₂²/2 does not close.")
        elif len(missing) == 1:
            k = missing[0]
            if k == "h2":
                changed |= self._set(b, "h", h1 + (V1 ** 2 - V2 ** 2) / 2 + pe, name)
            elif k == "h1":
                changed |= self._set(a, "h", h2 + (V2 ** 2 - V1 ** 2) / 2 - pe, name)
            else:
                v_known, sign = (V1, 1) if k == "V2" else (V2, -1)
                ke = v_known ** 2 + sign * 2 * (h1 - h2 + pe)
                if ke < -1e-6:
                    raise SolveError(
                        f"{name}: the energy balance would need a negative kinetic energy for {k}. "
                        f"Check which way the fluid flows and the pressures at each end."
                    )
                changed |= self._set(a if k == "V1" else b, "V", math.sqrt(max(ke, 0.0)), name)
        return changed

    def _air_states(self, node_id, name, params):
        """Inlet and outlet air of a cooling tower from the device inputs, via
        CoolProp's humid-air functions. Per kg of dry air: h (J), w (kg water),
        v (m^3). None until all four air inputs are given."""
        if node_id in self.air:
            return self.air[node_id]
        p_txt = params.get("Patm")
        p = 101325.0 if _number(p_txt, f"{name} P") is None else to_si(self.system, "P", _number(p_txt, f"{name} P"))
        vals = {}
        for key, label in (("Tin", "air inlet T"), ("RHin", "air inlet RH"),
                           ("Tout", "air outlet T"), ("RHout", "air outlet RH")):
            vals[key] = _number(params.get(key), f"{name} {label}")
            if vals[key] is None:
                return None
        states = []
        for t_key, rh_key, where in (("Tin", "RHin", "inlet"), ("Tout", "RHout", "outlet")):
            rh = vals[rh_key] / 100.0
            if not 0 <= rh <= 1:
                raise SolveError(f"{name}: air {where} relative humidity {vals[rh_key]:g}% must be between 0 and 100%.")
            t = to_si(self.system, "T", vals[t_key])
            try:
                states.append({
                    "h": self.ha_props("H", "T", t, "P", p, "R", rh),
                    "w": self.ha_props("W", "T", t, "P", p, "R", rh),
                    "v": self.ha_props("Vda", "T", t, "P", p, "R", rh),
                    "Twb": self.ha_props("B", "T", t, "P", p, "R", rh),
                })
            except Exception as e:
                raise SolveError(f"{name}: could not evaluate the {where} air ({e}).")
        if states[1]["w"] < states[0]["w"]:
            raise SolveError(f"{name}: the outlet air holds less moisture than the inlet air, so no water evaporates.")
        self.air[node_id] = states
        return states

    def _cooling_tower(self, node_id, node, name, ports):
        """Wet cooling tower. Water enters at `in` and leaves cooler at `out`;
        air passes through (its states are device inputs, not diagram states)
        and carries off the evaporated water.

        Energy, per the textbook balance (makeup, if added, enters at the
        outlet water temperature, so it cancels):
            m_air (h_a2 - h_a1 - (w2 - w1) h4) = m3 (h3 - h4)
        Mass: m4 = m3 - m_air (w2 - w1), or m4 = m3 with makeup added."""
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        a, b = ports["in"], ports["out"]
        va, vb = self._v(a), self._v(b)
        dev = self.ct.setdefault(node_id, {})
        changed = False

        if asm.get("isobaric"):
            changed |= self._propagate(a, b, "P", name)
        if asm.get("makeup"):
            changed |= self._propagate(a, b, "m", name)

        if "ma" not in dev:
            ma = _number(params.get("mAir"), f"{name} dry air flow")
            if ma is not None:
                dev["ma"] = to_si(self.system, "m", ma)
                changed = True

        # Air ports piped to humid-air states take the place of the air inputs.
        ai, ao = ports.get("airIn"), ports.get("airOut")
        unused = set(node.get("unused") or [])
        linked = ai is not None and ao is not None and not ({"airIn", "airOut"} & unused)
        if linked:
            air = self._linked_air(node_id, name, ai, ao)
            changed |= self._propagate(ai, ao, "m", name)        # the dry air passes through
            if "ma" not in dev and self._v(ai).get("m") is not None:
                dev["ma"] = self._v(ai)["m"]
                changed = True
        else:
            air = self._air_states(node_id, name, params)
        if air is None:
            return changed
        (h1, w1), (h2, w2) = (air[0]["h"], air[0]["w"]), (air[1]["h"], air[1]["w"])
        dw = w2 - w1

        ma, m3, h3, h4 = dev.get("ma"), va.get("m"), va.get("h"), vb.get("h")
        # Makeup water at its own temperature (Moran Ex. 12.15): the evaporated
        # water is replaced by a stream at h5, so m_air (h_a2 - h_a1 - dw h5)
        # = m3 (h3 - h4). Blank: it enters at the outlet water temperature.
        t_mk = _number(params.get("Tmakeup"), f"{name} makeup water T") if asm.get("makeup") else None
        h_mk = None
        if t_mk is not None:
            p_mk = vb.get("P") or va.get("P") or 101325.0
            try:
                h_mk = self.props("H", "P", p_mk, "T", to_si(self.system, "T", t_mk), self._fluid(b))
            except Exception as e:
                raise SolveError(f"{name}: could not evaluate the makeup water ({e}).")
            dev["hMakeup"] = h_mk

        def hw():
            """enthalpy of the water that replaces the evaporated water"""
            return h_mk if h_mk is not None else h4

        missing = [k for k, v in (("ma", ma), ("m3", m3), ("h3", h3), ("h4", h4)) if v is None]
        if not missing:
            if not _close(ma * (h2 - h1 - dw * hw()), m3 * (h3 - h4)):
                raise SolveError(f"{name}: the energy balance does not close.")
        elif len(missing) == 1:
            k = missing[0]
            if k == "ma":
                denom = h2 - h1 - dw * hw()
                if abs(denom) < 1e-9:
                    raise SolveError(f"{name}: the air leaves with no more energy than it entered, so it cannot cool the water.")
                dev["ma"] = m3 * (h3 - h4) / denom
                changed = True
            elif k == "m3":
                if _close(h3, h4):
                    raise SolveError(f"{name}: water inlet and outlet are at the same state.")
                changed |= self._set(a, "m", ma * (h2 - h1 - dw * hw()) / (h3 - h4), name)
            elif k == "h3":
                if m3:
                    changed |= self._set(a, "h", h4 + ma * (h2 - h1 - dw * hw()) / m3, name)
            elif h_mk is not None:
                if m3:
                    changed |= self._set(b, "h", h3 - ma * (h2 - h1 - dw * h_mk) / m3, name)
            else:
                denom = m3 - ma * dw
                if abs(denom) > 1e-12:
                    changed |= self._set(b, "h", (m3 * h3 - ma * (h2 - h1)) / denom, name)

        ma = dev.get("ma")
        if ma is not None:
            if ma < 0:
                raise SolveError(f"{name}: the dry air flow came out negative ({ma:.4g} kg/s). Check that the water is cooled and the air is heated.")
            if not asm.get("makeup"):
                changed |= self._linear([(a, 1), (b, -1)], "m", name, "water mass", -ma * dw)
            h_rep = h_mk if h_mk is not None else (vb.get("h") if vb.get("h") is not None else 0.0)
            dev.update(mMakeup=ma * dw, Vair=ma * air[0]["v"], Q=ma * (h2 - h1 - dw * h_rep))
            if linked:
                changed |= self._set(ai, "m", ma, name)
        dev["Twb"] = air[0]["Twb"]
        return changed

    def _linked_air(self, node_id, name, ai, ao):
        """The cooling tower's air from the humid-air states on its air ports,
        in the shape _air_states gives, or None until both are known."""
        if not (psychro.is_humid(self, ai) and psychro.is_humid(self, ao)):
            raise SolveError(f"{name}: the air ports carry air, so set S{ai} and S{ao} to Humid air.")
        vi, vo = self._v(ai), self._v(ao)
        if not all(k in v for v in (vi, vo) for k in ("h", "w", "T", "P")):
            return None
        if vo["w"] < vi["w"]:
            raise SolveError(f"{name}: the outlet air holds less moisture than the inlet air, so no water evaporates.")
        ha = self.ha_props
        try:
            extra = {"v": ha("Vda", "T", vi["T"], "P", vi["P"], "W", vi["w"]),
                     "Twb": ha("B", "T", vi["T"], "P", vi["P"], "W", vi["w"])}
        except Exception as e:
            raise SolveError(f"{name}: could not evaluate the inlet air ({e}).")
        return [{"h": vi["h"], "w": vi["w"], **extra}, {"h": vo["h"], "w": vo["w"], "v": None, "Twb": None}]

    def _atmosphere(self, node, name, ports):
        """Intake or exhaust of an open cycle: the state there is at the ambient
        pressure (and, for an intake, the ambient temperature) when given."""
        params = node.get("params") or {}
        s = ports.get("out", ports.get("in"))
        changed = False
        keys = (("P0", "P"), ("T0", "T")) if "out" in ports else (("P0", "P"),)
        for param, key in keys:
            x = _number(params.get(param), f"{name} ambient {key}")
            if x is not None:
                changed |= self._set(s, key, to_si(self.system, key, x), name)
        return changed

    def _mixer(self, node, name, ports):
        """Any number of inlets (in1, in2, ...) mixing into one outlet. Mass
        and energy balances; heat Q̇ may be added unless Q̇ = 0 is checked."""
        ins, c = _numbered(ports, "in"), ports["out"]
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}

        changed = False
        if asm.get("isobaric"):
            for s in ins:
                changed |= self._propagate(s, c, "P", name)

        signed = [(s, 1) for s in ins] + [(c, -1)]
        q = None if asm.get("adiabatic") else _number(params.get("Qdot"), f"{name} Q̇")
        if asm.get("adiabatic") or q is not None:
            q_w = 0.0 if q is None else q * power_factor(self.system)
            changed |= self._cv(name, signed, [ins + [c]], q_w)
        else:
            changed |= self._linear(signed, "m", name, "mass")
        if psychro.is_humid(self, c):
            # humid air: ṁ is dry air and h is per kg of it; the water balances too
            changed |= psychro.mix_moisture(self, name, ins, c)
        return changed

    def _splitter(self, node, name, ports):
        """One inlet, any number of outlets (out1, out2, ...). Equal pressure on
        every port; with "same state" the outlets also share T, h, s and x."""
        a = ports["in"]
        outs = _numbered(ports, "out")
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}

        changed = False
        for o in outs:
            if asm.get("isobaric"):
                changed |= self._propagate(a, o, "P", name)
            if asm.get("sameState"):
                for key in ("T", "h", "s", "x") + (HUMID_FIELDS if psychro.is_humid(self, a) else ()):
                    changed |= self._propagate(a, o, key, name)

        changed |= self._linear([(a, 1)] + [(o, -1) for o in outs], "m", name, "mass")

        for i, o in enumerate(outs, 1):
            y = _number(params.get(f"y{i}"), f"{name} fraction to outlet {i}")
            if y is not None:
                if not 0 <= y <= 1:
                    raise SolveError(f"{name}: fraction to outlet {i} = {y:g} must be between 0 and 1.")
                changed |= self._linear([(o, 1), (a, -y)], "m", name, "split")
        return changed

    def _flash(self, name, ports):
        """Separator at one pressure: saturated vapor out one side, saturated liquid out the other."""
        a, v, l = ports["in"], ports["vapor"], ports["liquid"]

        changed = self._propagate(a, v, "P", name)
        changed |= self._propagate(a, l, "P", name)
        changed |= self._set(v, "x", 1.0, name)
        changed |= self._set(l, "x", 0.0, name)
        changed |= self._linear([(a, 1), (v, -1), (l, -1)], "m", name, "mass")

        x = self._v(a).get("x")
        if x is not None:
            changed |= self._linear([(v, 1), (a, -x)], "m", name, "vapor fraction")
            changed |= self._linear([(l, 1), (a, -(1 - x))], "m", name, "liquid fraction")
        return changed

    def _hx_side(self, name, ports, prefix):
        return _numbered(ports, f"{prefix}_in"), _numbered(ports, f"{prefix}_out")

    def _hx(self, node, name, ports):
        """Modular heat exchanger. One stream (a_*) covers boilers, condensers,
        heaters and open feedwater heaters; two streams (a_* and b_*) that
        never mix cover closed feedwater heaters and two-stream exchangers.

        Within a stream the inlets mix and every outlet leaves at the same
        state. One stream: q is heat added from outside (Q̇ = 0 if adiabatic).
        Two streams: Q̇ is heat passed from stream A to stream B."""
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        sides = {p: self._hx_side(name, ports, p) for p in ("a", "b")}
        two = bool(sides["b"][0] or sides["b"][1])

        changed = False
        for p, (ins, outs) in sides.items():
            if not ins and not outs:
                continue
            for o in outs[1:]:
                for key in INTENSIVE:
                    changed |= self._propagate(outs[0], o, key, name)
            if asm.get(f"{p}Isobaric"):
                every = ins + outs
                for s in every[1:]:
                    changed |= self._propagate(every[0], s, "P", name)
            for o in outs:
                if asm.get(f"{p}SatLiquidOut"):
                    changed |= self._set(o, "x", 0.0, name)
                if asm.get(f"{p}SatVaporOut"):
                    changed |= self._set(o, "x", 1.0, name)
            if ins and outs:
                changed |= self._linear([(s, 1) for s in ins] + [(s, -1) for s in outs], "m", name, "mass")

        def signed(p):
            ins, outs = sides[p]
            return [(s, 1) for s in ins] + [(s, -1) for s in outs]

        q = _number(params.get("Qdot"), f"{name} Q̇")
        q_w = None if q is None else q * power_factor(self.system)

        if not two:
            ins, outs = sides["a"]
            if not ins or not outs:
                raise SolveError(f"{name}: needs at least one inlet and one outlet.")
            # heat per kg, q: the first law without a mass flow, h₂ = h₁ + q
            q_kg = _number(params.get("q"), f"{name} q")
            if asm.get("adiabatic"):
                for given, text in ((q_w, f"Q̇ = {q:g}" if q is not None else ""),
                                    (q_kg, f"q = {params.get('q')}")):
                    if given is not None and not _close(given, 0.0):
                        raise SolveError(f"{name}: 'Q̇ = 0' conflicts with {text}. Uncheck one of them.")
                changed |= self._cv(name, signed("a"), [ins + outs], 0.0)
                return changed
            if q_kg is not None:
                if len(ins) != 1:
                    raise SolveError(f"{name}: heat per kg (q) needs a single inlet; give Q̇ instead.")
                q_si = to_si(self.system, "h", q_kg)
                x_in = self._extra(ins[0])
                for o in outs:
                    x_out = self._extra(o)
                    if x_in is not None and x_out is not None:
                        # h_out + ke/pe_out = h_in + ke/pe_in + q
                        changed |= self._linear([(ins[0], -1), (o, 1)], "h", name, "energy", x_out - x_in - q_si)
                if q_w is not None:
                    if _close(q_si, 0.0):
                        raise SolveError(f"{name}: q = 0 but Q̇ = {q:g}. Give one of them.")
                    changed |= self._set(ins[0], "m", q_w / q_si, f"{name}: ṁ = Q̇ / q")
            if q_w is not None:
                changed |= self._cv(name, signed("a"), [ins + outs], q_w)
            return changed

        for p in ("a", "b"):
            if not sides[p][0] or not sides[p][1]:
                raise SolveError(f"{name}: stream {p.upper()} needs at least one inlet and one outlet.")
        (a_in, a_out), (b_in, b_out) = sides["a"], sides["b"]

        # Which stream is hot follows from the inlet temperatures, so either
        # side can be the hot one. Until both are known, the rules that need it wait.
        hot = self._hot_side(a_in, b_in)
        if asm.get("idealOutlet") and hot is not None:
            # the cold stream leaves at the hot stream's outlet temperature
            h_out, c_out = (a_out, b_out) if hot == "a" else (b_out, a_out)
            changed |= self._propagate(h_out[0], c_out[0], "T", name)
        if asm.get("noLoss"):
            changed |= self._cv(name, signed("a") + signed("b"), [a_in + a_out, b_in + b_out])

        # Q̇ is the heat passed from the hot stream to the cold one. Older
        # diagrams gave it signed, from A to B; the direction now comes from
        # the temperatures, so only its size counts.
        q_w = None if q_w is None else abs(q_w)
        eps = self._fraction(params.get("eps"), f"{name} effectiveness")
        if q_w is None and eps is not None and hot is not None:
            if len(a_in) > 1 or len(a_out) > 1 or len(b_in) > 1 or len(b_out) > 1:
                raise SolveError(f"{name}: effectiveness needs one inlet and one outlet per stream.")
            if hot == "a":
                qmax = self._max_heat(name, a_in[0], a_out[0], b_in[0], b_out[0])
            else:
                qmax = self._max_heat(name, b_in[0], b_out[0], a_in[0], a_out[0])
            q_w = None if qmax is None else eps * qmax

        ua = _number(params.get("UA"), f"{name} UA")
        if q_w is None and ua is not None and hot is not None:
            if ua <= 0:
                raise SolveError(f"{name}: UA must be positive.")
            if len(a_in) > 1 or len(a_out) > 1 or len(b_in) > 1 or len(b_out) > 1:
                raise SolveError(f"{name}: UA needs one inlet and one outlet per stream.")
            ua_si = ua * power_factor(self.system) / to_si(self.system, "dT", 1.0)
            layout = (node.get("config") or {}).get("layout", "counter")
            if hot == "a":
                q_w = self._ua_heat(name, ua_si, layout, a_in[0], a_out[0], b_in[0], b_out[0])
            else:
                q_w = self._ua_heat(name, ua_si, layout, b_in[0], b_out[0], a_in[0], a_out[0])

        if q_w is not None and hot is not None:
            into_b = q_w if hot == "a" else -q_w
            changed |= self._cv(name, signed("a"), [a_in + a_out], -into_b)
            changed |= self._cv(name, signed("b"), [b_in + b_out], into_b)
        return changed

    def _hot_side(self, a_in, b_in):
        """"a" or "b", whichever stream comes in hotter, or None until both
        inlet temperatures are known (the first inlet of each stream counts)."""
        ta, tb = self._v(a_in[0]).get("T"), self._v(b_in[0]).get("T")
        if ta is None or tb is None:
            return None
        return "a" if ta >= tb else "b"

    def _fraction(self, value, label):
        x = _number(value, label)
        if x is None:
            return None
        if not 0 < x <= 1:
            raise SolveError(f"{label} = {x:g} must be between 0 and 1.")
        return x

    def _ua_heat(self, name, ua, layout, hi, ho, ci, co):
        """Rating a heat exchanger of known UA (Penoncello Sec. 5.6.4): the heat
        Q̇ for which Q̇ = UA · LMTD, with each outlet temperature found from
        its stream's energy balance and real properties (so no average cp is
        needed). Counter or parallel flow. None until both inlets, both outlet
        pressures and both flows are known."""
        qmax = self._max_heat(name, hi, ho, ci, co)
        if qmax is None or qmax <= 0:
            return None
        vh, vc = self._v(hi), self._v(ci)
        p_ho, p_co = self._v(ho)["P"], self._v(co)["P"]
        fh, fc = self._fluid(hi), self._fluid(ci)

        def residual(q):
            try:
                t_ho = self.props("T", "P", p_ho, "H", vh["h"] - q / vh["m"], fh)
                t_co = self.props("T", "P", p_co, "H", vc["h"] + q / vc["m"], fc)
            except Exception as e:
                raise SolveError(f"{name}: could not evaluate the outlet states for UA ({e}).")
            if layout == "parallel":
                d1, d2 = vh["T"] - vc["T"], t_ho - t_co
            else:
                d1, d2 = vh["T"] - t_co, t_ho - vc["T"]
            if d1 <= 0 or d2 <= 0:
                return -q                    # past what this layout can transfer
            lmtd = d1 if abs(d1 - d2) < 1e-9 * max(d1, 1.0) else (d1 - d2) / math.log(d1 / d2)
            return ua * lmtd - q

        lo, hi_q = 0.0, qmax * (1 - 1e-9)
        for _ in range(100):
            mid = 0.5 * (lo + hi_q)
            if residual(mid) > 0:
                lo = mid
            else:
                hi_q = mid
        return 0.5 * (lo + hi_q)

    def _max_heat(self, name, hi, ho, ci, co):
        """Qmax: the most heat the streams could exchange (W). Needs both inlets
        fully known plus both outlet pressures."""
        vh, vc = self._v(hi), self._v(ci)
        p_ho, p_co = self._v(ho).get("P"), self._v(co).get("P")
        mh, mc = vh.get("m"), vc.get("m")
        needed = [vh.get("h"), vh.get("T"), vc.get("h"), vc.get("T"), p_ho, p_co, mh, mc]
        if any(x is None for x in needed):
            return None
        try:
            h_hot_min = self.props("H", "P", p_ho, "T", vc["T"], self._fluid(hi))
            h_cold_max = self.props("H", "P", p_co, "T", vh["T"], self._fluid(ci))
        except Exception as e:
            raise SolveError(f"{name}: could not evaluate the maximum possible heat transfer ({e}).")
        return min(mh * (vh["h"] - h_hot_min), mc * (h_cold_max - vc["h"]))

    # ---------- run + report ----------
    def run(self):
        errors = []
        try:
            for _ in range(MAX_PASSES):
                changed = self.comb.spread()
                for n in list(self.values):
                    changed |= self._resolve_state(n)
                for node_id, node in self.nodes.items():
                    changed |= self._apply_device(node_id, node)
                if not changed:
                    break
        except SolveError as e:
            self.failure = e
            errors.append(str(e))
        if not errors:
            errors += self._check_hx_temperatures()
        return errors

    def unknown_flows(self):
        """States on a device port whose mass flow is still unknown."""
        out = []
        for node in self.nodes.values():
            for n in _ports(node).values():
                if "m" not in self._v(n) and n not in out:
                    out.append(n)
        return out

    def _check_hx_temperatures(self):
        """Second-law checks for two-stream heat exchangers once the solve is
        done: heat only flows from hot to cold, and in parallel flow the hot
        stream can't leave colder than the cold one."""
        problems = []
        for node in self.nodes.values():
            if node.get("type") != "hx":
                continue
            ports = _ports(node)
            sides = {p: self._hx_side(None, ports, p) for p in ("a", "b")}
            if not (sides["b"][0] and sides["b"][1]):
                continue
            temps = [self._v(s[0]).get("T") for s in (sides["a"][0], sides["a"][1], sides["b"][0], sides["b"][1])]
            if any(t is None for t in temps):
                continue
            ai, ao, bi, bo = temps
            hi, ho, ci, co = (ai, ao, bi, bo) if ai >= bi else (bi, bo, ai, ao)
            name = node.get("name") or "Heat Exchanger"
            tol = 1e-3
            fmt = lambda t: self._fmt("T", t)  # noqa: E731
            if ho < ci - tol or co > hi + tol:
                problems.append(
                    f"{name}: the temperatures cross. The hot stream would leave at {fmt(ho)} and the cold "
                    f"stream at {fmt(co)}, but heat can only flow from hot ({fmt(hi)} in) to cold ({fmt(ci)} in)."
                )
            elif (node.get("config") or {}).get("layout") == "parallel" and ho < co - tol:
                problems.append(
                    f"{name}: in parallel flow the hot stream can't leave colder than the cold stream "
                    f"({fmt(ho)} vs {fmt(co)}). Switch to counter flow or change the outlet temperatures."
                )
        return problems

    def report_states(self):
        out = {}
        for n in sorted(self.values):
            v = self.values[n]
            out[str(n)] = {
                f: (_round(from_si(self.system, f, v[f])) if f in v else "") for f in FIELDS + MOTION + HUMID_FIELDS
            }
        return out

    def report_devices(self):
        out = {}
        pf = power_factor(self.system)
        v = self.values

        def flows(states):
            """[(m, e)] for the states, e = h + V²/2 + g z, or None if any is unknown."""
            out_ = []
            for s in states:
                m, e = v.get(s, {}).get("m"), self._e(s)
                if m is None or e is None:
                    return None
                out_.append((m, e))
            return out_

        for node_id, node in self.nodes.items():
            t = node.get("type")
            ports = _ports(node)
            r = {}

            if t in ("compressor", "pump"):
                e1, e2 = self._e(ports["in"]), self._e(ports["out"])
                m = v.get(ports["in"], {}).get("m")
                if e1 is not None and e2 is not None:
                    w = e2 - e1                                 # work in
                    r["w"] = _round(from_si(self.system, "h", w))
                    if m is not None:
                        r["Wdot"] = _round(w * m / pf)
            elif t == "turbine":
                a = v.get(ports["in"], {})
                outlets = [ports[k] for k in ports if k == "out" or k.startswith("ext")]
                hs = [self._e(o) for o in outlets]
                fl = flows([ports["in"]] + outlets)
                if fl:
                    (m_in, h_in), rest = fl[0], fl[1:]
                    total = m_in * h_in - sum(m * h for m, h in rest)
                    tp = node.get("params") or {}
                    w_given = None if (node.get("assumptions") or {}).get("adiabatic") \
                        else _number(tp.get("Wdot"), "Ẇ")
                    if w_given is not None:
                        # not adiabatic: Ẇ as typed, the rest of the energy drop is heat
                        r["Wdot"] = _round(w_given)
                        r["Qdot"] = _round(w_given - total / pf)       # heat added (negative = lost)
                        total = w_given * pf
                    else:
                        r["Wdot"] = _round(total / pf)
                    if m_in:
                        r["w"] = _round(from_si(self.system, "h", total / m_in))
                elif self._e(ports["in"]) is not None and all(h is not None for h in hs) and all(_close(h, hs[0]) for h in hs):
                    r["w"] = _round(from_si(self.system, "h", self._e(ports["in"]) - hs[0]))   # per kg in
            elif is_combustor(node):
                r.update(self.comb.report(node_id))
            elif t == "ac":
                r.update(psychro.ac_report(self, node, ports))
            elif t == "hx":
                two = any(k.startswith("b_") for k in ports)
                side = "b" if two else "a"
                ins = _numbered(ports, f"{side}_in")
                outs = _numbered(ports, f"{side}_out")
                fi, fo = flows(ins), flows(outs)
                if fi and fo:
                    q = sum(m * h for m, h in fo) - sum(m * h for m, h in fi)   # heat into this stream
                    if two:
                        # heat passed from the hot stream to the cold one, and which is hot
                        r["Qdot"] = _round(abs(q) / pf)
                        r["hot"] = "A" if q >= 0 else "B"
                    else:
                        r["Qdot"] = _round(q / pf)
                elif not two and len(ins) == 1:
                    e_in, e_outs = self._e(ins[0]), [self._e(o) for o in outs]
                    if e_in is not None and e_outs and all(e is not None for e in e_outs):
                        r["q"] = _round(from_si(self.system, "h", e_outs[0] - e_in))   # per kg, no flows known

            elif t == "coolingTower":
                ct = self.ct.get(node_id, {})
                va, vb = v.get(ports["in"], {}), v.get(ports["out"], {})
                for key, unit in (("ma", "m"), ("mMakeup", "m"), ("Vair", "vol")):
                    if ct.get(key) is not None:
                        r[key] = _round(from_si(self.system, unit, ct[key]))
                if ct.get("Q") is not None and vb.get("h") is not None:
                    r["Qdot"] = _round(ct["Q"] / pf)
                if "T" in va and "T" in vb:
                    r["range"] = _round(from_si(self.system, "dT", va["T"] - vb["T"]))
                if "T" in vb and ct.get("Twb") is not None:
                    r["approach"] = _round(from_si(self.system, "dT", vb["T"] - ct["Twb"]))

            if t in ("nozzle", "diffuser"):
                for key, port in (("V1", "in"), ("V2", "out")):
                    vel = v.get(ports[port], {}).get("V")
                    if vel is not None:
                        r[key] = _round(from_si(self.system, "V", vel))
                for key, port in (("A1", "in"), ("A2", "out")):
                    s = v.get(ports[port], {})
                    if s.get("V") and s.get("m") is not None and "P" in s and "h" in s:
                        try:
                            area = s["m"] / self.props("D", "P", s["P"], "H", s["h"], self._fluid(ports[port])) / s["V"]
                        except Exception:
                            continue
                        r[key] = _round(from_si(self.system, "A", area))
                nd = self.dev.get(node_id, {})
                if t == "nozzle" and "Pstar" in nd:
                    r["Pstar"] = _round(from_si(self.system, "P", nd["Pstar"]))
                    r["chokedText"] = "choked (sonic exit)" if nd["choked"] else "not choked"
                s2 = v.get(ports["out"], {})
                if t == "nozzle" and s2.get("V") is not None and "P" in s2 and "h" in s2:
                    try:
                        c = self.props("A", "P", s2["P"], "H", s2["h"], self._fluid(ports["out"]))
                        r["Mach2"] = _round(s2["V"] / c)
                    except Exception:
                        pass
            if t == "mixer":
                fi, fo = flows(_numbered(ports, "in")), flows([ports["out"]])
                if fi and fo:
                    r["Qdot"] = _round((fo[0][0] * fo[0][1] - sum(m * h for m, h in fi)) / pf)   # heat added
                m_out = v.get(ports["out"], {}).get("m")
                for i, s_in in enumerate(_numbered(ports, "in"), 1):
                    m_i = v.get(s_in, {}).get("m")
                    if m_i is not None and m_out:
                        r[f"y{i}"] = _round(m_i / m_out)
            eta = self.dev.get(node_id, {}).get("eta")
            if eta is not None:
                r["eta"] = _round(eta)

            if r:
                out[node_id] = r
        return out

    def report_messages(self):
        """Combustors still missing their fuel input. States that aren't fixed
        yet are reported by hints.diagnose, with what would close them."""
        msgs = []
        for node_id, node in self.comb.combustors.items():
            if self.comb.res.get(node_id, {}).get("f") is None:
                msgs.insert(0, f"{node.get('name') or 'Combustor'}: give the air-fuel ratio, the fuel flow, "
                               "or the exit temperature (T on the outlet state).")
        return msgs


WORK_OUT = ("turbine",)
WORK_IN = ("compressor", "pump")


def _net_power(solver):
    """Net power (W) of every turbine, compressor and pump, or None while any
    of them is missing a flow or an enthalpy."""
    total, found = 0.0, False
    for node in solver.nodes.values():
        t = node.get("type")
        if t not in WORK_OUT + WORK_IN:
            continue
        ports = _ports(node)
        ins = [ports["in"]]
        outs = [ports[k] for k in ports if k == "out" or k.startswith("ext")]
        ms = [solver.values.get(s, {}).get("m") for s in ins + outs]
        es = [solver._e(s) for s in ins + outs]
        if any(x is None for x in ms + es):
            return None
        # energy in minus energy out: work produced (negative for a pump or compressor)
        total += ms[0] * es[0] - sum(m * e for m, e in zip(ms[1:], es[1:]))
        found = True
    return total if found else None


def _seed_state(payload):
    """Where to put a trial mass flow: the inlet of the first turbine, else of
    the first compressor or pump."""
    nodes = (payload.get("nodes") or {}).values()
    for types in (WORK_OUT, WORK_IN):
        for node in nodes:
            if node.get("type") in types:
                return str(_ports(node)["in"])
    return None


def _english_humid_reference(ha):
    """Humid-air enthalpy on the English-unit reference. CoolProp (like SI
    charts and tables) puts h = 0 for dry air at 0 °C; English charts, the
    textbooks' English tables and EES in English units put it at 0 °F, so
    their h reads about 7.7 Btu/lbm higher. Only dry air's zero moves (water's
    stays at the triple point either way), and it moves by the same amount
    per kg of dry air everywhere, so every balance is unchanged; the h a user
    reads or types just matches their chart."""
    off = ha("H", "T", 273.15, "P", 101325.0, "W", 0.0) - ha("H", "T", 255.3722222, "P", 101325.0, "W", 0.0)

    def wrapped(out, *args):
        args = list(args)
        for i in (0, 2, 4):
            if args[i] in ("H", "Hda"):
                args[i + 1] = args[i + 1] - off
        v = ha(out, *args)
        return v + off if out in ("H", "Hda") else v
    return wrapped


def _determined(payload):
    """True when the equation count says the diagram has enough inputs (so a
    flow left unknown is a loop to iterate, not a missing input)."""
    try:
        from dof import count
        c = count(payload)
    except Exception:
        return False
    if not c.get("supported"):
        return True          # the count can't tell (cooling towers, humid air): try
    return c.get("stateMissing", 1) <= 0 and not c.get("flowFree")


def _tear_flows(payload, props, ha_props, first):
    """A loop the pass-by-pass solve can't untie: every flow in it depends on
    another flow in it (a cooling-water loop whose makeup is mixed back in,
    say). Tear the loop at one unknown flow: guess it, solve, and read how far
    the balance that closes the loop is off (the Mismatch residual); a secant
    on that residual finds the flow that closes it. Returns (solver, state)
    or None when no single torn flow solves the diagram."""
    known = [abs(v["m"]) for v in first.values.values() if v.get("m")]
    if not known:
        return None
    m_ref = max(known)
    system = first.system
    budget = [120]               # solves in all, so a diagram that is simply missing a flow stays quick
    for n in first.unknown_flows()[:12]:
        cache = {}

        def attempt(m_si):
            if m_si in cache:
                return cache[m_si]
            if budget[0] <= 0:
                return None, None
            budget[0] -= 1
            s = _Solver(_with_flow(payload, str(n), from_si(system, "m", m_si)), props, ha_props)
            errs = s.run()
            if not errs and not s.unknown_flows():
                res = 0.0
            elif errs and isinstance(getattr(s, "failure", None), Mismatch):
                res = s.failure.residual
            else:
                res = None
            cache[m_si] = (s, res)
            return s, res

        pts = []
        # 1.10.12: a re-solve of a nearly identical diagram (the optimizer, the
        # sweep) passes the flow that closed the loop last time as `tearHint`;
        # a tight bracket around it saves most of the wide scan below.
        hint = payload.get("tearHint") or {}
        guesses = (1.0, 3.0, 10.0, 30.0, 100.0, 0.3, 0.1, 0.01)
        if str(hint.get("state")) == str(n) and _number(hint.get("m"), "tear hint"):
            m_hint = to_si(system, "m", float(hint["m"]))
            tight = [m_hint * k for k in (1.0, 1.002, 0.998, 1.02, 0.98)]
            seeds = tight + [m_ref * k for k in guesses]
        else:
            seeds = [m_ref * k for k in guesses]
        for x in seeds:
            s, r = attempt(x)
            if r == 0.0:
                return s, n
            if r is not None:
                pts.append((x, r))
                srt = sorted(pts)
                if any(a[1] * b[1] < 0 for a, b in zip(srt, srt[1:])):
                    break       # bracketed: no need to try the rest
        pts.sort()
        brackets = [(a, b) for a, b in zip(pts, pts[1:]) if a[1] * b[1] < 0]
        if not brackets:
            continue
        (x0, r0), (x1, r1) = min(brackets, key=lambda ab: ab[1][0] - ab[0][0])   # the narrowest
        side = 0
        for _ in range(80):
            x = x1 - r1 * (x1 - x0) / (r1 - r0) if r1 != r0 else 0.5 * (x0 + x1)
            if not (min(x0, x1) < x < max(x0, x1)):
                x = 0.5 * (x0 + x1)
            s, r = attempt(x)
            if r == 0.0:
                return s, n
            if r is None:
                break
            # Illinois rule: when the same end stays put twice, halve its
            # residual, so false position keeps closing in from both sides
            if r * r0 < 0:
                x1, r1 = x, r
                if side == -1:
                    r0 *= 0.5
                side = -1
            else:
                x0, r0 = x, r
                if side == 1:
                    r1 *= 0.5
                side = 1
            if abs(x1 - x0) < 1e-12 * abs(x):
                break
    return None


def _with_flow(payload, state, m):
    out = dict(payload)
    out["states"] = {k: dict(v) for k, v in (payload.get("states") or {}).items()}
    out["states"].setdefault(state, {})["m"] = m
    return out


def _on_target(w, target):
    """Net power matched: much tighter than _close, so the summary shows
    exactly the power asked for."""
    return math.isclose(w, target, rel_tol=1e-9, abs_tol=1e-6)


def _solve_for_net_power(payload, props, ha_props, target_display):
    """Mass flows that give the requested net power. Every flow in a cycle
    follows the flow at one seed state (a turbine inlet, else a compressor or
    pump inlet), so search that one flow until the net power matches. Usually
    net power is proportional to flow and the second try lands exactly, but a
    fixed heat rate (a boiler's Q) makes it nonlinear, and a poor first guess
    can push a state off the property tables, so the search is a guarded
    secant."""
    first = _Solver(payload, props, ha_props)
    errors = first.run()
    if errors:
        return first, errors
    target = target_display * power_factor(first.system)
    w = _net_power(first)
    if w is not None:
        if not _close(w, target):
            errors.append(
                f"Net power wanted (Cycle Summary): the cycle as given makes {w / power_factor(first.system):.6g} "
                f"{label(first.system, 'power')}, not the {target_display:g} asked for. "
                f"Clear a mass flow (or the net power) so one can be found from the other."
            )
        return first, errors

    seed = _seed_state(payload)
    if seed is None:
        return first, ["Net power wanted (Cycle Summary): add a turbine, compressor or pump so there is work to match."]

    tried = {}

    def attempt(m):
        solver = _Solver(_with_flow(payload, seed, m), props, ha_props)
        errs = solver.run()
        w = None if errs else _net_power(solver)
        tried[m] = (solver, errs, w)
        return w

    # a flow at which the cycle solves at all
    m0 = w0 = None
    for m in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 0.1, 0.01):
        w = attempt(m)
        if w is not None and abs(w) > 1e-9:
            m0, w0 = m, w
            break
    if m0 is None:
        solver, errs, _ = tried[1.0]
        return solver, errs or [
            "Net power wanted (Cycle Summary): the cycle isn't defined enough yet to find its work per kg. Solve it without "
            "the net power first and fill in what's missing."
        ]

    m1 = m0 * target / w0
    for _ in range(40):
        if m1 <= 0:
            break
        w1 = attempt(m1)
        # off the property tables or unsolvable: step back toward the last good flow
        backoff = 0
        while w1 is None and backoff < 30:
            m1 = math.sqrt(m0 * m1)
            w1 = attempt(m1)
            backoff += 1
        if w1 is None:
            break
        if _on_target(w1, target):
            return tried[m1][0], tried[m1][1]
        if abs(w1 - w0) < 1e-12 * max(abs(w1), 1.0):
            break
        m0, w0, m1 = m1, w1, m1 + (target - w1) * (m1 - m0) / (w1 - w0)

    # The secant can stall where the work hardly changes with the flow (a
    # turbine fed liquid, say). Scan the flow over a wide range for a sign
    # change and bisect it.
    grid = [10 ** (k / 4) for k in range(-12, 21)]
    points = []
    for m in grid:
        w = tried[m][2] if m in tried else attempt(m)
        if w is not None:
            points.append((m, w - target))
    for (ma, fa), (mb, fb) in zip(points, points[1:]):
        if fa * fb > 0:
            continue
        for _ in range(80):
            mid = math.sqrt(ma * mb)
            w = attempt(mid)
            if w is None:
                break
            if _on_target(w, target):
                return tried[mid][0], tried[mid][1]
            if (w - target) * fa > 0:
                ma, fa = mid, w - target
            else:
                mb, fb = mid, w - target

    best = min((m for m in tried if tried[m][2] is not None), key=lambda m: abs(tried[m][2] - target))
    solver = tried[best][0]
    pf = power_factor(solver.system)
    return solver, [
        f"Net power wanted (Cycle Summary): no mass flow makes {target_display:g} {label(solver.system, 'power')}; the closest "
        f"is {tried[best][2] / pf:.6g} {label(solver.system, 'power')}. Check the fixed heat rates and "
        f"temperatures against the power asked for."
    ]


# ---------- T-s and P-h diagrams ----------

TS_SAMPLES = 32


def _dome(props, fluid):
    """Saturation dome as [[s, T, h, P], ...] (SI), liquid side up to the
    critical point and back down the vapor side. Empty for fluids CoolProp has
    no saturation curve for."""
    try:
        t_lo = props("Ttriple", "", 0, "", 0, fluid)
        t_crit = props("Tcrit", "", 0, "", 0, fluid)
    except Exception:
        return []
    t_lo = max(t_lo, t_crit * 0.4) + 0.01
    liq, vap = [], []
    n = 60
    for i in range(n + 1):
        # crowd the points near the critical point, where the dome turns
        f = 1 - (1 - i / n) ** 2
        t = t_lo + (t_crit - t_lo) * f * 0.9999
        try:
            p = props("P", "T", t, "Q", 0, fluid)
            liq.append([props("S", "T", t, "Q", 0, fluid), t, props("H", "T", t, "Q", 0, fluid), p])
            vap.append([props("S", "T", t, "Q", 1, fluid), t, props("H", "T", t, "Q", 1, fluid), p])
        except Exception:
            continue
    return liq + vap[::-1]


def _path(solver, a, b):
    """Process line between states a and b as [[s, T, h, P], ...] (SI): along
    the isobar when the pressures match, along the isenthalp when the
    enthalpies match (a valve), otherwise a straight line between the ends."""
    va, vb = solver.values.get(a, {}), solver.values.get(b, {})
    if not all(k in v for v in (va, vb) for k in ("s", "T", "h", "P")):
        return None
    fluid = solver._fluid(a)
    if solver._fluid(b) != fluid:
        return solver.comb.path(a, b) if b in solver.comb.fluid else None   # combustor or mixer
    pts = []
    try:
        if _close(va["P"], vb["P"]) and not _close(va["s"], vb["s"]):
            for i in range(TS_SAMPLES + 1):
                sv = va["s"] + (vb["s"] - va["s"]) * i / TS_SAMPLES
                pts.append([sv, solver.props("T", "P", va["P"], "S", sv, fluid),
                            solver.props("H", "P", va["P"], "S", sv, fluid), va["P"]])
        elif _close(va["h"], vb["h"]) and not _close(va["P"], vb["P"]):
            for i in range(TS_SAMPLES + 1):
                pv = va["P"] * (vb["P"] / va["P"]) ** (i / TS_SAMPLES)
                pts.append([solver.props("S", "P", pv, "H", va["h"], fluid),
                            solver.props("T", "P", pv, "H", va["h"], fluid), va["h"], pv])
    except Exception:
        pts = []
    return pts or [[va[k] for k in ("s", "T", "h", "P")], [vb[k] for k in ("s", "T", "h", "P")]]


def _path_kind(node, a, b, solver):
    """What crosses the boundary along this path: "Win", "Wout", "Qin", "Qout" or ""."""
    t = node.get("type")
    if t in ("compressor", "pump"):
        return "Win"
    if t == "turbine":
        return "Wout"
    if is_combustor(node):
        return "Qin"
    if t == "ac" and a == _ports(node).get("refIn"):
        # the refrigerant takes up (or gives) the air's heat
        ha, hb = solver.values.get(a, {}).get("h"), solver.values.get(b, {}).get("h")
        if ha is not None and hb is not None and not _close(ha, hb):
            return "Qin" if hb > ha else "Qout"
        return ""
    if t == "hx" and not any(k.startswith("b_") for k in _ports(node)):
        ha, hb = solver.values.get(a, {}).get("h"), solver.values.get(b, {}).get("h")
        if ha is not None and hb is not None and not _close(ha, hb):
            return "Qin" if hb > ha else "Qout"
    return ""


def _device_pairs(node):
    """(inlet state, outlet state) pairs that are one fluid going through the device."""
    t = node.get("type")
    ports = _ports(node)
    if t == "turbine":
        outs = [ports[k] for k in ports if k.startswith("ext")] + [ports["out"]]
        if not (node.get("assumptions") or {}).get("sameOutlet") and len(outs) > 1:
            chain = [ports["in"]] + outs
            return list(zip(chain, chain[1:]))
        return [(ports["in"], o) for o in outs]
    if t == "hx":
        pairs = []
        for p in ("a", "b"):
            for i in _numbered(ports, f"{p}_in"):
                pairs += [(i, o) for o in _numbered(ports, f"{p}_out")]
        return pairs
    if t == "mixer":
        return [(i, ports["out"]) for i in _numbered(ports, "in")]
    if t == "splitter":
        return [(ports["in"], o) for o in _numbered(ports, "out")]
    if t == "flash":
        return [(ports["in"], ports["vapor"]), (ports["in"], ports["liquid"])]
    pairs = []
    if "in" in ports and "out" in ports:
        pairs.append((ports["in"], ports["out"]))
    # 1.10.2: an air-conditioning coil's refrigerant side is its own process
    # line (the evaporator or condenser leg of the refrigerant's cycle)
    if t == "ac" and "refIn" in ports and "refOut" in ports:
        pairs.append((ports["refIn"], ports["refOut"]))
    return pairs


def _diagram_data(solver):
    """Everything the T-s, P-h and P-v views draw, in display units:
    { ts: { fluids: { name: { dome: [[s, T]], states: { n: [s, T] }, paths: [{ from, to, kind, points }] } } },
      ph: the same with [h, P] points, pv: with [v, P] points }"""
    sys_ = solver.system
    axes = {"ts": (("s", 0), ("T", 1)), "ph": (("h", 2), ("P", 3)), "pv": (("v", 4), ("P", 3)),
            "hs": (("s", 0), ("h", 2))}           # hs: the Mollier chart
    out = {k: {} for k in axes}

    def with_v(pts, fluid):
        """[s, T, h, P] -> [s, T, h, P, v]; None for v when it can't be found."""
        res = []
        for p in pts:
            try:
                v = 1 / solver.props("D", "P", p[3], "H", p[2], fluid)
            except Exception:
                v = None
            res.append(list(p[:4]) + [v])
        return res

    def conv(view, pts):
        (kx, ix), (ky, iy) = axes[view]
        return [[from_si(sys_, kx, p[ix]), from_si(sys_, ky, p[iy])] for p in pts if p[ix] is not None]

    domes = {}

    def entry(view, fluid):
        if fluid not in out[view]:
            if fluid not in domes:
                domes[fluid] = with_v(_dome(solver.props, fluid), fluid)
            out[view][fluid] = {"dome": conv(view, domes[fluid]), "states": {}, "paths": []}
        return out[view][fluid]

    for n, v in solver.values.items():
        if all(k in v for k in ("s", "T", "h", "P")):
            pt = with_v([[v["s"], v["T"], v["h"], v["P"]]], solver._fluid(n))
            for view in axes:
                c = conv(view, pt)
                if c:
                    entry(view, solver.comb.chart(n))["states"][str(n)] = c[0]
    seen = set()
    for node in solver.nodes.values():
        for a, b in _device_pairs(node):
            if (a, b) in seen or a == b:
                continue
            seen.add((a, b))
            pts = _path(solver, a, b)
            if pts:
                pts = with_v(pts, solver._fluid(a))
                kind = _path_kind(node, a, b, solver)
                for view in axes:
                    entry(view, solver.comb.chart(a))["paths"].append(
                        {"from": a, "to": b, "kind": kind, "points": conv(view, pts)})
    return {view: {"fluids": out[view]} for view in axes}


def solve_system(payload, props=None, ha_props=None, lean=False):
    """lean (1.10.12): the answers only (states, devices, errors, messages),
    without the charts (dome, process paths), the psychrometric chart and the
    Show my work write-up. For the optimizer and the sweep, which re-solve the
    cycle dozens of times and never look at those: drawing the charts was about
    90 % of each solve."""
    if props is None:
        props = _cached_props()
    if payload.get("mode") == "closed":
        from closed_system import solve_closed
        return solve_closed(payload, props)
    if ha_props is None:
        try:
            from CoolProp.HumidAirProp import HAPropsSI
            ha_props = HAPropsSI
        except ImportError:
            pass   # only cooling towers need it; they report the error
        else:
            if "F" in label(payload.get("unitFactors") or payload.get("unitSystem", "SI"), "T"):
                ha_props = _english_humid_reference(ha_props)

    try:
        target = _number((payload.get("cycle") or {}).get("netPower"), "Net power")
        torn = None
        if target is None:
            solver = _Solver(payload, props, ha_props)
            errors = solver.run()
            if not errors and solver.unknown_flows() and solver.values and _determined(payload):
                found = _tear_flows(payload, props, ha_props, solver)
                if found:
                    solver, torn = found
        else:
            solver, errors = _solve_for_net_power(payload, props, ha_props, target)
    except SolveError as e:
        hints = _with_reasons(payload, None, [str(e)], diagnose(payload, None, [str(e)]))
        return {"states": {}, "devices": {}, "errors": [str(e)], "messages": [], "hints": hints}

    hints = _with_reasons(payload, solver, errors, diagnose(payload, solver, errors))
    messages = [] if errors else solver.report_messages() + hints["messages"]
    if torn is not None and not errors:
        messages.insert(0, f"The flows form a loop, so the flow at S{torn} was found by iterating until "
                           f"every balance around the loop closed.")
    messages += [w["text"] for w in hints["why"] if w["kind"] == "warning"]
    fixed = _fixed_efficiency(payload, target)
    if fixed is not None:
        note = (
            f"Both the heat added (Q̇_in) and the net power (Ẇ_net) are given, so the thermal efficiency "
            f"is fixed at Ẇ_net / Q̇_in = {fixed * 100:.4g}% whatever the device efficiencies are. "
            f"Leave one of them blank to let the cycle's efficiency come from its devices."
        )
        if errors:
            errors = errors + [note]
        else:
            messages = [note] + messages

    out = {
        "states": solver.report_states(),
        "devices": solver.report_devices(),
        "units": {"power": label(solver.system, "power")},
        "errors": errors,
        "messages": messages,
        "hints": hints,
        "products": solver.comb.report_states(),
    }
    if torn is not None and not errors and "m" in solver.values.get(torn, {}):
        # the flow that closed the loop: a warm start for the next re-solve
        out["tear"] = {"state": torn, "m": from_si(solver.system, "m", solver.values[torn]["m"])}
    if lean:
        return out
    return {
        **out,
        **_diagram_data(solver),
        "psy": psychro.chart(solver, from_si),
        "work": _show_work(payload, solver),
    }


_PROPS = None


def _cached_props():
    """CoolProp's PropsSI (with the app's ideal-gas air), remembering answers
    (1.10.12). A property call is a pure function of its inputs, and the same
    calls repeat a great deal: every pass of a solve re-checks its states, a
    loop of unknown flows re-solves the whole diagram several times with only
    the flows changed, and the optimizer and sweep re-solve it dozens of times.
    Errors are not remembered, so they are raised again as before."""
    global _PROPS
    if _PROPS is None:
        import functools
        from CoolProp.CoolProp import PropsSI
        from ideal_gas import with_ideal_air
        base = with_ideal_air(PropsSI)
        cached = functools.lru_cache(maxsize=200_000)(base)

        def props(out, k1, v1, k2, v2, fluid):
            return cached(out, k1, float(v1), k2, float(v2), fluid)
        props.base = getattr(base, "base", base)
        props.cache = cached
        _PROPS = props
    return _PROPS


def _show_work(payload, solver):
    """View > Show my work (1.10.7): the worked solution, from workings.py.
    A help, never a reason for a solve to fail."""
    try:
        from workings import show_work
        return show_work(payload, solver)
    except Exception as e:
        return {"states": [], "devices": [], "cycle": None, "notes": [f"The worked solution could not be written ({e})."]}


def _with_reasons(payload, solver, errors, hints):
    """Adds the degrees-of-freedom count (hints["dof"]) and the plain-language
    reasons (hints["why"], from hints.explain) to diagnose()'s hints, and
    marks the states and devices those reasons name. Never lets a problem in
    the explaining stop a solve from being reported."""
    try:
        from dof import count
        dof = count(payload)
    except Exception as e:                  # the count is a help, not a requirement
        dof = {"supported": False, "reason": f"The count could not be made ({e})."}
    try:
        why = explain(payload, solver, errors, dof)
    except Exception:
        why = []
    kinds = {}
    for item in why:
        if item["kind"] == "info":
            continue
        kind = "conflict" if item["kind"] == "error" else "missing"
        for n in item["states"]:
            st = hints["states"].setdefault(str(n), {"kind": kind, "text": item["text"]})
            if kind == "conflict":
                st.update(kind="conflict", text=item["text"])
        for d in item["devices"]:
            if d is None:
                continue
            hints["devices"][d] = item["text"] if kind == "conflict" or d not in hints["devices"] else hints["devices"][d]
            if kind == "conflict" or d not in kinds:
                kinds[d] = kind
    for d in hints["devices"]:
        kinds.setdefault(d, "conflict" if errors else "missing")
    hints["deviceKinds"] = kinds
    hints["dof"] = dof
    hints["why"] = why
    return hints


def _fixed_efficiency(payload, net_target):
    """Ẇ_net / Q̇_in when both are inputs (net power and the heat added to
    one-stream heat exchangers and mixers), else None."""
    if not net_target:
        return None
    q_in = 0.0
    for node in (payload.get("nodes") or {}).values():
        if node.get("type") not in ("hx", "mixer"):
            continue
        if any(k.startswith("b_") for k in _ports(node)) or (node.get("assumptions") or {}).get("adiabatic"):
            continue
        try:
            q = _number((node.get("params") or {}).get("Qdot"), "Q̇")
        except SolveError:
            return None
        if q is not None and q > 0:
            q_in += q
    return net_target / q_in if q_in > 0 else None