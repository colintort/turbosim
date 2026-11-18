"""Show my work (1.10.7): the solution written out the way a textbook does it.

For every state: what was given, which property table fixes it and how the
rest is read (h = hf + x hfg with the numbers, for a two-phase state). For
every device: the mass and energy rate balances in their general form, the
assumptions that reduce them, the reduced equation, the numbers plugged in
with their units and the answer. Then the cycle: net power, thermal
efficiency or coefficient of performance.

The equations, their numbers and the table numbers follow Moran, Shapiro,
Boettner and Bailey, "Fundamentals of Engineering Thermodynamics", 9th ed.
(checked against the book):
    Eq. 3.7    h = hf + x (hg - hf)              two-phase mixture
    Eq. 4.6    sum m_in = sum m_out              mass rate balance, steady state
    Eq. 4.18   0 = Q - W + sum m_i(h + V^2/2 + gz)_i - sum m_e(...)_e
    Eq. 4.20a  the same for one inlet and one exit
    Eq. 4.22   h2 = h1                           throttling
    Eq. 6.20a  s2 - s1 = s°(T2) - s°(T1) - R ln(p2/p1)
    Eq. 6.41   pr2/pr1 = p2/p1                   isentropic, air tables
    Eq. 6.43   T2/T1 = (p2/p1)^((k-1)/k)         isentropic, constant cp
    Eq. 6.46   eta_t = (h1 - h2)/(h1 - h2s)      turbine
    Eq. 6.47   eta_nozzle = (V2^2/2)/(V2s^2/2)
    Eq. 6.48   eta_c = (h2s - h1)/(h2 - h1)      compressor and pump
    Eq. 8.7b   W_p/m ≈ v1 (p2 - p1)              pump work
    Eq. 2.42   eta = W_cycle/Q_in;  Eq. 2.45 beta = Q_in/W_cycle;  Eq. 2.47 gamma = Q_out/W_cycle
    Eq. 8.6    bwr = W_p/W_t (back work ratio)
Tables: water A-2 (saturated, temperature), A-3 (saturated, pressure), A-4
(superheated), A-5 (compressed liquid); R-22 A-7..A-9; R-134a A-10..A-12;
ammonia A-13..A-15; propane A-16..A-18; air as an ideal gas A-22.

Everything is reported in the units on screen. The result:
    { states: [item], devices: [item], cycle: item | None, notes: [str] }
    item = { id, title, kind, steps: [{ head, lines: [[cls, text]] }] }
    cls: "law" (general form), "asm" (assumptions), "eq" (reduced equation),
         "num" (numbers plugged in), "res" (the answer), "note"
"""

import math

from units import from_si, label, power_factor, to_si

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
MINUS = "−"
G = 9.80665
SI_LIKE = 1e-3          # display and SI powers agree to this, so no conversion line is needed

# Moran 9th ed. property tables, by CoolProp fluid name.
TABLES = {
    "Water": {"what": "water", "satT": "A-2", "satP": "A-3", "sup": "A-4", "comp": "A-5"},
    "R22": {"what": "Refrigerant 22", "satT": "A-7", "satP": "A-8", "sup": "A-9"},
    "R134a": {"what": "Refrigerant 134a", "satT": "A-10", "satP": "A-11", "sup": "A-12"},
    "Ammonia": {"what": "ammonia", "satT": "A-13", "satP": "A-14", "sup": "A-15"},
    "Propane": {"what": "propane", "satT": "A-16", "satP": "A-17", "sup": "A-18"},
}
AIR_TABLES = "Air (ideal gas, variable cp)"
IDEAL_AIR = "Air (ideal gas, k = 1.4)"
HUMID_AIR = "Humid air"
PR_300 = 1.3860          # Moran Table A-22: pr at 300 K, which fixes the scale of pr

# What each kind of device is called in the headings and its work symbol.
WORK_SYMBOL = {"turbine": "Ẇt", "compressor": "Ẇc", "pump": "Ẇp"}


def sub(n):
    return str(n).translate(SUB)


def num(x, sig=5):
    """A number the way a worked solution prints it: about five significant
    figures, no exponent between 0.001 and ten million, a real minus sign."""
    if x is None:
        return "?"
    if abs(x) < 1e-10 or not math.isfinite(x):     # round-off noise such as −1e-13 °C is zero
        return "0" if math.isfinite(x) else str(x)
    a = abs(x)
    if 1e-3 <= a < 1e7:
        d = max(0, sig - 1 - int(math.floor(math.log10(a))))
        text = f"{a:,.{d}f}" if a >= 1e4 else f"{a:.{d}f}"    # 101,169 as the book prints it
        if "." in text:
            text = text.rstrip("0").rstrip(".")
    else:
        text = f"{a:.{sig - 1}e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    return (MINUS if x < 0 else "") + text


def par(x, sig=5):
    """A number to substitute after an operator: negative ones in parentheses."""
    t = num(x, sig)
    return f"({t})" if x is not None and x < 0 else t


def step(head, *lines):
    return {"head": head, "lines": [list(l) for l in lines if l and l[1]]}


class _Work:
    def __init__(self, payload, solver):
        self.s = solver
        self.sys = solver.system
        self.pf = power_factor(self.sys)
        self.v = solver.values
        self.typed = {}
        for key, cells in (payload.get("states") or {}).items():
            got = set()
            for f, x in (cells or {}).items():
                if x is not None and str(x).strip() != "":
                    got.add(f)
            self.typed[int(key)] = got
        self.nodes = payload.get("nodes") or {}
        self.cycle = payload.get("cycle") or {}
        self.notes = []
        # English units: the book's E tables (A-2E, A-4E, ..., A-22E)
        self.en = "Btu" in label(self.sys, "h")
        e = "E" if self.en else ""
        self.tables = {fl: {k: (t if k == "what" else t + e) for k, t in tab.items()} for fl, tab in TABLES.items()}
        self.a22 = "A-22" + e
        # kinetic and potential energy only appear in the balances when some state has them
        self.kepe = any(abs(v.get("V", 0.0)) > 1e-9 or abs(v.get("z", 0.0)) > 1e-9 for v in self.v.values())

    # ---------- units ----------
    def d(self, f, si):
        return None if si is None else from_si(self.sys, f, si)

    def u(self, f):
        if f == "power":
            return label(self.sys, "power") or self._power_natural()
        return label(self.sys, f)

    def q(self, f, si, sig=5):
        """value and unit, in display units"""
        if si is None:
            return "?"
        unit = self.u(f)
        if f == "x":
            unit = ""
        if f == "T" and unit in ("K", "R"):
            return f"{num(self.d(f, si), sig)} {unit}"
        return f"{num(self.d(f, si), sig)} {unit}".strip()

    def dh(self, si):
        """an enthalpy difference in display units (no offset)"""
        return si / to_si(self.sys, "h", 1.0) if to_si(self.sys, "h", 0.0) == 0 else from_si(self.sys, "h", si)

    def _power_natural(self):
        """(enthalpy unit) x (mass flow unit), e.g. kJ/s, Btu/s"""
        h, m = label(self.sys, "h"), label(self.sys, "m")
        top = h.split("/")[0] if "/" in h else h
        bottom = m.split("/")[-1] if "/" in m else "s"
        return f"{top}/{bottom}"

    def power(self, w_si):
        """a power (W) in the display power unit"""
        return w_si / self.pf

    def rate_result(self, m_si, dh_si, sym):
        """'= (ṁ)(Δh) = Ẇ' with the units, converting when the power unit isn't (h unit)(m unit)."""
        m_d, h_d = self.d("m", m_si), self.dh(dh_si)
        natural = m_d * h_d
        shown = self.power(m_si * dh_si)
        line = f"{sym} = ({num(m_d)} {self.u('m')})({num(h_d)} {self.u('h')})"
        if abs(natural - shown) <= SI_LIKE * max(abs(shown), 1e-12):
            return line + f" = {num(shown)} {self.u('power')}"
        return line + f" = {num(natural)} {self._power_natural()} = {num(shown)} {self.u('power')}"

    # ---------- state values ----------
    def val(self, n, f):
        return self.v.get(n, {}).get(f)

    def T_abs(self, si):
        """an absolute temperature for ratios: K, or R in English units"""
        return si * 1.8 if "F" in label(self.sys, "T") or label(self.sys, "T") == "R" else si

    def T_abs_unit(self):
        return "R" if "F" in label(self.sys, "T") or label(self.sys, "T") == "R" else "K"

    def fluid(self, n):
        return self.s._fluid(n)

    def props(self, *args):
        return self.s.props(*args)

    def sym(self, f, n):
        return {"P": "p", "T": "T", "h": "h", "s": "s", "x": "x", "m": "ṁ", "V": "V", "z": "z",
                "w": "ω", "RH": "φ"}.get(f, f) + sub(n)

    def has_kepe(self, *states):
        return self.kepe and any(abs(self.val(n, "V") or 0.0) > 1e-9 or abs(self.val(n, "z") or 0.0) > 1e-9
                                 for n in states)

    def extra(self, n):
        return (self.val(n, "V") or 0.0) ** 2 / 2 + G * (self.val(n, "z") or 0.0)

    # ---------- states ----------
    def states(self):
        out = []
        for n in sorted(self.v):
            try:
                steps = self.state_steps(n)
            except Exception as e:      # one state that can't be written out is a note, not a failure
                steps = [step("Properties", ("note", f"This state could not be written out ({e})."))]
            if steps:
                out.append({"id": str(n), "title": f"State {n}", "kind": "state", "fluid": self.fluid(n), "steps": steps})
        return out

    def given_line(self, n):
        typed = [f for f in ("P", "T", "x", "h", "s", "m", "V", "z", "w", "RH") if f in self.typed.get(n, ())
                 and self.val(n, f) is not None]
        if not typed:
            return None
        return ("eq", "Given: " + ", ".join(f"{self.sym(f, n)} = {self.q(f, self.val(n, f))}" for f in typed))

    def summary_line(self, n, keys=("T", "P", "h", "s", "x")):
        parts = [f"{self.sym(f, n)} = {self.q(f, self.val(n, f))}" for f in keys if self.val(n, f) is not None]
        return ("res", ", ".join(parts)) if parts else None

    def state_steps(self, n):
        v = self.v.get(n, {})
        fl = self.fluid(n)
        given = self.given_line(n)
        if fl == HUMID_AIR:
            return [step(f"Humid air, per kg of dry air", given,
                         ("note", "Psychrometric properties (Moran Ch. 12) at p, from two of T, ω, φ and h."),
                         self.summary_line(n, ("T", "P", "w", "RH", "h")))]
        comb = getattr(self.s, "comb", None)
        if comb is not None and n in getattr(comb, "fluid", {}):
            return [step("Combustion products", given,
                         ("note", "An ideal-gas mixture of the products (Moran Ch. 13); h and s are those of the mixture."),
                         self.summary_line(n, ("T", "P", "h", "s")))]
        pair = self.s.pairs.get(n)
        if pair is None or not all(k in v for k in ("P", "T", "h", "s")):
            if not given:
                return []
            return [step("Not fixed yet", given, ("note", "Two independent properties are needed to fix this state."))]
        found = [k for k in pair if k not in self.typed.get(n, ())]
        lines = [given]
        if found:
            lines.append(("note", "From the device equations below: "
                          + ", ".join(f"{self.sym(k, n)} = {self.q(k, v[k])}" for k in found) + "."))
        lines += self.lookup(n, pair, fl)
        lines.append(self.summary_line(n))
        return [step("Fixing the state", *lines)]

    def lookup(self, n, pair, fl):
        """How the rest of the state is read from the tables."""
        v = self.v[n]
        p_s, t_s = self.sym("P", n), self.sym("T", n)
        pairtxt = " and ".join(f"{self.sym(k, n)}" for k in pair)
        if fl == AIR_TABLES:
            return self.air_table_lookup(n, pair)
        if fl == IDEAL_AIR:
            return [("note", f"Air as an ideal gas with constant specific heats (cp = 1.005 kJ/kg·K, k = 1.4), "
                             f"fixed by {pairtxt}.")]
        tab = self.tables.get(fl)
        x = v.get("x")
        if x is not None and -1e-9 <= x <= 1 + 1e-9:
            sat = (tab or {}).get("satP" if "P" in pair or "T" not in pair else "satT")
            where = f"Table {sat} (saturated {tab['what']}, " + ("pressure" if sat == tab["satP"] else "temperature") \
                + " table)" if tab else f"saturation properties of {fl} (CoolProp)"
            at = f"{p_s} = {self.q('P', v['P'])}" if "P" in pair or "T" not in pair else f"{t_s} = {self.q('T', v['T'])}"
            hf, hg = self.props("H", "P", v["P"], "Q", 0, fl), self.props("H", "P", v["P"], "Q", 1, fl)
            sf, sg = self.props("S", "P", v["P"], "Q", 0, fl), self.props("S", "P", v["P"], "Q", 1, fl)
            H, S, hu, su = lambda z: num(self.d("h", z)), lambda z: num(self.d("s", z), 5), self.u("h"), self.u("s")
            lines = [("eq", f"Two-phase at {at}: from {where}, T = Tsat = {self.q('T', v['T'])}, "
                            f"hf = {H(hf)}, hg = {H(hg)} {hu}, sf = {S(sf)}, sg = {S(sg)} {su}")]
            xs = "x" + sub(n)
            if x <= 1e-9:
                lines.append(("num", f"Saturated liquid ({xs} = 0): h{sub(n)} = hf = {H(hf)} {hu}, s{sub(n)} = sf = {S(sf)} {su}"))
            elif x >= 1 - 1e-9:
                lines.append(("num", f"Saturated vapor ({xs} = 1): h{sub(n)} = hg = {H(hg)} {hu}, s{sub(n)} = sg = {S(sg)} {su}"))
            else:
                if "x" in pair:
                    pass
                elif "h" in pair:
                    lines.append(("num", f"{xs} = (h{sub(n)} {MINUS} hf)/(hg {MINUS} hf) = ({H(v['h'])} {MINUS} {H(hf)})/"
                                         f"({H(hg)} {MINUS} {H(hf)}) = {num(x, 4)}"))
                elif "s" in pair:
                    lines.append(("num", f"{xs} = (s{sub(n)} {MINUS} sf)/(sg {MINUS} sf) = ({S(v['s'])} {MINUS} {S(sf)})/"
                                         f"({S(sg)} {MINUS} {S(sf)}) = {num(x, 4)}"))
                if "h" not in pair:
                    lines.append(("num", f"h{sub(n)} = hf + {xs}(hg {MINUS} hf) = {H(hf)} + ({num(x, 4)})({H(hg - hf)}) "
                                         f"= {H(v['h'])} {hu}   (Eq. 3.7)"))
                if "s" not in pair:
                    lines.append(("num", f"s{sub(n)} = sf + {xs}(sg {MINUS} sf) = {S(sf)} + ({num(x, 4)})({S(sg - sf)}) "
                                         f"= {S(v['s'])} {su}"))
            self.zero_point_note(fl)
            return lines
        # single phase
        region, table = None, None
        try:
            t_sat = self.props("T", "P", v["P"], "Q", 0, fl)
        except Exception:
            t_sat = None
        if t_sat is not None and v["T"] < t_sat - 1e-6:
            region = "compressed liquid"
            if tab and tab.get("comp"):
                table = f"Table {tab['comp']} (compressed liquid {tab['what']})"
        elif t_sat is not None:
            region = "superheated vapor"
            if tab:
                table = f"Table {tab['sup']} (superheated {tab['what']})"
        else:
            region = "single phase"
        lines = []
        if t_sat is not None:
            cmp = "<" if region == "compressed liquid" else ">"
            lines.append(("eq", f"At {p_s} = {self.q('P', v['P'])}, Tsat = {self.q('T', t_sat)}; {t_s} {cmp} Tsat, "
                                f"so the state is {region}."))
        if table:
            how = "interpolating with" if "s" in pair or "h" in pair else "at"
            lines.append(("num", f"From {table}, {how} {pairtxt}."))
        elif region == "compressed liquid" and tab:
            lines.append(("num", f"Compressed liquid: no table for {tab['what']}, so h ≈ hf(T) (Moran Sec. 3.10.1) "
                                 f"or the exact value from CoolProp, used here."))
        else:
            lines.append(("num", f"Properties of {fl} (CoolProp), fixed by {pairtxt}."))
        self.zero_point_note(fl)
        return lines

    def zero_point_note(self, fl):
        if fl in self.tables and fl != "Water":
            note = (f"{fl}: CoolProp's zero point for h and s may differ from the book's Table "
                    f"{self.tables[fl]['satT']}–{self.tables[fl]['sup']}, which shifts every h and s by a constant. "
                    "The differences, and so every answer, are the same.")
            if note not in self.notes:
                self.notes.append(note)

    def s0(self, n):
        """s°(T) and pr for an air-table state."""
        from ideal_gas import AIR_VAR, P0
        AIR_VAR._build()
        T = self.val(n, "T")
        s0 = AIR_VAR._at(AIR_VAR.s0, T)
        s0_300 = AIR_VAR._at(AIR_VAR.s0, 300.0)
        pr = PR_300 * math.exp((s0 - s0_300) / AIR_VAR.r)
        return s0, pr

    def air_table_lookup(self, n, pair):
        v = self.v[n]
        s0, pr = self.s0(n)
        T = v["T"]
        lines = [("eq", f"Air as an ideal gas, Table {self.a22} at T{sub(n)} = {num(self.T_abs(T))} {self.T_abs_unit()}: "
                        f"h{sub(n)} = {self.q('h', v['h'])}, s°{sub(n)} = {self.q('s', s0)}, pr{sub(n)} = {num(pr, 4)}")]
        if "T" not in pair:
            lines.insert(0, ("note", f"T{sub(n)} is read back from Table {self.a22} with "
                                     + " and ".join(self.sym(k, n) for k in pair) + "."))
        return lines

    # ---------- devices ----------
    def devices(self):
        from system_solver import _ports
        out = []
        for node_id, node in self.nodes.items():
            t = node.get("type")
            name = node.get("name") or t
            try:
                ports = _ports(node)
                steps = self.device_steps(node_id, node, t, ports)
            except Exception as e:
                steps = [step("Balances", ("note", f"This device could not be written out ({e})."))]
            steps = [s for s in steps if s and s["lines"]]
            if not steps:
                steps = [step("Balances", ("note", "Not solved far enough yet to write the balances out."))]
            out.append({"id": node_id, "title": name, "kind": t, "steps": steps})
        return out

    def device_steps(self, node_id, node, t, ports):
        from combustion import is_combustor
        if is_combustor(node):
            return self.combustor(node_id, node, ports)
        fn = {
            "turbine": self.turbine, "compressor": self.compressor, "pump": self.compressor,
            "valve": self.valve, "nozzle": self.nozzle, "diffuser": self.nozzle, "hx": self.hx,
            "mixer": self.mixer, "splitter": self.splitter, "flash": self.flash,
            "coolingTower": self.cooling_tower, "atmosphere": self.atmosphere, "ac": self.ac,
        }.get(t)
        return fn(node_id, node, ports) if fn else []

    def mass_step(self, ins, outs, extra=None):
        """Eq. 4.6 with the numbers."""
        lhs = " + ".join(f"ṁ{sub(i)}" for i in ins)
        rhs = " + ".join(f"ṁ{sub(o)}" for o in outs)
        lines = [("law", "Σ ṁi = Σ ṁe   (Eq. 4.6, steady state)"), ("eq", f"{lhs} = {rhs}")]
        ms = [self.val(n, "m") for n in ins + outs]
        if all(m is not None for m in ms):
            if len(ins) == 1 and len(outs) == 1:
                lines.append(("res", f"ṁ{sub(ins[0])} = ṁ{sub(outs[0])} = {self.q('m', ms[0])}"))
            else:
                lines.append(("res", ", ".join(f"ṁ{sub(n)} = {self.q('m', m)}" for n, m in zip(ins + outs, ms))))
        if extra:
            lines.append(extra)
        return step("Mass rate balance", *lines)

    def kepe_line(self, a, b):
        """(V₁² − V₂²)/2 + g(z₁ − z₂) in the enthalpy unit, when it isn't zero"""
        if not self.has_kepe(a, b):
            return None, 0.0
        x = self.extra(a) - self.extra(b)
        return ("num", f"(V{sub(a)}² {MINUS} V{sub(b)}²)/2 + g(z{sub(a)} {MINUS} z{sub(b)}) = {num(self.dh(x))} {self.u('h')}"), x

    def isentropic_lines(self, a, b, h2s):
        """How h₂s is found: s₂s = s₁ at p₂, with the air-table and constant-cp routes."""
        fl = self.fluid(a)
        pa, pb = self.val(a, "P"), self.val(b, "P")
        s2 = f"{sub(b)}s"
        lines = []
        if fl == AIR_TABLES:
            _, pr1 = self.s0(a)
            pr2 = pr1 * pb / pa
            from ideal_gas import AIR_VAR
            T2s = self.props("T", "P", pb, "S", self.val(a, "s"), fl)
            lines.append(("num", f"pr,{s2} = pr{sub(a)}(p{sub(b)}/p{sub(a)}) = ({num(pr1, 4)})({num(pb / pa, 4)}) = {num(pr2, 4)}   (Eq. 6.41)"))
            lines.append(("num", f"Table {self.a22} at pr = {num(pr2, 4)}: T{s2} = {num(self.T_abs(T2s))} {self.T_abs_unit()}, "
                                 f"h{s2} = {self.q('h', h2s)}"))
            return lines
        if fl == IDEAL_AIR:
            T1 = self.val(a, "T")
            T2s = T1 * (pb / pa) ** (0.4 / 1.4)
            u = self.T_abs_unit()
            lines.append(("num", f"T{s2} = T{sub(a)}(p{sub(b)}/p{sub(a)})^((k{MINUS}1)/k) = ({num(self.T_abs(T1))} {u})"
                                 f"({num(pb / pa, 4)})^(0.4/1.4) = {num(self.T_abs(T2s))} {u}   (Eq. 6.43)"))
            lines.append(("num", f"h{s2} = {self.q('h', h2s)} at T{s2}"))
            return lines
        tab = self.tables.get(fl)
        where = ""
        try:
            x2s = self.props("Q", "P", pb, "S", self.val(a, "s"), fl)
        except Exception:
            x2s = None
        if x2s is not None and 0 <= x2s <= 1:
            sf = self.props("S", "P", pb, "Q", 0, fl)
            sg = self.props("S", "P", pb, "Q", 1, fl)
            hf = self.props("H", "P", pb, "Q", 0, fl)
            hg = self.props("H", "P", pb, "Q", 1, fl)
            S, H = lambda z: num(self.d("s", z)), lambda z: num(self.d("h", z))
            if tab:
                where = f" (Table {tab['satP']})"
            lines.append(("num", f"x{s2} = (s{sub(a)} {MINUS} sf)/(sg {MINUS} sf) = ({S(self.val(a, 's'))} {MINUS} {S(sf)})/"
                                 f"({S(sg)} {MINUS} {S(sf)}) = {num(x2s, 4)}{where}"))
            lines.append(("num", f"h{s2} = hf + x{s2}(hg {MINUS} hf) = {H(hf)} + ({num(x2s, 4)})({H(hg - hf)}) = "
                                 f"{self.q('h', h2s)}"))
        else:
            if tab:
                try:
                    t_sat = self.props("T", "P", pb, "Q", 0, fl)
                    t2s = self.props("T", "P", pb, "S", self.val(a, "s"), fl)
                    where = f" from Table {tab['sup'] if t2s >= t_sat else tab.get('comp', tab['sup'])}"
                except Exception:
                    where = ""
            lines.append(("num", f"At p{sub(b)} = {self.q('P', pb)} and s{s2} = s{sub(a)} = {self.q('s', self.val(a, 's'))}"
                                 f"{where}: h{s2} = {self.q('h', h2s)}"))
        return lines

    def h2s(self, a, b):
        try:
            return self.props("H", "P", self.val(b, "P"), "S", self.val(a, "s"), self.fluid(a))
        except Exception:
            return None

    def turbine(self, node_id, node, ports):
        from system_solver import _numbered, _efficiency
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        a, out = ports["in"], ports["out"]
        exts = _numbered(ports, "ext")
        outlets = exts + [out]
        steps = [self.mass_step([a], outlets)]
        h1 = self.val(a, "h")
        adiabatic = asm.get("adiabatic", True)

        if len(outlets) == 1:
            b = out
            h2 = self.val(b, "h")
            law = ("law", f"0 = Q̇cv {MINUS} Ẇt + ṁ[(h{sub(a)} {MINUS} h{sub(b)}) + (V{sub(a)}² {MINUS} V{sub(b)}²)/2 + "
                          f"g(z{sub(a)} {MINUS} z{sub(b)})]   (Eq. 4.20a)")
            ke, x = self.kepe_line(a, b)
            q_txt = "Q̇cv = 0 (adiabatic)" if adiabatic else "heat crosses the casing"
            asm_line = ("asm", f"Steady state, {q_txt}" + ("" if ke else ", Δke = Δpe = 0"))
            lines = [law, asm_line]
            if h1 is not None and h2 is not None:
                lines.append(("eq", f"Ẇt/ṁ = h{sub(a)} {MINUS} h{sub(b)}" + (" + Δke + Δpe" if ke else "")))
                lines.append(ke)
                w = h1 - h2 + x
                lines.append(("num", f"Ẇt/ṁ = {num(self.d('h', h1))} {MINUS} {par(self.d('h', h2))}"
                                     + (f" + {par(self.dh(x))}" if ke else "") + f" = {num(self.dh(w))} {self.u('h')}"))
                m = self.val(a, "m")
                if m is not None and adiabatic:
                    lines.append(("res", self.rate_result(m, w, "Ẇt")))
                elif m is not None:
                    wd = params.get("Wdot")
                    lines.append(("res", f"Ẇt = {wd} {self.u('power')} (given), so Q̇cv = Ẇt {MINUS} ṁ(h{sub(a)} {MINUS} h{sub(b)}) = "
                                         f"{num(float(wd) - self.power(m * w))} {self.u('power')}"))
                else:
                    lines.append(("res", f"Ẇt/ṁ = {num(self.dh(w))} {self.u('h')} (per kg; no mass flow given)"))
            steps.append(step("Energy rate balance", *lines))
            steps.append(self.efficiency_step(node_id, node, a, b, compressor=False, sym="ηt", eq="6.46"))
        else:
            lines = [("law", f"0 = Q̇cv {MINUS} Ẇt + ṁ{sub(a)}h{sub(a)} {MINUS} "
                             + f" {MINUS} ".join(f"ṁ{sub(o)}h{sub(o)}" for o in outlets) + "   (Eq. 4.18)"),
                     ("asm", "Steady state, Q̇cv = 0 (adiabatic), Δke = Δpe = 0")]
            for i, e in enumerate(exts, 1):
                y = params.get(f"y{i}")
                if y not in (None, ""):
                    lines.append(("eq", f"ṁ{sub(e)} = y{sub(i)}ṁ{sub(a)}, y{sub(i)} = {y} (given)"))
            lines.append(("eq", f"Ẇt = ṁ{sub(a)}h{sub(a)} {MINUS} " + f" {MINUS} ".join(f"ṁ{sub(o)}h{sub(o)}" for o in outlets)))
            ms = [self.val(n, "m") for n in [a] + outlets]
            hs = [self.val(n, "h") for n in [a] + outlets]
            if all(z is not None for z in ms + hs):
                terms = [f"({num(self.d('m', ms[0]))})({num(self.d('h', hs[0]))})"] + \
                        [f"({num(self.d('m', m))})({num(self.d('h', h))})" for m, h in zip(ms[1:], hs[1:])]
                w = ms[0] * hs[0] - sum(m * h for m, h in zip(ms[1:], hs[1:]))
                lines.append(("num", f"Ẇt = {f' {MINUS} '.join(terms)}  [{self.u('m')} × {self.u('h')}]"))
                lines.append(("res", f"Ẇt = {num(self.power(w))} {self.u('power')}"))
            steps.append(step("Energy rate balance", *lines))
            if not asm.get("sameOutlet"):
                chain = [a] + exts + [out]
                for x_, y_ in zip(chain, chain[1:]):
                    steps.append(self.efficiency_step(None, node, x_, y_, compressor=False, sym="ηt", eq="6.46",
                                                      head=f"Stage {x_}–{y_}: isentropic efficiency"))
            else:
                steps.append(self.efficiency_step(node_id, node, a, out, compressor=False, sym="ηt", eq="6.46"))
                if exts:
                    steps.append(step("Extractions", ("eq", "Every outlet leaves at the exhaust state: "
                                      + ", ".join(f"h{sub(e)} = h{sub(out)}" for e in exts))))
        return steps

    def efficiency_step(self, node_id, node, a, b, compressor, sym, eq, head="Isentropic efficiency"):
        from system_solver import _efficiency
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        h1, h2 = self.val(a, "h"), self.val(b, "h")
        if h1 is None or h2 is None or self.val(a, "s") is None or self.val(b, "P") is None:
            return None
        h2s = self.h2s(a, b)
        if h2s is None:
            return None
        H = lambda z: num(self.d("h", z))  # noqa: E731
        hu = self.u("h")
        s_ = f"{sub(b)}s"          # the isentropic end state, 2s
        if compressor:
            law = ("law", f"{sym} = (h{s_} {MINUS} h{sub(a)})/(h{sub(b)} {MINUS} h{sub(a)})   (Eq. {eq})")
        else:
            law = ("law", f"{sym} = (h{sub(a)} {MINUS} h{sub(b)})/(h{sub(a)} {MINUS} h{s_})   (Eq. {eq})")
        lines = [law, ("asm", f"The isentropic end state {b}s: p{s_} = p{sub(b)}, s{s_} = s{sub(a)}")]
        lines += self.isentropic_lines(a, b, h2s)
        ideal = asm.get("isentropic") or asm.get("eta100")
        eta_typed = None if ideal else _efficiency(params.get("eta"), "η")
        if ideal:
            lines.append(("res", f"Isentropic ({sym} = 100 %): s{sub(b)} = s{sub(a)}, so h{sub(b)} = h{s_} = {H(h2s)} {hu}"))
        elif eta_typed is not None and "h" not in self.typed.get(b, ()) and "T" not in self.typed.get(b, ()):
            if compressor:
                lines.append(("eq", f"h{sub(b)} = h{sub(a)} + (h{s_} {MINUS} h{sub(a)})/{sym}"))
                lines.append(("res", f"h{sub(b)} = {H(h1)} + ({H(h2s)} {MINUS} {H(h1)})/{num(eta_typed, 4)} = {H(h2)} {hu}"))
            else:
                lines.append(("eq", f"h{sub(b)} = h{sub(a)} {MINUS} {sym}(h{sub(a)} {MINUS} h{s_})"))
                lines.append(("res", f"h{sub(b)} = {H(h1)} {MINUS} ({num(eta_typed, 4)})({H(h1)} {MINUS} {H(h2s)}) = {H(h2)} {hu}"))
        else:
            num_, den = (h2s - h1, h2 - h1) if compressor else (h1 - h2, h1 - h2s)
            if abs(den) < 1e-9:
                return None
            if compressor:
                txt = f"({H(h2s)} {MINUS} {H(h1)})/({H(h2)} {MINUS} {H(h1)})"
            else:
                txt = f"({H(h1)} {MINUS} {H(h2)})/({H(h1)} {MINUS} {H(h2s)})"
            lines.append(("res", f"{sym} = {txt} = {num(num_ / den, 4)} ({num(100 * num_ / den, 3)} %)"))
        return step(head, *lines)

    def compressor(self, node_id, node, ports):
        a, b = ports["in"], ports["out"]
        t = node.get("type")
        sym = WORK_SYMBOL[t]
        steps = [self.mass_step([a], [b])]
        params = node.get("params") or {}
        if t == "compressor" and params.get("rp") not in (None, ""):
            steps.append(step("Pressure ratio", ("eq", f"p{sub(b)} = r p{sub(a)}"),
                              ("res", f"p{sub(b)} = ({params.get('rp')})({self.q('P', self.val(a, 'P'))}) = {self.q('P', self.val(b, 'P'))}")))
        h1, h2 = self.val(a, "h"), self.val(b, "h")
        lines = [("law", f"0 = Q̇cv {MINUS} Ẇcv + ṁ[(h{sub(a)} {MINUS} h{sub(b)}) + (V{sub(a)}² {MINUS} V{sub(b)}²)/2 + "
                         f"g(z{sub(a)} {MINUS} z{sub(b)})]   (Eq. 4.20a)")]
        ke, x = self.kepe_line(a, b)
        lines.append(("asm", "Steady state, Q̇cv = 0 (adiabatic)" + ("" if ke else ", Δke = Δpe = 0")
                      + f", and the power input {sym} = {MINUS}Ẇcv"))
        if h1 is not None and h2 is not None:
            lines.append(("eq", f"{sym}/ṁ = h{sub(b)} {MINUS} h{sub(a)}" + (f" {MINUS} [Δke + Δpe]" if ke else "")))
            lines.append(ke)
            w = h2 - h1 - x
            lines.append(("num", f"{sym}/ṁ = {num(self.d('h', h2))} {MINUS} {par(self.d('h', h1))}"
                                 + (f" {MINUS} {par(self.dh(x))}" if ke else "") + f" = {num(self.dh(w))} {self.u('h')}"))
            m = self.val(a, "m")
            lines.append(("res", self.rate_result(m, w, sym) if m is not None
                          else f"{sym}/ṁ = {num(self.dh(w))} {self.u('h')} (per kg; no mass flow given)"))
        steps.append(step("Energy rate balance", *lines))
        steps.append(self.efficiency_step(node_id, node, a, b, compressor=True, sym="ηp" if t == "pump" else "ηc", eq="6.48"))
        if t == "pump" and all(self.val(n, k) is not None for n in (a, b) for k in ("P", "h")):
            try:
                rho = self.props("D", "P", self.val(a, "P"), "H", self.val(a, "h"), self.fluid(a))
            except Exception:
                rho = None
            if rho:
                v1 = 1.0 / rho
                dp = self.val(b, "P") - self.val(a, "P")
                w_s = v1 * dp
                # (v unit)(P unit) is not the enthalpy unit outside kPa and m³/kg
                # (ft³·psi/lbm needs 144 in²/ft² and 778 ft·lbf/Btu): show the factor
                v_d, dp_d = self.d("v", v1), self.dh(dp) * to_si(self.sys, "h", 1.0) / to_si(self.sys, "P", 1.0)
                factor = self.dh(w_s) / (v_d * dp_d) if v_d and dp_d else 1.0
                conv_txt = "" if abs(factor - 1) < 1e-6 else f" × {num(factor, 5)} {self.u('h')} per {self.u('v')}·{self.u('P')}"
                lines = [("law", f"(Ẇp/ṁ)s ≈ v{sub(a)}(p{sub(b)} {MINUS} p{sub(a)})   (Eq. 8.7b, a check on the table values)"),
                         ("num", f"= ({num(v_d, 4)} {self.u('v')})({num(self.d('P', self.val(b, 'P')))} {MINUS} "
                                 f"{num(self.d('P', self.val(a, 'P')))}) {self.u('P')}{conv_txt} = {num(self.dh(w_s))} {self.u('h')} (isentropic)")]
                steps.append(step("Pump work from the specific volume", *lines))
        return steps

    def valve(self, node_id, node, ports):
        a, b = ports["in"], ports["out"]
        steps = [self.mass_step([a], [b])]
        if (node.get("assumptions") or {}).get("isenthalpic"):
            lines = [("law", f"0 = Q̇cv {MINUS} Ẇcv + ṁ[(h{sub(a)} {MINUS} h{sub(b)}) + ...]   (Eq. 4.20a)"),
                     ("asm", "No work, Q̇cv = 0, Δke = Δpe = 0: a throttling process"),
                     ("eq", f"h{sub(b)} = h{sub(a)}   (Eq. 4.22)")]
            if self.val(a, "h") is not None:
                lines.append(("res", f"h{sub(b)} = h{sub(a)} = {self.q('h', self.val(a, 'h'))}"))
            steps.append(step("Energy rate balance", *lines))
        return steps

    def nozzle(self, node_id, node, ports):
        t = node.get("type")
        a, b = ports["in"], ports["out"]
        steps = [self.mass_step([a], [b])]
        h1, h2, V1, V2 = (self.val(a, "h"), self.val(b, "h"), self.val(a, "V"), self.val(b, "V"))
        lines = [("law", f"0 = Q̇cv {MINUS} Ẇcv + ṁ[(h{sub(a)} {MINUS} h{sub(b)}) + (V{sub(a)}² {MINUS} V{sub(b)}²)/2 + "
                         f"g(z{sub(a)} {MINUS} z{sub(b)})]   (Eq. 4.20a)"),
                 ("asm", "Steady state, Q̇cv = 0, Ẇcv = 0, Δpe = 0")]
        if None not in (h1, h2, V1, V2):
            # 1 kJ/kg = 1000 m²/s² (1 Btu/lbm = 25,037 ft²/s²)
            conv = to_si(self.sys, "h", 1.0) / to_si(self.sys, "V", 1.0) ** 2
            vu, hu = self.u("V"), self.u("h")
            note = ("note", f"Units: 1 {hu} = {num(conv)} {vu.replace('/s', '')}²/s²")
            if t == "nozzle":
                lines.append(("eq", f"V{sub(b)} = √[V{sub(a)}² + 2(h{sub(a)} {MINUS} h{sub(b)})]"))
                lines.append(("res", f"V{sub(b)} = √[({num(self.d('V', V1))})² + 2({num(self.d('h', h1))} {MINUS} "
                                     f"{par(self.d('h', h2))})({num(conv)})] = {self.q('V', V2)}"))
            else:
                lines.append(("eq", f"h{sub(b)} = h{sub(a)} + (V{sub(a)}² {MINUS} V{sub(b)}²)/2"))
                lines.append(("res", f"h{sub(b)} = {num(self.d('h', h1))} + [({num(self.d('V', V1))})² {MINUS} "
                                     f"({num(self.d('V', V2))})²]/(2 × {num(conv)}) = {self.q('h', h2)}"))
            lines.append(note)
        steps.append(step("Energy rate balance", *lines))
        if t == "nozzle" and None not in (h1, h2, V2) and self.val(a, "s") is not None:
            h2s = self.h2s(a, b)
            if h2s is not None:
                h0 = h1 + (V1 or 0.0) ** 2 / 2
                ke_s = h0 - h2s
                ke = V2 ** 2 / 2
                if ke_s > 0:
                    lines = [("law", f"ηnozzle = (V{sub(b)}²/2)/(V{sub(b)}s²/2)   (Eq. 6.47)")]
                    lines += self.isentropic_lines(a, b, h2s)
                    lines.append(("res", f"ηnozzle = ({num(self.dh(ke))})/({num(self.dh(ke_s))}) = {num(ke / ke_s, 4)}"
                                         f"   [V²/2 in {self.u('h')}]"))
                    steps.append(step("Nozzle efficiency", *lines))
        for key, n in (("A1", a), ("A2", b)):
            m, V = self.val(n, "m"), self.val(n, "V")
            if m and V and self.val(n, "P") is not None and self.val(n, "h") is not None:
                try:
                    vs = 1.0 / self.props("D", "P", self.val(n, "P"), "H", self.val(n, "h"), self.fluid(n))
                except Exception:
                    continue
                A = m * vs / V
                if (node.get("params") or {}).get(key) not in (None, "") or True:
                    steps.append(step(f"Flow area at state {n}",
                                      ("law", f"ṁ = AV/v   (Eq. 4.4b)"),
                                      ("res", f"A{sub(n)} = ṁv/V = ({num(self.d('m', m))})({num(self.d('v', vs), 4)})/"
                                              f"({num(self.d('V', V))}) = {self.q('A', A)}")))
        return steps

    def hx_sides(self, ports):
        from system_solver import _numbered
        return {p: (_numbered(ports, f"{p}_in"), _numbered(ports, f"{p}_out")) for p in ("a", "b")}

    def hx(self, node_id, node, ports):
        sides = self.hx_sides(ports)
        (ai, ao), (bi, bo) = sides["a"], sides["b"]
        two = bool(bi or bo)
        steps = []
        for p, (ins, outs) in sides.items():
            if ins and outs:
                s = self.mass_step(ins, outs)
                if two:
                    s["head"] = f"Mass rate balance, stream {p.upper()}"
                steps.append(s)
        H = lambda z: num(self.d("h", z))  # noqa: E731
        asm = node.get("assumptions") or {}
        if not two:
            ins, outs = ai, ao
            lines = [("law", f"0 = Q̇cv {MINUS} Ẇcv + Σṁi hi {MINUS} Σṁe he   (Eq. 4.18)"),
                     ("asm", "Steady state, Ẇcv = 0, Δke = Δpe = 0" + (", Q̇cv = 0 (adiabatic)" if asm.get("adiabatic") else ""))]
            if len(ins) == 1 and len(outs) == 1:
                a, b = ins[0], outs[0]
                h1, h2, m = self.val(a, "h"), self.val(b, "h"), self.val(a, "m")
                lines.append(("eq", f"Q̇cv = ṁ(h{sub(b)} {MINUS} h{sub(a)})"))
                if h1 is not None and h2 is not None:
                    q = h2 - h1
                    lines.append(("num", f"Q̇cv/ṁ = {H(h2)} {MINUS} {par(self.d('h', h1))} = {num(self.dh(q))} {self.u('h')}"))
                    lines.append(("res", self.rate_result(m, q, "Q̇cv") if m is not None
                                  else f"q = {num(self.dh(q))} {self.u('h')} (per kg)"))
                    lines.append(("note", "Positive: heat added to the fluid. Negative: heat leaves it."))
            else:
                lines.append(("eq", "Q̇cv = " + " + ".join(f"ṁ{sub(o)}h{sub(o)}" for o in outs) + f" {MINUS} "
                              + f" {MINUS} ".join(f"ṁ{sub(i)}h{sub(i)}" for i in ins)))
                ms = [self.val(n, "m") for n in ins + outs]
                hs = [self.val(n, "h") for n in ins + outs]
                if all(z is not None for z in ms + hs):
                    qv = sum(self.val(o, "m") * self.val(o, "h") for o in outs) - sum(self.val(i, "m") * self.val(i, "h") for i in ins)
                    terms_o = " + ".join(f"({num(self.d('m', self.val(o, 'm')))})({H(self.val(o, 'h'))})" for o in outs)
                    terms_i = f" {MINUS} ".join(f"({num(self.d('m', self.val(i, 'm')))})({H(self.val(i, 'h'))})" for i in ins)
                    lines.append(("num", f"Q̇cv = {terms_o} {MINUS} {terms_i}"))
                    lines.append(("res", f"Q̇cv = {num(self.power(qv))} {self.u('power')}"))
            steps.append(step("Energy rate balance", *lines))
            return steps

        # two streams that never mix
        lines = [("law", f"0 = Q̇cv {MINUS} Ẇcv + Σṁi hi {MINUS} Σṁe he   (Eq. 4.18)"),
                 ("asm", "Steady state, no heat lost to the surroundings (Q̇cv = 0), Ẇcv = 0, Δke = Δpe = 0")]
        if len(ai) == 1 and len(ao) == 1 and len(bi) == 1 and len(bo) == 1:
            a1, a2, b1, b2 = ai[0], ao[0], bi[0], bo[0]
            lines.append(("eq", f"0 = ṁ{sub(a1)}(h{sub(a1)} {MINUS} h{sub(a2)}) + ṁ{sub(b1)}(h{sub(b1)} {MINUS} h{sub(b2)})"))
            vals = [self.val(n, k) for n in (a1, a2, b1, b2) for k in ("m", "h")]
            if all(z is not None for z in vals):
                ta, tb = self.val(a1, "T"), self.val(b1, "T")
                hot = "a" if (ta or 0) >= (tb or 0) else "b"
                (h_in, h_out), (c_in, c_out) = ((a1, a2), (b1, b2)) if hot == "a" else ((b1, b2), (a1, a2))
                q = self.val(h_in, "m") * (self.val(h_in, "h") - self.val(h_out, "h"))
                lines.append(("num", f"Hot stream: Q̇ = ṁ{sub(h_in)}(h{sub(h_in)} {MINUS} h{sub(h_out)}) = "
                                     f"({num(self.d('m', self.val(h_in, 'm')))})({H(self.val(h_in, 'h'))} {MINUS} "
                                     f"{par(self.d('h', self.val(h_out, 'h')))}) = {num(self.power(q))} {self.u('power')}"))
                qc = self.val(c_in, "m") * (self.val(c_out, "h") - self.val(c_in, "h"))
                lines.append(("num", f"Cold stream: Q̇ = ṁ{sub(c_in)}(h{sub(c_out)} {MINUS} h{sub(c_in)}) = "
                                     f"({num(self.d('m', self.val(c_in, 'm')))})({H(self.val(c_out, 'h'))} {MINUS} "
                                     f"{par(self.d('h', self.val(c_in, 'h')))}) = {num(self.power(qc))} {self.u('power')}"))
                lines.append(("res", f"Heat passed from hot to cold: Q̇ = {num(self.power(q))} {self.u('power')}"))
                steps.append(step("Energy rate balance", *lines))
                lm = self.lmtd(node, h_in, h_out, c_in, c_out)
                if lm:
                    steps.append(lm)
                return steps
        # several inlets on a stream (a closed feedwater heater with drains cascaded in)
        flows = {p: [(n, self.val(n, "m"), self.val(n, "h")) for n in sides[p][0] + sides[p][1]] for p in ("a", "b")}
        term = lambda ns: " + ".join(f"ṁ{sub(n)}h{sub(n)}" for n in ns)  # noqa: E731
        lines.append(("eq", f"{term(ai + bi)} = {term(ao + bo)}"))
        if all(m is not None and h is not None for p in flows for _, m, h in flows[p]):
            ta, tb = self.val(ai[0], "T"), self.val(bi[0], "T")
            hot = "a" if (ta or 0) >= (tb or 0) else "b"
            for p in (hot, "b" if hot == "a" else "a"):
                ins, outs = sides[p]
                q = sum(self.val(n, "m") * self.val(n, "h") for n in ins) - sum(self.val(n, "m") * self.val(n, "h") for n in outs)
                if p != hot:
                    q = -q
                nums = lambda ns: " + ".join(f"({num(self.d('m', self.val(n, 'm')))})({H(self.val(n, 'h'))})" for n in ns)  # noqa: E731
                if p == hot:
                    txt = f"Hot stream: Q̇ = Σṁi hi {MINUS} Σṁe he = [{nums(ins)}] {MINUS} [{nums(outs)}]"
                else:
                    txt = f"Cold stream: Q̇ = Σṁe he {MINUS} Σṁi hi = [{nums(outs)}] {MINUS} [{nums(ins)}]"
                lines.append(("num", f"{txt} = {num(self.power(q))} {self.u('power')}"))
                if p == hot:
                    q_hot = q
            lines.append(("res", f"Heat passed from hot to cold: Q̇ = {num(self.power(q_hot))} {self.u('power')}"))
        steps.append(step("Energy rate balance", *lines))
        return steps

    def lmtd(self, node, hi, ho, ci, co):
        T = [self.val(n, "T") for n in (hi, ho, ci, co)]
        if any(t is None for t in T):
            return None
        layout = (node.get("config") or {}).get("layout", "counter")
        Thi, Tho, Tci, Tco = T
        if layout == "parallel":
            d1, d2 = Thi - Tci, Tho - Tco
            l1 = f"ΔT₁ = T{sub(hi)} {MINUS} T{sub(ci)}", f"ΔT₂ = T{sub(ho)} {MINUS} T{sub(co)}"
        else:
            d1, d2 = Thi - Tco, Tho - Tci
            l1 = f"ΔT₁ = T{sub(hi)} {MINUS} T{sub(co)}", f"ΔT₂ = T{sub(ho)} {MINUS} T{sub(ci)}"
        if d1 <= 0 or d2 <= 0:
            return None
        lm = d1 if abs(d1 - d2) < 1e-9 * max(d1, 1.0) else (d1 - d2) / math.log(d1 / d2)
        du = self.u("dT")
        D = lambda z: num(self.d("dT", z))  # noqa: E731
        lines = [("law", "LMTD = (ΔT₁ − ΔT₂)/ln(ΔT₁/ΔT₂)" + f"   ({layout} flow)"),
                 ("num", f"{l1[0]} = {D(d1)} {du}, {l1[1]} = {D(d2)} {du}"),
                 ("res", f"LMTD = ({D(d1)} {MINUS} {D(d2)})/ln({D(d1)}/{D(d2)}) = {D(lm)} {du}")]
        q = self.val(hi, "m") * (self.val(hi, "h") - self.val(ho, "h"))
        if lm > 0:
            ua = q / lm
            pu = self.u("power")
            lines.append(("num", f"UA = Q̇/LMTD = {num(self.power(q))} {pu} / {D(lm)} {du} = "
                                 f"{num(self.power(ua) * to_si(self.sys, 'dT', 1.0))} {pu}/{du}"))
        return step("Log mean temperature difference", *lines)

    def mixer(self, node_id, node, ports):
        from system_solver import _numbered
        ins, c = _numbered(ports, "in"), ports["out"]
        steps = [self.mass_step(ins, [c])]
        asm = node.get("assumptions") or {}
        H = lambda z: num(self.d("h", z))  # noqa: E731
        lines = [("law", f"0 = Q̇cv {MINUS} Ẇcv + Σṁi hi {MINUS} ṁe he   (Eq. 4.18)"),
                 ("asm", "Steady state, Ẇcv = 0, Δke = Δpe = 0" + (", Q̇cv = 0 (adiabatic)" if asm.get("adiabatic") else ""))]
        if asm.get("adiabatic"):
            lines.append(("eq", " + ".join(f"ṁ{sub(i)}h{sub(i)}" for i in ins) + f" = ṁ{sub(c)}h{sub(c)}"))
        hs = [self.val(n, "h") for n in ins + [c]]
        if len(ins) == 2 and all(h is not None for h in hs) and asm.get("adiabatic"):
            i1, i2 = ins
            h1, h2, h3 = hs
            if abs(h1 - h2) > 1e-6:
                y = (h3 - h2) / (h1 - h2)
                lines.append(("eq", f"Dividing by ṁ{sub(c)}, with y = ṁ{sub(i1)}/ṁ{sub(c)}: y = (h{sub(c)} {MINUS} h{sub(i2)})/"
                                    f"(h{sub(i1)} {MINUS} h{sub(i2)})"))
                lines.append(("res", f"y = ({H(h3)} {MINUS} {par(self.d('h', h2))})/({H(h1)} {MINUS} {par(self.d('h', h2))}) = {num(y, 4)}"))
        ms = [self.val(n, "m") for n in ins + [c]]
        if all(m is not None for m in ms) and all(h is not None for h in hs):
            terms = " + ".join(f"({num(self.d('m', self.val(i, 'm')))})({H(self.val(i, 'h'))})" for i in ins)
            lines.append(("num", f"{terms} = ({num(self.d('m', ms[-1]))})({H(hs[-1])})   [{self.u('m')} × {self.u('h')}]"))
            q = ms[-1] * hs[-1] - sum(m * h for m, h in zip(ms[:-1], hs[:-1]))
            if not asm.get("adiabatic"):
                lines.append(("res", f"Q̇cv = ṁ{sub(c)}h{sub(c)} {MINUS} Σṁi hi = {num(self.power(q))} {self.u('power')}"))
        steps.append(step("Energy rate balance", *lines))
        return steps

    def splitter(self, node_id, node, ports):
        from system_solver import _numbered
        a, outs = ports["in"], _numbered(ports, "out")
        steps = [self.mass_step([a], outs)]
        params = node.get("params") or {}
        asm = node.get("assumptions") or {}
        lines = []
        for i, o in enumerate(outs, 1):
            y = params.get(f"y{i}")
            if y not in (None, "") and self.val(a, "m") is not None:
                lines.append(("num", f"ṁ{sub(o)} = y{sub(i)}ṁ{sub(a)} = ({y})({num(self.d('m', self.val(a, 'm')))}) = "
                                     f"{self.q('m', self.val(o, 'm'))}"))
        if asm.get("sameState"):
            lines.append(("asm", "Every outlet leaves at the inlet state: " + ", ".join(
                f"T{sub(o)} = T{sub(a)}, h{sub(o)} = h{sub(a)}" for o in outs)))
        if lines:
            steps.append(step("Split", *lines))
        return steps

    def flash(self, node_id, node, ports):
        a, v, l = ports["in"], ports["vapor"], ports["liquid"]
        steps = [self.mass_step([a], [v, l])]
        x = self.val(a, "x")
        if x is not None and self.val(a, "m") is not None:
            m = self.val(a, "m")
            steps.append(step("Separation at one pressure",
                              ("asm", f"Saturated vapor leaves at {v} (x = 1), saturated liquid at {l} (x = 0), p constant"),
                              ("eq", f"ṁ{sub(v)} = x{sub(a)}ṁ{sub(a)}, ṁ{sub(l)} = (1 {MINUS} x{sub(a)})ṁ{sub(a)}"),
                              ("res", f"ṁ{sub(v)} = ({num(x, 4)})({num(self.d('m', m))}) = {self.q('m', x * m)}, "
                                      f"ṁ{sub(l)} = ({num(1 - x, 4)})({num(self.d('m', m))}) = {self.q('m', (1 - x) * m)}")))
        return steps

    def cooling_tower(self, node_id, node, ports):
        a, b = ports["in"], ports["out"]
        ct = getattr(self.s, "ct", {}).get(node_id, {})
        air = getattr(self.s, "air", {}).get(node_id)
        ai, ao = ports.get("airIn"), ports.get("airOut")
        if air is None and ai is not None and ao is not None:
            # air ports piped to humid-air states
            if all(self.val(n, k) is not None for n in (ai, ao) for k in ("h", "w")):
                air = [{"h": self.val(ai, "h"), "w": self.val(ai, "w")}, {"h": self.val(ao, "h"), "w": self.val(ao, "w")}]
        H = lambda z: num(self.d("h", z))  # noqa: E731
        lines = [("law", f"ṁa(ha2 {MINUS} ha1 {MINUS} (ω₂ {MINUS} ω₁)hw) = ṁ{sub(a)}(h{sub(a)} {MINUS} h{sub(b)})   "
                         "(energy rate balance on the tower, Moran Sec. 12.9)"),
                 ("asm", "Steady state, adiabatic, Ẇcv = 0, Δke = Δpe = 0; the evaporated water is made up at hw")]
        if air and ct.get("ma") is not None and None not in (self.val(a, "h"), self.val(b, "h"), self.val(a, "m")):
            (h1, w1), (h2, w2) = (air[0]["h"], air[0]["w"]), (air[1]["h"], air[1]["w"])
            hw = ct.get("hMakeup", self.val(b, "h"))
            lines.append(("num", f"ṁa = ṁ{sub(a)}(h{sub(a)} {MINUS} h{sub(b)})/(ha2 {MINUS} ha1 {MINUS} (ω₂ {MINUS} ω₁)hw) = "
                                 f"({num(self.d('m', self.val(a, 'm')))})({H(self.val(a, 'h'))} {MINUS} {H(self.val(b, 'h'))})/"
                                 f"({H(h2)} {MINUS} {H(h1)} {MINUS} ({num(w2 - w1, 4)})({H(hw)}))"))
            lines.append(("res", f"ṁa = {self.q('m', ct['ma'])} of dry air; makeup water ṁa(ω₂ {MINUS} ω₁) = {self.q('m', ct.get('mMakeup'))}"))
        return [self.mass_step([a], [b]), step("Energy rate balance", *lines)]

    def atmosphere(self, node_id, node, ports):
        n = ports.get("out", ports.get("in"))
        lines = [("eq", "The state here is the surroundings'"
                  + (f": p{sub(n)} = {self.q('P', self.val(n, 'P'))}" if self.val(n, "P") is not None else ""))]
        return [step("Surroundings", *lines)]

    def ac(self, node_id, node, ports):
        a, b = ports["in"], ports["out"]
        mode = (node.get("config") or {}).get("mode", "heating")
        H = lambda z: num(self.d("h", z))  # noqa: E731
        steps = [step("Mass rate balances", ("law", "Dry air: ṁa1 = ṁa2. Water: ṁa1ω₁ ± ṁw = ṁa2ω₂   (Moran Sec. 12.8)"),
                      ("res", f"ṁa = {self.q('m', self.val(a, 'm'))} of dry air" if self.val(a, "m") is not None else None))]
        w1, w2, h1, h2 = (self.val(a, "w"), self.val(b, "w"), self.val(a, "h"), self.val(b, "h"))
        lines = [("law", f"0 = Q̇cv + ṁa(ha1 {MINUS} ha2) ± ṁw hw, h per kg of dry air"),
                 ("asm", "Steady state, Ẇcv = 0, Δke = Δpe = 0")]
        if None not in (w1, w2):
            lines.append(("num", f"ω{sub(a)} = {num(w1, 4)}, ω{sub(b)} = {num(w2, 4)} kg water/kg dry air"
                                 + (f", so ṁw = ṁa|ω₂ {MINUS} ω₁| = {self.q('m', abs(w2 - w1) * self.val(a, 'm'))}"
                                    if self.val(a, "m") is not None and abs(w2 - w1) > 1e-9 else "")))
        if None not in (h1, h2):
            lines.append(("num", f"ha1 = {H(h1)}, ha2 = {H(h2)} {self.u('h')} (per kg of dry air)"))
        steps.append(step(f"Energy rate balance ({mode})", *lines))
        return steps

    def combustor(self, node_id, node, ports):
        rep = {}
        try:
            rep = self.s.comb.report(node_id) or {}
        except Exception:
            pass
        lines = [("law", "Σ ṁi hi = Σ ṁe he with the reactants and products as ideal-gas mixtures (Moran Ch. 13)"),
                 ("asm", "Steady state, adiabatic, Ẇcv = 0, Δke = Δpe = 0, complete combustion")]
        for k, lab in (("AF", "air-fuel ratio"), ("f", "fuel-air ratio"), ("mFuel", "fuel flow")):
            if rep.get(k) not in (None, ""):
                lines.append(("res", f"{lab}: {rep[k]}"))
        return [step("Energy rate balance", *lines)]

    # ---------- cycle ----------
    def cycle_item(self):
        from system_solver import _ports
        from combustion import is_combustor
        w_out, w_in, q_in, q_out = 0.0, 0.0, 0.0, 0.0
        rate = True
        terms_t, terms_in, terms_qin, terms_qout = [], [], [], []
        any_work = False
        for node in self.nodes.values():
            t = node.get("type")
            p = _ports(node)
            name = node.get("name") or t
            if t in ("turbine", "compressor", "pump"):
                ins = [p["in"]]
                outs = [p[k] for k in p if k == "out" or k.startswith("ext")]
                ms = [self.val(n, "m") for n in ins + outs]
                es = [None if self.val(n, "h") is None else self.val(n, "h") + self.extra(n) for n in ins + outs]
                if any(e is None for e in es):
                    return None
                if any(m is None for m in ms):
                    rate = False
                    if len(outs) > 1:
                        return None
                    w = es[0] - es[1]
                else:
                    w = ms[0] * es[0] - sum(m * e for m, e in zip(ms[1:], es[1:]))
                any_work = True
                if w >= 0:
                    w_out += w
                    terms_t.append((name, w))
                else:
                    w_in += -w
                    terms_in.append((name, -w))
            elif t == "hx" and not any(k.startswith("b_") for k in p) and not is_combustor(node):
                sides = self.hx_sides(p)
                ins, outs = sides["a"]
                ms = [self.val(n, "m") for n in ins + outs]
                hs = [self.val(n, "h") for n in ins + outs]
                if any(h is None for h in hs):
                    return None
                if any(m is None for m in ms):
                    rate = False
                    if len(ins) != 1:
                        return None
                    q = hs[len(ins)] - hs[0]
                else:
                    q = sum(self.val(o, "m") * self.val(o, "h") for o in outs) - sum(self.val(i, "m") * self.val(i, "h") for i in ins)
                if q >= 0:
                    q_in += q
                    terms_qin.append((name, q))
                else:
                    q_out += -q
                    terms_qout.append((name, -q))
        if not any_work:
            return None

        if rate:
            fmt = lambda x: f"{num(self.power(x))} {self.u('power')}"  # noqa: E731
            W, Q = "Ẇ", "Q̇"
        else:
            fmt = lambda x: f"{num(self.dh(x))} {self.u('h')}"  # noqa: E731
            W, Q = "w", "q"          # per kg: w = Ẇ/ṁ, q = Q̇/ṁ
        lines = []
        net = w_out - w_in
        parts = " + ".join(f"{fmt(w)} ({n})" for n, w in terms_t) or "0"
        if terms_in:
            parts += f" {MINUS} " + f" {MINUS} ".join(f"{fmt(w)} ({n})" for n, w in terms_in)
        lines.append(("law", f"{W}cycle = Σ{W}out {MINUS} Σ{W}in"))
        lines.append(("num", f"{W}cycle = {parts}"))
        lines.append(("res", f"{W}cycle = {fmt(net)}"))
        steps = [step("Net power" if rate else "Net work per kg (no mass flow given)", *lines)]
        if terms_qin or terms_qout:
            ql = []
            if terms_qin:
                ql.append(("num", f"{Q}in = " + " + ".join(f"{fmt(q)} ({n})" for n, q in terms_qin) + f" = {fmt(q_in)}"))
            if terms_qout:
                ql.append(("num", f"{Q}out = " + " + ".join(f"{fmt(q)} ({n})" for n, q in terms_qout) + f" = {fmt(q_out)}"))
            if terms_qin and terms_qout:
                ql.append(("note", f"Check (Eq. 2.41): {Q}in {MINUS} {Q}out = {fmt(q_in - q_out)}, against {W}cycle = {fmt(net)}"))
            steps.append(step("Heat transfer", *ql))
        if net > 0 and q_in > 0:
            steps.append(step("Thermal efficiency",
                              ("law", f"η = {W}cycle/{Q}in   (Eq. 2.42)"),
                              ("res", f"η = {fmt(net)} / {fmt(q_in)} = {num(net / q_in, 4)} ({num(100 * net / q_in, 3)} %)")))
            if w_in > 0 and w_out > 0:
                steps.append(step("Back work ratio",
                                  ("law", f"bwr = {W}in/{W}out   (Eq. 8.6)"),
                                  ("res", f"bwr = {fmt(w_in)} / {fmt(w_out)} = {num(w_in / w_out, 4)}")))
        elif net < 0:
            heating = (self.cycle.get("purpose") or "") == "heating"    # CycleSummary's pick
            if q_in > 0 and not heating:
                steps.append(step("Coefficient of performance (refrigerator)",
                                  ("law", f"β = {Q}in/{W}cycle   (Eq. 2.45; Eq. 10.7 for vapor compression)"),
                                  ("res", f"β = {fmt(q_in)} / {fmt(-net)} = {num(q_in / -net, 4)}")))
            if q_out > 0 and heating:
                steps.append(step("Coefficient of performance (heat pump)",
                                  ("law", f"γ = {Q}out/{W}cycle   (Eq. 2.47)"),
                                  ("res", f"γ = {fmt(q_out)} / {fmt(-net)} = {num(q_out / -net, 4)}")))
        return {"id": "cycle", "title": "Cycle performance", "kind": "cycle", "steps": steps}


def show_work(payload, solver):
    w = _Work(payload, solver)
    try:
        cycle = w.cycle_item()
    except Exception as e:
        cycle = None
        w.notes.append(f"The cycle summary could not be written out ({e}).")
    return {"states": w.states(), "devices": w.devices(), "cycle": cycle, "notes": w.notes,
            "source": "Moran, Shapiro, Boettner and Bailey, Fundamentals of Engineering Thermodynamics, 9th ed."}
