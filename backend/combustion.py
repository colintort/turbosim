"""Combustion with real fuels: the Combustor (a one-stream heat exchanger with
a fuel), the combustion products that leave it, and those products flowing on
through the turbine, afterburner, mixer and nozzle downstream.

A combustor burns a fuel, picked from FUELS, in the air (or exhaust) that flows
through it. It needs one of
    AF     the air-fuel ratio: kg of gas in per kg of fuel
    ṁ_f    the fuel flow (with the gas flow in known)
    T_out  the exit temperature, typed on the outlet state
and gives the other two, the equivalence ratio φ and the heat the fuel
releases, Q̇ = η_b ṁ_f ΔH_R (ΔH_R = LHV for a lean mixture). Q̇ is what the
cycle summary and the propulsion panel count as Q̇_in, so η_th and TSFC come
from the fuel.

Chemistry: complete combustion, no dissociation. C burns to CO2 and H to H2O
(vapor); leftover O2 stays. A rich mixture (φ > 1) has too little O2 for that,
so all H still becomes H2O and the C splits between CO and CO2. The fuel
enters at 25 °C. η_b (combustion efficiency) is the share of the heat of
reaction released; the products keep the complete-combustion make-up.

Gases: every species is an ideal gas with cp(T) from the NASA 7-coefficient
polynomials (GRI-Mech 3.0, checked against JANAF in tests/check_combustion.py),
so the heat capacity of the products rises with temperature and with the CO2
and H2O they carry. Air is 78.08 % N2, 20.95 % O2, 0.97 % Ar by mole.

Products are a new fluid for the states downstream. Their h and s are shifted
so that at 25 °C and 1 atm they equal those of the air that came in (CoolProp
Air or the k = 1.4 ideal gas), so the state table, the T-s diagram and every
Δh read as usual across the combustor.

All math is in SI.
"""

import math

R_U = 8.314462618         # J/mol·K
T0, P0 = 298.15, 101325.0

# NASA 7-coefficient polynomials: (200-1000 K, 1000-3500 K) a1..a7, and molar mass (kg/mol).
_NASA = {
    "N2": ([3.298677, 1.4082404e-3, -3.963222e-6, 5.641515e-9, -2.444854e-12, -1020.8999, 3.950372],
           [2.92664, 1.4879768e-3, -5.68476e-7, 1.0097038e-10, -6.753351e-15, -922.7977, 5.980528]),
    "O2": ([3.78245636, -2.99673416e-3, 9.84730201e-6, -9.68129509e-9, 3.24372837e-12, -1063.94356, 3.65767573],
           [3.28253784, 1.48308754e-3, -7.57966669e-7, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129]),
    "Ar": ([2.5, 0, 0, 0, 0, -745.375, 4.366], [2.5, 0, 0, 0, 0, -745.375, 4.366]),
    "CO2": ([2.35677352, 8.98459677e-3, -7.12356269e-6, 2.45919022e-9, -1.43699548e-13, -48371.9697, 9.90105222],
            [3.85746029, 4.41437026e-3, -2.21481404e-6, 5.23490188e-10, -4.72084164e-14, -48759.166, 2.27163806]),
    "H2O": ([4.19864056, -2.0364341e-3, 6.52040211e-6, -5.48797062e-9, 1.77197817e-12, -30293.7267, -0.849032208],
            [3.03399249, 2.17691804e-3, -1.64072518e-7, -9.7041987e-11, 1.68200992e-14, -30004.2971, 4.9667701]),
    "CO": ([3.57953347, -6.1035368e-4, 1.01681433e-6, 9.07005884e-10, -9.04424499e-13, -14344.086, 3.50840928],
           [2.71518561, 2.06252743e-3, -9.98825771e-7, 2.30053008e-10, -2.03647716e-14, -14151.8724, 7.81868772]),
}
MOLAR_MASS = {"N2": 0.0280134, "O2": 0.0319988, "Ar": 0.039948, "CO2": 0.0440095, "H2O": 0.0180153, "CO": 0.0280101}
SPECIES = tuple(_NASA)
T_MIN, T_MAX = 200.0, 4000.0
_C, _H, _O = 0.0120107, 0.00100794, 0.0159994

AIR_MOLES = {"N2": 0.7808, "O2": 0.2095, "Ar": 0.0097}

# Fuels offered in the Combustor. formula (C, H, O) per molecule; LHV in kJ/kg
# with the water as vapor (Çengel, Table A-27; jet fuel as in his turbojet examples).
FUELS = {
    "Jet fuel (Jet-A)": {"C": 12, "H": 23, "O": 0, "LHV": 42_800},
    "Natural gas (methane)": {"C": 1, "H": 4, "O": 0, "LHV": 50_050},
    "Propane": {"C": 3, "H": 8, "O": 0, "LHV": 46_340},
    "Gasoline (octane)": {"C": 8, "H": 18, "O": 0, "LHV": 44_430},
    "Diesel (dodecane)": {"C": 12, "H": 26, "O": 0, "LHV": 44_150},
    "Hydrogen": {"C": 0, "H": 2, "O": 0, "LHV": 120_000},
    "Ethanol": {"C": 2, "H": 6, "O": 1, "LHV": 26_810},
    "Methanol": {"C": 1, "H": 4, "O": 1, "LHV": 19_920},
}
DEFAULT_FUEL = "Jet fuel (Jet-A)"


class CombustionError(Exception):
    pass


# ---------- species ----------

def _coef(sp, T):
    return _NASA[sp][0] if T < 1000.0 else _NASA[sp][1]


def _h_molar(sp, T):
    """Absolute enthalpy, J/mol, formation included."""
    a = _coef(sp, T)
    return R_U * T * (a[0] + a[1] * T / 2 + a[2] * T ** 2 / 3 + a[3] * T ** 3 / 4 + a[4] * T ** 4 / 5 + a[5] / T)


def _cp_molar(sp, T):
    a = _coef(sp, T)
    return R_U * (a[0] + a[1] * T + a[2] * T ** 2 + a[3] * T ** 3 + a[4] * T ** 4)


def _s_molar(sp, T):
    """Standard-state entropy at 1 atm, J/mol·K."""
    a = _coef(sp, T)
    return R_U * (a[0] * math.log(T) + a[1] * T + a[2] * T ** 2 / 2 + a[3] * T ** 3 / 3 + a[4] * T ** 4 / 4 + a[6])


def fuel_molar_mass(fuel):
    return fuel["C"] * _C + fuel["H"] * _H + fuel["O"] * _O


def fuel_enthalpy(fuel):
    """Enthalpy of formation of the fuel at 25 °C, J/kg, chosen so that burning
    it completely to CO2 and H2O vapor releases exactly its LHV."""
    m = fuel_molar_mass(fuel)
    products = fuel["C"] * _h_molar("CO2", T0) + fuel["H"] / 2 * _h_molar("H2O", T0)
    return (products + fuel["LHV"] * 1e3 * m) / m


def get_fuel(name):
    name = name or DEFAULT_FUEL
    if name not in FUELS:
        raise CombustionError(f"'{name}' is not one of the fuels: {', '.join(FUELS)}.")
    return name, FUELS[name]


# ---------- gas mixtures ----------

class GasMixture:
    """An ideal-gas mixture, by mass fraction y. h and s are on the display
    basis: equal to (h_ref, s_ref) at 25 °C and 1 atm. fuel_frac is the mass of
    fuel burned per kg of mixture so far (0 for air)."""

    def __init__(self, y, name="Air", h_ref=0.0, s_ref=0.0, fuel=None, fuel_frac=0.0):
        total = sum(y.values())
        self.y = {k: v / total for k, v in y.items() if v > 0}
        self.name, self.fuel, self.fuel_frac = name, fuel, fuel_frac
        self.n = {k: v / MOLAR_MASS[k] for k, v in self.y.items()}          # mol per kg
        self.r = R_U * sum(self.n.values())
        n_tot = sum(self.n.values())
        mixing = -R_U * sum(n * math.log(n / n_tot) for n in self.n.values())
        self._mix_s = mixing
        self.h_ref, self.s_ref = h_ref, s_ref
        self._h0, self._s0 = self.h_abs(T0), self.s_abs(T0, P0)

    @classmethod
    def from_moles(cls, moles, **kw):
        return cls({k: n * MOLAR_MASS[k] for k, n in moles.items()}, **kw)

    def rebased(self, h_ref, s_ref, name=None):
        return GasMixture(self.y, name or self.name, h_ref, s_ref, self.fuel, self.fuel_frac)

    # absolute basis (formation enthalpies included), per kg
    def h_abs(self, T):
        return sum(n * _h_molar(k, T) for k, n in self.n.items())

    def cp(self, T):
        return sum(n * _cp_molar(k, T) for k, n in self.n.items())

    def s_abs(self, T, P):
        return sum(n * _s_molar(k, T) for k, n in self.n.items()) + self._mix_s - self.r * math.log(P / P0)

    def T_from_h_abs(self, h):
        return self._invert(lambda T: self.h_abs(T) - h, self.cp)

    # display basis
    def h(self, T):
        return self.h_abs(T) - self._h0 + self.h_ref

    def s(self, T, P):
        return self.s_abs(T, P) - self._s0 + self.s_ref

    def to_abs(self, h_display):
        return h_display - self.h_ref + self._h0

    def _invert(self, f, dfdT, T=1000.0):
        for _ in range(60):
            step = f(T) / dfdT(T)
            T = min(max(T - step, T_MIN * 0.5), T_MAX * 1.5)
            if abs(step) < 1e-9 * T:
                break
        if not T_MIN - 1 <= T <= T_MAX:
            raise ValueError(f"temperature {T:.0f} K is out of range for {self.name} (200 to 4000 K)")
        return T

    def _T_from_s(self, s, P):
        return self._invert(lambda T: self.s(T, P) - s, lambda T: self.cp(T) / T)

    def _state(self, inputs):
        """(T, P) from any two of T, P, D, H, U, S."""
        if "Q" in inputs:
            raise ValueError("combustion products are an ideal gas here, with no two-phase states")
        T = inputs.get("T")
        if T is None and "H" in inputs:
            T = self.T_from_h_abs(self.to_abs(inputs["H"]))
        if T is None and "U" in inputs:
            u = inputs["U"]
            T = self._invert(lambda t: self.h(t) - self.r * t - u, lambda t: self.cp(t) - self.r)
        P, D, S = inputs.get("P"), inputs.get("D"), inputs.get("S")
        if T is not None:
            if P is not None:
                return T, P
            if D is not None:
                return T, D * self.r * T
            if S is not None:
                # s(T, P) = s(T, P0) - R ln(P / P0)
                return T, P0 * math.exp((self.s(T, P0) - S) / self.r)
            raise ValueError("needs a second property besides temperature")
        if P is not None and S is not None:
            return self._T_from_s(S, P), P
        if P is not None and D is not None:
            return P / (D * self.r), P
        if D is not None and S is not None:
            T = self._invert(lambda t: self.s(t, D * self.r * t) - S, lambda t: (self.cp(t) - self.r) / t)
            return T, D * self.r * T
        raise ValueError("this pair of inputs does not fix the state")

    def props(self, out, k1, v1, k2, v2):
        if out in ("Tcrit", "Ttriple", "pcrit", "ptriple"):
            raise ValueError("an ideal gas has no saturation curve")
        T, P = self._state({k1: v1, k2: v2})
        if P <= 0:
            raise ValueError(f"pressure {P:g} Pa is out of range")
        cp = self.cp(T)
        values = {
            "T": T, "P": P,
            "D": P / (self.r * T),
            "H": self.h(T),
            "U": self.h(T) - self.r * T,
            "S": self.s(T, P),
            "C": cp, "CPMASS": cp, "CVMASS": cp - self.r,
            "A": math.sqrt(cp / (cp - self.r) * self.r * T),
        }
        if out not in values:
            raise ValueError(f"{out} is not defined for combustion products here")
        return values[out]


AIR = GasMixture.from_moles(AIR_MOLES)


def _elements(gas, fuel, f):
    """Moles of C, H, O, N, Ar in 1 kg of gas plus f kg of fuel."""
    nf = f / fuel_molar_mass(fuel)
    n = gas.n
    return {
        "C": n.get("CO2", 0) + n.get("CO", 0) + fuel["C"] * nf,
        "H": 2 * n.get("H2O", 0) + fuel["H"] * nf,
        "O": 2 * n.get("O2", 0) + 2 * n.get("CO2", 0) + n.get("CO", 0) + n.get("H2O", 0) + fuel["O"] * nf,
        "N": 2 * n.get("N2", 0),
        "Ar": n.get("Ar", 0),
    }


def stoich_far(gas, fuel):
    """kg of fuel per kg of gas that uses up all its O2."""
    nf_per_kg = 1 / fuel_molar_mass(fuel)
    need = fuel["C"] * 2 + fuel["H"] / 2 - fuel["O"]          # O atoms per molecule of fuel
    e = _elements(gas, fuel, 0.0)
    spare_o = e["O"] - 2 * e["C"] - e["H"] / 2                  # O not yet bound in CO2/H2O
    if need <= 0:
        return math.inf
    return spare_o / (need * nf_per_kg)


def burn(gas, fuel, f):
    """Moles of product species per (1 + f) kg, complete combustion of 1 kg of
    gas with f kg of fuel. Rich: H to H2O first, C split between CO and CO2."""
    e = _elements(gas, fuel, f)
    C, H, O = e["C"], e["H"], e["O"]
    out = {"N2": e["N"] / 2, "Ar": e["Ar"], "H2O": H / 2}
    if O >= 2 * C + H / 2 - 1e-12:
        out.update(CO2=C, O2=(O - 2 * C - H / 2) / 2)
    else:
        co2 = O - C - H / 2
        if co2 < -1e-12:
            raise CombustionError("the mixture is too rich to burn (not enough oxygen even for CO). Use more air.")
        out.update(CO2=max(co2, 0.0), CO=C - max(co2, 0.0))
    return {k: v for k, v in out.items() if v > 1e-15}


def _products(gas, fuel_name, fuel, f):
    frac = (gas.fuel_frac + f) / (1 + f)            # fuel burned per kg of products
    return GasMixture.from_moles(burn(gas, fuel, f), h_ref=gas.h_ref, s_ref=gas.s_ref,
                                 fuel=fuel_name if gas.fuel in (None, fuel_name) else "mixed",
                                 fuel_frac=frac)


def heat_of_reaction(gas, fuel, f):
    """J per kg of fuel released at 25 °C (the LHV for a lean mixture)."""
    if f <= 0:
        return fuel["LHV"] * 1e3
    p = _products(gas, "", fuel, f)
    return (gas.h_abs(T0) + f * fuel_enthalpy(fuel) - (1 + f) * p.h_abs(T0)) / f


def exit_state(gas, fuel_name, f, T_in, eta_b=1.0, ke_in=0.0, ke_out=0.0):
    """(products, T_out) for f kg of fuel per kg of gas entering at T_in."""
    fuel = FUELS[fuel_name]
    p = _products(gas, fuel_name, fuel, f)
    h_in = gas.h_abs(T_in) + ke_in + f * fuel_enthalpy(fuel) - (1 - eta_b) * f * heat_of_reaction(gas, fuel, f)
    return p, p.T_from_h_abs(h_in / (1 + f) - ke_out)


def far_for_exit(gas, fuel_name, T_in, T_out, eta_b=1.0, ke_in=0.0, ke_out=0.0):
    """Fuel-air ratio (lean side) that brings gas from T_in to T_out."""
    fuel = FUELS[fuel_name]
    if T_out <= T_in + 1e-9:
        if T_out < T_in - 1e-6:
            raise CombustionError(
                f"the exit ({T_out - 273.15:.1f} °C) is colder than the inlet ({T_in - 273.15:.1f} °C), "
                "and burning fuel can only heat the gas.")
        return 0.0
    f_st = stoich_far(gas, fuel)
    if f_st <= 0:
        raise CombustionError("the gas coming in has no oxygen left to burn fuel with.")

    def excess(f):        # > 0 while the exit would be hotter than wanted
        return exit_state(gas, fuel_name, f, T_in, eta_b, ke_in, ke_out)[1] - T_out

    hi = f_st if math.isfinite(f_st) else 1.0
    if excess(hi) < 0:
        T_ad = exit_state(gas, fuel_name, hi, T_in, eta_b, ke_in, ke_out)[1]
        raise CombustionError(
            f"{T_out - 273.15:.0f} °C is hotter than {fuel_name} can reach with this inlet "
            f"(at most {T_ad - 273.15:.0f} °C, burning with exactly the theoretical air).")
    lo = 0.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if excess(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2


def with_products(props, registry):
    """props, except that fluids named in registry (combustion products) are
    answered by their GasMixture."""
    def wrapped(out, k1, v1, k2, v2, fluid):
        mix = registry.get(fluid)
        if mix is not None:
            return mix.props(out, k1, v1, k2, v2)
        return props(out, k1, v1, k2, v2, fluid)
    wrapped.base = getattr(props, "base", props)
    return wrapped


# ---------- the solver's side ----------

AIR_FLUIDS = ("Air",)   # plus ideal_gas.IDEAL_AIR, added below


def is_combustor(node):
    return node.get("type") == "hx" and bool((node.get("config") or {}).get("combustor"))


def _num(x):
    if x is None or (isinstance(x, str) and x.strip() == ""):
        return None
    return float(x)


class Combustion:
    """Everything the open-system solver needs for combustors. One per solve.

    waiting(n)  state n is downstream of a combustor whose products aren't
                known yet: neither it nor any device touching it can be solved
    spread()    carries products on through every device that passes its
                fluid straight through (not mixers or combustors)
    combustor() and mixer() are the device rules."""

    def __init__(self, solver):
        from ideal_gas import IDEAL_AIR, AIR_TABLES
        self.s = solver
        self.air_fluids = AIR_FLUIDS + (IDEAL_AIR, AIR_TABLES)
        self.registry = {}           # fluid name -> GasMixture
        self.fluid = {}              # state -> products fluid name
        self.res = {}                # combustor id -> {f, phi, fuel, q (J per kg gas), ...}
        self.combustors = {i: n for i, n in solver.nodes.items() if is_combustor(n)}
        self.active = bool(self.combustors)
        self.pending = set()
        if not self.active:
            return
        solver.props = with_products(solver.props, self.registry)
        self.pending = self._downstream()
        for node_id, node in self.combustors.items():
            if any(i in self._reach(self._sides(node)[1]) for i in self._sides(node)[0]):
                name = node.get("name") or "Combustor"
                raise _solve_error(
                    f"{name}: a combustor burns fuel in air, so it can't sit in a closed loop (its products "
                    "would come back round to it). For a closed Brayton cycle use a Heater (Q̇ added) instead.")

    # ---- graph ----
    @staticmethod
    def _sides(node):
        from system_solver import _ports, _numbered
        p = _ports(node)
        return _numbered(p, "a_in"), _numbered(p, "a_out")

    def _pairs(self):
        from system_solver import _device_pairs
        for node_id, node in self.s.nodes.items():
            if node_id in self.combustors:
                continue
            for a, b in _device_pairs(node):
                yield node_id, node, a, b

    def _reach(self, starts):
        seen, stack = set(), list(starts)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            for node_id, node, a, b in self._pairs():
                if a == n:
                    stack.append(b)
            for node in self.combustors.values():
                ins, outs = self._sides(node)
                if n in ins:
                    stack.extend(outs)
        return seen

    def _downstream(self):
        outs = [o for node in self.combustors.values() for o in self._sides(node)[1]]
        return self._reach(outs)

    # ---- fluids ----
    def waiting(self, n):
        return n in self.pending and n not in self.fluid

    def blocks(self, node):
        from system_solver import _ports
        return any(self.waiting(n) for n in _ports(node).values())

    def mixture(self, n):
        """The GasMixture at state n (air for an air stream), or None."""
        if n in self.fluid:
            return self.registry[self.fluid[n]]
        base = self.s.state_fluids.get(n, self.s.fluid)
        if base in self.air_fluids:
            return self._air(base)
        return None

    def _air(self, base):
        props = self.s.props
        h_ref = props("H", "T", T0, "P", P0, base)
        s_ref = props("S", "T", T0, "P", P0, base)
        return AIR.rebased(h_ref, s_ref)

    def _register(self, n, mix, where):
        if n in self.fluid:
            return False
        name = mix.name
        if mix.fuel and mix.fuel_frac > 0:
            af = (1 - mix.fuel_frac) / mix.fuel_frac
            fuel = "Mixed fuel" if mix.fuel == "mixed" else mix.fuel.split(" (")[0]
            name = f"{fuel} products (AF {af:.4g})"
        if name in self.registry and self.registry[name].y != mix.y:
            name = f"{name} at S{n}"
        mix.name = name
        self.registry.setdefault(name, mix)
        self.fluid[n] = name
        return True

    def spread(self):
        changed = False
        for node_id, node, a, b in self._pairs():
            if node.get("type") == "mixer":
                continue
            if a in self.fluid and b in self.pending and b not in self.fluid:
                self.fluid[b] = self.fluid[a]
                changed = True
        return changed

    # ---- device rules ----
    def combustor(self, node_id, node, name, ports):
        s = self.s
        ins, outs = self._sides(node)
        if len(ins) != 1 or len(outs) != 1:
            raise _solve_error(f"{name}: a combustor has one inlet (air) and one outlet (products).")
        a, b = ins[0], outs[0]
        params = node.get("params") or {}
        asm = node.get("assumptions") or {}
        try:
            fuel_name, fuel = get_fuel(params.get("fuel"))
        except CombustionError as e:
            raise _solve_error(f"{name}: {e}")
        from system_solver import _efficiency, _number
        from units import to_si
        eta_b = _efficiency(params.get("etaB"), f"{name} combustion") or 1.0
        AF = _number(params.get("AF"), f"{name} air-fuel ratio")
        mf = _number(params.get("mf"), f"{name} fuel flow")
        if AF is not None and AF <= 0:
            raise _solve_error(f"{name}: the air-fuel ratio must be positive.")
        if mf is not None:
            mf = to_si(s.system, "m", mf)
            if mf < 0:
                raise _solve_error(f"{name}: the fuel flow can't be negative.")

        changed = False
        if asm.get("aIsobaric"):
            changed |= s._propagate(a, b, "P", name)
        if self.waiting(a):              # an afterburner waits for the products upstream
            return changed
        gas = self.mixture(a)
        if gas is None:
            raise _solve_error(
                f"{name}: fuel needs air to burn in, but S{a} is {s._fluid(a)}. "
                "Set the stream to Air, or use a Heater (Q̇ added) instead of a combustor.")
        va, vb = s._v(a), s._v(b)
        ke_a, ke_b = s._extra(a) or 0.0, s._extra(b) or 0.0
        r = self.res.setdefault(node_id, {"fuel": fuel_name})

        f = r.get("f")
        if f is None:
            if AF is not None:
                f = 1 / AF
            elif mf is not None and va.get("m") is not None:
                f = mf / va["m"] if va["m"] > 0 else 0.0
            elif "T" in va and "T" in vb:
                try:
                    f = far_for_exit(gas, fuel_name, va["T"], vb["T"], eta_b, ke_a, ke_b)
                except CombustionError as e:
                    raise _solve_error(f"{name}: {e}")
            if f is not None:
                r["f"] = f
                changed = True
        if f is None:
            return changed

        if f > 0 and AF is not None and mf is not None and va.get("m") is not None:
            if not math.isclose(mf, f * va["m"], rel_tol=1e-4):
                raise _solve_error(f"{name}: the fuel flow and the air-fuel ratio disagree with ṁ at S{a}. Give one of them.")
        if mf is not None and f > 0:
            changed |= s._linear([(a, 1)], "m", name, "fuel flow", -mf / f)
        changed |= s._linear([(b, 1), (a, -(1 + f))], "m", name, "mass")

        try:
            products = _products(gas, fuel_name, fuel, f)
        except CombustionError as e:
            raise _solve_error(f"{name}: {e}")
        changed |= self._register(b, products, name)
        r["phi"] = f / stoich_far(gas, fuel) if math.isfinite(stoich_far(gas, fuel)) else 0.0
        r["dHR"] = heat_of_reaction(gas, fuel, f) if f > 0 else fuel["LHV"] * 1e3
        r["eta_b"] = eta_b

        if "T" in va:
            try:
                _, T_out = exit_state(gas, fuel_name, f, va["T"], eta_b, ke_a, ke_b)
            except ValueError as e:
                raise _solve_error(f"{name}: {e}")
            changed |= s._set(b, "T", T_out, name)
        return changed

    def mixes_products(self, node):
        from system_solver import _numbered, _ports
        ins = _numbered(_ports(node), "in")
        return any(i in self.pending or i in self.fluid for i in ins)

    def mixer(self, node, name, ports):
        """Products mixing with air (a mixed-exhaust turbofan) or with other
        products. Mass, composition and energy on one basis; Q̇ as in the
        plain mixer."""
        from system_solver import _numbered, _number
        from units import power_factor
        s = self.s
        ins, c = _numbered(ports, "in"), ports["out"]
        asm = node.get("assumptions") or {}
        params = node.get("params") or {}
        changed = False
        if asm.get("isobaric"):
            for i in ins:
                changed |= s._propagate(i, c, "P", name)
        changed |= s._linear([(i, 1) for i in ins] + [(c, -1)], "m", name, "mass")
        if any(self.waiting(i) for i in ins):
            return changed
        gases = [self.mixture(i) for i in ins]
        if any(g is None for g in gases):
            bad = next(i for i, g in zip(ins, gases) if g is None)
            raise _solve_error(f"{name}: combustion products can only mix with air here, but S{bad} is {s._fluid(bad)}.")
        ms = [s._v(i).get("m") for i in ins]
        if any(m is None for m in ms):
            return changed
        total = sum(ms)
        y = {}
        for g, m in zip(gases, ms):
            for k, v in g.y.items():
                y[k] = y.get(k, 0.0) + v * m / total
        fuels = {g.fuel for g in gases if g.fuel}
        frac = sum(g.fuel_frac * m for g, m in zip(gases, ms)) / total
        ref = next(g for g in gases if g.fuel)
        mix = GasMixture(y, h_ref=ref.h_ref, s_ref=ref.s_ref,
                         fuel=fuels.pop() if len(fuels) == 1 else "mixed", fuel_frac=frac)
        changed |= self._register(c, mix, name)
        if any("T" not in s._v(i) for i in ins):
            return changed
        q = 0.0
        if not asm.get("adiabatic"):
            qd = _number(params.get("Qdot"), f"{name} Q̇")
            if qd is None:
                return changed
            q = qd * power_factor(s.system)
        H = sum(m * (g.h_abs(s._v(i)["T"]) + (s._extra(i) or 0.0)) for g, m, i in zip(gases, ms, ins)) + q
        try:
            T = mix.T_from_h_abs(H / total - (s._extra(c) or 0.0))
        except ValueError as e:
            raise _solve_error(f"{name}: {e}")
        changed |= s._set(c, "T", T, name)
        return changed

    # ---- reporting ----
    def report(self, node_id):
        """Device results for a combustor, in display units."""
        from system_solver import _round
        from units import from_si, power_factor
        s = self.s
        r = self.res.get(node_id)
        if not r or r.get("f") is None:
            return {}
        a = self._sides(self.combustors[node_id])[0][0]
        f = r["f"]
        out = {"far": _round(f), "phi": _round(r["phi"])}
        if f > 0:
            out["AF"] = _round(1 / f)
        m = s._v(a).get("m")
        if m is not None:
            out["mf"] = _round(from_si(s.system, "m", f * m))
            out["Qdot"] = _round(r["eta_b"] * f * m * r["dHR"] / power_factor(s.system))
        out["q"] = _round(from_si(s.system, "h", r["eta_b"] * f * r["dHR"]))     # per kg of gas in
        return out

    def report_states(self):
        """{state: {name, fuel, y}} for every state carrying products."""
        out = {}
        for n, name in sorted(self.fluid.items()):
            mix = self.registry[name]
            out[str(n)] = {"name": name, "fuel": mix.fuel, "fuelFrac": mix.fuel_frac,
                           "y": mix.y, "hRef": mix.h_ref, "sRef": mix.s_ref}
        return out

    def chart(self, n):
        """Which fluid's T-s / P-h chart state n goes on: products share the
        chart of the air they came from (same basis at 25 °C)."""
        return self.s.state_fluids.get(n, self.s.fluid) if n in self.fluid else self.s._fluid(n)

    def path(self, a, b, samples=24):
        """Process line through a combustor (air in, products out) at the
        outlet pressure: T from T_a to T_b, s and h spread as the products'
        ∫cp dT / T and ∫cp dT."""
        s = self.s
        va, vb = s._v(a), s._v(b)
        if not all(k in v for v in (va, vb) for k in ("s", "T", "h", "P")):
            return None
        mix = self.mixture(b)
        if mix is None:
            return None
        Ta, Tb = va["T"], vb["T"]
        if math.isclose(Ta, Tb):
            return [[va[k] for k in ("s", "T", "h", "P")], [vb[k] for k in ("s", "T", "h", "P")]]
        ts = [Ta + (Tb - Ta) * i / samples for i in range(samples + 1)]
        ds = [0.0]
        dh = [0.0]
        for t0, t1 in zip(ts, ts[1:]):
            tm = (t0 + t1) / 2
            ds.append(ds[-1] + mix.cp(tm) * math.log(t1 / t0))
            dh.append(dh[-1] + mix.cp(tm) * (t1 - t0))
        pts = []
        for t, x, y in zip(ts, ds, dh):
            w_s = x / ds[-1] if ds[-1] else 0.0
            w_h = y / dh[-1] if dh[-1] else 0.0
            P = va["P"] + (vb["P"] - va["P"]) * w_h
            pts.append([va["s"] + (vb["s"] - va["s"]) * w_s, t, va["h"] + (vb["h"] - va["h"]) * w_h, P])
        return pts


# ---------- for code that reads a solve result (propulsion.py) ----------

def products_props(props, result):
    """(props that also knows the products in result, {state: products fluid name})."""
    registry, fluid = {}, {}
    for n, p in (result.get("products") or {}).items():
        if p["name"] not in registry:
            registry[p["name"]] = GasMixture(p["y"], p["name"], p["hRef"], p["sRef"], p["fuel"], p["fuelFrac"])
        fluid[int(n)] = p["name"]
    return with_products(props, registry), fluid


def fuel_flows(nodes, devices, system):
    """{combustor id: fuel flow in kg/s} from a solve result, for TSFC."""
    from units import to_si
    out = {}
    for node_id, node in nodes.items():
        mf = (devices.get(node_id) or {}).get("mf")
        if is_combustor(node) and mf is not None:
            out[node_id] = to_si(system, "m", mf)
    return out


def _solve_error(msg):
    from system_solver import SolveError
    return SolveError(msg)
