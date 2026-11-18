"""Air as an ideal gas with constant specific heats (the cold-air-standard
assumptions of most textbook gas-cycle problems): cp = 1.005 kJ/kg·K,
k = 1.4, so cv = cp / k and R = cp - cv.

`with_ideal_air(props)` wraps a PropsSI-style function so the fluid named
IDEAL_AIR is answered here and every other fluid goes to CoolProp as before.

    u = cv T,  h = cp T  (zero at 0 K, so h - u = R T = P v exactly)
    s = cp ln(T / T0) - R ln(P / P0),  zero at 25 °C and 1 atm
"""
import math

IDEAL_AIR = "Air (ideal gas, k = 1.4)"

CP = 1005.0
K = 1.4
CV = CP / K
R = CP - CV
T0, P0 = 298.15, 101325.0


class IdealGas:
    """An ideal gas with constant specific heats cp and R (J/kg·K)."""

    def __init__(self, cp=CP, r=R):
        self.cp, self.r = cp, r
        self.cv = cp - r

    @property
    def k(self):
        return self.cp / self.cv

    def _state(self, inputs):
        """(T, P) from any two of T, P, D (density), H, U, S."""
        cp, cv, r = self.cp, self.cv, self.r
        T = inputs.get("T")
        if T is None and "H" in inputs:
            T = inputs["H"] / cp
        if T is None and "U" in inputs:
            T = inputs["U"] / cv
        if "Q" in inputs:
            raise ValueError("an ideal gas has no two-phase states")
        if T is not None:
            if T <= 0:
                raise ValueError(f"temperature {T:g} K is out of range")
            if "P" in inputs:
                return T, inputs["P"]
            if "D" in inputs:
                return T, inputs["D"] * r * T
            if "S" in inputs:
                return T, P0 * math.exp((cp * math.log(T / T0) - inputs["S"]) / r)
            raise ValueError("needs a second property besides temperature")
        P, D, S = inputs.get("P"), inputs.get("D"), inputs.get("S")
        if P is not None and D is not None:
            return P / (D * r), P
        if P is not None and S is not None:
            return T0 * math.exp((S + r * math.log(P / P0)) / cp), P
        if D is not None and S is not None:
            T = math.exp((S + cp * math.log(T0) + r * math.log(D * r / P0)) / cv)
            return T, D * r * T
        raise ValueError("this pair of inputs does not fix the state")

    def props(self, out, k1, v1, k2, v2):
        if out in ("Tcrit", "Ttriple", "pcrit"):
            raise ValueError("an ideal gas has no saturation curve")
        T, P = self._state({k1: v1, k2: v2})
        if P <= 0:
            raise ValueError(f"pressure {P:g} Pa is out of range")
        values = {
            "T": T,
            "P": P,
            "D": P / (self.r * T),
            "H": self.cp * T,
            "U": self.cv * T,
            "S": self.cp * math.log(T / T0) - self.r * math.log(P / P0),
            "A": math.sqrt(self.k * self.r * T),      # speed of sound
        }
        if out not in values:
            raise ValueError(f"{out} is not defined for an ideal gas here")
        return values[out]


AIR = IdealGas()


# ---------------------------------------------------------------------------
# Air as an ideal gas with variable specific heats: the "air tables" of the
# textbooks (Moran Table A-22, Çengel Table A-17). cp°(T) is CoolProp's
# ideal-gas part of the Lemmon air model; h(T) and s°(T) are its integrals,
# tabulated once at 1 K steps. Pressure enters only through s = s°(T) - R ln(P/P0),
# so P v = R T holds at every pressure, as the air-standard analysis assumes.
# The zero points are set to the book's: h = 300.19 kJ/kg and s° = 1.70203
# kJ/kg·K at 300 K, so the app's h and s read like the table's.
AIR_TABLES = "Air (ideal gas, variable cp)"
R_AIR = 8314.46 / 28.97          # J/kg·K, the book's molar mass


class AirTables:
    T_MIN, T_MAX, STEP = 100.0, 4000.0, 1.0

    def __init__(self):
        self._built = False
        self.r = R_AIR

    def _build(self):
        if self._built:
            return
        from CoolProp.CoolProp import PropsSI
        n = int((self.T_MAX - self.T_MIN) / self.STEP) + 1
        self.T = [self.T_MIN + i * self.STEP for i in range(n)]
        self.cp = [PropsSI("Cp0mass", "T", t, "P", 101325.0, "Air") for t in self.T]
        h, s = [0.0], [0.0]
        for i in range(1, n):
            dt = self.T[i] - self.T[i - 1]
            h.append(h[-1] + 0.5 * (self.cp[i] + self.cp[i - 1]) * dt)
            s.append(s[-1] + 0.5 * (self.cp[i] / self.T[i] + self.cp[i - 1] / self.T[i - 1]) * dt)
        h300, s300 = self._interp(self.T, h, 300.0), self._interp(self.T, s, 300.0)
        self.h = [x - h300 + 300190.0 for x in h]
        self.s0 = [x - s300 + 1702.03 for x in s]
        self.u = [hh - self.r * t for hh, t in zip(self.h, self.T)]
        self._built = True

    @staticmethod
    def _interp(xs, ys, x):
        """Linear interpolation in increasing xs."""
        if not xs[0] <= x <= xs[-1]:
            raise ValueError(f"value {x:g} is outside the air table")
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        t = (x - xs[lo]) / (xs[hi] - xs[lo])
        return ys[lo] + t * (ys[hi] - ys[lo])

    def _at(self, ys, T):
        if not self.T_MIN <= T <= self.T_MAX:
            raise ValueError(f"temperature {T:g} K is outside the air table ({self.T_MIN:g}-{self.T_MAX:g} K)")
        return self._interp(self.T, ys, T)

    def _T_from(self, ys, y):
        return self._interp(ys, self.T, y)

    def _state(self, inputs):
        self._build()
        r = self.r
        if "Q" in inputs:
            raise ValueError("an ideal gas has no two-phase states")
        T = inputs.get("T")
        if T is None and "H" in inputs:
            T = self._T_from(self.h, inputs["H"])
        if T is None and "U" in inputs:
            T = self._T_from(self.u, inputs["U"])
        P, D, S = inputs.get("P"), inputs.get("D"), inputs.get("S")
        if T is not None:
            if P is not None:
                return T, P
            if D is not None:
                return T, D * r * T
            if S is not None:
                return T, P0 * math.exp((self._at(self.s0, T) - S) / r)
            raise ValueError("needs a second property besides temperature")
        if P is not None and D is not None:
            return P / (D * r), P
        if P is not None and S is not None:
            return self._T_from(self.s0, S + r * math.log(P / P0)), P
        if D is not None and S is not None:
            # s = s°(T) - R ln(D R T / P0) rises with T (ds/dT = cv/T): bisect
            lo, hi = self.T_MIN, self.T_MAX
            f = lambda t: self._at(self.s0, t) - r * math.log(D * r * t / P0) - S  # noqa: E731
            if f(lo) > 0 or f(hi) < 0:
                raise ValueError("this state is outside the air table")
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            T = 0.5 * (lo + hi)
            return T, D * r * T
        raise ValueError("this pair of inputs does not fix the state")

    def props(self, out, k1, v1, k2, v2):
        if out in ("Tcrit", "Ttriple", "pcrit"):
            raise ValueError("an ideal gas has no saturation curve")
        T, P = self._state({k1: v1, k2: v2})
        if P <= 0:
            raise ValueError(f"pressure {P:g} Pa is out of range")
        if out == "T":
            return T
        if out == "P":
            return P
        if out == "D":
            return P / (self.r * T)
        if out == "H":
            return self._at(self.h, T)
        if out == "U":
            return self._at(self.u, T)
        if out == "S":
            return self._at(self.s0, T) - self.r * math.log(P / P0)
        if out in ("C", "Cpmass"):
            return self._at(self.cp, T)
        if out == "A":
            cp = self._at(self.cp, T)
            return math.sqrt(cp / (cp - self.r) * self.r * T)
        raise ValueError(f"{out} is not defined for an ideal gas here")


AIR_VAR = AirTables()


def ideal_air_props(out, k1, v1, k2, v2):
    return AIR.props(out, k1, v1, k2, v2)


def with_ideal_air(props, gas=AIR):
    """props, except that IDEAL_AIR is answered by `gas` (dry air by default)
    and AIR_TABLES by the variable-cp air tables."""
    def wrapped(out, k1, v1, k2, v2, fluid):
        if fluid == IDEAL_AIR:
            return gas.props(out, k1, v1, k2, v2)
        if fluid == AIR_TABLES:
            return AIR_VAR.props(out, k1, v1, k2, v2)
        return props(out, k1, v1, k2, v2, fluid)
    wrapped.base = getattr(props, "base", props)
    return wrapped


# Water vapor as an ideal gas, for humid air (Çengel, Table A-2).
CP_VAPOR, R_VAPOR = 1872.0, 461.5


def humid_air(omega):
    """Air carrying omega kg of water vapor per kg of dry air, per kg of mixture."""
    ya, yv = 1 / (1 + omega), omega / (1 + omega)
    return IdealGas(ya * CP + yv * CP_VAPOR, ya * R + yv * R_VAPOR)
