"""Textbook validation: solves worked examples from Colin's course books with
the app's own solver (solve_system, the code behind the Solve button) and
compares the answers with the book's.

    python validate_textbook.py            prints a pass/fail table
    python validate_textbook.py --md FILE  also writes the table as Markdown

Run it from the backend folder, or use validate.bat in the ThermoAnal folder.
build_exe_<version>.bat runs it first and stops if any check fails.

Books
  Moran      Moran, Shapiro, Boettner, Bailey, Fundamentals of Engineering
             Thermodynamics, 9th ed. (Wiley). Chapters 8 to 12.
  Penoncello Penoncello, Thermal Energy Systems: Design and Analysis.
  Dick       Dick, Fundamentals of Turbomachines (Springer).

Why answers differ a little from the book: the book reads steam, R-134a and
air tables (or the psychrometric chart) and rounds as it goes; the app uses
CoolProp's reference equations of state. The tolerances below allow for that
(typically 1 to 2 %), and are kept tight enough that a real modelling error
(a wrong sign, a missing term, a wrong state) still fails.

English-unit examples are converted to SI here; the book values are converted
the same way, so nothing is re-derived.
"""
import argparse
import math
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from system_solver import solve_system  # noqa: E402
from pressure_optimizer import cycle_totals  # noqa: E402
from ideal_gas import IDEAL_AIR, AIR_TABLES  # noqa: E402

HA = "Humid air"
ISEN = {"eta100": True, "isentropic": True, "adiabatic": True}
TON = 3.516852842           # kW per ton of refrigeration
R_TO_K = 5 / 9
PSI = 6.894757              # kPa per lbf/in²
ATM = 101.325
FT = 0.3048
LB = 0.45359237
BTU_LB = 2.326              # kJ/kg per Btu/lb


# ---------------------------------------------------------------- helpers
def node(t, ports, assumptions=None, params=None, name=None, config=None):
    n = {"type": t, "name": name or t, "ports": ports,
         "assumptions": assumptions or {}, "params": params or {}}
    if config:
        n["config"] = config
    return n


def solve(states, nodes, fluid="Water", fluids=None, cycle=None, mode=None):
    payload = {"fluid": fluid, "unitSystem": "SI", "states": states, "nodes": nodes}
    if fluids:
        payload["stateFluids"] = fluids
    if cycle:
        payload["cycle"] = cycle
    if mode:
        payload["mode"] = mode
    r = solve_system(payload, lean=True)     # the numbers only: the checks never read the charts
    if r.get("errors"):
        raise RuntimeError("the solver reported: " + "; ".join(r["errors"]))
    r["_payload"] = payload
    return r


def totals(r):
    t = cycle_totals(r["_payload"], r)
    if t is None:
        raise RuntimeError("the cycle summary could not be formed (a device is unsolved)")
    return t


def f(x):
    if x in ("", None):
        raise RuntimeError("a value the check needs was left unsolved")
    return float(x)


def K(t_c):
    return f(t_c) + 273.15


class Check:
    """One compared quantity: book value, app value, tolerance (relative,
    or absolute when abs_tol is given)."""

    def __init__(self, what, app, book, unit="", rel=0.015, abs_tol=None):
        self.what, self.app, self.book, self.unit = what, app, book, unit
        self.rel, self.abs_tol = rel, abs_tol

    @property
    def ok(self):
        if self.app is None or (isinstance(self.app, float) and math.isnan(self.app)):
            return False
        err = abs(self.app - self.book)
        return err <= self.abs_tol if self.abs_tol is not None else err <= self.rel * abs(self.book)

    def tol_text(self):
        return f"±{self.abs_tol:g}" if self.abs_tol is not None else f"±{self.rel * 100:g}%"


CASES = []


def case(book, example, title):
    def deco(fn):
        CASES.append((book, example, title, fn))
        return fn
    return deco


# ================================================================ Moran Ch. 8: vapour power
def rankine(eta_t=None, eta_p=None, reheat=None, net=100000):
    """Moran's 8.0 MPa / 0.008 MPa plant. reheat = (P2, T3) for Ex. 8.3/8.4."""
    turb_a = ISEN if eta_t is None else {"adiabatic": True}
    pump_a = ISEN if eta_p is None else {"adiabatic": True}
    tp = {} if eta_t is None else {"eta": eta_t}
    pp = {} if eta_p is None else {"eta": eta_p}
    if reheat is None:
        states = {"1": {"P": 8000, "x": 1}, "2": {"P": 8}, "3": {"x": 0}}
        nodes = {
            "t": node("turbine", {"in": 1, "out": 2}, turb_a, tp, "Turbine"),
            "c": node("hx", {"a_in1": 2, "a_out1": 3}, {"aIsobaric": True}, name="Condenser"),
            "p": node("pump", {"in": 3, "out": 4}, pump_a, pp, "Pump"),
            "b": node("hx", {"a_in1": 4, "a_out1": 1}, {"aIsobaric": True}, name="Boiler"),
        }
    else:
        p_rh, t_rh = reheat
        states = {"1": {"P": 8000, "T": 480}, "2": {"P": p_rh}, "3": {"T": t_rh}, "4": {"P": 8}, "5": {"x": 0}}
        nodes = {
            "t1": node("turbine", {"in": 1, "out": 2}, turb_a, tp, "HP turbine"),
            "rh": node("hx", {"a_in1": 2, "a_out1": 3}, {"aIsobaric": True}, name="Reheater"),
            "t2": node("turbine", {"in": 3, "out": 4}, turb_a, tp, "LP turbine"),
            "c": node("hx", {"a_in1": 4, "a_out1": 5}, {"aIsobaric": True}, name="Condenser"),
            "p": node("pump", {"in": 5, "out": 6}, pump_a, pp, "Pump"),
            "b": node("hx", {"a_in1": 6, "a_out1": 1}, {"aIsobaric": True}, name="Boiler"),
        }
    return solve(states, nodes, cycle={"netPower": net})


def mass_flow(r, n=1):
    return f(r["states"][str(n)]["m"])


@case("Moran", "8.1", "Ideal Rankine cycle, 8 MPa sat. vapour to 8 kPa, 100 MW")
def moran_8_1():
    r = rankine()
    t = totals(r)
    return [
        Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.371, abs_tol=0.003),
        Check("Back work ratio", t["Win"] / t["Wout"], 0.0084, abs_tol=0.0003),
        Check("Steam flow", mass_flow(r) * 3600, 3.77e5, "kg/h"),
        Check("Boiler heat input", t["Qin"] / 1000, 269.8, "MW"),
        Check("Condenser heat rejected", t["Qout"] / 1000, 169.8, "MW"),
    ]


@case("Moran", "8.2", "Rankine cycle with 85 % turbine and pump")
def moran_8_2():
    r = rankine(eta_t=0.85, eta_p=0.85)
    t = totals(r)
    return [
        Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.314, abs_tol=0.003),
        Check("Steam flow", mass_flow(r) * 3600, 4.449e5, "kg/h"),
        Check("Boiler heat input", t["Qin"] / 1000, 318.2, "MW"),
        Check("Condenser heat rejected", t["Qout"] / 1000, 218.2, "MW"),
    ]


@case("Moran", "8.3", "Ideal reheat cycle, 8 MPa 480 °C, reheat at 0.7 MPa to 440 °C")
def moran_8_3():
    r = rankine(reheat=(700, 440))
    t = totals(r)
    return [
        Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.403, abs_tol=0.003),
        Check("Steam flow", mass_flow(r) * 3600, 2.363e5, "kg/h"),
        Check("Quality leaving LP turbine", f(r["states"]["4"]["x"]), 0.9382, abs_tol=0.003),
        Check("Condenser heat rejected", t["Qout"] / 1000, 148.0, "MW"),
    ]


@case("Moran", "8.4", "Reheat cycle with 85 % turbine stages")
def moran_8_4():
    r = rankine(eta_t=0.85, reheat=(700, 440))
    t = totals(r)
    return [Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.351, abs_tol=0.003)]


@case("Moran", "8.5", "Regenerative cycle, open feedwater heater at 0.7 MPa, 85 % turbine stages")
def moran_8_5():
    states = {"1": {"P": 8000, "T": 480}, "2": {"P": 700}, "3": {"P": 8}}
    nodes = {
        "t": node("turbine", {"in": 1, "ext1": 2, "out": 3}, {"adiabatic": True}, {"eta": 0.85}, "Turbine"),
        "c": node("hx", {"a_in1": 3, "a_out1": 4}, {"aIsobaric": True, "aSatLiquidOut": True}, name="Condenser"),
        "p1": node("pump", {"in": 4, "out": 5}, ISEN, name="Pump 1"),
        "fwh": node("hx", {"a_in1": 5, "a_in2": 2, "a_out1": 6},
                    {"aIsobaric": True, "adiabatic": True, "aSatLiquidOut": True}, name="Open FWH"),
        "p2": node("pump", {"in": 6, "out": 7}, ISEN, name="Pump 2"),
        "b": node("hx", {"a_in1": 7, "a_out1": 1}, {"aIsobaric": True}, name="Boiler"),
    }
    r = solve(states, nodes, cycle={"netPower": 100000})
    t = totals(r)
    return [
        Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.369, abs_tol=0.003),
        Check("Extraction fraction y", f(r["states"]["2"]["m"]) / mass_flow(r), 0.1966, abs_tol=0.002),
        Check("Steam flow into turbine", mass_flow(r) * 3600, 3.69e5, "kg/h"),
    ]


# ================================================================ Moran Ch. 9: gas power
def brayton(fluid=AIR_TABLES, eta=None, regen=None, m=5.807):
    """Moran Ex. 9.4: 100 kPa, 300 K, rp = 10, turbine inlet 1400 K."""
    ma = ISEN if eta is None else {"adiabatic": True}
    pa = {} if eta is None else {"eta": eta}
    if regen is None:
        states = {"1": {"P": 100, "T": 300 - 273.15, "m": m}, "3": {"T": 1400 - 273.15}}
        nodes = {
            "c": node("compressor", {"in": 1, "out": 2}, ma, dict(pa, rp=10), "Compressor"),
            "b": node("hx", {"a_in1": 2, "a_out1": 3}, {"aIsobaric": True}, name="Heater"),
            "t": node("turbine", {"in": 3, "out": 4}, ma, pa, "Turbine"),
            "x": node("hx", {"a_in1": 4, "a_out1": 1}, {"aIsobaric": True}, name="Cooler"),
        }
    else:
        # 1 -> C -> 2 -> regen cold -> 5 -> heater -> 3 -> T -> 4 -> regen hot -> 6 -> cooler -> 1
        states = {"1": {"P": 100, "T": 300 - 273.15, "m": m}, "3": {"T": 1400 - 273.15}}
        nodes = {
            "c": node("compressor", {"in": 1, "out": 2}, ma, dict(pa, rp=10), "Compressor"),
            "r": node("hx", {"a_in1": 4, "a_out1": 6, "b_in1": 2, "b_out1": 5},
                      {"aIsobaric": True, "bIsobaric": True, "noLoss": True}, {"eps": regen}, "Regenerator"),
            "b": node("hx", {"a_in1": 5, "a_out1": 3}, {"aIsobaric": True}, name="Heater"),
            "t": node("turbine", {"in": 3, "out": 4}, ma, pa, "Turbine"),
            "x": node("hx", {"a_in1": 6, "a_out1": 1}, {"aIsobaric": True}, name="Cooler"),
        }
    return solve(states, nodes, fluid=fluid)


@case("Moran", "9.4", "Ideal Brayton cycle, rp = 10, 300 K / 1400 K (variable cp)")
def moran_9_4():
    r = brayton()
    t, s = totals(r), r["states"]
    return [
        Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.457, abs_tol=0.005),
        Check("Back work ratio", t["Win"] / t["Wout"], 0.396, abs_tol=0.005),
        Check("Net power", t["Wnet"], 5.807 * (706.9 - 279.7), "kW", rel=0.02),
        Check("Compressor exit T2", K(s["2"]["T"]), 574.1, "K", rel=0.01),
        Check("Turbine exit T4", K(s["4"]["T"]), 787.7, "K", rel=0.01),
    ]


@case("Moran", "9.4", "Same cycle on the cold-air standard (k = 1.4)")
def moran_9_4_cold():
    r = brayton(fluid=IDEAL_AIR)
    t, s = totals(r), r["states"]
    return [
        Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.482, abs_tol=0.002),
        Check("Back work ratio", t["Win"] / t["Wout"], 0.414, abs_tol=0.002),
        Check("Compressor exit T2", K(s["2"]["T"]), 579.2, "K", rel=0.002),
        Check("Turbine exit T4", K(s["4"]["T"]), 725.1, "K", rel=0.002),
    ]


@case("Moran", "9.6", "Brayton cycle with 80 % compressor and turbine")
def moran_9_6():
    r = brayton(eta=0.8)
    t = totals(r)
    return [
        Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.249, abs_tol=0.005),
        Check("Back work ratio", t["Win"] / t["Wout"], 0.618, abs_tol=0.008),
        Check("Net power", t["Wnet"], 5.807 * (565.5 - 349.6), "kW", rel=0.03),
    ]


@case("Moran", "9.7", "Ideal Brayton cycle with an 80 % effective regenerator")
def moran_9_7():
    r = brayton(regen=0.8)
    t = totals(r)
    return [Check("Thermal efficiency", t["Wnet"] / t["Qin"], 0.568, abs_tol=0.006)]


@case("Moran", "9.11", "Regenerative gas turbine with intercooling and reheat, 80 % stages")
def moran_9_11():
    T300, T1400 = 300 - 273.15, 1400 - 273.15
    eta = {"adiabatic": True}
    states = {"1": {"P": 100, "T": T300, "m": 5.807}, "2": {"P": 300}, "3": {"T": T300}, "4": {"P": 1000},
              "6": {"T": T1400}, "7": {"P": 300}, "8": {"T": T1400}, "9": {"P": 100}}
    iso = {"aIsobaric": True}
    nodes = {
        "c1": node("compressor", {"in": 1, "out": 2}, eta, {"eta": 0.8}, "Compressor 1"),
        "ic": node("hx", {"a_in1": 2, "a_out1": 3}, iso, name="Intercooler"),
        "c2": node("compressor", {"in": 3, "out": 4}, eta, {"eta": 0.8}, "Compressor 2"),
        "rg": node("hx", {"a_in1": 9, "a_out1": 10, "b_in1": 4, "b_out1": 5},
                   {"aIsobaric": True, "bIsobaric": True, "noLoss": True}, {"eps": 0.8}, "Regenerator"),
        "cb": node("hx", {"a_in1": 5, "a_out1": 6}, iso, name="Combustor"),
        "t1": node("turbine", {"in": 6, "out": 7}, eta, {"eta": 0.8}, "Turbine 1"),
        "rh": node("hx", {"a_in1": 7, "a_out1": 8}, iso, name="Reheater"),
        "t2": node("turbine", {"in": 8, "out": 9}, eta, {"eta": 0.8}, "Turbine 2"),
        "x": node("hx", {"a_in1": 10, "a_out1": 1}, iso, name="Cooler"),
    }
    r = solve(states, nodes, fluid=AIR_TABLES)
    t, d = totals(r), r["devices"]
    q_in = f(d["cb"]["Qdot"]) + f(d["rh"]["Qdot"])
    return [
        Check("Thermal efficiency", t["Wnet"] / q_in, 0.443, abs_tol=0.006),
        Check("Back work ratio", t["Win"] / t["Wout"], 0.454, abs_tol=0.006),
        Check("Net power", t["Wnet"], 5.807 * (645.8 - 293.4), "kW", rel=0.02),
        Check("Heat added (combustor + reheater)", q_in, 5.807 * 795.9, "kW", rel=0.02),
    ]


def closed(states, nodes, fluid=AIR_TABLES):
    return solve(states, nodes, fluid=fluid, mode="closed")


def proc(t, a, b, path, params=None, assumptions=None):
    return {"type": t, "name": t, "ports": {"in": a, "out": b}, "config": {"path": path},
            "params": params or {}, "assumptions": assumptions or {}}


def otto(fluid):
    T1 = 540 * R_TO_K - 273.15
    T3 = 3600 * R_TO_K - 273.15
    return closed({"1": {"P": ATM, "T": T1}, "3": {"T": T3}}, {
        "c": proc("compression", 1, 2, "isentropic", {"r": 8}),
        "h": proc("heatAddition", 2, 3, "isochoric"),
        "e": proc("expansion", 3, 4, "isentropic", {"r": 8}),
        "x": proc("heatRejection", 4, 1, "isochoric"),
    }, fluid)


@case("Moran", "9.1", "Otto cycle, r = 8, 540 °R / 1 atm, Tmax 3600 °R (variable cp)")
def moran_9_1():
    r = otto(AIR_TABLES)
    s, c = r["states"], r["cycle"]
    return [
        Check("T2", K(s["2"]["T"]), 1212 * R_TO_K, "K", rel=0.01),
        Check("p2", f(s["2"]["P"]) / ATM, 17.96, "atm", rel=0.01),
        Check("p3", f(s["3"]["P"]) / ATM, 53.3, "atm", rel=0.01),
        Check("T4", K(s["4"]["T"]), 1878 * R_TO_K, "K", rel=0.01),
        Check("Thermal efficiency", f(c["eta"]), 0.51, abs_tol=0.008),
        Check("Mean effective pressure", f(c["mep"]) / ATM, 8.03, "atm", rel=0.02),
    ]


@case("Moran", "9.1", "Same Otto cycle on the cold-air standard (k = 1.4)")
def moran_9_1_cold():
    r = otto(IDEAL_AIR)
    c = r["cycle"]
    return [
        Check("Thermal efficiency", f(c["eta"]), 0.565, abs_tol=0.002),
        Check("Mean effective pressure", f(c["mep"]) / ATM, 7.05, "atm", rel=0.01),
    ]


@case("Moran", "9.2", "Diesel cycle, r = 18, cutoff 2, 300 K / 0.1 MPa (variable cp)")
def moran_9_2():
    r = closed({"1": {"P": 100, "T": 300 - 273.15}}, {
        "c": proc("compression", 1, 2, "isentropic", {"r": 18}),
        "h": proc("heatAddition", 2, 3, "isobaric", {"rc": 2}),
        "e": proc("expansion", 3, 4, "isentropic"),
        "x": proc("heatRejection", 4, 1, "isochoric"),
    })
    s, c = r["states"], r["cycle"]
    return [
        Check("T2", K(s["2"]["T"]), 898.3, "K", rel=0.01),
        Check("p2", f(s["2"]["P"]) / 1000, 5.39, "MPa", rel=0.01),
        Check("T3", K(s["3"]["T"]), 1796.6, "K", rel=0.01),
        Check("T4", K(s["4"]["T"]), 887.7, "K", rel=0.01),
        Check("Thermal efficiency", f(c["eta"]), 0.578, abs_tol=0.006),
        Check("Mean effective pressure", f(c["mep"]) / 1000, 0.76, "MPa", rel=0.02),
    ]


@case("Moran", "9.3", "Dual cycle, r = 18, pressure ratio 1.5, cutoff 1.2 (variable cp)")
def moran_9_3():
    r = closed({"1": {"P": 100, "T": 300 - 273.15}}, {
        "c": proc("compression", 1, 2, "isentropic", {"r": 18}),
        "h1": proc("heatAddition", 2, 3, "isochoric", {"rp": 1.5}),
        "h2": proc("heatAddition", 3, 4, "isobaric", {"rc": 1.2}),
        "e": proc("expansion", 4, 5, "isentropic"),
        "x": proc("heatRejection", 5, 1, "isochoric"),
    })
    c = r["cycle"]
    return [
        Check("Thermal efficiency", f(c["eta"]), 0.635, abs_tol=0.006),
        Check("Mean effective pressure", f(c["mep"]) / 1000, 0.56, "MPa", rel=0.03),
    ]


@case("Moran", "9.13", "Ideal turbojet: 11.8 psia, 430 °R, 620 mph, rp = 8, 2150 °R")
def moran_9_13():
    pa = 11.8 * PSI
    states = {"1": {"P": pa, "T": 430 * R_TO_K - 273.15, "V": 909.3 * FT},
              "4": {"T": 2150 * R_TO_K - 273.15}, "6": {"P": pa}}
    nodes = {
        "d": node("diffuser", {"in": 1, "out": 2}, dict(ISEN, negV2=True), name="Diffuser"),
        "c": node("compressor", {"in": 2, "out": 3}, ISEN, {"rp": 8}, "Compressor"),
        "b": node("hx", {"a_in1": 3, "a_out1": 4}, {"aIsobaric": True}, name="Combustor"),
        "t": node("turbine", {"in": 4, "out": 5}, dict(ISEN, drivesCompressor=True), name="Turbine"),
        "n": node("nozzle", {"in": 5, "out": 6}, dict(ISEN, negV1=True), name="Nozzle"),
    }
    r = solve(states, nodes, fluid=AIR_TABLES)
    s = r["states"]
    v_exit = math.sqrt(2 * (449.5 - 265.8) * BTU_LB * 1000) / FT     # from the book's h4, h5
    return [
        Check("Pressure after diffuser", f(s["2"]["P"]) / PSI, 19.79, "psia", rel=0.015),
        Check("Pressure after compressor", f(s["3"]["P"]) / PSI, 158.3, "psia", rel=0.015),
        Check("Pressure after turbine", f(s["5"]["P"]) / PSI, 77.2, "psia", rel=0.02),
        Check("Nozzle exit velocity", f(s["6"]["V"]) / FT, v_exit, "ft/s", rel=0.015),
    ]


def conv_nozzle(p_back):
    states = {"1": {"P": 1000, "T": 360 - 273.15}}
    nodes = {"n": node("nozzle", {"in": 1, "out": 2}, dict(ISEN, negV1=True), {"A2": 0.001, "Pb": p_back}, "Nozzle")}
    return solve(states, nodes, fluid=IDEAL_AIR)


@case("Moran", "9.14b", "Converging nozzle, 1 MPa / 360 K, A = 0.001 m², back pressure 784 kPa")
def moran_9_14b():
    r = conv_nozzle(784)
    s, d = r["states"]["2"], r["devices"]["n"]
    return [
        Check("Mass flow", f(s["m"]), 1.79, "kg/s", rel=0.01),
        Check("Exit velocity", f(s["V"]), 220.5, "m/s", rel=0.01),
        Check("Exit temperature", K(s["T"]), 336, "K", rel=0.005),
        Check("Exit Mach number", f(d["Mach2"]), 0.60, abs_tol=0.01),
    ]


@case("Moran", "9.14a", "Same nozzle, back pressure 500 kPa (below critical: the nozzle chokes)")
def moran_9_14a():
    r = conv_nozzle(500)
    s, d = r["states"]["2"], r["devices"]["n"]
    return [
        Check("Mass flow", f(s["m"]), 2.13, "kg/s", rel=0.01),
        Check("Exit pressure (critical)", f(s["P"]), 528, "kPa", rel=0.005),
        Check("Exit velocity (sonic)", f(s["V"]), 347.2, "m/s", rel=0.01),
        Check("Exit Mach number", f(d["Mach2"]), 1.0, abs_tol=0.01),
    ]


# ================================================================ Moran Ch. 10: refrigeration
def vcr(t_evap, cond, eta=None, sub_T=None, m=0.08):
    """R-134a cycle: sat. vapour at t_evap into the compressor, condenser at
    cond = ("T", 26) or ("P", 900); liquid leaves saturated or at sub_T."""
    s3 = {"x": 0} if sub_T is None else {"T": sub_T}
    if cond[0] == "T":
        s3 = {"T": cond[1], "x": 0}
    else:
        s3["P"] = cond[1]
    states = {"1": {"T": t_evap, "x": 1, "m": m}, "3": s3}
    comp_a = ISEN if eta is None else {"adiabatic": True}
    nodes = {
        "c": node("compressor", {"in": 1, "out": 2}, comp_a, {} if eta is None else {"eta": eta}, "Compressor"),
        "k": node("hx", {"a_in1": 2, "a_out1": 3}, {"aIsobaric": True}, name="Condenser"),
        "v": node("valve", {"in": 3, "out": 4}, {"isenthalpic": True}, name="Valve"),
        "e": node("hx", {"a_in1": 4, "a_out1": 1}, {"aIsobaric": True}, name="Evaporator"),
    }
    r = solve(states, nodes, fluid="R134a")
    t = totals(r)
    return [
        Check("Compressor power", t["Win"], None, "kW"),
        Check("Refrigeration capacity", t["Qin"] / TON, None, "ton"),
        Check("COP", t["Qin"] / t["Win"], None),
    ]


def vcr_checks(rows, w, tons, cop, rel=0.02):
    vals = (w, tons, cop)
    for row, v in zip(rows, vals):
        row.book, row.rel = v, rel
    return rows


@case("Moran", "10.1", "Ideal vapour-compression cycle, R-134a, 0 °C / 26 °C, 0.08 kg/s")
def moran_10_1():
    # the book rounds the power to 1.4 kW: (0.08)(264.7 - 247.23) = 1.398 kW
    return vcr_checks(vcr(0, ("T", 26)), 0.08 * (264.7 - 247.23), 3.67, 9.24)


@case("Moran", "10.2", "Ideal cycle, R-134a, -10 °C evaporator, 9 bar condenser")
def moran_10_2():
    return vcr_checks(vcr(-10, ("P", 900)), 2.48, 3.23, 4.57)


@case("Moran", "10.3", "Actual cycle: 80 % compressor, liquid leaves condenser at 30 °C")
def moran_10_3():
    return vcr_checks(vcr(-10, ("P", 900), eta=0.8, sub_T=30), 0.08 * (280.15 - 241.35), 3.41, 3.86)


@case("Moran", "10.5", "Ideal Brayton refrigeration: 1 atm, 480 °R, rp = 3, turbine inlet 540 °R")
def moran_10_5():
    m = 248 * LB / 60
    states = {"1": {"P": ATM, "T": 480 * R_TO_K - 273.15, "m": m}, "3": {"T": 540 * R_TO_K - 273.15}}
    nodes = {
        "c": node("compressor", {"in": 1, "out": 2}, ISEN, {"rp": 3}, "Compressor"),
        "h": node("hx", {"a_in1": 2, "a_out1": 3}, {"aIsobaric": True}, name="Heat rejection"),
        "t": node("turbine", {"in": 3, "out": 4}, ISEN, name="Turbine"),
        "l": node("hx", {"a_in1": 4, "a_out1": 1}, {"aIsobaric": True}, name="Refrigerated space"),
    }
    r = solve(states, nodes, fluid=AIR_TABLES)
    t = totals(r)
    btu_min = 60 / 1.055056
    return [
        Check("Net power input", -t["Wnet"] * btu_min, 248 * ((157.1 - 114.69) - (129.06 - 94.1)), "Btu/min", rel=0.03),
        Check("Refrigeration capacity", t["Qin"] * btu_min, 248 * (114.69 - 94.1), "Btu/min", rel=0.02),
        Check("COP", t["Qin"] / -t["Wnet"], 2.76, rel=0.03),
    ]


# ================================================================ Moran Ch. 12: psychrometrics
def ac(mode, a, b, water=None, params=None, asm=None):
    ports = {"in": a, "out": b}
    if water is not None:
        ports["water"] = water
    return {"type": "ac", "name": "AC", "ports": ports, "config": {"mode": mode},
            "params": params or {}, "assumptions": asm or {"isobaric": True}}


@case("Moran", "12.10", "Heating moist air in a duct: 10 °C, 80 % to 30 °C at 1 bar")
def moran_12_10():
    states = {"1": {"P": 100, "T": 10, "RH": 80, "m": 182.9 / 60}, "2": {"T": 30}}
    r = solve(states, {"h": ac("heating", 1, 2)}, fluids={"1": HA, "2": HA})
    q = 182.9 * ((303.2 - 283.1) + 0.00616 * (2556.3 - 2519.8))
    return [
        Check("Heat rate", f(r["devices"]["h"]["Qdot"]) * 60, q, "kJ/min", rel=0.01),
        Check("Exit relative humidity", f(r["states"]["2"]["RH"]), 23.1, "%", abs_tol=0.3),
    ]


@case("Moran", "12.11", "Dehumidifier: 30 °C, 50 % in, saturated at 10 °C out, 1.013 bar")
def moran_12_11():
    states = {"1": {"P": 101.3, "T": 30, "RH": 50, "m": 319.35 / 60}, "2": {"T": 10, "RH": 100}}
    r = solve(states, {"c": ac("cooling", 1, 2, 3, asm={"isobaric": True, "condensateAtOutletT": True})},
              fluids={"1": HA, "2": HA, "3": "Water"})
    s, d = r["states"], r["devices"]["c"]
    return [
        Check("Inlet humidity ratio", f(s["1"]["w"]), 0.0133, "kg/kg", abs_tol=0.0002),
        Check("Exit humidity ratio", f(s["2"]["w"]), 0.0076, "kg/kg", abs_tol=0.0001),
        Check("Condensate per kg dry air", f(d["mWater"]) / (319.35 / 60), 0.0057, "kg/kg", abs_tol=0.0002),
        Check("Refrigerating capacity", f(d["tons"]), 52.5, "ton", rel=0.015),
    ]


@case("Moran", "12.12", "Steam-spray humidifier: 22 °C, ω = 0.002, sat. steam at 110 °C, 52 kg/h")
def moran_12_12():
    states = {"1": {"P": 100, "T": 22, "w": 0.002, "m": 1.5}, "3": {"T": 110, "x": 1, "m": 52 / 3600}}
    r = solve(states, {"s": ac("steam", 1, 2, 3)}, fluids={"1": HA, "2": HA, "3": "Water"})
    s = r["states"]["2"]
    return [
        Check("Exit humidity ratio", f(s["w"]), 0.0116, "kg/kg", abs_tol=0.0002),
        Check("Exit temperature", f(s["T"]), 23.4, "°C", abs_tol=0.3),
    ]


@case("Moran", "12.13", "Evaporative cooler: 100 °F, 10 % in, 70 °F out, water at 70 °F")
def moran_12_13():
    t_in, t_out = (100 - 32) / 1.8, (70 - 32) / 1.8
    ma = 352.1 * LB / 60
    states = {"1": {"P": ATM, "T": t_in, "RH": 10, "m": ma}, "2": {"T": t_out}, "3": {"T": t_out, "x": 0}}
    r = solve(states, {"e": ac("evaporative", 1, 2, 3, asm={"isobaric": True, "adiabatic": True})},
              fluids={"1": HA, "2": HA, "3": "Water"})
    return [
        Check("Water added", f(r["states"]["3"]["m"]) * 3600 / LB, 144.7, "lb/h", rel=0.03),
        Check("Exit relative humidity", f(r["states"]["2"]["RH"]), 69.7, "%", abs_tol=1.0),
    ]


@case("Moran", "12.14", "Adiabatic mixing of two moist air streams at 1 bar")
def moran_12_14():
    states = {"1": {"P": 100, "T": 5, "w": 0.002, "m": 180 / 60},
              "2": {"P": 100, "T": 24, "RH": 50, "m": 497 / 60}}
    mixer = node("mixer", {"in1": 1, "in2": 2, "out": 3}, {"isobaric": True, "adiabatic": True}, name="Mixer")
    r = solve(states, {"m": mixer}, fluids={"1": HA, "2": HA, "3": HA})
    s = r["states"]["3"]
    return [
        Check("Mixed humidity ratio", f(s["w"]), 0.00745, "kg/kg", abs_tol=0.0001),
        Check("Mixed temperature", f(s["T"]), 19.01, "°C", abs_tol=0.2),
    ]


@case("Moran", "12.15", "Power plant cooling tower, 38 → 30 °C, makeup water at 20 °C")
def moran_12_15():
    m_w = 4.5e7 / 3600
    tower = node("coolingTower", {"in": 1, "out": 2}, {"isobaric": True, "makeup": True},
                 {"Tin": 25, "RHin": 35, "Tout": 35, "RHout": 90, "Patm": ATM, "Tmakeup": 20}, "Cooling tower")
    r = solve({"1": {"P": ATM, "T": 38, "m": m_w}, "2": {"T": 30}}, {"ct": tower})
    d = r["devices"]["ct"]
    return [
        Check("Dry air flow", f(d["ma"]) * 3600, 2.03e7, "kg/h", rel=0.01),
        Check("Makeup water", f(d["mMakeup"]) * 3600, 5.24e5, "kg/h", rel=0.015),
    ]


# ================================================================ Penoncello Ch. 3
def penoncello_turbine(tb=None):
    states = {"1": {"P": 1600, "T": 350, "m": 16}, "2": {"T": 30, "x": 1}}
    params = {"Wdot": 9000}
    if tb is not None:
        params["Tb"] = tb
    nodes = {"t": node("turbine", {"in": 1, "out": 2}, {}, params, "Turbine")}
    return solve(states, nodes, cycle={"exergy": {"T0": 25, "P0": 100}})


@case("Penoncello", "3.2", "Steam turbine losing heat: 1.6 MPa 350 °C to sat. vapour at 30 °C, 16 kg/s, 9 MW")
def pen_3_2():
    r = penoncello_turbine()
    return [Check("Heat transfer rate (lost)", -f(r["devices"]["t"]["Qdot"]), 446.4, "kW", rel=0.01)]


def exergy_rows(r):
    from exergy import exergy_summary
    ex = exergy_summary(r["_payload"], r)
    if not ex or not ex.get("devices"):
        raise RuntimeError("the exergy panel produced nothing" + (f" (needs {ex['need']})" if ex else ""))
    return {row["id"]: row for row in ex["devices"]}


@case("Penoncello", "3.5 / 3.11", "Same turbine, casing at 70 °C, dead state 25 °C / 100 kPa: entropy and exergy")
def pen_3_11():
    r = penoncello_turbine(tb=70)
    row = exergy_rows(r)["t"]
    x_dest = f(row["Xdest"])
    return [
        Check("Entropy generation (Ex. 3.5)", x_dest / 298.15, 23.39, "kW/K", rel=0.01),
        Check("Exergy destroyed (Ex. 3.11)", x_dest, 6974, "kW", rel=0.01),
        Check("Exergy lost with the heat", f(row["Xheat"]), 59, "kW", rel=0.03),
        Check("Exergetic efficiency", f(row["etaII"]), 0.561, abs_tol=0.005),
    ]


@case("Penoncello", "3.6", "Air compressor 25 °C/100 kPa to 460 °C/1600 kPa, after-cooled by 2 kg/s water")
def pen_3_6():
    states = {"1": {"P": 100, "T": 25}, "2": {"P": 1600, "T": 460}, "3": {"T": 300},
              "4": {"P": 200, "T": 25, "m": 2}, "5": {"T": 110}}
    nodes = {
        "c": node("compressor", {"in": 1, "out": 2}, {"adiabatic": True}, name="Compressor"),
        "x": node("hx", {"a_in1": 2, "a_out1": 3, "b_in1": 4, "b_out1": 5},
                  {"aIsobaric": True, "bIsobaric": True, "noLoss": True}, name="Aftercooler"),
    }
    r = solve(states, nodes, fluid="Air", fluids={"4": "Water", "5": "Water"},
              cycle={"exergy": {"T0": 25, "P0": 100}})
    d = r["devices"]
    rows = exergy_rows(r)
    return [
        Check("Compressor isentropic efficiency", f(d["c"]["eta"]), 0.7996, abs_tol=0.005),
        Check("Air flow", f(r["states"]["1"]["m"]), 4.17, "kg/s", rel=0.01),
        Check("Compressor power", f(d["c"]["Wdot"]), 1884, "kW", rel=0.01),
        Check("Compressor entropy generation", f(rows["c"]["Xdest"]) / 298.15, 0.5469, "kW/K", rel=0.02),
    ]


def acetone_hx(layout):
    states = {"1": {"P": 300, "T": 80, "m": 7.1979}, "2": {"T": 60}, "3": {"P": 200, "T": 20, "m": 3.4936}}
    n = node("hx", {"a_in1": 1, "a_out1": 2, "b_in1": 3, "b_out1": 4},
             {"aIsobaric": True, "bIsobaric": True, "noLoss": True}, name="Heat exchanger", config={"layout": layout})
    r = solve(states, {"x": n}, fluid="Acetone", fluids={"3": "Water", "4": "Water"})
    s = r["states"]
    return f(s["4"]["T"]), f(s["1"]["T"]), f(s["2"]["T"]), f(s["3"]["T"])


@case("Penoncello", "3.7", "Counter-flow acetone cooler, 80 → 60 °C, water in at 20 °C: pinch point")
def pen_3_7():
    tco, thi, tho, tci = acetone_hx("counter")
    return [
        Check("Water outlet temperature", tco, 42.43, "°C", abs_tol=0.3),
        Check("Pinch-point temperature difference", min(thi - tco, tho - tci), 37.6, "K", abs_tol=0.3),
    ]


@case("Penoncello", "3.8", "Same exchanger in parallel flow: pinch at the outlet")
def pen_3_8():
    tco, thi, tho, tci = acetone_hx("parallel")
    return [Check("Pinch-point temperature difference", tho - tco, 17.57, "K", abs_tol=0.3)]


# ================================================================ Penoncello Ch. 5
@case("Penoncello", "5.2", "Film coefficient in a 1-std type L copper tube, 40 % methanol-water (Gnielinski)")
def pen_5_2():
    import convection
    D, k, Re, Pr = 0.026035, 0.38047, 47246, 24.976      # the book's tube and property values
    rr = 1.5e-6 / D
    nu = convection.nu_tube(Re, Pr, rr)
    return [
        Check("Friction factor (Swamee-Jain)", convection.swamee_jain(Re, rr), 0.021257, rel=0.002),
        Check("Nusselt number (Gnielinski)", nu, 516.77, rel=0.002),
        Check("Film coefficient h = Nu k / D", nu * k / D, 516.77 * k / D, "W/m²·K", rel=0.002),
    ]


@case("Penoncello", "5.5", "Rating a counter-flow hexane cooler, UA = 2.7 kW/K, 5 m³/h of each fluid")
def pen_5_5():
    # 5 m³/h each, turned into mass flows with the density at each fluid's
    # average temperature, as the book evaluates its properties
    from CoolProp.CoolProp import PropsSI
    m_h = PropsSI("D", "T", 273.15 + (90 + 39.9) / 2, "P", 3e5, "Hexane") * 5 / 3600
    m_c = PropsSI("D", "T", 273.15 + (15 + 33.0) / 2, "P", 2e5, "Water") * 5 / 3600
    states = {"1": {"P": 300, "T": 90, "m": m_h}, "3": {"P": 200, "T": 15, "m": m_c}}
    n = node("hx", {"a_in1": 1, "a_out1": 2, "b_in1": 3, "b_out1": 4},
             {"aIsobaric": True, "bIsobaric": True, "noLoss": True}, {"UA": 2.7}, "Hexane cooler",
             config={"layout": "counter"})
    r = solve(states, {"x": n}, fluid="Hexane", fluids={"3": "Water", "4": "Water"})
    s = r["states"]
    return [
        Check("Hexane outlet temperature", f(s["2"]["T"]), 39.9, "°C", abs_tol=0.4),
        Check("Water outlet temperature", f(s["4"]["T"]), 33.0, "°C", abs_tol=0.3),
        Check("Heat transfer rate", f(r["devices"]["x"]["Qdot"]), 104.5, "kW", rel=0.01),
    ]


@case("Penoncello", "5.8", "Gas turbine regenerator, UA = 48,000 Btu/h·°R, 100 lbm/s of air each side")
def pen_5_8():
    def F(t):
        return (t - 32) / 1.8
    m = 100 * LB
    ua = 48000 * 1.05505585 / 3600 / (5 / 9)        # kW/K
    states = {"1": {"P": ATM, "T": F(1000), "m": m}, "3": {"P": 200 * PSI, "T": F(600), "m": m}}
    n = node("hx", {"a_in1": 1, "a_out1": 2, "b_in1": 3, "b_out1": 4},
             {"aIsobaric": True, "bIsobaric": True, "noLoss": True}, {"UA": ua}, "Regenerator",
             config={"layout": "counter"})
    r = solve(states, {"x": n}, fluid="Air")
    s = r["states"]
    # the book takes one average cp for both streams; real air's cp rises with
    # T, so the changes are compared (2 %) rather than the absolute outlet T
    t_ho, t_co = f(s["2"]["T"]) * 1.8 + 32, f(s["4"]["T"]) * 1.8 + 32
    return [
        Check("Hot air temperature drop", 1000 - t_ho, 1000 - 863.5, "°F", rel=0.02),
        Check("Cold air temperature rise", t_co - 600, 736.5 - 600, "°F", rel=0.02),
    ]


# ================================================================ Dick Ch. 4: compressible flow
def dick_nozzle(p1):
    A1, A2 = math.pi * 0.40 ** 2 / 4, math.pi * 0.20 ** 2 / 4
    states = {"1": {"P": p1, "T": 300 - 273.15}}
    nodes = {"n": node("nozzle", {"in": 1, "out": 2}, ISEN, {"A1": A1, "A2": A2, "Pb": 100}, "Convergent pipe")}
    r = solve(states, nodes, fluid=IDEAL_AIR)
    return r["states"]["2"], r["devices"]["n"]


@case("Dick", "4.8.1 / 4.8.2", "Convergent pipe 0.40 → 0.20 m, air at 150 kPa / 300 K into 100 kPa (subcritical)")
def dick_4_8_2a():
    s, d = dick_nozzle(150)
    return [
        Check("Mass flow", f(s["m"]), 10.71, "kg/s", rel=0.005),
        Check("Throat Mach number", f(d["Mach2"]), 0.80, abs_tol=0.01),
    ]


@case("Dick", "4.8.2", "Same pipe at 200 kPa supply (slightly supercritical: chokes)")
def dick_4_8_2b():
    s, d = dick_nozzle(200)
    return [
        Check("Mass flow", f(s["m"]), 14.85, "kg/s", rel=0.005),
        Check("Throat Mach number", f(d["Mach2"]), 1.0, abs_tol=0.005),
        Check("Throat pressure", f(s["P"]), 107.26, "kPa", rel=0.003),
    ]


@case("Dick", "4.8.2", "Same pipe at 500 kPa supply (strongly supercritical)")
def dick_4_8_2c():
    s, d = dick_nozzle(500)
    return [
        Check("Mass flow", f(s["m"]), 37.13, "kg/s", rel=0.005),
        Check("Throat Mach number", f(d["Mach2"]), 1.0, abs_tol=0.005),
    ]


# ================================================================ Dick Ch. 6: steam turbines
@case("Dick", "6.10.4", "Flamanville 3 HP turbine: sat. steam at 75 bar expanded to 11 bar, 2500 kg/s")
def dick_6_10_4():
    states = {"1": {"P": 7500, "x": 1, "m": 2500}, "2": {"P": 1100}}
    r = solve(states, {"t": node("turbine", {"in": 1, "out": 2}, ISEN, name="HP part")})
    w = f(r["devices"]["t"]["w"])
    return [
        Check("Inlet temperature (saturated)", f(r["states"]["1"]["T"]), 290, "°C", abs_tol=1.0),
        Check("Isentropic enthalpy drop of the HP part", w, 340, "kJ/kg", rel=0.02),
        Check("Stages at 38 kJ/kg each", w / 38, 9, abs_tol=0.5),
    ]


# ================================================================ Part 2: the ME 313 project
PROJECT = "ME 313 project"
_P313 = {}


def _project_solved():
    if "r" not in _P313:
        import project_313
        p, r = project_313.solve()
        if r.get("errors"):
            raise RuntimeError("the solver reported: " + "; ".join(r["errors"]))
        _P313.update(p=p, r=r, mod=project_313)
    return _P313


TOL_313 = {"T": dict(abs_tol=0.3), "h": dict(rel=0.005), "s": dict(abs_tol=0.002), "ṁ": dict(rel=0.01),
           "x": dict(abs_tol=0.002), "ω": dict(abs_tol=0.0002)}
NAME_313 = {"T": "Temperature", "h": "Enthalpy", "s": "Entropy", "ṁ": "Mass flow", "x": "Quality",
            "ω": "Humidity ratio"}


@case(PROJECT, "Part 2, states", "State points 1 to 32 of the nuclear Rankine cycle and cooling tower vs EES")
def project_states():
    d = _project_solved()
    out = []
    for n, q, app, ees, unit in d["mod"].compare(d["r"]):
        tol = dict(TOL_313[q])
        if q == "h" and n in (31, 32):
            tol = dict(rel=0.01)          # humid air: CoolProp vs EES's AirH2O, see the notes
        out.append(Check(f"State {n}: {NAME_313[q]}", app, ees, unit, **tol))
    return out


@case(PROJECT, "Part 2, results", "Turbine and pump powers, net power, efficiency, heat rejected, tower flows")
def project_results():
    d = _project_solved()
    sm = d["mod"].summary(d["p"], d["r"])
    e = d["mod"].EES_SUMMARY
    rows = [("W_HPT_1", "HP turbine 1 power", "Btu/h"), ("W_HPT_2", "HP turbine 2 power", "Btu/h"),
            ("W_LPT_1", "LP turbine 1 power", "Btu/h"), ("W_LPT_2", "LP turbine 2 power", "Btu/h"),
            ("W_P_1", "Pump 1 power", "Btu/h"), ("W_P_2", "Pump 2 power", "Btu/h"),
            ("W_net", "Net power", "Btu/h"), ("W_net_MW", "Net power", "MW"), ("Q_out", "Heat rejected", "Btu/h"),
            ("m_total", "Steam flow", "lbm/h"), ("m_CW", "Cooling water flow", "lbm/h"),
            ("m_makeup", "Makeup water", "lbm/h"), ("m_air", "Dry air through the tower", "lbm/h")]
    out = [Check(label, sm[k], e[k], unit, rel=0.01) for k, label, unit in rows]
    out.insert(8, Check("Thermal efficiency", sm["N_therm"], e["N_therm"], abs_tol=0.0005))
    return out


@case(PROJECT, "Part 2, optimization", "FWH 2 pressure for the best efficiency (EES parametric table, 20 to 30 psia)")
def project_sweep():
    import project_313
    rows, p_opt, eta_opt = project_313.sweep()
    out = [Check(f"η at P_FWH2 = {p:g} psia", app, ees, abs_tol=0.0001) for p, app, ees in rows]
    out.append(Check("Best FWH 2 pressure", p_opt, 24.0, "psia", abs_tol=0.5))
    out.append(Check("Best efficiency", eta_opt, 0.388402138, abs_tol=0.0001))
    return out


# ---------------------------------------------------------------- given data, for the report
GIVEN = {
    "moran_8_1": "Steam. Saturated vapour enters the turbine at 8.0 MPa; saturated liquid leaves the condenser at "
                 "0.008 MPa. Turbine and pump isentropic. Net power 100 MW.",
    "moran_8_2": "As Example 8.1, but the turbine and the pump each have an isentropic efficiency of 85 %.",
    "moran_8_3": "Steam enters the first turbine at 8.0 MPa, 480 °C and expands to 0.7 MPa; reheated to 440 °C; "
                 "expands to 0.008 MPa. Ideal (isentropic) turbines and pump. Net power 100 MW.",
    "moran_8_4": "As Example 8.3, with each turbine stage at 85 % isentropic efficiency.",
    "moran_8_5": "Steam at 8.0 MPa, 480 °C; extraction at 0.7 MPa to an open feedwater heater leaving saturated "
                 "liquid at 0.7 MPa; condenser 0.008 MPa. Turbine stages 85 %, pumps isentropic. Net power 100 MW.",
    "moran_9_4": "Air enters the compressor at 100 kPa, 300 K, 5.807 kg/s (5 m³/s). Pressure ratio 10. Turbine inlet "
                 "1400 K. Ideal compressor and turbine. Air as an ideal gas with variable specific heats (Table A-22).",
    "moran_9_4_cold": "As Example 9.4 on the cold-air standard: cp = 1.005 kJ/kg·K, k = 1.4.",
    "moran_9_6": "As Example 9.4 with compressor and turbine isentropic efficiencies of 80 %.",
    "moran_9_7": "As Example 9.4 with a regenerator of effectiveness 80 %.",
    "moran_9_11": "Air at 100 kPa, 300 K, 5.807 kg/s. Two compressor stages (overall ratio 10) with intercooling "
                  "to 300 K at 300 kPa; two turbine stages (overall ratio 10) with reheat to 1400 K at 300 kPa; "
                  "turbine inlets 1400 K. Every stage 80 %. Regenerator effectiveness 80 %.",
    "moran_9_1": "Air-standard Otto cycle, compression ratio 8. Start of compression 540 °R (300 K), 1 atm. "
                 "Maximum temperature 3600 °R (2000 K). Variable specific heats (Table A-22E).",
    "moran_9_1_cold": "As Example 9.1 on the cold-air standard, k = 1.4.",
    "moran_9_2": "Air-standard Diesel cycle, compression ratio 18, cutoff ratio 2. Start of compression 300 K, "
                 "0.1 MPa. Variable specific heats.",
    "moran_9_3": "Air-standard dual cycle, compression ratio 18, start at 300 K, 0.1 MPa. Constant-volume heat "
                 "addition with pressure ratio 1.5, then constant-pressure heat addition with volume ratio 1.2.",
    "moran_9_13": "Air enters at 11.8 lbf/in², 430 °R, 620 mph (909.3 ft/s). Compressor pressure ratio 8. Turbine "
                  "inlet 2150 °R. Nozzle exit 11.8 lbf/in². Turbine drives the compressor. All processes "
                  "isentropic, no combustor pressure drop. Kinetic energy only at engine inlet and exit.",
    "moran_9_14b": "Converging nozzle, exit area 0.001 m². Air (ideal gas, k = 1.4) enters with negligible "
                   "velocity at 1.0 MPa, 360 K. Isentropic. Back pressure 784 kPa.",
    "moran_9_14a": "As above with back pressure 500 kPa, below the critical pressure.",
    "moran_10_1": "R-134a. Saturated vapour enters the compressor at 0 °C; saturated liquid leaves the condenser "
                  "at 26 °C. Isentropic compressor, throttling valve. 0.08 kg/s.",
    "moran_10_2": "R-134a. Saturated vapour at -10 °C into the compressor; saturated liquid leaves the condenser "
                  "at 9 bar. Isentropic compressor. 0.08 kg/s.",
    "moran_10_3": "As Example 10.2 with a compressor isentropic efficiency of 80 % and liquid leaving the "
                  "condenser at 30 °C (9 bar).",
    "moran_10_5": "Air enters the compressor at 1 atm, 480 °R, 50 ft³/s (248 lb/min). Pressure ratio 3. Turbine "
                  "inlet 540 °R. Ideal compressor and turbine.",
    "moran_12_10": "Moist air at 10 °C, 80 % relative humidity, 150 m³/min (182.9 kg/min of dry air) heated at "
                   "1 bar to 30 °C. No moisture added or removed.",
    "moran_12_11": "Moist air at 30 °C, 50 %, 280 m³/min (319.35 kg/min dry air), 1.013 bar, cooled to saturated "
                   "air at 10 °C. Condensate leaves saturated at 10 °C.",
    "moran_12_12": "Moist air at 22 °C, ω = 0.002 (wet bulb 9 °C), 90 kg/min dry air, 1 bar. Saturated steam at "
                   "110 °C injected at 52 kg/h. Adiabatic.",
    "moran_12_13": "Air at 100 °F, 10 %, 5000 ft³/min (352.1 lb/min dry air), 1 atm, leaves at 70 °F. Liquid water "
                   "at 70 °F evaporates into it. Adiabatic.",
    "moran_12_14": "142 m³/min of moist air at 5 °C, ω = 0.002 (180 kg/min dry air) mixed adiabatically at 1 bar "
                   "with 425 m³/min at 24 °C, 50 % (497 kg/min dry air).",
    "moran_12_15": "Cooling tower water 38 °C in, 30 °C out, 4.5 × 10⁷ kg/h. Makeup water at 20 °C. Air in at "
                   "25 °C, 35 %; out at 35 °C, 90 %; 1 atm.",
    "pen_3_2": "Steam at 1.6 MPa, 350 °C, 16 kg/s enters a turbine and leaves as saturated vapour at 30 °C. "
               "Power delivered 9 MW. The turbine is not adiabatic.",
    "pen_3_11": "Same turbine. Casing (insulation surface) at 70 °C. Dead state 25 °C, 100 kPa.",
    "pen_3_6": "Air at 25 °C, 100 kPa compressed adiabatically to 460 °C, 1600 kPa, then cooled to 300 °C at "
               "constant pressure by water entering at 25 °C, 200 kPa, 2 kg/s and leaving at 110 °C. "
               "Dead state 25 °C, 100 kPa.",
    "pen_3_7": "Liquid acetone 80 → 60 °C (7.198 kg/s, from 10 L/s) cooled in counter flow by water entering at "
               "20 °C (3.494 kg/s, from 3.5 L/s).",
    "pen_3_8": "As Example 3.7 in parallel flow.",
    "pen_5_2": "40 % methanol-water at 2.6 L/s in a 1-std type L copper tube (ID 26.035 mm, drawn tubing "
               "roughness 0.0015 mm). Book properties at 10 °C give Re = 47,246, Pr = 24.976, k = 0.38047 W/m·K.",
    "pen_5_5": "Counter flow. Hexane 5 m³/h in at 90 °C; water 5 m³/h in at 15 °C. UA = 2.7 kW/K. Mass flows "
               "from the density at each fluid's average temperature, as in the book.",
    "pen_5_8": "Counter-flow regenerator, 100 lbm/s of air each side. Cold air in at 600 °F, 200 psia; hot "
               "exhaust in at 1000 °F, 1 atm. UA = 48,000 Btu/h·°R.",
    "dick_4_8_2a": "Pipe narrowing from D = 0.40 m to d = 0.20 m. Air at 150 kPa, 300 K exhausting to 100 kPa. "
                   "Lossless, no jet contraction, ideal gas γ = 1.4.",
    "dick_4_8_2b": "As above with a supply pressure of 200 kPa.",
    "dick_4_8_2c": "As above with a supply pressure of 500 kPa.",
    "dick_6_10_4": "Saturated steam at 75 bar (290 °C) expanded isentropically to 11 bar, 2500 kg/s. "
                   "Stage enthalpy drop about 38 kJ/kg.",
    "project_states": "The full Part 2 diagram (see the givens above), solved by TurboSim from the givens only, "
                      "state by state against the EES Arrays Table. Humid-air flows at 31 and 32 are moist air "
                      "(dry air × (1 + ω)), as in the EES table; the last column at 31 and 32 is ω. "
                      "Tolerances: T ± 0.3 °F, h ± 0.5 % (± 1 % for humid air), s ± 0.002 Btu/lbm·R, "
                      "ṁ ± 1 %, x ± 0.002, ω ± 0.0002.",
    "project_results": "Device powers and cycle results against the EES Solution window.",
    "project_sweep": "FWH 2 pressure varied from 20 to 30 psia with pump 1 discharging 5 psi above it, as in the "
                     "EES code; TurboSim solves the cycle at each pressure and finds the best one by a "
                     "golden-section search.",
}


# ---------------------------------------------------------------- runner
def run(selected=None, names=None):
    rows = []
    for book, ex, title, fn in CASES:
        if selected and not any(s in book.lower() for s in selected):
            continue
        if names is not None:
            names.append(fn.__name__)
        try:
            checks = fn()
            err = None
        except Exception as e:            # a crash is a failure, with the reason shown
            checks, err = [], f"{type(e).__name__}: {e}"
            if "--trace" in sys.argv:
                traceback.print_exc()
        rows.append((book, ex, title, checks, err))
    return rows


def fmt(x):
    if x is None:
        return "-"
    a = abs(x)
    if a >= 1e5:
        return f"{x:.4g}"
    if a >= 100:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def print_table(rows):
    n_ok = n_all = 0
    for book, ex, title, checks, err in rows:
        ok = err is None and all(c.ok for c in checks)
        n_all += 1
        n_ok += ok
        print(f"\n{'PASS' if ok else 'FAIL'}  {book} {'' if book == PROJECT else 'Example '}{ex}: {title}")
        if err:
            print(f"      could not solve: {err}")
        for c in checks:
            mark = "ok " if c.ok else "OFF"
            print(f"      {mark} {c.what:34} book {fmt(c.book):>10}  app {fmt(c.app):>10} {c.unit:8} ({c.tol_text()})")
    print(f"\n{n_ok} of {n_all} checks pass (textbook examples and the ME 313 project).")
    return n_ok == n_all


def markdown(rows):
    out = ["| Result | Book | Example | What is checked | Worst miss |", "|---|---|---|---|---|"]
    for book, ex, title, checks, err in rows:
        ok = err is None and all(c.ok for c in checks)
        worst = ""
        if err:
            worst = "could not solve"
        elif checks:
            def miss(c):
                return abs(c.app - c.book) / abs(c.book) if c.book else 0.0
            w = max(checks, key=miss)
            worst = f"{w.what}: book {fmt(w.book)}, app {fmt(w.app)} {w.unit}".strip()
        out.append(f"| {'Pass' if ok else 'FAIL'} | {book} | {ex} | {title} | {worst} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--md", help="also write the summary table to this Markdown file")
    ap.add_argument("--book", action="append", help="only this book (Moran, Penoncello, Dick)")
    ap.add_argument("--trace", action="store_true", help="print a traceback for examples that crash")
    ap.add_argument("--pdf", help="write the printable report to this PDF, only if every example passes")
    args = ap.parse_args()
    names = []
    rows = run([b.lower() for b in args.book] if args.book else None, names)
    ok = print_table(rows)
    if args.md:
        pathlib.Path(args.md).write_text(markdown(rows), encoding="utf-8")
    if args.pdf:
        if ok:
            from validation_report import write_pdf
            import project_313
            write_pdf(args.pdf, rows, {"order": names, "text": GIVEN},
                      project={"name": PROJECT, "givens": project_313.GIVENS, "system": project_313.SYSTEM,
                               "notes": project_313.FACT_CHECK})
            print(f"Report written: {args.pdf}")
        else:
            print("Report NOT written: it is only made when every example passes.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
