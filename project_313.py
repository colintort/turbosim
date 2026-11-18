"""Part 2 of the validation: Colin's ME 313 project (nuclear-heated Rankine
cycle with three closed feedwater heaters, a moisture separator and a wet
cooling tower), rebuilt in TurboSim with the project's own state numbers
and compared with the EES solution.

    python project_313.py           prints the state table, TurboSim vs EES

The diagram is built from the givens only (pressures, efficiencies, heat
input, the saturated-liquid drains, the feedwater leaving each heater at the
drain temperature and the cooling-tower air and water conditions). No EES
result is fed in.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from system_solver import solve_system  # noqa: E402
from pressure_optimizer import cycle_totals  # noqa: E402

PATM = 14.6959          # psia, EES 1 atm
HA = "Humid air"

GIVENS = [
    ("Reactor heat input", "Q̇in = 1 × 10¹⁰ Btu/h"),
    ("Steam generator outlet", "1050 psia, saturated vapour"),
    ("Condenser", "1 psia, saturated liquid out"),
    ("Closed feedwater heaters", "FWH 3 at 140 psia, FWH 2 at 24 psia, FWH 1 at 5 psia; drains leave "
                                 "saturated; feedwater leaves at the drain temperature; drains cascade "
                                 "backwards through valves"),
    ("Moisture separator", "at 24 psia between the HP and LP turbines"),
    ("Pump 1 discharge", "29 psia (24 + 5 psi)"),
    ("Isentropic efficiencies", "HP turbine 0.95, LP turbine 0.92, pumps 0.89"),
    ("Cooling tower", "water 100 °F in, 72 °F out; air 72 °F, 60 % in; 85 °F, 85 % out; 1 atm; "
                      "makeup water at 60 °F mixed into the return water"),
]


SYSTEM = [
    ("Reactor (steam generator)", "6 → 7", "feedwater heated to saturated vapour at 1050 psia by Q̇in"),
    ("HP turbine, two stages", "7 → 8, 9 → 10", "η = 0.95; bleeds to FWH 3 at 140 psia (16) and FWH 2 at 24 psia (17)"),
    ("Moisture separator", "11 → 12 + 18", "saturated vapour to the LP turbine, saturated liquid drained"),
    ("LP turbine, two stages", "12 → 13, 14 → 15", "η = 0.92; bleed to FWH 1 at 5 psia (19)"),
    ("Condenser", "15 + 26 → 1", "saturated liquid at 1 psia; cooling water 28 → 29"),
    ("Pump 1, pump 2", "1 → 2, 4 → 5", "η = 0.89; to 29 psia and 1050 psia"),
    ("Closed FWH 1, 2, 3", "2 → 3, 3 → 4, 5 → 6", "drains 25, 23, 21 saturated; feedwater leaves at the drain T"),
    ("Drain valves", "21 → 22, 23 → 24, 25 → 26, 18 → 20", "throttling, cascading the drains backwards"),
    ("Cooling tower and makeup", "29 → 27, 27 + 30 → 28; air 31 → 32", "makeup at 60 °F replaces the water evaporated"),
]

FACT_CHECK = [
    ("Every state point, flow and result in the EES solution is reproduced by an independent model in "
     "TurboSim within the tolerances listed (steam states within 0.03 %, net power and efficiency within 0.01 %). "
     "The EES equations, mass balances and feedwater-heater energy balances are correct."),
    ("Report Table 4, state 5: the temperature is printed as 139.2 °F. The EES solution (and TurboSim) give "
     "239.2 °F; the pump only adds about 1.4 °F to the 237.8 °F feedwater. This is a typo in the table."),
    ("EES variable T_WB is computed with the DewPoint function, so the values 57.37 °F (state 31) and 79.98 °F "
     "(state 32) are dew points, not wet-bulb temperatures. The wet-bulb temperatures are 62.7 °F and 81.2 °F. "
     "Nothing else uses T_WB, so no result is affected; rename it T_dp or use WetBulb()."),
    ("Report page 11: \"the net power is equivalent to the rate of heat addition\" should read \"to the net heat "
     "transfer (Q̇in − Q̇out)\"; 3.884 × 10⁹ Btu/h of net power against 1 × 10¹⁰ Btu/h of heat added."),
    ("Figure 3 (hand derivation): \"ṁ18 = 1 − ṁ12\" should be ṁ18 = ṁ11 − ṁ12 (= (1 − x11) ṁ11). The EES code "
     "has it right."),
    ("The pumps use the incompressible-liquid estimate h2s = h1 + v1 ΔP. TurboSim takes the isentropic outlet from "
     "the steam tables instead; the two differ by 0.17 % in pump 2 power (4.15 × 10⁷ Btu/h), which changes the "
     "net power by 0.002 %."),
    ("The cooling-tower dry-air flow differs by 0.34 % because EES's AirH2O and CoolProp's humid-air model give "
     "humidity ratios 0.5 % apart at the same T and φ (CoolProp includes the real-gas enhancement factor). "
     "The water-side flows agree within 0.15 %."),
    ("The efficiency is quoted as 0.388402138; property data and pump approximations make only the first four "
     "digits meaningful (0.3884). The optimum FWH 2 pressure, 24 psia, is confirmed: TurboSim finds 24.0 psia "
     "and the curve is flat (η changes by only 0.00001 between 23 and 25 psia)."),
]


def _n(t, ports, asm=None, params=None, name=None, config=None):
    d = {"type": t, "name": name or t, "ports": ports, "assumptions": asm or {}, "params": params or {}}
    if config:
        d["config"] = config
    return d


def payload(p_fwh2=24, tower=True):
    """The diagram as /solve receives it, in English units (°F, psia, Btu/lbm,
    lbm/s, Btu/s). p_fwh2 moves FWH 2 (and pump 1, 5 psi above it, as in the
    EES code); tower=False leaves out the cooling water, for quick sweeps."""
    turb = lambda eta: ({"adiabatic": True}, {"eta": eta})  # noqa: E731
    pump = ({"adiabatic": True}, {"eta": 0.89})
    split = {"isobaric": True, "sameState": True}
    # TurboSim's Closed FWH preset: stream A is the bleed steam and drains,
    # leaving as saturated liquid; stream B is the feedwater, leaving at the
    # drain temperature
    fwh = {"aIsobaric": True, "bIsobaric": True, "noLoss": True, "aSatLiquidOut": True, "idealOutlet": True}
    states = {
        "1": {"P": 1, "x": 0},
        "2": {"P": p_fwh2 + 5},
        "5": {"P": 1050},
        "7": {"P": 1050, "x": 1},
        "8": {"P": 140},
        "10": {"P": p_fwh2},
        "13": {"P": 5},
        "15": {"P": 1},
        "20": {"P": 5},
        "22": {"P": p_fwh2},
        "24": {"P": 5},
        "26": {"P": 1},
        "27": {"P": PATM, "T": 72},
        "28": {"P": PATM},
        "29": {"P": PATM, "T": 100},
        "30": {"P": PATM, "T": 60},
        "31": {"P": PATM, "T": 72, "RH": 60},
        "32": {"P": PATM, "T": 85, "RH": 85},
    }
    nodes = {
        "reactor": _n("hx", {"a_in1": 6, "a_out1": 7}, {"aIsobaric": True}, {"Qdot": 1e10 / 3600}, "Reactor"),
        "hpt1": _n("turbine", {"in": 7, "out": 8}, *turb(0.95), name="HP turbine 1"),
        "sp1": _n("splitter", {"in": 8, "out1": 9, "out2": 16}, split, name="Bleed 1"),
        "hpt2": _n("turbine", {"in": 9, "out": 10}, *turb(0.95), name="HP turbine 2"),
        "sp2": _n("splitter", {"in": 10, "out1": 11, "out2": 17}, split, name="Bleed 2"),
        "sep": _n("flash", {"in": 11, "vapor": 12, "liquid": 18}, name="Moisture separator"),
        "lpt1": _n("turbine", {"in": 12, "out": 13}, *turb(0.92), name="LP turbine 1"),
        "sp3": _n("splitter", {"in": 13, "out1": 14, "out2": 19}, split, name="Bleed 3"),
        "lpt2": _n("turbine", {"in": 14, "out": 15}, *turb(0.92), name="LP turbine 2"),
        "cond": _n("hx", {"a_in1": 15, "a_in2": 26, "a_out1": 1, "b_in1": 28, "b_out1": 29},
                   {"aIsobaric": True, "bIsobaric": True, "noLoss": True, "aSatLiquidOut": True}, name="Condenser"),
        "p1": _n("pump", {"in": 1, "out": 2}, *pump, name="Pump 1"),
        "fwh1": _n("hx", {"a_in1": 19, "a_in2": 20, "a_in3": 24, "a_out1": 25, "b_in1": 2, "b_out1": 3}, fwh,
                   name="FWH 1"),
        "v1": _n("valve", {"in": 25, "out": 26}, {"isenthalpic": True}, name="Drain valve 1"),
        "vs": _n("valve", {"in": 18, "out": 20}, {"isenthalpic": True}, name="Separator drain valve"),
        "fwh2": _n("hx", {"a_in1": 17, "a_in2": 22, "a_out1": 23, "b_in1": 3, "b_out1": 4}, fwh, name="FWH 2"),
        "v2": _n("valve", {"in": 23, "out": 24}, {"isenthalpic": True}, name="Drain valve 2"),
        "p2": _n("pump", {"in": 4, "out": 5}, *pump, name="Pump 2"),
        "fwh3": _n("hx", {"a_in1": 16, "a_out1": 21, "b_in1": 5, "b_out1": 6}, fwh, name="FWH 3"),
        "v3": _n("valve", {"in": 21, "out": 22}, {"isenthalpic": True}, name="Drain valve 3"),
        "tower": _n("coolingTower", {"in": 29, "out": 27, "airIn": 31, "airOut": 32}, {"isobaric": True},
                    name="Cooling tower"),
        "mix": _n("mixer", {"in1": 27, "in2": 30, "out": 28}, {"isobaric": True, "adiabatic": True},
                  name="Makeup mixer"),
        "airIn": _n("atmosphere", {"out": 31}, name="Air in", config={"mode": "intake"}),
        "airOut": _n("atmosphere", {"in": 32}, name="Air out", config={"mode": "exhaust"}),
    }
    if not tower:
        for k in ("tower", "mix", "airIn", "airOut"):
            nodes.pop(k)
        for k in ("27", "28", "29", "30", "31", "32"):
            states.pop(k)
        cond = nodes["cond"]
        cond["ports"] = {"a_in1": 15, "a_in2": 26, "a_out1": 1}
        cond["assumptions"] = {"aIsobaric": True, "aSatLiquidOut": True}
    return {"fluid": "Water", "unitSystem": "EN", "states": states, "nodes": nodes,
            "stateFluids": {"31": HA, "32": HA} if tower else {}}


def efficiency(p_fwh2):
    p = payload(p_fwh2, tower=False)
    r = solve_system(p, lean=True)     # the totals only, no charts (1.10.12)
    if r["errors"]:
        return None
    t = cycle_totals(p, r)
    return t["Wnet"] / t["Qin"]


# EES parametric table (Table 2): P_FWH,2 [psia] -> eta_th
EES_SWEEP = {20: 0.388225098, 21: 0.388306787, 22: 0.388361483, 23: 0.388392363, 24: 0.388402138,
             25: 0.388393142, 26: 0.388367390, 27: 0.388326634, 28: 0.388272403, 29: 0.388206040,
             30: 0.388128727}


def sweep():
    """eta_th at each EES table pressure, and the TurboSim optimum found by a
    golden-section search on the same equations (pump 1 moves with FWH 2)."""
    rows = [(p, efficiency(p), e) for p, e in EES_SWEEP.items()]
    lo, hi = 20.0, 30.0
    g = (5 ** 0.5 - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = efficiency(a), efficiency(b)
    for _ in range(30):
        if fa > fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = efficiency(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = efficiency(b)
    p_opt = 0.5 * (lo + hi)
    return rows, p_opt, efficiency(p_opt)


# EES solution (313_Project_Part_2.EES, Arrays Table): T °F, P psia, h Btu/lbm, s Btu/lbm·R, m lbm/h, x
EES = {
    1: (1, 101.7, 69.72, 0.1326, 1.157e7, 0), 2: (29, 101.7, 69.82, 0.1326, 1.157e7, None),
    3: (29, 162.2, 130.2, 0.2349, 1.157e7, None), 4: (29, 237.8, 206.3, 0.3502, 1.157e7, None),
    5: (1050, 239.2, 209.8, 0.3508, 1.157e7, None), 6: (1050, 353, 326.3, 0.5051, 1.157e7, None),
    7: (1050, 550.6, 1191, 1.385, 1.157e7, 1), 8: (140, 353, 1046, 1.394, 1.157e7, 0.8301),
    9: (140, 353, 1046, 1.394, 9.700e6, 0.8301), 10: (24, 237.8, 939.9, 1.402, 9.700e6, 0.7694),
    11: (24, 237.8, 939.9, 1.402, 8.804e6, 0.7694), 12: (24, 237.8, 1160, 1.717, 6.774e6, 1),
    13: (5, 162.2, 1061, 1.731, 6.774e6, 0.93), 14: (5, 162.2, 1061, 1.731, 6.414e6, 0.93),
    15: (1, 101.7, 974.6, 1.745, 6.414e6, 0.8737), 16: (140, 353, 1046, 1.394, 1.869e6, 0.8301),
    17: (24, 237.8, 939.9, 1.402, 896324, 0.7694), 18: (24, 237.8, 206.2, 0.3502, 2.030e6, 0),
    19: (5, 162.2, 1061, 1.731, 359305, 0.93), 20: (5, 162.2, 206.2, 0.3572, 2.030e6, 0.07602),
    21: (140, 353, 324.9, 0.5071, 1.869e6, 0), 22: (24, 237.8, 324.9, 0.5204, 1.869e6, 0.1245),
    23: (24, 237.8, 206.2, 0.3502, 2.766e6, 0), 24: (5, 162.2, 206.2, 0.3572, 2.766e6, 0.07602),
    25: (5, 162.2, 130.2, 0.2349, 5.155e6, 0), 26: (1, 101.7, 130.2, 0.2403, 5.155e6, 0.05838),
    27: (14.7, 72, 40.11, None, 2.122e8, None), 28: (14.7, 71.75, 39.86, None, 2.168e8, None),
    29: (14.7, 100, 68.07, None, 2.168e8, None), 30: (14.7, 60, 28.12, None, 4.594e6, None),
    31: (14.7, 72, 28.23, None, 3.806e8, None), 32: (14.7, 85, 44.8, None, 3.852e8, None),
}
EES_HUMID = {31: 0.01003, 32: 0.02222}          # ω
EES_SUMMARY = {   # Btu/h unless noted
    "W_HPT_1": 1.676e9, "W_HPT_2": 1.027e9, "W_LPT_1": 6.711e8, "W_LPT_2": 5.522e8,
    "W_P_1": 1.087e6, "W_P_2": 4.152e7, "W_net": 3.884e9, "Q_out": 6.116e9, "N_therm": 0.388402138,
    "m_total": 1.157e7, "m_air": 3.768e8, "m_CW": 2.168e8, "m_makeup": 4.594e6, "W_net_MW": 1138,
}


def solve():
    p = payload()
    r = solve_system(p)
    return p, r


def compare(r):
    """Rows (state, quantity, TurboSim, EES, unit) for every value the EES table lists."""
    s, d = r["states"], r["devices"]
    rows = []
    for n, (P, T, h, sv, m, x) in EES.items():
        st = s.get(str(n), {})

        def g(k):
            v = st.get(k)
            return None if v in ("", None) else float(v)
        m_app = g("m")
        # humid-air states carry ṁ of dry air; EES lists the moist-air flow ṁa(1 + ω)
        if n in EES_HUMID and m_app is not None and g("w") is not None:
            m_app *= 1 + g("w")
        rows.append((n, "T", g("T"), T, "°F"))
        rows.append((n, "h", g("h"), h, "Btu/lbm"))
        if sv is not None:
            rows.append((n, "s", g("s"), sv, "Btu/lbm·R"))
        rows.append((n, "ṁ", None if m_app is None else m_app * 3600, m, "lbm/h"))
        if x is not None and n not in (1, 7, 12, 18, 21, 23, 25):
            rows.append((n, "x", g("x"), x, "-"))
        if n in EES_HUMID:
            rows.append((n, "ω", g("w"), EES_HUMID[n], "lbm/lbm"))
    return rows


def summary(p, r):
    d = r["devices"]
    t = cycle_totals(p, r)
    w = lambda k: float(d[k]["Wdot"]) * 3600  # noqa: E731
    out = {
        "W_HPT_1": w("hpt1"), "W_HPT_2": w("hpt2"), "W_LPT_1": w("lpt1"), "W_LPT_2": w("lpt2"),
        "W_P_1": w("p1"), "W_P_2": w("p2"),
        "W_net": w("hpt1") + w("hpt2") + w("lpt1") + w("lpt2") - w("p1") - w("p2"),
        "Q_out": abs(float(d["cond"]["Qdot"])) * 3600,
        "m_total": float(r["states"]["7"]["m"]) * 3600,
        "m_air": float(d["tower"].get("ma") or r["states"]["31"]["m"]) * 3600,
        "m_CW": float(r["states"]["29"]["m"]) * 3600,
        "m_makeup": float(r["states"]["30"]["m"]) * 3600,
    }
    out["N_therm"] = out["W_net"] / 1e10
    out["W_net_MW"] = out["W_net"] * 1055.05585262 / 3600 / 1e6
    out["_totals"] = t
    return out


if __name__ == "__main__":
    p, r = solve()
    if r["errors"]:
        print("ERRORS:", r["errors"])
    for n, q, app, ees, unit in compare(r):
        diff = "" if app is None or not ees else f"{(app - ees) / abs(ees) * 100:+.2f} %"
        print(f"{n:>3} {q:2} app {app if app is None else round(app, 5)!s:>14}  EES {ees!s:>10} {unit:10} {diff}")
    if not r["errors"]:
        sm = summary(p, r)
        for k, v in EES_SUMMARY.items():
            print(f"{k:10} app {sm[k]:.5g}  EES {v:.5g}  {(sm[k] - v) / v * 100:+.2f} %")
