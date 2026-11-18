"""Convective film coefficients from Nusselt-number correlations.

Single-phase correlations are the ones in Penoncello, Thermal Energy Systems:
Design and Analysis (2nd ed.), chapter 5:

  inside a tube, turbulent   Gnielinski (Eq. 5.26)
      Nu = (f/8)(Re - 1000) Pr / [1 + 12.7 (f/8)^½ (Pr^⅔ - 1)]
      with the Darcy friction factor f from Swamee-Jain (Table 4.2)
  inside a tube, laminar     fully developed, Nu = 3.66 (the limit of Eq. 5.24)
  annulus                    the tube correlation on the equivalent diameter
                             D_equ = (ID_a² - OD_t²) / OD_t (Eq. 5.81)
  cross flow over a tube     Churchill-Bernstein (Eq. 5.20)
  shell side, baffled        McAdams, Nu = 0.36 Re^0.55 Pr^⅓ (Eq. 5.115) on the
                             shell equivalent diameter (Eq. 5.119), with the
                             flow area A_s = D_s C B / P_T (Eq. 5.120)

The book leaves boiling and condensing to the handbooks (section 5.7); the
standard correlations used here are

  condensing inside a tube   Shah (1979)
      h = h_lo [(1-x)^0.8 + 3.8 x^0.76 (1-x)^0.04 / p_r^0.38]
  boiling inside a tube      Gungor-Winterton (1987)
      h = h_l [1 + 3000 Bo^0.86 + 1.12 (x/(1-x))^0.75 (ρ_l/ρ_v)^0.41]
  condensing on a bank of    Nusselt film theory for N tubes in a column
  horizontal tubes               h = 0.729 [g ρ_l (ρ_l-ρ_v) k_l³ h'_fg / (N μ_l ΔT D)]^¼

h_lo and h_l are Dittus-Boelter liquid coefficients, 0.023 Re^0.8 Pr^0.4 k/D.
Everything is SI.
"""
import math

G0 = 9.80665
ROUGHNESS = 1.5e-6      # drawn tubing, m (Penoncello Table 4.3)


def swamee_jain(Re, rel_rough=0.0):
    """Darcy friction factor, turbulent flow."""
    return 0.25 / math.log10(rel_rough / 3.7 + 5.74 / Re ** 0.9) ** 2


def nu_tube(Re, Pr, rel_rough=0.0):
    """Inside a tube (or an annulus on its equivalent diameter)."""
    if Re < 2300:
        return 3.66
    f = swamee_jain(Re, rel_rough)
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))


def nu_cross_cylinder(Re, Pr):
    """Churchill-Bernstein, cross flow over a cylinder (Re Pr >= 0.2)."""
    return 0.3 + 0.62 * Re ** 0.5 * Pr ** (1 / 3) / (1 + (0.4 / Pr) ** (2 / 3)) ** 0.25 * \
        (1 + (Re / 282000) ** (5 / 8)) ** (4 / 5)


def nu_shell(Re, Pr):
    """McAdams, shell side of a baffled shell-and-tube exchanger."""
    return 0.36 * Re ** 0.55 * Pr ** (1 / 3)


def shell_equivalent_diameter(PT, Do, triangular=True):
    """Eq. 5.119: D_e = X P_T² / (π OD) - OD, X = 4 (square) or 2√3 (triangular)."""
    X = 2 * math.sqrt(3) if triangular else 4.0
    return X * PT * PT / (math.pi * Do) - Do


def _dittus_boelter(Re, Pr, k, D):
    return 0.023 * Re ** 0.8 * Pr ** 0.4 * k / D


def _mean(fn, x0, x1, n=12):
    lo, hi = min(x0, x1), max(x0, x1)
    return sum(fn(lo + (hi - lo) * (i + 0.5) / n) for i in range(n)) / n


def h_condensing_tube(G, D, x_in, x_out, sat):
    """Shah (1979), averaged over the quality range. sat: saturated
    properties {mu_l, k_l, Pr_l, p_r}."""
    h_lo = _dittus_boelter(G * D / sat["mu_l"], sat["Pr_l"], sat["k_l"], D)

    def h(x):
        x = min(max(x, 1e-4), 1 - 1e-4)
        return h_lo * ((1 - x) ** 0.8 + 3.8 * x ** 0.76 * (1 - x) ** 0.04 / sat["p_r"] ** 0.38)
    return _mean(h, x_in, x_out)


def h_boiling_tube(G, D, x_in, x_out, q, sat):
    """Gungor-Winterton (1987), averaged over the quality range (capped at
    x = 0.95, where the wall dries out). q: heat flux on the inside wall."""
    Bo = q / (G * sat["hfg"])

    def h(x):
        x = min(max(x, 1e-4), 0.95)
        h_l = _dittus_boelter(G * (1 - x) * D / sat["mu_l"], sat["Pr_l"], sat["k_l"], D)
        return h_l * (1 + 3000 * Bo ** 0.86 + 1.12 * (x / (1 - x)) ** 0.75 * (sat["rho_l"] / sat["rho_v"]) ** 0.41)
    return _mean(h, x_in, x_out)


def h_condensing_bank(D, dT, N, sat):
    """Film condensation on N horizontal tubes in a vertical column.
    dT: saturation minus wall temperature."""
    hfg = sat["hfg"] + 0.68 * sat["cp_l"] * dT
    return 0.729 * (G0 * sat["rho_l"] * (sat["rho_l"] - sat["rho_v"]) * sat["k_l"] ** 3 * hfg /
                    (N * sat["mu_l"] * dT * D)) ** 0.25
