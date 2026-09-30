# Shared friction-factor closures.
import numpy as np


# Colebrook-White, iterated, floored at 64/Re. Valid for Re >= ~4000.
def colebrook_f(Re, eps_mm, D_mm, iters=60, f0=0.03):
    with np.errstate(invalid="ignore", divide="ignore"):
        Re = np.asarray(Re, dtype=float)
        f = np.full(np.broadcast(Re, D_mm).shape, f0, dtype=float)
        eps_D = eps_mm / D_mm
        for _ in range(iters):
            f = (-2 * np.log10(eps_D / 3.7 + 2.51 / (Re * np.sqrt(f)))) ** -2
        return np.maximum(f, 64.0 / Re)


# Churchill (1977), continuous across all regimes, no iteration.
def churchill_f(Re, eps_mm, D_mm):
    Re = np.asarray(Re, dtype=float)
    eps_D = eps_mm / D_mm
    term1 = (8.0 / Re) ** 12
    A = (2.457 * np.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * eps_D))) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * (term1 + (A + B) ** -1.5) ** (1.0 / 12.0)


# Closed-form inversion of churchill_f for eps_mm. Ill-conditioned below Re ~2000.
def eps_from_churchill(Re, f, D_mm):
    Re = np.asarray(Re, dtype=float)
    AB = ((f / 8.0) ** 12 - (8.0 / Re) ** 12) ** (-2.0 / 3.0)
    A = AB - (37530.0 / Re) ** 16
    eps_D = (np.exp(-(A ** (1.0 / 16.0)) / 2.457) - (7.0 / Re) ** 0.9) / 0.27
    return eps_D * D_mm


# Polyflo power law f = a0*Re^-n, no laminar floor.
def polyflo_f(Re, a0=0.140370, n=0.151571):
    return a0 * Re ** -n
