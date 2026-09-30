# Parker Hannifin f-Re closures: regime fits, Churchill and Cheng blends, the
# all-size collapse, the capacity-safe envelope, H2 tables, and the measured
# corrugated geometry.
# Run: python parker_closures.py regime_fits envelope h2_tables ...  (no args = all)
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import csst_manufacturer_data as md
from friction import churchill_f

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FITS = os.path.join(HERE, "output", "plots", "parker_hannifin", "regime_fits")
OUT_SMOOTH = os.path.join(HERE, "output", "plots", "parker_hannifin", "smooth_closure")
OUT_H2 = os.path.join(HERE, "output", "plots", "parker_hannifin", "h2_tables")
OUT_PLOTS = os.path.join(HERE, "output", "plots")
CONST_JSON = os.path.join(OUT_SMOOTH, "parker_envelope_constants.json")

RE_LAM, RE_TURB = 2300.0, 4000.0
S_NG, MU_NG = 0.60, 1.2e-5
S_H2, MU_H2 = 0.0696, 8.76e-6
B_RATIO = 37.26 / 12.10          # HHV NG / HHV H2 per unit volume
SCFH_TO_M3S = 7.8658e-6
R_air, Pb, Tb = md.R_air, md.Pb, md.Tb

REGIMES = [("laminar", 0.0, RE_LAM, "#d62728"),
           ("transitional", RE_LAM, RE_TURB, "#ff9500"),
           ("turbulent", RE_TURB, np.inf, "#1a9850")]


def _pipes():
    return sorted(md.load("parker").items(), key=lambda kv: kv[1]["ehd"])


def _clean(arr):
    Re, f = arr["Re"].ravel(), arr["f_moody"].ravel()
    m = np.isfinite(Re) & np.isfinite(f) & (Re > 0) & (f > 0)
    return Re[m], f[m]


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _log_axes(ax, title=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.set_xlabel("Re (-)")
    ax.set_ylabel("f (-)")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10)


# Geometric blend (Cheng 2008): laminar asymptote handed off to a plateau.
def blend_f(Re, f_inf, Re_c, m, c_lam=1.0, c_t=1.0):
    alpha = 1.0 / (1.0 + (Re / Re_c) ** m)
    return (c_lam * 64.0 / Re) ** alpha * (c_t * f_inf) ** (1.0 - alpha)


def _f_inf(Re, f):
    return float(np.exp(np.mean(np.log(f[Re >= RE_TURB]))))


# =============================================================== regime fits

def _powfit(Re, f):
    lRe, lf = np.log(Re), np.log(f)
    slope, ic = np.polyfit(lRe, lf, 1)
    ss_tot = np.sum((lf - lf.mean()) ** 2)
    r2 = float(1 - np.sum((lf - (slope * lRe + ic)) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return float(np.exp(ic)), float(-slope), r2


# A different diameter slides every point along slope -5 in (log Re, log f),
# so it can move a branch's level but never its slope.
def regime_fits():
    order = _pipes()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    rows = []
    for ax, (label, pipe) in zip(axes.flat, order):
        arr = md.pipe_arrays(pipe)
        Re, f = _clean(arr)
        ehd, D_mm = int(pipe["ehd"]), float(arr["D_mm"])

        ax.scatter(Re, f, s=18, color="#4a4a4a", alpha=0.55, zorder=3)
        Re_ref = np.logspace(np.log10(Re.min() / 1.3), np.log10(RE_LAM), 60)
        ax.plot(Re_ref, 64.0 / Re_ref, "k-", lw=1.5, label="f = 64/Re", zorder=2)
        ax.axvspan(RE_LAM, RE_TURB, color="lightyellow", alpha=0.6, zorder=0)

        for name, lo, hi, col in REGIMES:
            mm = (Re >= lo) & (Re < hi)
            row = dict(EHD=ehd, regime=name, N=int(mm.sum()))
            if mm.sum() >= 3:
                a, n, r2 = _powfit(Re[mm], f[mm])
                row.update(a=a, n=n, r2=r2, Re_lo=float(Re[mm].min()), Re_hi=float(Re[mm].max()))
                Rf = np.logspace(np.log10(Re[mm].min()), np.log10(Re[mm].max()), 60)
                ax.plot(Rf, a * Rf ** -n, color=col, lw=2.6, zorder=5,
                        label=f"{name} (N={mm.sum()}): f = {a:.3g}·Re$^{{{-n:+.3f}}}$   R²={r2:.2f}")
                if name == "turbulent":
                    f_bar = float(np.exp(np.mean(np.log(f[mm]))))
                    ksd = 3.7 / 10 ** (1.0 / (2.0 * np.sqrt(f_bar)))
                    row.update(f_bar=f_bar, ks_over_D=ksd)
                    ax.axhline(f_bar, color=col, ls=":", lw=1.3, alpha=0.9,
                               label=f"plateau f̄ = {f_bar:.3f}  →  ks/D = {ksd:.3f}")
                if name == "laminar":
                    # D that would make each laminar point exactly Hagen-Poiseuille.
                    gam = (64.0 / (f[mm] * Re[mm])) ** 0.25
                    row.update(gamma_med=float(np.median(gam)),
                               D_HP_med=float(np.median(gam) * D_mm),
                               D_HP_lo=float(gam.min() * D_mm), D_HP_hi=float(gam.max() * D_mm))
            rows.append(row)

        _log_axes(ax, f"Parker Hannifin - EHD {ehd}  (D = {D_mm:.2f} mm, EHD-nominal)")
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

    fig.suptitle("Parker Hannifin - per-regime power-law fits f = a·Re$^{-n}$\n"
                 "laminar (Re<2300) / transitional (2300-4000) / turbulent (≥4000), "
                 "fitted independently per EHD", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, os.path.join(OUT_FITS, "parker_regime_fits_by_ehd.png"))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for ax, (label, pipe) in zip(axes.flat, order):
        Re, f = _clean(md.pipe_arrays(pipe))
        ax.scatter(Re, f, s=18, color="#4a4a4a", alpha=0.55, zorder=3)
        _log_axes(ax)
    fig.tight_layout()
    _save(fig, os.path.join(OUT_FITS, "f_vs_Re_by_ehd_scatteronly.png"))

    _, pipe13 = next((l, p) for l, p in order if int(p["ehd"]) == 13)
    arr13 = md.pipe_arrays(pipe13)
    Re, f = _clean(arr13)
    D_mm = float(arr13["D_mm"])
    lam = Re < RE_LAM
    gam = float(np.median((64.0 / (f[lam] * Re[lam])) ** 0.25))

    fig, ax = plt.subplots(figsize=(9.5, 7))
    ax.scatter(Re, f, s=24, color="#4a4a4a", alpha=0.6, zorder=3,
               label=f"nominal D = {D_mm:.2f} mm (EHD·25.4/32)")
    ax.scatter(Re / gam, f * gam ** 5, s=24, color="#c02428", alpha=0.6, zorder=4,
               label=f"rescaled D = {gam * D_mm:.2f} mm (γ = {gam:.3f}, laminar-matched)")
    i0 = np.argmin(Re)
    g_line = np.array([0.85, 1.10])
    ax.plot(Re[i0] / g_line, f[i0] * g_line ** 5, color="#2a78d6", lw=1.4, ls="--",
            label="direction a point moves when D changes (slope -5)")
    ax.axvspan(RE_LAM, RE_TURB, color="lightyellow", alpha=0.6, zorder=0)
    _log_axes(ax)
    ax.set_title("Parker EHD 13 - what a different diameter can and cannot fix\n"
                 "every point slides along slope -5 when D changes: the laminar LEVEL "
                 "can be matched to 64/Re,\nbut the laminar SLOPE (and the plateau shape) "
                 "is diameter-invariant", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, os.path.join(OUT_FITS, "parker_D_rescale_demo_ehd13.png"))

    cols = ["EHD", "regime", "N", "a", "n", "r2", "Re_lo", "Re_hi", "f_bar",
            "ks_over_D", "gamma_med", "D_HP_med", "D_HP_lo", "D_HP_hi"]
    print("\n--- Per-regime fits (f = a*Re^-n) + turbulent plateau + implied "
          "Hagen-Poiseuille diameter ---")
    print(pd.DataFrame(rows).reindex(columns=cols).round(4).to_string(index=False))


# ============================================================ smooth closures

# One calibrated constant per EHD: eps/D set so Churchill's fully-rough
# plateau equals the measured turbulent level. No regression.
def churchill_closure():
    print("--- Churchill closure per EHD ---")
    print(f"{'EHD':>4} {'D_mm':>7} {'f_bar':>7} {'eps/D':>7} {'eps_mm':>7} "
          f"{'rms_lam':>8} {'rms_trans':>9} {'rms_turb':>8} {'rms_all':>8}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for ax, (label, pipe) in zip(axes.flat, _pipes()):
        arr = md.pipe_arrays(pipe)
        Re, f = _clean(arr)
        ehd, D_mm = int(pipe["ehd"]), float(arr["D_mm"])

        mt = Re >= RE_TURB
        f_bar = _f_inf(Re, f)
        eps_D = 3.7037 * np.exp(-np.sqrt(8.0 / f_bar) / 2.457)
        eps_mm = eps_D * D_mm

        resid = np.log(f / churchill_f(Re, eps_mm, D_mm))
        def rms(m):
            return float(np.sqrt(np.mean(resid[m] ** 2))) if m.any() else np.nan
        m_lam, m_tr = Re < RE_LAM, (Re >= RE_LAM) & (Re < RE_TURB)
        print(f"{ehd:>4} {D_mm:>7.2f} {f_bar:>7.4f} {eps_D:>7.4f} {eps_mm:>7.3f} "
              f"{rms(m_lam):>8.3f} {rms(m_tr):>9.3f} {rms(mt):>8.3f} "
              f"{float(np.sqrt(np.mean(resid ** 2))):>8.3f}")

        Re_line = np.logspace(np.log10(Re.min() / 1.5), np.log10(Re.max() * 1.5), 400)
        ax.scatter(Re, f, s=18, color="#4a4a4a", alpha=0.55, zorder=3,
                   label="published cells (back-calc)")
        ax.plot(Re_line, churchill_f(Re_line, eps_mm, D_mm), color="#c02428", lw=2.6,
                zorder=5, label=f"Churchill 1977, eps/D = {eps_D:.3f}")
        ax.plot(Re_line, 64.0 / Re_line, "k--", lw=1.2, alpha=0.7, label="f = 64/Re")
        ax.axhline(f_bar, color="#1a9850", ls=":", lw=1.3, alpha=0.9,
                   label=f"fully-rough plateau f = {f_bar:.3f}")
        ax.axvspan(RE_LAM, RE_TURB, color="lightyellow", alpha=0.6, zorder=0)
        _log_axes(ax, f"Parker Hannifin - EHD {ehd}  (D = {D_mm:.2f} mm)")
        ax.set_ylim(min(f.min(), 64.0 / Re.max()) / 1.6,
                    max(f.max(), 64.0 / Re.min() * 1.2) * 1.4)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig.suptitle("Parker Hannifin - single smooth closure per EHD (Churchill 1977)\n"
                 "one calibrated constant per EHD: ε/D set so the fully-rough limit "
                 "equals the measured plateau", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, os.path.join(OUT_SMOOTH, "parker_churchill_closure_by_ehd.png"))


# Churchill dips onto 64/Re at Re~2300; Cheng's geometric blend hands off
# monotonically instead. Re_c and m fitted per EHD on ln f.
def cheng_closure():
    print("--- Cheng-type blend per EHD: f = (64/Re)^a * f_inf^(1-a) ---")
    print(f"{'EHD':>4} {'f_inf':>7} {'ks/D':>7} {'Re_c':>7} {'m':>6} {'rms_lam':>8} "
          f"{'rms_trans':>9} {'rms_turb':>8} {'rms_all':>8} {'rms_Churchill':>13}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for ax, (label, pipe) in zip(axes.flat, _pipes()):
        arr = md.pipe_arrays(pipe)
        Re, f = _clean(arr)
        ehd, D_mm = int(pipe["ehd"]), float(arr["D_mm"])

        mt = Re >= RE_TURB
        f_inf = _f_inf(Re, f)
        ksD = 3.7 / 10 ** (1.0 / (2.0 * np.sqrt(f_inf)))
        eps_mm = ksD * D_mm
        lf = np.log(f)

        def cost(x):
            Re_c, m = np.exp(x)
            return float(np.sum((np.log(blend_f(Re, f_inf, Re_c, m)) - lf) ** 2))

        best = min((minimize(cost, np.log([rc0, m0]), method="Nelder-Mead",
                             options=dict(xatol=1e-4, fatol=1e-8, maxiter=2000))
                    for rc0 in (1000.0, 2000.0, 2720.0) for m0 in (1.5, 3.0, 6.0)),
                   key=lambda r: r.fun)
        Re_c, m = np.exp(best.x)

        resid = lf - np.log(blend_f(Re, f_inf, Re_c, m))
        resid_ch = lf - np.log(churchill_f(Re, eps_mm, D_mm))
        def rms(r, msk):
            return float(np.sqrt(np.mean(r[msk] ** 2))) if msk.any() else np.nan
        m_lam, m_tr = Re < RE_LAM, (Re >= RE_LAM) & (Re < RE_TURB)
        allm = np.ones_like(mt, bool)
        print(f"{ehd:>4} {f_inf:>7.4f} {ksD:>7.4f} {Re_c:>7.0f} {m:>6.2f} "
              f"{rms(resid, m_lam):>8.3f} {rms(resid, m_tr):>9.3f} {rms(resid, mt):>8.3f} "
              f"{rms(resid, allm):>8.3f} {rms(resid_ch, allm):>13.3f}")

        Re_line = np.logspace(np.log10(Re.min() / 1.5), np.log10(Re.max() * 1.5), 400)
        ax.scatter(Re, f, s=18, color="#4a4a4a", alpha=0.55, zorder=3,
                   label="published cells (back-calc)")
        ax.plot(Re_line, churchill_f(Re_line, eps_mm, D_mm), color="#9a9a9a", lw=1.6,
                alpha=0.8, zorder=4, label="Churchill 1977 (the kink)")
        ax.plot(Re_line, blend_f(Re_line, f_inf, Re_c, m), color="#c02428", lw=2.6,
                zorder=5, label=f"Cheng blend: Re$_c$={Re_c:.0f}, m={m:.2f}")
        ax.plot(Re_line, 64.0 / Re_line, "k--", lw=1.2, alpha=0.7, label="f = 64/Re")
        ax.axhline(f_inf, color="#1a9850", ls=":", lw=1.3, alpha=0.9,
                   label=f"plateau f$_\\infty$ = {f_inf:.3f}  (ks/D = {ksD:.3f})")
        ax.axvspan(RE_LAM, RE_TURB, color="lightyellow", alpha=0.6, zorder=0)
        _log_axes(ax, f"Parker Hannifin - EHD {ehd}  (D = {D_mm:.2f} mm)")
        ax.set_ylim(min(f.min(), f_inf) / 2.0, max(f.max(), 64.0 / Re.min()) * 1.5)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    fig.suptitle("Parker Hannifin - kink-free smooth closure per EHD (Cheng-2008-type blend)\n"
                 r"f = (64/Re)$^{\alpha}$ · f$_\infty^{(1-\alpha)}$,   "
                 r"$\alpha$ = 1/(1+(Re/Re$_c$)$^m$);  Churchill shown grey for comparison",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, os.path.join(OUT_SMOOTH, "parker_cheng_closure_by_ehd.png"))


COLLAPSE_RE_C, COLLAPSE_M = 1681, 2.900


# Below Re_c the size-dependent term's weight vanishes, so every EHD must
# collapse onto the same 64/Re branch.
def all_ehd_collapse():
    cmap = plt.get_cmap("viridis")
    order = _pipes()
    ehds = [int(p["ehd"]) for _, p in order]
    norm = plt.Normalize(vmin=min(ehds), vmax=max(ehds))

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    Re_lo_all = np.inf
    for label, pipe in order:
        arr = md.pipe_arrays(pipe)
        Re, f = _clean(arr)
        col = cmap(norm(int(pipe["ehd"])))
        f_inf = _f_inf(Re, f)
        Re_lo_all = min(Re_lo_all, Re.min())
        ax.scatter(Re, f, s=22, color=col, alpha=0.65, zorder=3,
                   label=f"EHD {int(pipe['ehd'])}  (D = {arr['D_mm']:.1f} mm, "
                         f"f$_\\infty$ = {f_inf:.3f})")
        Re_line = np.logspace(np.log10(Re.min() / 1.3), np.log10(Re.max() * 1.3), 300)
        ax.plot(Re_line, blend_f(Re_line, f_inf, COLLAPSE_RE_C, COLLAPSE_M),
                color=col, lw=1.9, alpha=0.9, zorder=4)

    Re_ref = np.logspace(np.log10(Re_lo_all / 1.5), np.log10(3000.0), 100)
    ax.plot(Re_ref, 64.0 / Re_ref, "k--", lw=1.6, zorder=5,
            label="f = 64/Re (common laminar asymptote)")
    ax.axvline(COLLAPSE_RE_C, color="grey", ls=":", lw=1.2)
    ax.annotate(f"Re$_c$ = {COLLAPSE_RE_C:.0f}\n(shared, all sizes)",
                xy=(COLLAPSE_RE_C, 0.55), fontsize=9, ha="center", va="top", color="dimgrey")
    for xy, text in (((500, 0.16), "one common branch:\nsize does not enter below Re$_c$"),
                     ((2.5e4, 0.19), "branches fan out by f$_\\infty$(EHD) only")):
        ax.annotate(text, xy=xy, fontsize=10, color="black",
                    bbox=dict(boxstyle="round", fc="lightyellow", ec="grey", alpha=0.9))

    _log_axes(ax)
    ax.set_xlabel("Reynolds number Re (-)", fontsize=11)
    ax.set_ylabel("Darcy-Weisbach friction factor f (-)", fontsize=11)
    ax.set_title("Parker Hannifin - all pipe sizes on one axes\n"
                 f"universal closure, $\\alpha$ = 1/(1+(Re/{COLLAPSE_RE_C:.0f})"
                 f"$^{{{COLLAPSE_M:.2f}}}$) - identical below transition, plateau "
                 "height is the only size effect", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, os.path.join(OUT_SMOOTH, "parker_all_ehd_collapse.png"))


# ============================================================ upper envelope

COVERAGE = 0.90      # Q-space coverage target, enforced per EHD
Q_TOL = 1.02         # a cell counts as covered if Q_pred <= Q_TOL * Q_pub


def _envelope_cells():
    cells = {k: [] for k in ("ehd", "D_mm", "f_inf", "P1", "P2", "L_m", "Q_pub", "f", "Re")}
    finfo = {}
    for label, pipe in _pipes():
        arr = md.pipe_arrays(pipe)
        ehd, D_mm = int(pipe["ehd"]), float(arr["D_mm"])
        Re2, f2, Q2 = arr["Re"], arr["f_moody"], arr["Q_scfh"]
        L_ft = arr["L_ft"]
        P12 = np.array([md.get_pressures_kPa(c) for c in pipe["conditions"]])
        ok = np.isfinite(Re2) & np.isfinite(f2) & (Re2 > 0) & (f2 > 0)
        f_inf = float(np.exp(np.mean(np.log(f2[ok & (Re2 >= RE_TURB)]))))
        finfo[ehd] = dict(D_mm=D_mm, f_inf=f_inf)
        for i in range(Re2.shape[0]):
            for j in range(Re2.shape[1]):
                if not ok[i, j]:
                    continue
                cells["ehd"].append(ehd)
                cells["D_mm"].append(D_mm)
                cells["f_inf"].append(f_inf)
                cells["P1"].append(P12[i, 0])
                cells["P2"].append(P12[i, 1])
                cells["L_m"].append(L_ft[j] * 0.3048)
                cells["Q_pub"].append(Q2[i, j] * SCFH_TO_M3S)
                cells["f"].append(f2[i, j])
                cells["Re"].append(Re2[i, j])
    return {k: np.array(v) for k, v in cells.items()}, finfo


# Isothermal GFE forward solve under an arbitrary f(Re) closure, Picard iterated.
def gfe_forward(P1, P2, L_m, D_mm, f_inf, S, mu, shape_params):
    Re_c, m, c_lam, c_t = shape_params
    D_m = D_mm / 1000.0
    area = np.pi / 4 * D_m ** 2
    L_km = np.asarray(L_m, float) / 1000.0
    P1, P2 = np.asarray(P1, float), np.asarray(P2, float)
    P_avg = (2.0 / 3.0) * (P1 + P2 - (P1 * P2) / (P1 + P2))
    f = np.full(np.broadcast(P1, L_km).shape, 0.05, dtype=float)
    Q_base = Re = None
    for _ in range(300):
        with np.errstate(invalid="ignore", divide="ignore"):
            Q_day = (1.1494e-3 * (Tb / Pb)
                     * np.sqrt((P1 ** 2 - P2 ** 2) / (S * Tb * L_km * f)) * D_mm ** 2.5)
            Q_base = Q_day / 86400.0
            V = (Q_base * (Pb / P_avg)) / area
            Re = (S * P_avg * 1000 / (R_air * Tb)) * V * D_m / mu
            f_new = blend_f(Re, f_inf, Re_c, m, c_lam, c_t)
        step = np.nanmax(np.abs(f_new / f - 1.0))
        f = f + 0.5 * (f_new - f)
        if step < 1e-8:
            break
    return Q_base, Re


# Best fit on ln f, then the tightest upper envelope whose forward-solved
# capacity stays under the published value (within the rounding band).
def envelope():
    C, finfo = _envelope_cells()
    EHDS = sorted(finfo)

    def fit_shape(keep):
        lf = np.log(C["f"][keep])

        def cost(x):
            Re_c, m = np.exp(x)
            r = lf - np.log(blend_f(C["Re"][keep], C["f_inf"][keep], Re_c, m))
            return float(np.sum(r ** 2))

        best = min((minimize(cost, np.log([rc0, m0]), method="Nelder-Mead",
                             options=dict(xatol=1e-4, fatol=1e-8, maxiter=2000))
                    for rc0 in (1200.0, 1800.0, 2600.0) for m0 in (2.0, 3.0, 5.0)),
                   key=lambda r: r.fun)
        return np.exp(best.x)

    keep = np.ones(C["f"].size, bool)
    Re_c, m = fit_shape(keep)
    resid = np.log(C["f"]) - np.log(blend_f(C["Re"], C["f_inf"], Re_c, m))
    sig = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    keep = np.abs(resid - np.median(resid)) <= 3.0 * sig     # 3 robust sigma
    n_out = int((~keep).sum())
    Re_c, m = fit_shape(keep)
    resid = np.log(C["f"]) - np.log(blend_f(C["Re"], C["f_inf"], Re_c, m))
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)))

    def cov_and_margin(p):
        Q_pred, _ = gfe_forward(C["P1"], C["P2"], C["L_m"], C["D_mm"], C["f_inf"],
                                S_NG, MU_NG, (p[2], p[3], p[0], p[1]))
        slack = np.log(C["Q_pub"] / Q_pred)        # >= 0 means capacity-safe
        tol = np.log(Q_TOL)
        covs = {e: float(np.mean(slack[keep & (C["ehd"] == e)] >= -tol)) for e in EHDS}
        return slack, covs

    def env_cost(x):
        c_lam, c_t, Re_c_e, m_e = np.exp(x)
        if not (1.0 <= c_lam < 4.0 and 1.0 <= c_t < 2.0
                and 300 < Re_c_e < 10000 and 0.5 < m_e < 25):
            return 1e9
        slack, covs = cov_and_margin((c_lam, c_t, Re_c_e, m_e))
        pen = sum(max(0.0, COVERAGE - c) for c in covs.values())
        return float(np.mean(slack[keep])) + 1e3 * pen

    best_e = min((minimize(env_cost, np.log([cl0, ct0, rc0, m0]), method="Nelder-Mead",
                           options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000, maxfev=4000))
                  for cl0 in (1.4, 1.8) for ct0 in (1.10, 1.25)
                  for rc0 in (1400.0, 1900.0) for m0 in (3.0,)),
                 key=lambda r: r.fun)
    c_lam, c_t, Re_c_e, m_e = np.exp(best_e.x)
    slack, covs = cov_and_margin((c_lam, c_t, Re_c_e, m_e))

    f_env_at = blend_f(C["Re"], C["f_inf"], Re_c_e, m_e, c_lam, c_t)
    fcov = {e: float(np.mean((C["f"] <= f_env_at)[keep & (C["ehd"] == e)])) for e in EHDS}

    print("--- Universal closure (mu_NG = 1.2e-5 basis) ---")
    print(f"best fit:  Re_c = {Re_c:.0f}, m = {m:.3f}  (rms ln-resid {rms:.3f} on "
          f"{int(keep.sum())} pts; {n_out} outliers removed at 3 robust sigma)")
    print("plateaus f_inf: " + ", ".join(f"EHD {e}: {finfo[e]['f_inf']:.4f}" for e in EHDS))
    print("\n--- UPPER envelope (capacity-safe) ---")
    print(f"c_lam = {c_lam:.4f}, c_t = {c_t:.4f}, Re_c = {Re_c_e:.0f}, m = {m_e:.3f}")
    print(f"criterion: Q_pred <= {Q_TOL:g}*Q_pub for >= {COVERAGE:.0%} of retained cells, "
          f"per EHD")
    print(f"\n{'EHD':>4} {'Q-coverage':>10} {'f-coverage':>10} {'mean margin %':>13} "
          f"{'median margin %':>15}")
    for e in EHDS:
        sl = slack[keep & (C["ehd"] == e)]
        print(f"{e:>4} {covs[e]:>10.2%} {fcov[e]:>10.2%} "
              f"{100 * (np.exp(np.mean(sl)) - 1):>13.1f} "
              f"{100 * (np.exp(np.median(sl)) - 1):>15.1f}")

    os.makedirs(OUT_SMOOTH, exist_ok=True)
    json.dump(dict(basis="mu_NG=1.2e-5, bare L, D=EHD*25.4/32 mm",
                   best_fit=dict(Re_c=float(Re_c), m=float(m), rms_ln=rms),
                   envelope=dict(c_lam=float(c_lam), c_t=float(c_t), Re_c=float(Re_c_e),
                                 m=float(m_e), coverage_target=COVERAGE,
                                 Q_tolerance=Q_TOL, space="Q (GFE forward solve)"),
                   f_inf={str(e): finfo[e]["f_inf"] for e in EHDS},
                   D_mm={str(e): finfo[e]["D_mm"] for e in EHDS}),
              open(CONST_JSON, "w"), indent=1)
    print(f"\nSaved: {CONST_JSON}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for ax, e in zip(axes.flat, EHDS):
        mm_ = C["ehd"] == e
        kk, oo = keep & mm_, ~keep & mm_
        d = finfo[e]
        ax.scatter(C["Re"][kk], C["f"][kk], s=18, color="#4a4a4a", alpha=0.55,
                   zorder=3, label="published cells (retained)")
        if oo.any():
            ax.scatter(C["Re"][oo], C["f"][oo], s=45, marker="x", color="#c02428",
                       zorder=4, label=f"outliers removed ({int(oo.sum())})")
        Re_line = np.logspace(np.log10(C["Re"][mm_].min() / 1.5),
                              np.log10(C["Re"][mm_].max() * 1.5), 400)
        ax.plot(Re_line, blend_f(Re_line, d["f_inf"], Re_c, m), color="#9a9a9a", lw=1.8,
                ls="--", zorder=4, label=f"best fit (Re$_c$={Re_c:.0f}, m={m:.2f})")
        ax.plot(Re_line, blend_f(Re_line, d["f_inf"], Re_c_e, m_e, c_lam, c_t),
                color="#b0322a", lw=2.6, zorder=5,
                label=f"upper envelope (c$_{{lam}}$={c_lam:.2f}, c$_t$={c_t:.2f})")
        ax.plot(Re_line, 64.0 / Re_line, "k--", lw=1.0, alpha=0.55, label="f = 64/Re")
        ax.axhline(d["f_inf"], color="#1a9850", ls=":", lw=1.2, alpha=0.85,
                   label=f"plateau f$_\\infty$ = {d['f_inf']:.3f}")
        _log_axes(ax, f"Parker Hannifin - EHD {e}  (D = {d['D_mm']:.2f} mm)")
        ax.set_ylim(min(C["f"][mm_].min(), d["f_inf"]) / 2.0, C["f"][mm_].max() * 1.8)
        ax.legend(fontsize=7.6, loc="lower left", framealpha=0.9)

    fig.suptitle("Parker Hannifin - universal closure + capacity-safe UPPER envelope\n"
                 f"envelope: c$_{{lam}}$={c_lam:.2f}, c$_t$={c_t:.2f}, Re$_c$={Re_c_e:.0f}, "
                 f"m={m_e:.2f}; forward-solved capacity ≤ {Q_TOL:g}·published for "
                 f"≥{COVERAGE:.0%} of cells in every size", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, os.path.join(OUT_SMOOTH, "parker_universal_conservative.png"))


# ============================================================== H2 tables

# The envelope closure is dimensionless in Re, so gas identity enters only
# through density and viscosity.
def h2_tables():
    os.makedirs(OUT_H2, exist_ok=True)
    const = json.load(open(CONST_JSON))
    env = const["envelope"]
    shape = (env["Re_c"], env["m"], env["c_lam"], env["c_t"])

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    for ax, (label, pipe) in zip(axes.flat, _pipes()):
        ehd = int(pipe["ehd"])
        D_mm = const["D_mm"][str(ehd)]
        f_inf = const["f_inf"][str(ehd)]
        L_ft = md.pipe_arrays(pipe)["L_ft"]
        Q_pub_ng = np.array(pipe["flows"], float)
        conds = pipe["conditions"]
        P12 = np.array([md.get_pressures_kPa(c) for c in conds])

        Q_h2, Re_h2 = gfe_forward(P12[:, 0:1], P12[:, 1:2], (L_ft * 0.3048)[None, :],
                                  D_mm, f_inf, S_H2, MU_H2, shape)
        Q_h2_scfh = Q_h2 / SCFH_TO_M3S

        cond_lbl = [f"{c[0]:g} psi in, {c[2]:g} inWC drop" for c in conds]
        pd.DataFrame(np.round(Q_h2_scfh, 1), index=cond_lbl,
                     columns=[f"{v:g} ft" for v in L_ft]).to_csv(
            os.path.join(OUT_H2, f"parker_h2_capacity_ehd{ehd}.csv"))

        cmap = plt.get_cmap("plasma")
        for i, c in enumerate(conds):
            col = cmap(i / max(1, len(conds) - 1) * 0.85)
            mask = np.isfinite(Q_pub_ng[i]) & (Q_pub_ng[i] > 0)
            o = np.argsort(L_ft[mask])
            ax.plot(L_ft[mask][o], Q_h2_scfh[i][mask][o], color=col, lw=1.9, zorder=4,
                    label=f"H2: {cond_lbl[i]}")
            ax.plot(L_ft[mask][o], Q_pub_ng[i][mask][o], color=col, lw=1.2, ls="--",
                    alpha=0.65, zorder=3)
            for j in np.where(mask)[0]:
                rows.append(dict(EHD=ehd, schedule=cond_lbl[i], L_ft=L_ft[j],
                                 Q_H2_scfh=Q_h2_scfh[i, j], Re_H2=Re_h2[i, j],
                                 Q_NG_pub_scfh=Q_pub_ng[i, j],
                                 energy_ratio_H2_over_NG=(Q_h2_scfh[i, j] / B_RATIO)
                                 / Q_pub_ng[i, j]))

        _log_axes(ax, f"EHD {ehd}  (D = {D_mm:.2f} mm)")
        ax.set_xlabel("Length (ft)")
        ax.set_ylabel("Q (SCFH)")
        ax.legend(fontsize=6.5, loc="lower left", framealpha=0.9)

    fig.suptitle("Parker Hannifin grid - HYDROGEN capacity (solid, capacity-safe envelope)\n"
                 "vs published natural-gas capacity (dashed, same colour = same schedule)\n"
                 f"H2: S = {S_H2}, mu = {MU_H2:g} Pa.s", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, os.path.join(OUT_H2, "parker_h2_vs_ng_by_ehd.png"))

    big = pd.DataFrame(rows)
    summary = os.path.join(OUT_H2, "parker_h2_capacity_summary.csv")
    big.to_csv(summary, index=False)
    print(f"Saved: {summary} + per-EHD CSVs")

    print("\n--- Hydrogen operating window and energy comparison (vs published NG) ---")
    print(f"{'EHD':>4} {'Re_H2 min':>10} {'Re_H2 max':>10} {'% cells Re<2300':>15} "
          f"{'median energy ratio':>19} {'min':>6} {'max':>6}")
    for ehd in sorted(big["EHD"].unique()):
        g = big[big["EHD"] == ehd]
        print(f"{ehd:>4} {g['Re_H2'].min():>10.0f} {g['Re_H2'].max():>10.0f} "
              f"{100 * np.mean(g['Re_H2'] < 2300):>14.1f}% "
              f"{g['energy_ratio_H2_over_NG'].median():>19.3f} "
              f"{g['energy_ratio_H2_over_NG'].min():>6.3f} "
              f"{g['energy_ratio_H2_over_NG'].max():>6.3f}")
    print("\nenergy ratio = (Q_H2/3.080)/Q_NG_pub: >1 means H2 delivers MORE energy than")
    print("the published NG rating at the same schedule and length.")


# ======================================================= measured geometry

COORD_DIR = os.path.join(HERE, "..", "cfd", "new", "coordinates")
GEOM_COLOR = {13: "#173753", 18: "#2E6B45", 23: "#8A5A2A", 31: "#8F3D3D"}


# ID = 2*r_min, OD = 2*r_max, D_eff = sqrt(4*A_mean/pi) over one 5-corrugation
# length, from the Creo profile CSVs. EHD*25.4/32 is not used here.
def profile_geometry(ehd, coord_dir=COORD_DIR):
    hits = glob.glob(os.path.join(coord_dir, f"EHD{ehd}", "*_5corr.csv"))
    if not hits:
        raise FileNotFoundError(f"no 5corr profile csv for EHD{ehd} under {coord_dir}")
    d = np.loadtxt(hits[0], delimiter=",")
    d = d[d[:, 1] > 0]                       # drop the axis points closing the section
    x, r = d[:, 0] / 1000.0, d[:, 1] / 1000.0
    L = x.max() - x.min()
    A_mean = np.trapezoid(np.pi * r ** 2, x) / L
    return dict(id_mm=2000.0 * r.min(), od_mm=2000.0 * r.max(),
                d_eff_mm=1000.0 * np.sqrt(4 * A_mean / np.pi),
                a_mean_mm2=1e6 * A_mean, L_m=L)


def measured_geometry():
    print("%5s %8s %9s %9s %9s %9s %10s"
          % ("EHD", "NPS", "OD in", "OD mm", "ID in", "ID mm", "D_eff mm"))
    print("-" * 64)
    data = {}
    for label, pipe in md.load("parker").items():
        ehd = pipe["ehd"]
        g = profile_geometry(ehd)
        print("%5d %8s %9.3f %9.3f %9.3f %9.3f %10.4f"
              % (ehd, label.replace(" inch", ""), g["od_mm"] / 25.4, g["od_mm"],
                 g["id_mm"] / 25.4, g["id_mm"], g["d_eff_mm"]))
        pipe["id_mm"] = g["d_eff_mm"]        # the only substitution
        data[ehd] = md.pipe_arrays(pipe)
    print()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for ehd in sorted(data):
        a = data[ehd]
        Re, f = a["Re"].ravel(), a["f_moody"].ravel()
        m = np.isfinite(Re) & np.isfinite(f)
        ax.scatter(Re[m], f[m], s=26, facecolor="none", edgecolor=GEOM_COLOR[ehd],
                   linewidth=1.2, label="EHD %d, D = %.2f mm" % (ehd, a["D_mm"]))
    ax.plot([50.0, 2300.0], [64.0 / 50.0, 64.0 / 2300.0], color="#5A6470", lw=1.0,
            ls="--", label="64/Re")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Re")
    ax.set_ylabel("Darcy friction factor f")
    ax.set_title("Parker Hannifin 1996 CSST, measured corrugated geometry")
    ax.grid(True, which="both", lw=0.4, alpha=0.4)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(OUT_PLOTS, "parker_f_vs_Re_measured_geometry.png"))

    for ehd in sorted(data):
        a = data[ehd]
        Re, f = a["Re"].ravel(), a["f_moody"].ravel()
        m = np.isfinite(Re) & np.isfinite(f) & (Re > 4000)
        if m.any():
            print("EHD %-3d turbulent plateau: f median %.4f over %d points, Re %.0f - %.0f"
                  % (ehd, np.median(f[m]), m.sum(), Re[m].min(), Re[m].max()))


STEPS = {
    "regime_fits": regime_fits,
    "churchill": churchill_closure,
    "cheng": cheng_closure,
    "collapse": all_ehd_collapse,
    "envelope": envelope,
    "h2_tables": h2_tables,
    "measured_geometry": measured_geometry,
}


if __name__ == "__main__":
    for name in ([a for a in sys.argv[1:] if a in STEPS] or list(STEPS)):
        STEPS[name]()
