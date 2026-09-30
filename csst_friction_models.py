# CSST friction and sizing-equation models fitted to the manufacturer tables:
# the universal (alpha, beta) EHD-only friction model, bi-regime power laws,
# the generator equation each sizing table encodes, and a table lookup.
# Run: python csst_friction_models.py universal | biregime | generator | lookup
#      python csst_friction_models.py generator 2,5     (generator step subset)
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar

import csst_manufacturer_data as md
from friction import churchill_f, colebrook_f, polyflo_f

HERE = md.HERE
PLOTS_DIR = os.path.join(HERE, "output", "plots")

SCFH_TO_M3S = 7.8658e-6
PSI_TO_PA = 6894.76
EHD_TO_MM = md.EHD_TO_MM
INWC_TO_PSI = md.INWC_TO_PSI
R_air, Pb, Tb = md.R_air, md.Pb, md.Tb
GAS_PROPERTIES = md.GAS_PROPERTIES

RE_TURBULENT = 4000.0
RE_LAMINAR = 2300.0

# Suspect Parker row (3/4 inch, 2 psi / 41.5 inWC): residuals +20/+25/-31% at
# 150/200/250 ft and a half-struck mark on the original sheet.
PARKER_SUSPECT_PIPE = "3/4 inch"
PARKER_SUSPECT_INWC = 41.5


def _load(source):
    return md.load(source)


# ===================================================== universal (alpha, beta)

FIT_LABELS = {
    "F2": "Parker Hannifin only",
    "F3": "Gastite only (primary reference - largest clean single-manufacturer set)",
    "F4": "TracPipe only",
    "F6": "Parker Parflex only",
    "F5": "All four manufacturers pooled (naive pooling, not a balanced joint fit)",
}
FIT_KEYS = ["F2", "F3", "F4", "F6", "F5"]
FIT_SOURCE = {"F2": "parker", "F3": "gastite", "F4": "tracpipe", "F6": "parflex"}

ALPHA_GRID = (0.7, 0.8, 0.9, 1.0, 1.1)
BETA_GRID = (0.01, 0.03, 0.05, 0.1, 0.2)
X0_DEFAULT = (0.9, 0.05)

# Bare tabulated L throughout: the published tables satisfy exact cross-table
# identities on bare L, so no fitting-length allowance is added.
BASELINE_FITTING_LENGTH_FT = md.BASELINE_FITTING_LENGTH_FT


def load_points(source):
    name = {"gastite": "Gastite", "parflex": "Parflex", "tracpipe": "TracPipe",
            "parker": "Parker Hannifin", "csa": "CSA/ANSI"}[source]
    allowance = 0.0 if source == "parker" else BASELINE_FITTING_LENGTH_FT
    points = []
    for pipe in _load(source).values():
        L_ft_all = md.pipe_arrays(pipe)["L_ft"]
        for cond, flow_row in zip(pipe["conditions"], pipe["flows"]):
            P1, P2 = md.get_pressures_kPa(cond)
            for L_ft, Q_scfh in zip(L_ft_all, flow_row):
                if not np.isfinite(Q_scfh) or Q_scfh == 0:
                    continue
                points.append(dict(source=name, EHD=float(pipe["ehd"]), gas=pipe["gas"],
                                   Q_pub=Q_scfh * SCFH_TO_M3S, condition=cond[3],
                                   P1_kPa=P1, P2_kPa=P2,
                                   L_m=(L_ft + allowance) * 0.3048))
    return points


def points_to_arrays(points):
    return dict(EHD=np.array([p["EHD"] for p in points], dtype=float),
                P1=np.array([p["P1_kPa"] for p in points], dtype=float),
                P2=np.array([p["P2_kPa"] for p in points], dtype=float),
                L=np.array([p["L_m"] for p in points], dtype=float),
                gas=np.array([p["gas"] for p in points], dtype=object),
                Qpub=np.array([p["Q_pub"] for p in points], dtype=float),
                source=np.array([p["source"] for p in points], dtype=object),
                condition=np.array([p["condition"] for p in points], dtype=object))


# GFE forward solve with D_eff = alpha*EHD*25.4/32 and eps = beta*EHD*25.4/32.
# Inside diameter is never used: EHD is the only size parameter.
def Q_gfe_churchill(alpha, beta, EHD, P1_kPa, P2_kPa, L_m, gas,
                    max_iters=200, f0=0.02, tol=1e-6):
    EHD = np.asarray(EHD, dtype=float)
    P1 = np.asarray(P1_kPa, dtype=float)
    P2 = np.asarray(P2_kPa, dtype=float)
    L_m = np.asarray(L_m, dtype=float)
    gas_arr = np.asarray(gas, dtype=object)

    S = np.vectorize(lambda g: GAS_PROPERTIES[g]["S"])(gas_arr)
    mu = np.vectorize(lambda g: GAS_PROPERTIES[g]["mu_Pas"])(gas_arr)

    D_eff_mm = alpha * EHD * EHD_TO_MM
    eps_mm = beta * EHD * EHD_TO_MM
    D_m = D_eff_mm / 1000.0
    area = np.pi / 4 * D_m ** 2
    L_km = L_m / 1000.0
    P_avg = (2.0 / 3.0) * (P1 + P2 - (P1 * P2) / (P1 + P2))      # Menon mean

    f = np.full(P1.shape, f0, dtype=float)
    Q_base = Re = None
    for _ in range(max_iters):
        with np.errstate(invalid="ignore", divide="ignore"):
            Q_day = (1.1494e-3 * (Tb / Pb)
                     * np.sqrt((P1 ** 2 - P2 ** 2) / (S * Tb * L_km * f)) * D_eff_mm ** 2.5)
            Q_base = Q_day / 86400.0
            V = Q_base * (Pb / P_avg) / area
            Re = (S * P_avg * 1000 / (R_air * Tb)) * V * D_m / mu
            f_new = churchill_f(Re, eps_mm, D_eff_mm)
        step = np.nanmax(np.abs(f_new / f - 1))
        f = f + 0.5 * (f_new - f)
        if step < tol:
            break
    else:
        print(f"WARNING: Q_gfe_churchill did not converge in {max_iters} iterations "
              f"(final step={step:.2e})")
    return Q_base, Re


def _errors(alpha, beta, points):
    Q_pred, Re = Q_gfe_churchill(alpha, beta, points["EHD"], points["P1"], points["P2"],
                                 points["L"], points["gas"])
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (Q_pred - points["Qpub"]) / points["Qpub"] * 100
    return e, Re


# RMSE of Q over turbulent points only; the mask is recomputed at each alpha.
def objective_rmse(params, points):
    alpha, beta = params
    if alpha <= 0 or np.any(np.asarray(beta) <= 0):
        return 1e6
    e, Re = _errors(alpha, beta, points)
    mask = (Re > RE_TURBULENT) & np.isfinite(e)
    if mask.sum() == 0:
        return 1e6
    return float(np.sqrt(np.mean(e[mask] ** 2)))


def compute_metrics(alpha, beta, points):
    e, Re = _errors(alpha, beta, points)
    mask = (Re > RE_TURBULENT) & np.isfinite(e)
    if mask.sum() == 0:
        return dict(rmse=np.nan, mae=np.nan, maxerr=np.nan, frac5=np.nan, frac10=np.nan, n=0)
    ev = e[mask]
    return dict(rmse=float(np.sqrt(np.mean(ev ** 2))), mae=float(np.mean(np.abs(ev))),
                maxerr=float(np.max(np.abs(ev))), frac5=float(np.mean(np.abs(ev) <= 5) * 100),
                frac10=float(np.mean(np.abs(ev) <= 10) * 100), n=int(mask.sum()))


def fit_alpha_beta(points, x0=X0_DEFAULT):
    opts = dict(xatol=1e-3, fatol=1e-3, maxiter=200, maxfev=200)
    res = minimize(objective_rmse, x0, args=(points,), method="Nelder-Mead", options=opts)
    if not res.success:
        best = min(((objective_rmse((a, b), points), a, b)
                    for a in ALPHA_GRID for b in BETA_GRID), key=lambda t: t[0])
        res = minimize(objective_rmse, (best[1], best[2]), args=(points,),
                       method="Nelder-Mead", options=opts)
    return float(res.x[0]), float(res.x[1])


# L2: one global alpha, one beta per EHD, fitted jointly.
def fit_alpha_beta_L2(points, x0_alpha=X0_DEFAULT[0], x0_beta=X0_DEFAULT[1]):
    ehds = sorted(set(points["EHD"]))
    idx = np.array([ehds.index(e) for e in points["EHD"]])

    def obj(params):
        alpha, betas = params[0], params[1:]
        if alpha <= 0 or np.any(betas <= 0):
            return 1e6
        return objective_rmse((alpha, betas[idx]), points)

    opts = dict(xatol=1e-3, fatol=1e-3, maxiter=300 * len(ehds), maxfev=300 * len(ehds))
    res = minimize(obj, [x0_alpha] + [x0_beta] * len(ehds), method="Nelder-Mead", options=opts)
    return (float(res.x[0]), {e: float(res.x[1 + i]) for i, e in enumerate(ehds)},
            res.x[1:][idx])


RE_BIN_EDGES = (100, 300, 500, 1000, 2000, 4000, 8000, 15000, 30000, 60000, 120000)
REGIME_COLORS = {"laminar": "#2a78d6", "transitional": "#eb6834", "turbulent": "#1baf7a"}


def _re_regime(re_val):
    return ("laminar" if re_val < RE_LAMINAR
            else "transitional" if re_val < RE_TURBULENT else "turbulent")


def bin_errors_by_re(Re, e):
    rows = []
    for lo, hi in zip(RE_BIN_EDGES[:-1], RE_BIN_EDGES[1:]):
        m = (Re >= lo) & (Re < hi)
        centre = float(np.sqrt(lo * hi))
        if not m.any():
            rows.append([centre, 0, np.nan, np.nan, np.nan, np.nan])
            continue
        ev = e[m]
        rows.append([centre, int(m.sum()), round(float(np.mean(ev)), 3),
                     round(float(np.std(ev)), 3), round(float(np.sqrt(np.mean(ev ** 2))), 3),
                     round(float(np.mean(np.abs(ev) <= 10) * 100), 3)])
    return pd.DataFrame(rows, columns=["Re_bin_centre", "n_points", "mean_error (%)",
                                       "std_error (%)", "RMSE (%)", "frac_within_10 (%)"])


def _print_re_bin_table(label, bin_df):
    df = bin_df.copy()
    df["note"] = ""
    markers = pd.DataFrame(
        [[RE_LAMINAR, np.nan, np.nan, np.nan, np.nan, np.nan, "<- Re=2300 (laminar threshold)"],
         [RE_TURBULENT, np.nan, np.nan, np.nan, np.nan, np.nan,
          "<- Re=4000 (turbulent threshold, used in fitting)"]], columns=df.columns)
    combined = pd.concat([df, markers], ignore_index=True).sort_values(
        "Re_bin_centre").reset_index(drop=True)
    combined["n_points"] = combined["n_points"].apply(lambda x: "" if pd.isna(x) else str(int(x)))
    print(f"\n{label}:")
    print(combined.to_string(index=False, na_rep=""))


def plot_rmse_vs_re(key, label, bin_df, Re, e, ehd, rmse_above, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    valid = bin_df.dropna(subset=["RMSE (%)"]).reset_index(drop=True)
    colors = [REGIME_COLORS[_re_regime(c)] for c in valid["Re_bin_centre"]]
    ax1.bar(range(len(valid)), valid["RMSE (%)"], color=colors)
    ax1.set_xticks(range(len(valid)))
    ax1.set_xticklabels([f"{c:.0f}" for c in valid["Re_bin_centre"]], rotation=45, ha="right")
    ax1.axhline(rmse_above, color="black", ls="--", lw=1)
    ax1.set_xlabel("Re (bin centre)")
    ax1.set_ylabel("RMSE (%)")
    ax1.set_title(f"{label}\nRMSE by Re bin")
    handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[r])
               for r in ("laminar", "transitional", "turbulent")]
    line = plt.Line2D([0], [0], color="black", ls="--", lw=1,
                      label=f"overall turbulent RMSE = {rmse_above:.2f}%")
    ax1.legend(handles + [line],
               ["laminar", "transitional", "turbulent", line.get_label()], fontsize=8)

    ehds = sorted(set(ehd))
    for ehd_val, c in zip(ehds, plt.cm.tab20(np.linspace(0, 1, max(len(ehds), 2)))):
        m = ehd == ehd_val
        ax2.scatter(Re[m], e[m], s=12, alpha=0.6, color=c, label=f"EHD {ehd_val:g}")
    ax2.axhspan(-10, 10, color="grey", alpha=0.15)
    ax2.axhline(0, color="black", lw=1, ls="--")
    ax2.set_xscale("log")
    xmin, xmax = ax2.get_xlim()
    band_lo = max(xmin, 500.0)
    for lo, hi, r in ((band_lo, RE_LAMINAR, "laminar"),
                      (RE_LAMINAR, RE_TURBULENT, "transitional"),
                      (RE_TURBULENT, xmax, "turbulent")):
        ax2.axvspan(lo, hi, color=REGIME_COLORS[r], alpha=0.08, zorder=0)
    trans = ax2.get_xaxis_transform()
    for txt, xpos in (("laminar", np.sqrt(band_lo * RE_LAMINAR)),
                      ("transitional", np.sqrt(RE_LAMINAR * RE_TURBULENT)),
                      ("turbulent (fitted regime)", np.sqrt(RE_TURBULENT * xmax))):
        ax2.text(xpos, 0.98, txt, transform=trans, ha="center", va="top", fontsize=7, alpha=0.75)
    ax2.axvline(RE_LAMINAR, color=REGIME_COLORS["transitional"], lw=1.5, ls=":", label="Re=2300")
    ax2.axvline(RE_TURBULENT, color=REGIME_COLORS["turbulent"], lw=1.5, ls=":", label="Re=4000")
    ax2.set_xlim(xmin, xmax)
    ax2.set_xlabel("Re")
    ax2.set_ylabel("relative error in Q (%)")
    ax2.set_title(f"{label}\nper-point error vs Re")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(outdir, f"rmse_vs_Re_{key}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_contour(key, points, alpha_opt, beta_opt, rmse_opt, outdir, n_alpha=23, n_beta=23):
    alpha_grid = np.linspace(0.6, 1.4, n_alpha)
    beta_grid = np.logspace(np.log10(0.001), np.log10(0.5), n_beta)
    RMSE = np.full((n_beta, n_alpha), np.nan)
    for i, a in enumerate(alpha_grid):
        for j, b in enumerate(beta_grid):
            RMSE[j, i] = objective_rmse((a, b), points)
    if not np.any(np.isfinite(RMSE)):
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6))
    finite = RMSE[np.isfinite(RMSE)]
    cs = ax.contourf(alpha_grid, beta_grid, RMSE,
                     levels=np.linspace(finite.min(), np.percentile(finite, 90), 25),
                     cmap="viridis_r", extend="max")
    fig.colorbar(cs, ax=ax, label="RMSE_Q (%)")
    ax.set_yscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta (log scale)")
    ax.set_title(f"{FIT_LABELS[key]}\nRMSE_Q(alpha, beta)")
    ax.plot(alpha_opt, beta_opt, marker="x", ms=14, mew=3, color="red",
            label=f"optimum: alpha={alpha_opt:.3f}, beta={beta_opt:.4f}")
    cl = ax.contour(alpha_grid, beta_grid, RMSE, levels=[rmse_opt + 1, rmse_opt + 5],
                    colors=["white", "yellow"], linewidths=1.5)
    ax.clabel(cl, fmt={rmse_opt + 1: "+1%", rmse_opt + 5: "+5%"}, fontsize=7)
    ax.legend(loc="upper right", fontsize=8)
    path = os.path.join(outdir, f"contour_{key}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residuals(key, points, alpha_opt, beta_opt, outdir):
    e, Re = _errors(alpha_opt, beta_opt, points)
    valid = np.isfinite(Re) & np.isfinite(e) & (points["Qpub"] > 0)
    if not np.any(valid):
        return None
    gases = sorted(set(points["gas"][valid]))
    fig, axes = plt.subplots(1, len(gases), figsize=(6.5 * len(gases), 5), squeeze=False)
    for ax, gas in zip(axes[0], gases):
        gmask = valid & (points["gas"] == gas)
        ehds = sorted(set(points["EHD"][gmask]))
        for ehd, c in zip(ehds, plt.cm.tab20(np.linspace(0, 1, max(len(ehds), 2)))):
            m = gmask & (points["EHD"] == ehd)
            ax.scatter(Re[m], e[m], s=14, alpha=0.7, color=c, label=f"EHD {ehd:g}")
        ax.axhspan(-10, 10, color="grey", alpha=0.15)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("Re")
        ax.set_ylabel("relative error in Q (%)")
        ax.set_title(f"{FIT_LABELS[key]} - {gas}")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(outdir, f"residuals_{key}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_per_ehd_table(points, alpha, beta):
    e, Re = _errors(alpha, beta, points)
    mask = (Re > RE_TURBULENT) & np.isfinite(e)
    df = pd.DataFrame({"EHD": points["EHD"][mask], "source": points["source"][mask],
                       "error": e[mask]})
    rows = [[ehd, src, g["error"].mean(), g["error"].std(), len(g)]
            for ehd, sub in df.groupby("EHD") for src, g in sub.groupby("source")]
    return pd.DataFrame(rows, columns=["EHD", "source", "mean_error", "std_error", "n_points"])


def plot_per_ehd_errors(df, title, filename, outdir):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sources = sorted(df["source"].unique())
    offsets = np.linspace(-0.3, 0.3, len(sources)) if len(sources) > 1 else [0.0]
    for src, c, off in zip(sources, plt.cm.tab10(np.linspace(0, 1, max(len(sources), 2))),
                           offsets):
        sub = df[df["source"] == src]
        ax.errorbar(sub["EHD"] + off, sub["mean_error"], yerr=sub["std_error"].fillna(0),
                    fmt="o", color=c, capsize=3, label=src)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("EHD")
    ax.set_ylabel("relative error in Q (%), mean +/- 1 std")
    ax.set_title(title)
    ax.legend(fontsize=8)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def run_universal():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pd.set_option("display.width", 200)

    pts = {s: load_points(s) for s in ("gastite", "parker", "tracpipe", "parflex")}
    print("\n--- STEP 1: data summary ---")
    print(pd.DataFrame([[p[0]["source"], len({(q["EHD"], q["gas"]) for q in p}), len(p)]
                        for p in pts.values()],
                       columns=["source", "n_pipes", "n_points"]).to_string(index=False))

    point_sets = {k: points_to_arrays(pts[FIT_SOURCE[k]]) for k in FIT_SOURCE}
    point_sets["F5"] = points_to_arrays(sum(pts.values(), []))

    results = {}
    for key, points in point_sets.items():
        a, b = fit_alpha_beta(points)
        results[key] = dict(alpha=a, beta=b, **compute_metrics(a, b, points))

    l2_alpha, l2_beta_by_ehd, l2_beta_per_point = fit_alpha_beta_L2(point_sets["F3"])
    l2_metrics = compute_metrics(l2_alpha, l2_beta_per_point, point_sets["F3"])

    rows = [[FIT_LABELS[k], "L1", results[k]["alpha"],
             f"{results[k]['beta'] / results[k]['alpha']:.4f}", results[k]["rmse"],
             results[k]["mae"], results[k]["maxerr"], results[k]["frac5"],
             results[k]["frac10"], results[k]["n"]] for k in FIT_KEYS]
    betas = list(l2_beta_by_ehd.values())
    rows.append([FIT_LABELS["F3"], "L2", l2_alpha, f"{min(betas):.4f}-{max(betas):.4f}",
                 l2_metrics["rmse"], l2_metrics["mae"], l2_metrics["maxerr"],
                 l2_metrics["frac5"], l2_metrics["frac10"], l2_metrics["n"]])
    print("\n--- STEP 4: universal friction fits (alpha, beta free, EHD only) ---")
    print(pd.DataFrame(rows, columns=["fit", "level", "alpha", "eps/D or beta_range",
                                      "RMSE_Q (%)", "MAE_Q (%)", "max_abs_error (%)",
                                      "frac within 5%", "frac within 10%",
                                      "n_points"]).round(4).to_string(index=False))
    print(f"\n  F3 L2 - beta per EHD (alpha = {l2_alpha:.4f}):")
    print(pd.DataFrame(sorted(l2_beta_by_ehd.items()),
                       columns=["EHD", "beta"]).round(4).to_string(index=False))

    # The Re>4000 mask is recomputed at every alpha, so check it does not swing.
    print("\n--- STEP 4b: Re > 4000 mask stability (n_turbulent vs alpha) ---")
    for key in ("F2", "F3"):
        p, r = point_sets[key], results[key]
        df = pd.DataFrame([[a, int(np.sum(Q_gfe_churchill(a, r["beta"], p["EHD"], p["P1"],
                                                          p["P2"], p["L"], p["gas"])[1]
                                          > RE_TURBULENT))] for a in ALPHA_GRID],
                          columns=["alpha", "n_turbulent"])
        n_min, n_max = int(df["n_turbulent"].min()), int(df["n_turbulent"].max())
        swing = (n_max - n_min) / r["n"] * 100
        print(f"\n{FIT_LABELS[key]} (own optimum alpha = {r['alpha']:.4f}, n = {r['n']}):")
        print(df.to_string(index=False))
        flag = "  <- FLAG: >5% swing across this alpha range" if swing > 5 else ""
        print(f"  n_turbulent range: {n_min}-{n_max}, swing = {swing:.2f}%{flag}")

    # Condition-dependent scatter points to a pressure-conversion issue rather
    # than EHD-dependent geometry.
    def condition_diagnostic(key):
        p, r = point_sets[key], results[key]
        e, Re = _errors(r["alpha"], r["beta"], p)
        mask = (Re > RE_TURBULENT) & np.isfinite(e)
        df = pd.DataFrame({"condition": p["condition"][mask], "error": e[mask]})
        return pd.DataFrame([[c, len(g), g["error"].mean(), g["error"].std(),
                              g["error"].min(), g["error"].max()]
                             for c, g in df.groupby("condition")],
                            columns=["condition", "n_points", "mean_error (%)",
                                     "std_error (%)", "min_error (%)", "max_error (%)"])

    tp_df, ga_df = condition_diagnostic("F4"), condition_diagnostic("F3")
    print("\n--- STEP 4c: TracPipe error by condition (F4 own optimum) ---")
    print(tp_df.round(3).to_string(index=False))
    print("\n--- STEP 4c: Gastite error by condition (F3 own optimum) ---")
    print(ga_df.round(3).to_string(index=False))
    merged = tp_df.merge(ga_df, on="condition", suffixes=("_tracpipe", "_gastite"))
    if merged.empty:
        print("\nNo matching condition labels between TracPipe and Gastite.")
    else:
        merged["std_ratio"] = merged["std_error (%)_tracpipe"] / merged["std_error (%)_gastite"]
        print("\n  TracPipe vs Gastite std_error ratio at matched conditions:")
        print(merged[["condition", "std_error (%)_tracpipe", "std_error (%)_gastite",
                      "std_ratio"]].round(3).to_string(index=False))
        print(f"\n  std_ratio across {len(merged)} matched conditions: "
              f"mean = {merged['std_ratio'].mean():.2f}, min = {merged['std_ratio'].min():.2f}, "
              f"max = {merged['std_ratio'].max():.2f}")
        if (merged["std_ratio"] >= 3).all():
            print("FLAG: TracPipe's condition-level std is >=3x Gastite's at every matched "
                  "condition - a condition-dependent (pressure conversion) issue.")

    alpha_ref, beta_ref = results["F3"]["alpha"], results["F3"]["beta"]
    rows = []
    for key in ("F2", "F4", "F6"):
        at_ref = objective_rmse((alpha_ref, beta_ref), point_sets[key])
        own = results[key]["rmse"]
        rows.append([FIT_LABELS[key], at_ref, own, at_ref - own,
                     "FLAG: >3pp degradation" if at_ref - own > 3 else ""])
    print("\n--- STEP 8: cross-manufacturer check (F3 optimum applied, not refitted) ---")
    print(pd.DataFrame(rows, columns=["source", "RMSE_at_F3_optimum (%)",
                                      "RMSE_at_own_optimum (%)", "degradation (pp)",
                                      "flag"]).round(3).to_string(index=False))

    f2, f3, f4, f5, f6 = (results[k] for k in ("F2", "F3", "F4", "F5", "F6"))
    eps_d = {k: results[k]["beta"] / results[k]["alpha"] for k in FIT_KEYS}
    print("\n--- STEP 9: cross-manufacturer conclusion ---")
    print(f"F1 (Gastite + Parker pooled) is retired: Gastite's {f3['n']} turbulent points "
          f"would outnumber Parker's {f2['n']} by >30:1, so F2 and F3, each fit at its own "
          f"optimum, are the primary cross-manufacturer comparison.")
    for key in ("F2", "F3", "F4", "F6"):
        r = results[key]
        print(f"\n{FIT_LABELS[key]} L1 optimum: alpha = {r['alpha']:.4f}, "
              f"beta = {r['beta']:.4f}, eps/D = {eps_d[key]:.4f}")
        if key != "F3":
            print(f"  eps/D difference ({key} - F3): {eps_d[key] - eps_d['F3']:+.4f}")
            gap = r["rmse"] - f3["rmse"]
            print(f"  F3 RMSE_Q: {f3['rmse']:.3f}%   {key} RMSE_Q: {r['rmse']:.3f}%   "
                  f"gap: {gap:+.3f} pp")
            if gap > 10:
                print(f"  FLAG: {FIT_LABELS[key]} does not conform to the same friction model "
                      f"as Gastite at matched EHD.")
    print(f"\nF5 (all four pooled) RMSE_Q: {f5['rmse']:.3f}%   vs F3: {f3['rmse']:.3f}%   "
          f"degradation from naive pooling: {f5['rmse'] - f3['rmse']:+.3f} pp")

    # No Re filter here: hydrogen at matched duty sits below the fitted regime.
    print("\n--- STEP 10: RMSE as a function of Re (no Re > 4000 filter) ---")
    groups = {k: (point_sets[k], results[k]["alpha"], results[k]["beta"], FIT_LABELS[k])
              for k in ("F2", "F3", "F4", "F6")}
    groups["F1"] = (points_to_arrays(pts["gastite"] + pts["parker"]), alpha_ref, beta_ref,
                    "Gastite + Parker combined (at F3 reference optimum)")
    step10 = {}
    for key, (points, alpha, beta, label) in groups.items():
        e, Re = _errors(alpha, beta, points)
        valid = np.isfinite(Re) & np.isfinite(e) & (points["Qpub"] > 0)
        Re, e, ehd = Re[valid], e[valid], points["EHD"][valid]
        bin_df = bin_errors_by_re(Re, e)
        _print_re_bin_table(label, bin_df)
        below, above = Re < RE_TURBULENT, Re >= RE_TURBULENT
        rmse_below = float(np.sqrt(np.mean(e[below] ** 2))) if below.any() else np.nan
        rmse_above = float(np.sqrt(np.mean(e[above] ** 2))) if above.any() else np.nan
        print(f"  Below Re={RE_TURBULENT:.0f} RMSE is {rmse_below:.2f}% vs "
              f"{rmse_above:.2f}% above.")
        step10[key] = (label, bin_df, Re, e, ehd, rmse_above)

    print("\n--- STEP 7: per-EHD error summary ---")
    gp_points = points_to_arrays(pts["gastite"] + pts["parker"])
    gp_df = build_per_ehd_table(gp_points, alpha_ref, beta_ref)
    print("Gastite + Parker, at F3 optimum:")
    print(gp_df.round(3).to_string(index=False))
    tp_ehd = build_per_ehd_table(point_sets["F4"], results["F4"]["alpha"], results["F4"]["beta"])
    print("\nTracPipe, at F4 (own) optimum:")
    print(tp_ehd.round(3).to_string(index=False))
    pf_ehd = build_per_ehd_table(point_sets["F6"], results["F6"]["alpha"], results["F6"]["beta"])
    print("\nParflex, at F6 (own) optimum:")
    print(pf_ehd.round(3).to_string(index=False))

    print("\n--- STEP 5/6: contour and residual plots ---")
    for key in FIT_KEYS:
        r = results[key]
        for path in (plot_contour(key, point_sets[key], r["alpha"], r["beta"], r["rmse"],
                                  PLOTS_DIR),
                     plot_residuals(key, point_sets[key], r["alpha"], r["beta"], PLOTS_DIR)):
            if path:
                print(f"  saved {path}")

    print("\n--- STEP 7: per-EHD error plots ---")
    for df, title, fname in (
            (gp_df, "Gastite + Parker - F3 reference optimum, per EHD by source",
             "per_ehd_error_GastriteParker.png"),
            (tp_ehd, "TracPipe - F4 (own) optimum, per EHD", "per_ehd_error_TracPipe.png"),
            (pf_ehd, "Parflex - F6 (own) optimum, per EHD", "per_ehd_error_Parflex.png")):
        path = plot_per_ehd_errors(df, title, fname, PLOTS_DIR)
        if path:
            print(f"  saved {path}")

    print("\n--- STEP 10: RMSE vs Re plots ---")
    for key, (label, bin_df, Re, e, ehd, rmse_above) in step10.items():
        print(f"  saved {plot_rmse_vs_re(key, label, bin_df, Re, e, ehd, rmse_above, PLOTS_DIR)}")

    return results, point_sets


# ================================================================ bi-regime

MIN_POINTS = 3


# Free power law f = a0*Re^-n, fitted on (log a0, n).
def fit_powerlaw_f(Re_v, f_v):
    def ssr(params):
        log_a, n_exp = params
        return np.sum((np.log(f_v) - (log_a - n_exp * np.log(Re_v))) ** 2)
    res = minimize(ssr, x0=[np.log(0.14), 0.15], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-8, maxiter=500, maxfev=500))
    return float(np.exp(res.x[0])), float(res.x[1])


# Same GFE-closed-by-power-law algebra as md.polyflo_Q_fwd_scfh, with a0/n free.
def polyflo_Q_fwd_generic(a0_fit, n_fit, D_mm, S, mu_Pas, P1_kPa, P2_kPa, L_ft):
    alpha_poly = 2.0 / (2.0 - n_fit)
    L_km = L_ft * 0.3048 / 1000.0
    K = 1250 * S * Pb / (27 * np.pi * R_air * Tb * mu_Pas)
    b1 = 1.1494e-3 * (Tb / Pb) * a0_fit ** -0.5 * K ** (n_fit / 2)
    b2 = (P1_kPa ** 2 - P2_kPa ** 2) / (S * Tb * L_km)
    Q_day = b1 ** alpha_poly * b2 ** (alpha_poly / 2) * D_mm ** (alpha_poly * (2.5 - n_fit / 2))
    return Q_day / 86400.0 / SCFH_TO_M3S


def _biregime_points(source):
    by_ehd = {}
    for pipe_label, pipe in _load(source).items():
        arr = md.pipe_arrays(pipe)
        props = GAS_PROPERTIES[pipe["gas"]]
        n_cond, n_len = arr["Re"].shape

        row_keep = np.ones(n_cond, dtype=bool)
        if source == "parker" and pipe_label == PARKER_SUSPECT_PIPE:
            for i, cond in enumerate(pipe["conditions"]):
                if abs(cond[2] - PARKER_SUSPECT_INWC) < 1e-6:
                    row_keep[i] = False
        grid = np.broadcast_to(row_keep[:, None], (n_cond, n_len))
        Re = np.where(grid, arr["Re"], np.nan)
        f_moody = np.where(grid, arr["f_moody"], np.nan)
        Q = np.where(grid, arr["Q_scfh"], np.nan)

        pts = by_ehd.setdefault(pipe["ehd"], dict(
            Re=[], f_moody=[], Q_pub=[], L_ft=[], P1=[], P2=[], S=[], mu_Pas=[],
            gas=[], condition=[], D_mm=arr["D_mm"]))
        pts["Re"].append(Re.ravel())
        pts["f_moody"].append(f_moody.ravel())
        pts["Q_pub"].append(Q.ravel())
        pts["L_ft"].append(np.broadcast_to(arr["L_ft"][None, :], (n_cond, n_len)).ravel())
        pts["P1"].append(np.broadcast_to(arr["P1"], (n_cond, n_len)).ravel())
        pts["P2"].append(np.broadcast_to(arr["P2"], (n_cond, n_len)).ravel())
        pts["S"].append(np.full(n_cond * n_len, props["S"]))
        pts["mu_Pas"].append(np.full(n_cond * n_len, props["mu_Pas"]))
        pts["gas"].append(np.full(n_cond * n_len, pipe["gas"], dtype=object))
        labels = np.array([c[3] for c in pipe["conditions"]], dtype=object)
        pts["condition"].append(np.broadcast_to(labels[:, None], (n_cond, n_len)).ravel())

    for pts in by_ehd.values():
        for key in ("Re", "f_moody", "Q_pub", "L_ft", "P1", "P2", "S", "mu_Pas",
                    "gas", "condition"):
            pts[key] = np.concatenate(pts[key])
    return by_ehd


def run_biregime():
    pd.set_option("display.width", 200)
    central_rows = []
    detail_rows = {}

    for source in ("gastite", "parflex", "tracpipe", "parker"):
        name = {"gastite": "Gastite", "parflex": "Parflex", "tracpipe": "TracPipe",
                "parker": "Parker Hannifin"}[source]
        detail_rows[name] = []
        by_ehd = _biregime_points(source)
        for ehd in sorted(by_ehd):
            pts = by_ehd[ehd]
            valid = np.isfinite(pts["Re"]) & np.isfinite(pts["f_moody"]) & (pts["Re"] > 0)
            for regime, mask in (("laminar", valid & (pts["Re"] < RE_LAMINAR)),
                                 ("turbulent", valid & (pts["Re"] >= RE_LAMINAR))):
                n_pts = int(mask.sum())
                if n_pts < MIN_POINTS:
                    plural = "s" if n_pts != 1 else ""
                    print(f"SKIP: {name} EHD {ehd:g} {regime} has {n_pts} point{plural}, "
                          f"need >={MIN_POINTS}")
                    continue

                Re_v, f_v = pts["Re"][mask], pts["f_moody"][mask]
                a0_fit, n_fit = fit_powerlaw_f(Re_v, f_v)
                rms_log_f = float(np.sqrt(np.mean(
                    (np.log(f_v) - np.log(a0_fit * Re_v ** -n_fit)) ** 2)))

                Q_pub_v = pts["Q_pub"][mask]
                L_v = pts["L_ft"][mask]
                with np.errstate(invalid="ignore", divide="ignore"):
                    Q_fit_v = polyflo_Q_fwd_generic(a0_fit, n_fit, pts["D_mm"],
                                                    pts["S"][mask], pts["mu_Pas"][mask],
                                                    pts["P1"][mask], pts["P2"][mask], L_v)
                    pct_err = (Q_fit_v - Q_pub_v) / Q_pub_v * 100

                central_rows.append([ehd, name, regime, a0_fit, n_fit, n_pts, rms_log_f,
                                     float(np.sqrt(np.mean(pct_err ** 2))),
                                     float(np.mean(np.abs(pct_err)))])
                gas_v, cond_v = pts["gas"][mask], pts["condition"][mask]
                for i in range(n_pts):
                    detail_rows[name].append([ehd, regime, gas_v[i], cond_v[i], L_v[i],
                                              Re_v[i], Q_pub_v[i], Q_fit_v[i], pct_err[i]])

    central = pd.DataFrame(central_rows, columns=["EHD", "manufacturer", "regime", "a0", "n",
                                                  "n_points", "rms_log_f", "RMSE_Q (%)",
                                                  "MAE_Q (%)"])
    central = central.sort_values(["EHD", "manufacturer", "regime"]).reset_index(drop=True)
    print("\n--- CENTRAL TABLE: bi-regime power-law friction fits f = a0*Re^-n ---")
    print(central.round({"a0": 5, "n": 4, "rms_log_f": 4, "RMSE_Q (%)": 2,
                         "MAE_Q (%)": 2}).to_string(index=False))

    for name, rows in detail_rows.items():
        df = pd.DataFrame(rows, columns=["EHD", "regime", "gas", "condition_label", "L_ft",
                                         "Re", "Q_published (SCFH)", "Q_powerfit (SCFH)",
                                         "%error"])
        df = df.sort_values(["EHD", "regime", "Re"]).reset_index(drop=True)
        print(f"\n--- {name}: per-point detail ---")
        print(df.round({"Re": 0, "Q_published (SCFH)": 2, "Q_powerfit (SCFH)": 2,
                        "%error": 2}).to_string(index=False))


# ============================================================ generator fits

POLYFLO_N = 0.151571
B_TURBULENT_ROUGH, B_LAMINAR = 0.5, 1.0
B_POLYFLO = 1.0 / (2.0 - POLYFLO_N)
A_TURBULENT_ROUGH, A_LAMINAR = 2.5, 4.0
A_POLYFLO = (5.0 - POLYFLO_N) / (2.0 - POLYFLO_N)

H2_MU_PAS, H2_S = 8.8e-6, 0.0696
NG_S = GAS_PROPERTIES["natural_gas"]["S"]
NG_MU_PAS = GAS_PROPERTIES["natural_gas"]["mu_Pas"]
Q_H2_OVER_Q_NG = 37.26 / 12.1
RE_H2_OVER_RE_NG = (H2_S / NG_S) * Q_H2_OVER_Q_NG * (NG_MU_PAS / H2_MU_PAS)

AB_N_TOL = 0.02
CHART_SOURCES = ("gastite", "parflex", "tracpipe", "csa")
SOURCE_NAME = {"gastite": "Gastite", "parflex": "Parflex", "tracpipe": "TracPipe",
               "csa": "CSA/ANSI", "parker": "Parker Hannifin"}
SOURCE_COLORS = {"Gastite": "#2a78d6", "Parflex": "#eb6834", "TracPipe": "#1baf7a",
                 "Parker Hannifin": "#800080", "CSA/ANSI": "#c9a227"}


# ln Q = ln C + a*ln D + b*ln(dP/L).
def fit_generator_lstsq(Q_m3s, D_m, dP_Pa, L_m):
    y = np.log(Q_m3s)
    X = np.column_stack([np.ones_like(y), np.log(D_m), np.log(dP_Pa / L_m)])
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return dict(C=float(np.exp(coef[0])), a=float(coef[1]), b=float(coef[2]),
                R2=float(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan),
                rms_log=float(np.sqrt(np.mean((y - y_pred) ** 2))), n=int(len(y)),
                rank=int(rank))


# Per-EHD point lists for one chart source: NG only, bare L.
def collect_ehd_points(pipes):
    by_ehd = {}
    for pipe in pipes.values():
        if pipe["gas"] != "natural_gas":
            continue
        ehd = pipe["ehd"]
        D_m = ehd * EHD_TO_MM / 1000.0
        L_ft_all = md.pipe_arrays(pipe)["L_ft"]
        for cond, flow_row in zip(pipe["conditions"], pipe["flows"]):
            dP_Pa = cond[1] * PSI_TO_PA
            for L_ft, Q_scfh in zip(L_ft_all, flow_row):
                if not np.isfinite(Q_scfh) or Q_scfh <= 0:
                    continue
                pts = by_ehd.setdefault(ehd, dict(Q=[], D=[], dP=[], L=[]))
                pts["Q"].append(Q_scfh * SCFH_TO_M3S)
                pts["D"].append(D_m)
                pts["dP"].append(dP_Pa)
                pts["L"].append(L_ft * 0.3048)
    return by_ehd


def _stack(by_ehd):
    return tuple(np.concatenate([np.asarray(by_ehd[e][k]) for e in sorted(by_ehd)])
                 for k in ("Q", "D", "dP", "L"))


def fit_generator_pooled(chart_pipes):
    rows = []
    for source, pipes in chart_pipes.items():
        fit = fit_generator_lstsq(*_stack(collect_ehd_points(pipes)))
        rows.append([SOURCE_NAME[source], fit["n"], fit["rank"], fit["C"], fit["a"],
                     fit["b"], fit["R2"], fit["rms_log"]])
    return pd.DataFrame(rows, columns=["Source", "n_pts", "rank", "C", "a", "b", "R2",
                                       "rms_log"])


def _parker_keep(label, pipe, shape):
    keep = np.ones(shape, dtype=bool)
    if label == PARKER_SUSPECT_PIPE:
        for i, cond in enumerate(pipe["conditions"]):
            if abs(cond[2] - PARKER_SUSPECT_INWC) < 1e-6:
                keep[i, :] = False
    return keep


# Finite and positive, suspect row excluded, no Re cut - describes the data.
def parker_valid_mask(label, pipe, arr):
    return (_parker_keep(label, pipe, arr["Re"].shape)
            & np.isfinite(arr["Q_scfh"]) & (arr["Q_scfh"] > 0))


# Re > 4000 on top of the above - for the friction-model comparison.
def parker_turb_mask(label, pipe, arr):
    return (parker_valid_mask(label, pipe, arr)
            & np.isfinite(arr["Re"]) & (arr["Re"] > RE_TURBULENT)
            & np.isfinite(arr["f_moody"]) & (arr["f_moody"] > 0))


# Parker's id_mm is computed from EHD, so D already matches the EHD/32 basis.
def collect_parker_points(parker_pipes, mask_fn):
    Q_l, D_l, dP_l, L_l = [], [], [], []
    for label, pipe in parker_pipes.items():
        arr = md.pipe_arrays(pipe)
        mask = mask_fn(label, pipe, arr)
        if mask.sum() < 3:
            continue
        dP_grid = np.broadcast_to(
            np.array([c[1] for c in pipe["conditions"]])[:, None] * PSI_TO_PA, arr["Re"].shape)
        L_grid = np.broadcast_to((arr["L_ft"] * 0.3048)[None, :], arr["Re"].shape)
        Q_l.append(arr["Q_scfh"][mask] * SCFH_TO_M3S)
        D_l.append(np.full(int(mask.sum()), arr["D_mm"] / 1000.0))
        dP_l.append(dP_grid[mask])
        L_l.append(L_grid[mask])
    return (np.concatenate(Q_l), np.concatenate(D_l), np.concatenate(dP_l),
            np.concatenate(L_l))


def fit_parker_pooled(parker_pipes):
    fit = fit_generator_lstsq(*collect_parker_points(parker_pipes, parker_valid_mask))
    return pd.DataFrame([["Parker Hannifin", fit["n"], fit["rank"], fit["C"], fit["a"],
                          fit["b"], fit["R2"], fit["rms_log"]]],
                        columns=["Source", "n_pts", "rank", "C", "a", "b", "R2", "rms_log"])


def _all_pooled(chart_pipes, parker_pipes):
    return pd.concat([fit_generator_pooled(chart_pipes), fit_parker_pooled(parker_pipes)],
                     ignore_index=True)


# Average (C, a, b) across all five brands, then back-test that one equation.
def build_universal_equation(chart_pipes, parker_pipes):
    pooled = _all_pooled(chart_pipes, parker_pipes)
    C_avg, a_avg, b_avg = pooled["C"].mean(), pooled["a"].mean(), pooled["b"].mean()

    by_source = {SOURCE_NAME[s]: _stack(collect_ehd_points(p)) for s, p in chart_pipes.items()}
    by_source["Parker Hannifin"] = collect_parker_points(parker_pipes, parker_valid_mask)

    rows = []
    for _, r in pooled.iterrows():
        Q, D, dP, L = by_source[r["Source"]]
        err_own = np.mean(np.abs((r["C"] * D ** r["a"] * (dP / L) ** r["b"] - Q) / Q * 100))
        Q_univ = C_avg * D ** a_avg * (dP / L) ** b_avg
        err_abs = np.mean(np.abs((Q_univ - Q) / Q * 100))
        rows.append([r["Source"], r["n_pts"], err_own, err_abs, err_abs / err_own,
                     np.mean((Q_univ - Q) / Q * 100)])
    return dict(C_avg=C_avg, a_avg=a_avg, b_avg=b_avg, pooled_df=pooled,
                result_df=pd.DataFrame(rows, columns=["Source", "n_pts", "own_MAE_pct",
                                                      "universal_MAE_pct", "inflation",
                                                      "universal_mean_err_pct"]))


# For f = a0*Re^-n: b = 1/(2-n) and a/b = 5-n must recover the same n.
def ab_n_consistency(a, b):
    n_from_b = 2.0 - 1.0 / b
    n_from_ab = 5.0 - a / b
    return n_from_b, n_from_ab, n_from_ab - n_from_b


def build_generator_table(chart_pipes):
    rows = []
    for source, pipes in chart_pipes.items():
        by_ehd = collect_ehd_points(pipes)
        for ehd in sorted(by_ehd):
            pts = by_ehd[ehd]
            fit = fit_generator_lstsq(np.array(pts["Q"]), np.array(pts["D"]),
                                      np.array(pts["dP"]), np.array(pts["L"]))
            rows.append([SOURCE_NAME[source], ehd, ehd * EHD_TO_MM, fit["n"], fit["R2"],
                         fit["C"], fit["a"], fit["b"], fit["rms_log"], fit["rank"]])
    return pd.DataFrame(rows, columns=["Source", "EHD", "D_mm", "n_pts", "R2", "C", "a", "b",
                                       "rms_log", "rank"]).sort_values(
        ["Source", "EHD"]).reset_index(drop=True)


def _fit_eps_masked(Re_v, f_v, D_mm, closure):
    def ssr(log_eps):
        return np.sum((np.log(f_v) - np.log(closure(Re_v, 10 ** log_eps, D_mm))) ** 2)
    return 10 ** minimize_scalar(ssr, bounds=(-3, 1), method="bounded").x


def build_parker_analysis(parker_pipes):
    gen_rows, friction_rows, powerlaw_rows = [], [], []
    pooled = {"Polyflo (fixed)": [], "Constant f (fitted)": [],
              "Colebrook-White (fitted)": [], "Churchill (fitted)": []}

    for label, pipe in parker_pipes.items():
        arr = md.pipe_arrays(pipe)
        turb = parker_turb_mask(label, pipe, arr)
        if turb.sum() < 3:
            continue

        dP_grid = np.broadcast_to(
            np.array([c[1] for c in pipe["conditions"]])[:, None] * PSI_TO_PA, arr["Re"].shape)
        L_grid = np.broadcast_to((arr["L_ft"] * 0.3048)[None, :], arr["Re"].shape)
        fit = fit_generator_lstsq(arr["Q_scfh"][turb] * SCFH_TO_M3S,
                                  np.full(int(turb.sum()), arr["D_mm"] / 1000.0),
                                  dP_grid[turb], L_grid[turb])
        gen_rows.append([label, pipe["ehd"], arr["D_mm"], fit["n"], fit["R2"], fit["b"],
                         fit["a"], fit["rms_log"]])

        Re_v, f_v = arr["Re"][turb], arr["f_moody"][turb]
        f_poly_pred = polyflo_f(Re_v)
        rms_poly = float(np.sqrt(np.mean((np.log(f_v) - np.log(f_poly_pred)) ** 2)))

        res_f = minimize_scalar(lambda lf: np.sum((np.log(f_v) - lf) ** 2),
                                bounds=(-6, 1), method="bounded")
        f_const, log_f_const = float(np.exp(res_f.x)), res_f.x
        rms_constf = float(np.sqrt(np.mean((np.log(f_v) - log_f_const) ** 2)))

        eps_fit = _fit_eps_masked(Re_v, f_v, arr["D_mm"], colebrook_f)
        f_cw_pred = colebrook_f(Re_v, eps_fit, arr["D_mm"])
        rms_cw = float(np.sqrt(np.mean((np.log(f_v) - np.log(f_cw_pred)) ** 2)))

        eps_ch = _fit_eps_masked(Re_v, f_v, arr["D_mm"], churchill_f)
        f_ch_pred = churchill_f(Re_v, eps_ch, arr["D_mm"])
        rms_ch = float(np.sqrt(np.mean((np.log(f_v) - np.log(f_ch_pred)) ** 2)))

        friction_rows.append([label, int(len(Re_v)), rms_poly, rms_constf, rms_cw, rms_ch,
                              f_const, eps_fit, eps_fit / arr["D_mm"], eps_ch,
                              eps_ch / arr["D_mm"], (eps_ch - eps_fit) / eps_fit * 100])
        pooled["Polyflo (fixed)"].append((np.log(f_v), np.log(f_poly_pred)))
        pooled["Constant f (fitted)"].append((np.log(f_v), np.full_like(f_v, log_f_const)))
        pooled["Colebrook-White (fitted)"].append((np.log(f_v), np.log(f_cw_pred)))
        pooled["Churchill (fitted)"].append((np.log(f_v), np.log(f_ch_pred)))

        a_coeff, n_exp = fit_powerlaw_f(Re_v, f_v)
        powerlaw_rows.append([label, a_coeff, n_exp])

    pooled_rms = {}
    for name, pairs in pooled.items():
        if pairs:
            obs = np.concatenate([p[0] for p in pairs])
            pred = np.concatenate([p[1] for p in pairs])
            pooled_rms[name] = float(np.sqrt(np.mean((obs - pred) ** 2)))

    return (pd.DataFrame(gen_rows, columns=["Pipe", "EHD", "D_mm", "n_pts", "R2", "b", "a",
                                            "rms_log"]),
            pd.DataFrame(friction_rows, columns=["Pipe", "n_pts", "rms_polyflo_fixed",
                                                 "rms_const_f", "rms_cw_fitted",
                                                 "rms_churchill_fitted", "f_const_fit",
                                                 "eps_cw_fit_mm", "eps_over_D",
                                                 "eps_churchill_fit_mm", "eps_over_D_churchill",
                                                 "eps_churchill_vs_cw_pct"]),
            pd.DataFrame(powerlaw_rows, columns=["Pipe", "a_coeff_fit", "n_exp_fit"]),
            pooled_rms)


def generator_context():
    chart_pipes = {s: _load(s) for s in CHART_SOURCES}
    parker_pipes = _load("parker")
    gen_df = build_generator_table(chart_pipes)
    parker_gen_df, friction_df, powerlaw_df, pooled_rms = build_parker_analysis(parker_pipes)
    return dict(chart_pipes=chart_pipes, parker_pipes=parker_pipes, gen_df=gen_df,
                parker_gen_df=parker_gen_df, friction_df=friction_df,
                powerlaw_df=powerlaw_df, pooled_rms=pooled_rms)


def _gen_step1(ctx):
    print("\n--- STEP 1: data summary (NG only for chart sources; Parker all conditions) ---")
    rows = []
    for source, pipes in ctx["chart_pipes"].items():
        ng = [p for p in pipes.values() if p["gas"] == "natural_gas"]
        n = sum(1 for p in ng for row in p["flows"] for v in row if np.isfinite(v) and v > 0)
        rows.append([SOURCE_NAME[source], len({p["ehd"] for p in ng}), n])
    n_parker = sum(1 for p in ctx["parker_pipes"].values() for row in p["flows"]
                   for v in row if np.isfinite(v) and v > 0)
    rows.append(["Parker Hannifin", len(ctx["parker_pipes"]), n_parker])
    print(pd.DataFrame(rows, columns=["source", "n_sizes", "n_points"]).to_string(index=False))


def _gen_step2(ctx):
    gen_df = ctx["gen_df"]
    print("\n--- STEP 2: generator equations, per manufacturer ---")
    print("(1) PER-EHD: one (C, b) pair per pipe size. b is meaningful; a is not identified")
    print("    (D is fixed within one EHD, so it is a minimum-norm artifact).")
    print("(2) POOLED: one (C, a, b) triple per brand across its sizes - a is identified here.")
    print("\n(1) PER-EHD fits:")
    print(gen_df[["Source", "EHD", "D_mm", "n_pts", "R2", "C", "b", "a",
                  "rms_log"]].round(5).to_string(index=False))
    pooled = _all_pooled(ctx["chart_pipes"], ctx["parker_pipes"])
    print("\n(2) POOLED fits (a identified):")
    print(pooled.round(5).to_string(index=False))

    print("\n--- (a, b) consistency with a single power-law closure f = a0*Re^-n ---")
    nb, nab, gap = ab_n_consistency(A_POLYFLO, B_POLYFLO)
    print(f"  Reference (Polyflo a={A_POLYFLO:.4f}, b={B_POLYFLO:.4f}): n_from_b={nb:.6f}, "
          f"n_from_ab={nab:.6f}, gap={gap:+.2e} -> CONSISTENT")
    for _, r in pooled.iterrows():
        nb, nab, gap = ab_n_consistency(r["a"], r["b"])
        print(f"  {r['Source']:16s} a={r['a']:.4f} b={r['b']:.4f} -> n_from_b={nb:+.4f}, "
              f"n_from_ab={nab:+.4f}, gap={gap:+.4f} -> "
              f"{'CONSISTENT' if abs(gap) < AB_N_TOL else 'INCONSISTENT'}")
    print("  No single n reconciles a and b: the dP/L scaling and the EHD size scaling are")
    print("  independently calibrated, so these tables encode no friction model.")

    print("\n--- (a, b) consistency per individual (non-pooled) fit ---")
    n_ok = n_tot = 0
    for _, r in gen_df.iterrows():
        nb, nab, gap = ab_n_consistency(r["a"], r["b"])
        print(f"  {r['Source']:9s} EHD={r['EHD']:>3} a={r['a']:.4f} b={r['b']:.4f} -> "
              f"n_from_b={nb:+.4f}, n_from_ab={nab:+.4f}, gap={gap:+.4f} -> "
              f"{'CONSISTENT' if abs(gap) < AB_N_TOL else 'INCONSISTENT'}")
        n_tot += 1
        n_ok += abs(gap) < AB_N_TOL
    for _, r in ctx["parker_gen_df"].iterrows():
        nb, nab, gap = ab_n_consistency(r["a"], r["b"])
        print(f"  {'Parker Hannifin':9s} Pipe={r['Pipe']:<12s} a={r['a']:.4f} b={r['b']:.4f} -> "
              f"n_from_b={nb:+.4f}, n_from_ab={nab:+.4f}, gap={gap:+.4f} -> "
              f"{'CONSISTENT' if abs(gap) < AB_N_TOL else 'INCONSISTENT'}")
        n_tot += 1
        n_ok += abs(gap) < AB_N_TOL
    print(f"\n  {n_ok}/{n_tot} individual fits consistent (|gap| < {AB_N_TOL}).")


def _exponent_table(gen_df, parker_gen_df, col):
    rows = [[s, g[col].mean(), g[col].min(), g[col].max(), len(g)]
            for s, g in gen_df.groupby("Source")]
    if not parker_gen_df.empty:
        rows.append(["Parker Hannifin", parker_gen_df[col].mean(), parker_gen_df[col].min(),
                     parker_gen_df[col].max(), len(parker_gen_df)])
    return pd.DataFrame(rows, columns=["Source", f"{col}_mean", f"{col}_min", f"{col}_max",
                                       "n_EHD"])


def _gen_step3(ctx):
    print("\n--- STEP 3: pressure-drop exponent b - theoretical references ---")
    print(f"  Fully turbulent (rough): b = {B_TURBULENT_ROUGH:.4f}")
    print(f"  Polyflo (n = {POLYFLO_N}): b = 1/(2-n) = {B_POLYFLO:.4f}")
    print(f"  Laminar: b = {B_LAMINAR:.4f}")
    df = _exponent_table(ctx["gen_df"], ctx["parker_gen_df"], "b")
    print("\nFitted b by source:")
    print(df.round(4).to_string(index=False))
    for _, row in df.iterrows():
        if row["b_min"] < 0.45 or row["b_max"] > 0.65:
            print(f"FLAG: {row['Source']} has a fitted b outside [0.45, 0.65] "
                  f"(min={row['b_min']:.4f}, max={row['b_max']:.4f}).")


def _gen_step4(ctx):
    print("\n--- STEP 4: diameter exponent a - theoretical references ---")
    print(f"  Fully turbulent: a = {A_TURBULENT_ROUGH:.4f}")
    print(f"  Polyflo: a = {A_POLYFLO:.4f}")
    print(f"  Laminar: a = {A_LAMINAR:.4f}")
    print("  Per-EHD 'a' below is NOT identified (see STEP 2); the pooled fit is.")
    print("\nFitted a by source:")
    print(_exponent_table(ctx["gen_df"], ctx["parker_gen_df"], "a").round(4).to_string(index=False))
    print("\nPooled across EHD (a identified):")
    print(_all_pooled(ctx["chart_pipes"], ctx["parker_pipes"])[["Source", "a", "b", "R2"]]
          .round(4).to_string(index=False))


def _gen_step5(ctx):
    print("\n--- STEP 5a: Parker generator fit (measured, Re > 4000, suspect row excluded) ---")
    print(ctx["parker_gen_df"].round(5).to_string(index=False))
    print("Expect lower R2 than the chart sources - this is measured data, not equation output.")
    print("\n--- STEP 5b: friction model comparison, rms log(f) residual ---")
    print(ctx["friction_df"].round(5).to_string(index=False))
    for _, row in ctx["friction_df"].iterrows():
        if row["eps_over_D"] > 0.05:
            print(f"  FLAG: {row['Pipe']} fitted eps/D = {row['eps_over_D']:.4f} exceeds "
                  f"Colebrook-White's validity limit of 0.05 - Churchill diverges by "
                  f"{row['eps_churchill_vs_cw_pct']:+.1f}%.")
    print("\nPooled (all Parker pipes):")
    for name, val in ctx["pooled_rms"].items():
        print(f"  {name}: rms log(f) = {val:.5f}")
    print("\n--- STEP 5c: free power law f = a_coeff * Re^-n_exp, per pipe ---")
    print(ctx["powerlaw_df"].round(5).to_string(index=False))
    print(f"\nPolyflo's fixed exponent: n = {POLYFLO_N:.6f}")


def _gen_step6(ctx):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    gen_df, parker_gen_df = ctx["gen_df"], ctx["parker_gen_df"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for source, g in gen_df.groupby("Source"):
        c = SOURCE_COLORS.get(source, "grey")
        ax1.scatter(g["EHD"], g["b"], marker="o", facecolors="none", edgecolors=c, label=source)
        ax2.scatter(g["EHD"], g["a"], marker="o", facecolors="none", edgecolors=c, label=source)
    if not parker_gen_df.empty:
        c = SOURCE_COLORS["Parker Hannifin"]
        ax1.scatter(parker_gen_df["EHD"], parker_gen_df["b"], marker="o", facecolors=c,
                    edgecolors=c, label="Parker Hannifin")
        ax2.scatter(parker_gen_df["EHD"], parker_gen_df["a"], marker="o", facecolors=c,
                    edgecolors=c, label="Parker Hannifin")
    for ax, refs, ylab, title in (
            (ax1, ((B_TURBULENT_ROUGH, "--", "b=0.5 (fully turbulent)"),
                   (B_POLYFLO, ":", f"b={B_POLYFLO:.4f} (Polyflo)"),
                   (B_LAMINAR, "-.", "b=1.0 (laminar)")),
             "b (pressure-drop exponent)", "Pressure-drop exponent b vs EHD"),
            (ax2, ((A_TURBULENT_ROUGH, "--", "a=2.5 (fully turbulent)"),
                   (A_POLYFLO, ":", f"a={A_POLYFLO:.4f} (Polyflo)"),
                   (A_LAMINAR, "-.", "a=4.0 (laminar)")),
             "a (diameter exponent)", "Diameter exponent a vs EHD (not identified)")):
        for val, ls, lbl in refs:
            ax.axhline(val, color="grey", linestyle=ls, linewidth=1, label=lbl)
        ax.set_xlabel("EHD")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Cross-brand CSST generator exponents", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "csst_generator_exponents.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _gen_step7(ctx):
    print("\n--- STEP 7: within-brand consistency (all sources) ---")
    groups = list(ctx["gen_df"].groupby("Source")) + [("Parker Hannifin", ctx["parker_gen_df"])]
    for source, g in groups:
        if len(g) < 2:
            continue
        mean, std = g["b"].mean(), g["b"].std()
        cv = std / mean if mean != 0 else np.nan
        print(f"  {source:16s} b: mean={mean:.4f}, std={std:.4f}, CV={cv:.4f}, n={len(g)}")
        if cv > 0.05:
            print(f"    GENERATOR/DATA EXPONENT VARIES WITH SIZE: b is not constant "
                  f"for {source}.")
    print("  'a' not tested - per-EHD 'a' is not identified (STEP 2/5a).")


def _gen_step8(ctx):
    print("\n--- STEP 8: hydrogen service note ---")
    print(f"  At matched energy duty, Q_H2 = Q_NG * {Q_H2_OVER_Q_NG:.3f}")
    print(f"  Turbulent Re ratio at that flow: Re_H2 = Re_NG * {RE_H2_OVER_RE_NG:.3f}")
    print(f"  (= (S_H2/S_NG)*(Q_H2/Q_NG)*(mu_NG/mu_H2), S_H2={H2_S}, mu_H2={H2_MU_PAS:.2e} Pa.s)")

    rows, all_below = [], []
    for pipe in ctx["chart_pipes"]["gastite"].values():
        if pipe["gas"] != "natural_gas":
            continue
        arr = md.pipe_arrays(pipe)
        valid = np.isfinite(arr["Re"])
        if not valid.any():
            continue
        Re_h2 = arr["Re"] * RE_H2_OVER_RE_NG
        rows.append([pipe["ehd"], arr["D_mm"], np.nanmin(arr["Re"][valid]),
                     np.nanmax(arr["Re"][valid]), np.nanmin(Re_h2[valid]),
                     np.nanmax(Re_h2[valid]),
                     float(np.mean(Re_h2[valid] < RE_TURBULENT) * 100)])
        all_below.append(Re_h2[valid] < RE_TURBULENT)

    print("\n  Gastite NG tables, H2 at matched energy duty:")
    print(pd.DataFrame(rows, columns=["EHD", "D_mm", "Re_NG_min", "Re_NG_max", "Re_H2_min",
                                      "Re_H2_max", "frac_below_4000 (%)"])
          .round(2).to_string(index=False))
    print(f"\n  {float(np.mean(np.concatenate(all_below)) * 100):.1f}% of the H2 service "
          f"envelope falls below Re=4000 where the generator equations were not validated.")


def _gen_step9(ctx):
    print("\n--- STEP 9: universal generator equation (average across all 5 brands) ---")
    uni = build_universal_equation(ctx["chart_pipes"], ctx["parker_pipes"])
    print(f"\n  C_avg = {uni['C_avg']:.5f}")
    print(f"  a_avg = {uni['a_avg']:.5f}")
    print(f"  b_avg = {uni['b_avg']:.5f}")
    print(f"\n  Universal equation:  Q = {uni['C_avg']:.4f} * D^{uni['a_avg']:.4f} * "
          f"(dP/L)^{uni['b_avg']:.4f}")
    print("\nPer-brand pooled fits going into the average:")
    print(uni["pooled_df"].round(5).to_string(index=False))
    print("\nBack-test, each brand's own pooled equation vs the universal equation:")
    print(uni["result_df"].round(3).to_string(index=False))


# Volume-equivalent diameter D_eff = sqrt(4*A_mean/pi) of the measured Creo profile, mm,
# the CFD Re basis. Same numbers as D_EFF_MM in report_figures.py and GEOM/PROFILE in
# cfd/new/analyse.py (EHD 13 = arc-arc profile).
D_EFF_MM = {13: 11.4922, 18: 16.1712, 23: 21.3971, 31: 28.8794}


# Parker measured cells re-expressed for H2 at matched energy duty (Re_H2 = Re_NG*ratio),
# on the EHD*25.4/32 bore and on D_eff. Only the diameter changes between the two.
def _gen_step10(ctx):
    print("\n--- STEP 10: Parker measured data, H2 Re coverage at matched energy duty ---")
    print(f"  Re_H2 = Re_NG * {RE_H2_OVER_RE_NG:.4f}")
    bases = (("EHD*25.4/32", ctx["parker_pipes"], False), ("D_eff", _load("parker"), True))
    for basis, pipes, use_deff in bases:
        for excl in (False, True):
            rows = []
            for label, pipe in sorted(pipes.items(), key=lambda kv: kv[1]["ehd"]):
                ehd = int(pipe["ehd"])
                if use_deff:
                    pipe["id_mm"] = D_EFF_MM[ehd]
                arr = md.pipe_arrays(pipe)
                m = np.isfinite(arr["Re"])
                if excl:
                    m &= parker_valid_mask(label, pipe, arr)
                Re_h2 = arr["Re"][m] * RE_H2_OVER_RE_NG
                rows.append([ehd, pipe["id_mm"], int(m.sum()),
                             float(np.mean(Re_h2 < RE_LAMINAR) * 100),
                             float(np.mean(Re_h2 < RE_TURBULENT) * 100), Re_h2.min()])
            print(f"\n  D basis: {basis}, "
                  f"{'suspect 3/4in 41.5 inWC row excluded' if excl else 'all cells'}")
            print(pd.DataFrame(rows, columns=["EHD", "D_mm", "n cells", "% Re_H2<2300",
                                              "% Re_H2<4000", "Re_H2 min"])
                  .round(1).to_string(index=False))


GEN_STEPS = {1: _gen_step1, 2: _gen_step2, 3: _gen_step3, 4: _gen_step4, 5: _gen_step5,
             6: _gen_step6, 7: _gen_step7, 8: _gen_step8, 9: _gen_step9,
             10: _gen_step10}


def run_generator(steps=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pd.set_option("display.width", 200)
    ctx = generator_context()
    for step in sorted(steps or GEN_STEPS):
        GEN_STEPS[step](ctx)
    return ctx


# ================================================================== lookup

# Source-document metadata only: which table id means which schedule.
TABLE_CATALOGS = {
    "gastite": {
        "7-1": dict(gas="natural_gas", dP_inWC=0.5), "7-2": dict(gas="natural_gas", dP_inWC=1.0),
        "7-3": dict(gas="natural_gas", dP_inWC=1.5), "7-4": dict(gas="natural_gas", dP_inWC=2.0),
        "7-5": dict(gas="natural_gas", dP_inWC=3.0), "7-6": dict(gas="natural_gas", dP_inWC=4.0),
        "7-7": dict(gas="natural_gas", dP_inWC=5.0), "7-8": dict(gas="natural_gas", dP_inWC=6.0),
        "7-9": dict(gas="natural_gas", dP_inWC=13.0), "7-10": dict(gas="natural_gas", dP_psi=1.0),
        "7-11": dict(gas="natural_gas", dP_psi=3.5), "7-12": dict(gas="natural_gas", dP_psi=7.0),
        "7-13": dict(gas="propane", dP_inWC=0.5), "7-14": dict(gas="propane", dP_inWC=1.0),
        "7-15": dict(gas="propane", dP_inWC=2.0), "7-16": dict(gas="propane", dP_inWC=2.5),
        "7-17": dict(gas="propane", dP_inWC=3.0), "7-18": dict(gas="propane", dP_psi=1.0),
        "7-19": dict(gas="propane", dP_psi=3.5), "7-20": dict(gas="propane", dP_psi=7.0),
    },
    "parflex": {
        "1": dict(gas="natural_gas", dP_inWC=0.5), "2": dict(gas="natural_gas", dP_inWC=1.0),
        "3": dict(gas="natural_gas", dP_inWC=1.5), "4": dict(gas="natural_gas", dP_inWC=2.0),
        "5": dict(gas="natural_gas", dP_inWC=3.0), "6": dict(gas="natural_gas", dP_inWC=4.0),
        "7": dict(gas="natural_gas", dP_inWC=5.0), "8": dict(gas="natural_gas", dP_inWC=6.0),
        "9": dict(gas="natural_gas", dP_psi=1.0), "10": dict(gas="natural_gas", dP_psi=1.5),
        "11": dict(gas="natural_gas", dP_psi=3.5), "12": dict(gas="propane", dP_inWC=0.5),
        "13": dict(gas="propane", dP_inWC=2.5), "14": dict(gas="propane", dP_psi=1.0),
        "15": dict(gas="propane", dP_psi=1.5), "16": dict(gas="propane", dP_psi=3.5),
    },
    "tracpipe": {
        "N-1": dict(gas="natural_gas", dP_inWC=0.5), "N-2": dict(gas="natural_gas", dP_inWC=1.0),
        "N-3": dict(gas="natural_gas", dP_psi=1.0), "N-4": dict(gas="natural_gas", dP_psi=3.5),
        "N-5": dict(gas="natural_gas", dP_psi=10.0), "P-1": dict(gas="propane", dP_inWC=1.0),
        "P-2": dict(gas="propane", dP_psi=1.0), "P-3": dict(gas="propane", dP_psi=10.0),
    },
    # Parker publishes the same 5 conditions for every pipe size.
    "parker": {
        "0.5inwc": dict(gas="natural_gas", drop_psi=0.0181),
        "3inwc": dict(gas="natural_gas", drop_psi=0.11),
        "6inwc": dict(gas="natural_gas", drop_psi=0.22),
        "41.5inwc": dict(gas="natural_gas", drop_psi=1.50),   # suspect for 3/4 inch
        "96.9inwc": dict(gas="natural_gas", drop_psi=3.50),
    },
    "csa": {
        "1a": dict(gas="natural_gas", dP_inWC=0.5), "1b": dict(gas="natural_gas", dP_psi=1.5),
        "1c": dict(gas="natural_gas", dP_psi=3.5),
    },
}

# Pooled (C, a, b) across all of a brand's sizes; False uses the per-EHD (C, b) fit,
# which reproduces one specific table far more closely.
POOLED_INSTEAD_OF_PER_EHD = True
SHOW_UNIVERSAL_COMPARISON = True

TABLE = "gastite_7-18"
EHD = 13


def find_pipe(pipes, ehd, gas=None):
    for label, pipe in pipes.items():
        if pipe["ehd"] != ehd:
            continue
        if gas is not None and pipe["gas"] != gas:
            continue
        return label, pipe
    raise KeyError(f"No pipe found for EHD={ehd} in this manufacturer's data.")


def _print_lookup(manufacturer, ehd, table_label, fit_desc, lengths_ft, Q_published,
                  Q_generator, Q_universal=None, universal_desc=None):
    print(f"\n--- {manufacturer}, EHD {ehd}, TABLE {table_label} ---")
    print(f"  {fit_desc}")
    if universal_desc:
        print(f"  {universal_desc}")
    valid = np.isfinite(Q_published) & (Q_published > 0)
    err = np.where(valid, (Q_generator - Q_published) / Q_published * 100.0, np.nan)
    cols = {"L_ft": lengths_ft, "Q_published_CFH": Q_published,
            "Q_generator_CFH": Q_generator, "error_pct": err}
    err_uni = None
    if Q_universal is not None:
        err_uni = np.where(valid, (Q_universal - Q_published) / Q_published * 100.0, np.nan)
        cols["Q_universal_CFH"] = Q_universal
        cols["error_pct_universal"] = err_uni
    print(pd.DataFrame(cols).round(3).to_string(index=False))
    if err[valid].size:
        print(f"\n  own fit:       mean error = {np.nanmean(err[valid]):+.2f}%   "
              f"max |error| = {np.nanmax(np.abs(err[valid])):.2f}%")
    if err_uni is not None and err_uni[valid].size:
        print(f"  universal fit: mean error = {np.nanmean(err_uni[valid]):+.2f}%   "
              f"max |error| = {np.nanmax(np.abs(err_uni[valid])):.2f}%")


def run_lookup(table=None, ehd=None):
    pd.set_option("display.width", 200)
    table = table or TABLE
    ehd = EHD if ehd is None else ehd
    key, table_id = table.split("_", 1)
    if key not in TABLE_CATALOGS:
        raise ValueError(f"Unknown manufacturer prefix '{key}'. "
                         f"Expected one of {sorted(TABLE_CATALOGS)}.")
    if table_id not in TABLE_CATALOGS[key]:
        raise ValueError(f"Unknown table id '{table_id}' for '{key}'. "
                         f"Available: {sorted(TABLE_CATALOGS[key])}")
    info = TABLE_CATALOGS[key][table_id]
    manufacturer = SOURCE_NAME[key]

    pipes = _load(key)
    gas = info["gas"]
    if gas == "propane":
        print("NOTE: the (C, a, b) fits are NG-only by design, and these propane tables are")
        print("convention-scaled NG data, so expect a roughly constant offset.")

    label, pipe = find_pipe(pipes, ehd, gas=gas if key != "parker" else None)
    arr = md.pipe_arrays(pipe)
    drop_psi = (info["drop_psi"] if "drop_psi" in info
                else info["dP_psi"] if "dP_psi" in info
                else info["dP_inWC"] * INWC_TO_PSI)
    cond_idx = next((i for i, c in enumerate(pipe["conditions"])
                     if abs(c[1] - drop_psi) < 1e-6), None)
    if cond_idx is None:
        raise ValueError(f"TABLE '{table}' (dP={drop_psi:.5f} psi) has no matching condition "
                         f"for EHD {ehd}.")
    if key == "parker" and label == PARKER_SUSPECT_PIPE and table_id == "41.5inwc":
        print("NOTE: this is the suspect row (residuals +20/+25/-31% at 150/200/250 ft). "
              "Shown, not excluded.")

    lengths_ft = arr["L_ft"]
    Q_published = np.array(arr["Q_scfh"][cond_idx], dtype=float)

    ctx = generator_context()
    if POOLED_INSTEAD_OF_PER_EHD:
        if key == "parker":
            row = fit_parker_pooled(ctx["parker_pipes"]).iloc[0]
            desc = "POOLED fit (all Parker pipes)"
        else:
            pooled = fit_generator_pooled(ctx["chart_pipes"])
            row = pooled[pooled["Source"] == manufacturer].iloc[0]
            desc = "POOLED fit"
        C, a, b = row["C"], row["a"], row["b"]
        fit_desc = f"{desc}: C={C:.5f}, a={a:.5f}, b={b:.5f}"
    elif key == "parker":
        # D is constant within one pipe, so 'a' is a degenerate minimum-norm split:
        # C*D^a, not C alone, reproduces the fitted value.
        turb = parker_turb_mask(label, pipe, arr)
        dP_grid = np.broadcast_to(
            (np.array([c[1] for c in pipe["conditions"]]) * PSI_TO_PA)[:, None], arr["Re"].shape)
        L_grid = np.broadcast_to((arr["L_ft"] * 0.3048)[None, :], arr["Re"].shape)
        fit = fit_generator_lstsq(arr["Q_scfh"][turb] * SCFH_TO_M3S,
                                  np.full(int(turb.sum()), arr["D_mm"] / 1000.0),
                                  dP_grid[turb], L_grid[turb])
        C, a = fit["C"], fit["a"]
        b = ctx["parker_gen_df"].set_index("Pipe").loc[label, "b"]
        fit_desc = (f"PER-PIPE fit (Re>4000, suspect row excluded): C={C:.5f}, a={a:.5f} "
                    f"(degenerate split, used as C*D^a), b={b:.5f}")
    else:
        row = ctx["gen_df"][(ctx["gen_df"]["Source"] == manufacturer)
                            & (ctx["gen_df"]["EHD"] == ehd)].iloc[0]
        C, a, b = row["C"], row["a"], row["b"]
        fit_desc = (f"PER-EHD fit: C={C:.5f}, a={a:.5f} (degenerate split, used as C*D^a), "
                    f"b={b:.5f}")

    D_m = ehd * EHD_TO_MM / 1000.0
    dP_Pa = drop_psi * PSI_TO_PA
    L_m = lengths_ft * 0.3048
    Q_gen = (C * D_m ** a * (dP_Pa / L_m) ** b) / SCFH_TO_M3S

    Q_uni, uni_desc = None, None
    if SHOW_UNIVERSAL_COMPARISON:
        uni = build_universal_equation(ctx["chart_pipes"], ctx["parker_pipes"])
        Q_uni = (uni["C_avg"] * D_m ** uni["a_avg"] * (dP_Pa / L_m) ** uni["b_avg"]) / SCFH_TO_M3S
        uni_desc = (f"UNIVERSAL fit (avg of all 5 brands): C={uni['C_avg']:.5f}, "
                    f"a={uni['a_avg']:.5f}, b={uni['b_avg']:.5f}")

    _print_lookup(manufacturer, ehd, table, fit_desc, lengths_ft, Q_published, Q_gen,
                  Q_uni, uni_desc)


MODELS = {"universal": run_universal, "biregime": run_biregime,
          "generator": run_generator, "lookup": run_lookup}


if __name__ == "__main__":
    args = sys.argv[1:]
    steps_arg = next((a for a in args if re.fullmatch(r"\d+(,\d+)*", a)), None)
    names = [a for a in args if a in MODELS] or list(MODELS)
    for name in names:
        if name == "generator" and steps_arg:
            run_generator({int(x) for x in steps_arg.split(",")})
        else:
            MODELS[name]()
