# Manufacturer-published CSST capacity vs the universal EHD-only friction model

import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import csst_manufacturer_data as md
import csst_friction_models as fm
from friction import colebrook_f

SCRIPT_DIR = md.HERE
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
PLOTS_ROOT = os.path.join(OUTPUT_DIR, "plots")
COMBINED_DIR = os.path.join(PLOTS_ROOT, "combined_re")

# Per-brand figure folders.
BRAND_FOLDER = {
    "Gastite":            "gastite",
    "Parker Hannifin":    "parker_hannifin",
    "TracPipe":           "tracpipe",
    "Parker Parflex":     "parflex",
    "CSA/ANSI LC 1:23":   "csa_tables",
}


SOURCE_BRAND = {"gastite": "Gastite", "parker": "Parker Hannifin",
                "tracpipe": "TracPipe", "parflex": "Parflex", "csa": "CSA/ANSI LC 1:23"}


def _fig_dir(*parts):
    d = os.path.join(*parts)
    os.makedirs(d, exist_ok=True)
    return d


SCFH_TO_M3S = fm.SCFH_TO_M3S
EHD_TO_MM = fm.EHD_TO_MM
RE_LAMINAR = fm.RE_LAMINAR
RE_TURBULENT = fm.RE_TURBULENT
BASELINE_FITTING_LENGTH_FT = fm.BASELINE_FITTING_LENGTH_FT

# Sources documenting the "4 bends + 2 end fittings already included" baseline.
SOURCES_WITH_BASELINE_FITTINGS = {"Gastite", "TracPipe", "Parflex"}

L_XLIM_FT = (30, 250)   # plotted length range, ft - same window as the source scripts' STEP 11

# Re_H2/Re_NG at matched energy duty - independent of D, one constant for all.
RE_SCALE_H2 = 0.489 #0.449

# EHDs to print console tables for; None prints every size. Figures ignore this.
PRINT_EHD_FILTER = {19}

# Raise the plotted window's Q_lo so the hydrogen point (at Re_H2) also lands inside band.
APPLY_H2_RE_WINDOW = True
# Also require Re_H2 >= this (turbulent floor). None disables.
RE_WINDOW_FLOOR = 4000.0
# None = every source present at an EHD governs that EHD's plotted window. Else a set of
# brand names restricting which sources govern it (all sources are still drawn).
XLIM_SOURCES = None


# STEP 1 - alpha, beta at the cross-manufacturer pooled optimum (F5)

def fit_universal_alpha_beta():
    pts = sum((fm.load_points(s) for s in
               ("gastite", "parker", "tracpipe", "parflex")), [])
    return fm.fit_alpha_beta(fm.points_to_arrays(pts))


def print_verification_table(alpha, beta, all_ehds):
    rows = []
    for ehd in sorted(all_ehds):
        D_eff = alpha * ehd * EHD_TO_MM
        eps = beta * ehd * EHD_TO_MM
        rows.append([ehd, D_eff, eps, eps / D_eff])
    df = pd.DataFrame(rows, columns=["EHD", "D_eff (mm)", "eps (mm)", "eps/D_eff"])
    print(f"\n--- Universal friction model: alpha = {alpha:.3f}, beta = {beta:.3f} "
          f"(F5 pooled fit - Gastite + Parker Hannifin + TracPipe + Parflex) ---")
    print(df.round(3).to_string(index=False))


# STEP 2 - per-pipe Q_published vs Q_calculated, plus that source's own Churchill/Polyflo.

# Re at the universal model's D_eff for an arbitrary base-condition flow.
def Re_from_Q(alpha, ehd, P1_kPa, P2_kPa, gas, Q_base):
    S = fm.GAS_PROPERTIES[gas]["S"]
    mu = fm.GAS_PROPERTIES[gas]["mu_Pas"]

    D_eff_mm = alpha * ehd * EHD_TO_MM
    D_m = D_eff_mm / 1000.0
    area = np.pi / 4 * D_m ** 2

    P1 = np.asarray(P1_kPa, dtype=float)
    P2 = np.asarray(P2_kPa, dtype=float)
    P_avg = (2.0 / 3.0) * (P1 + P2 - (P1 * P2) / (P1 + P2))

    with np.errstate(invalid="ignore", divide="ignore"):
        Q_act = np.asarray(Q_base, dtype=float) * (fm.Pb / P_avg)
        V = Q_act / area
        rho_act = S * P_avg * 1000 / (fm.R_air * fm.Tb)
        Re = rho_act * V * D_m / mu
    return Re


# Re is proportional to Q for a fixed pipe/gas - recover the ratio from the data.
def Q_to_Re_ratio(arr):
    mask = np.isfinite(arr["Re"]) & np.isfinite(arr["Q_scfh"]) & (arr["Q_scfh"] != 0)
    return np.median(arr["Re"][mask] / arr["Q_scfh"][mask])


def compute_pipe_comparison(pipe, gas, get_pressures_kPa_fn, alpha, beta, orig=None, apply_fitting_length=False):
    ehd = pipe["ehd"]

    L_ft = np.array([30.0 if v == "30S" else float(v) for v in pipe["lengths_ft"]])
    L_ft_gfe = L_ft + BASELINE_FITTING_LENGTH_FT if apply_fitting_length else L_ft
    L_m = L_ft_gfe[None, :] * 0.3048
    Q_pub_scfh = np.array(pipe["flows"], dtype=float)   # (n_cond, n_len)

    P1 = np.array([get_pressures_kPa_fn(c)[0] for c in pipe["conditions"]])[:, None]
    P2 = np.array([get_pressures_kPa_fn(c)[1] for c in pipe["conditions"]])[:, None]

    Q_pub_base = Q_pub_scfh * SCFH_TO_M3S
    Q_calc_base, Re_calc = fm.Q_gfe_churchill(alpha, beta, ehd, P1, P2, L_m, gas)
    Re_pub = Re_from_Q(alpha, ehd, P1, P2, gas, Q_pub_base)
    Q_calc_scfh = Q_calc_base / SCFH_TO_M3S

    with np.errstate(invalid="ignore", divide="ignore"):
        pct_diff = (Q_calc_scfh - Q_pub_scfh) / Q_pub_scfh * 100

    result = dict(L_ft=L_ft, Q_pub=Q_pub_scfh, Q_calc=Q_calc_scfh, pct_diff=pct_diff,
                  Re_pub=Re_pub, Re_calc=Re_calc,
                  D_eff_mm=alpha * ehd * EHD_TO_MM, eps_mm=beta * ehd * EHD_TO_MM)

    if orig is not None:
        ratio = Q_to_Re_ratio(orig["arr"])
        result["Q_churchill_old"] = orig["Q_churchill_fwd"]
        result["Q_poly_old"] = orig["Q_fwd"]
        result["Re_churchill_old"] = orig["Q_churchill_fwd"] * ratio
        result["Re_poly_old"] = orig["Q_fwd"] * ratio

    return result


# Pipe, that source's own compute() arrays, and the universal-model comparison.
def build_source_data(source, alpha, beta):
    brand = SOURCE_BRAND[source]
    pipes = md.load(source)
    orig_data, _, _ = md.compute(pipes)
    if source == "csa":
        # CSA is a standard, not a manufacturer schedule, so it takes no part in
        # the universal-model comparison - only the per-EHD curve comparison.
        return {label: dict(pipe=pipe, arr=orig_data[label]["arr"])
                for label, pipe in pipes.items()}
    apply_fl = brand in SOURCES_WITH_BASELINE_FITTINGS
    return {label: dict(pipe=pipe, arr=orig_data[label]["arr"],
                        cmp=compute_pipe_comparison(pipe, pipe["gas"],
                                                    md.get_pressures_kPa, alpha, beta,
                                                    orig=orig_data[label],
                                                    apply_fitting_length=apply_fl))
            for label, pipe in pipes.items()}


# STEP 3 (print) - Q_published / Q_calculated / % diff / Re tables, one block

def print_tables(source_name, data, idx_cols_fn, alpha, beta):
    # NOTE (review 2026-08-04): EHD values are NOT aligned across manufacturers
    if PRINT_EHD_FILTER is not None and not any(
            d["pipe"]["ehd"] in PRINT_EHD_FILTER for d in data.values()):
        print(f"\n--- {source_name}: NO pipes match PRINT_EHD_FILTER={sorted(PRINT_EHD_FILTER)} "
              f"(this source's EHDs: {sorted({d['pipe']['ehd'] for d in data.values()})}) - "
              f"nothing printed; note EHD is not size-aligned across manufacturers ---")
        return
    for label, d in data.items():
        pipe, cmp = d["pipe"], d["cmp"]
        if PRINT_EHD_FILTER is not None and pipe["ehd"] not in PRINT_EHD_FILTER:
            continue
        idx, cols = idx_cols_fn(pipe)
        print(f"\n--- {source_name} - {label}: EHD = {pipe['ehd']:g}, "
              f"D_eff = {cmp['D_eff_mm']:.3f} mm, eps = {cmp['eps_mm']:.3f} mm "
              f"(alpha = {alpha:.3f}, beta = {beta:.3f}) ---")
        print("\nQ_published (SCFH-equivalent):")
        print(pd.DataFrame(np.round(cmp["Q_pub"], 3), idx, cols))
        print("\nQ_calculated (SCFH-equivalent, GFE + Churchill):")
        print(pd.DataFrame(np.round(cmp["Q_calc"], 3), idx, cols))
        print("\n% diff (calculated vs published):")
        print(pd.DataFrame(np.round(cmp["pct_diff"], 3), idx, cols))
        print("\nRe (published Q):")
        print(pd.DataFrame(np.round(cmp["Re_pub"], 3), idx, cols))
        print("\nRe (calculated Q):")
        print(pd.DataFrame(np.round(cmp["Re_calc"], 3), idx, cols))


# STEP 4 (plot) - Q vs length, one figure per (gas, schedule), one subplot per
# size. Verticals mark where each curve's own Re crosses 2300/4000.

def grid_shape(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def re_crossings(L, Re, thresholds=(RE_LAMINAR, RE_TURBULENT)):
    mask = np.isfinite(L) & np.isfinite(Re)
    L_v, Re_v = L[mask], Re[mask]
    order = np.argsort(L_v)
    L_v, Re_v = L_v[order], Re_v[order]

    out = {}
    for thresh in thresholds:
        idxs = np.where(np.diff(np.sign(Re_v - thresh)))[0]
        crossings = []
        for k in idxs:
            L0, L1 = L_v[k], L_v[k + 1]
            Re0, Re1 = Re_v[k], Re_v[k + 1]
            crossings.append(L0 + (thresh - Re0) * (L1 - L0) / (Re1 - Re0))
        if crossings:
            out[thresh] = crossings
    return out


# Q_field, Re_field, color, linestyle, legend label.
def simple_series(eps_over_d):
    return [
        ("Q_pub", "Re_pub", "#2a78d6", "-", "Q published"),
        ("Q_calc", "Re_calc", "#800080", "--", f"Q calculated (universal, eps/D={eps_over_d:.3f})"),
    ]


def full_series(eps_over_d):
    return [
        ("Q_pub", "Re_pub", "#2a78d6", "-", "Q published"),
        ("Q_churchill_old", "Re_churchill_old", "#eb6834", ":", "Q Churchill (source's own fitted eps)"),
        ("Q_poly_old", "Re_poly_old", "#1baf7a", "--", "Q Polyflo"),
        ("Q_calc", "Re_calc", "#800080", "-.", f"Q calculated (universal, eps/D={eps_over_d:.3f})"),
    ]


def plot_Q_vs_L(cmp_subset, series, brand, gas, cond_index, cond_label, out_path, suptitle_extra):
    labels = list(cmp_subset.keys())
    rows, cols = grid_shape(len(labels))

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for ax, label in zip(axes, labels):
        cmp = cmp_subset[label]
        L_ft = cmp["L_ft"]

        for Q_field, Re_field, color, ls, _ in series:
            y = cmp[Q_field][cond_index, :]
            re_arr = cmp[Re_field][cond_index, :]
            mask = (np.isfinite(L_ft) & np.isfinite(y) & (y > 0)
                    & (L_ft >= L_XLIM_FT[0]) & (L_ft <= L_XLIM_FT[1]))
            if not mask.any():
                continue
            order = np.argsort(L_ft[mask])
            ax.plot(L_ft[mask][order], y[mask][order],
                    color=color, linestyle=ls, linewidth=1.8,
                    marker="o", markersize=4, alpha=0.9)

            for thresh, vls in ((RE_LAMINAR, ":"), (RE_TURBULENT, "-.")):
                for L_cross in re_crossings(L_ft, re_arr).get(thresh, []):
                    ax.axvline(L_cross, color=color, linewidth=1.1, linestyle=vls, alpha=0.8)

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("L (ft)", fontsize=8)
        ax.set_ylabel("Q (SCFH-equiv.)", fontsize=8)
        ax.set_xlim(*L_XLIM_FT)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

    for ax in axes[len(labels):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], color=color, linestyle=ls, marker="o",
                          markersize=4, linewidth=1.8, label=lbl)
               for _, _, color, ls, lbl in series]
    handles += [
        plt.Line2D([0], [0], color="grey", linewidth=1.1, linestyle=":",
                  label=f"Re = {RE_LAMINAR:g} (color = curve)"),
        plt.Line2D([0], [0], color="grey", linewidth=1.1, linestyle="-.",
                  label=f"Re = {RE_TURBULENT:g} (color = curve)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, framealpha=0.9)
    fig.suptitle(f"{brand} - {suptitle_extra}\n({gas}, {cond_label}, by pipe size, x = tube length)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def group_by_gas(data):
    groups = {}
    for label, d in data.items():
        gas = d["pipe"].get("gas", "natural_gas")
        groups.setdefault(gas, {})[label] = d
    return groups


def plot_gastite_or_tracpipe(source_name, data, eps_over_d, filename_prefix, brand_key):
    out_dir = _fig_dir(PLOTS_ROOT, BRAND_FOLDER[brand_key], "capacity_comparison_universal_model")
    for gas, subset in group_by_gas(data).items():
        ref_pipe = subset[next(iter(subset))]["pipe"]
        cmp_subset = {lbl: d["cmp"] for lbl, d in subset.items()}
        n_cond = cmp_subset[next(iter(cmp_subset))]["Q_pub"].shape[0]
        for i in range(n_cond):
            cond_label = ref_pipe["conditions"][i][2]
            safe = re.sub(r"[^\w.\-]+", "_", str(cond_label))
            out_simple = os.path.join(out_dir, f"{filename_prefix}_capacity_comparison_{gas}_{safe}.png")
            plot_Q_vs_L(cmp_subset, simple_series(eps_over_d), source_name, gas, i, cond_label, out_simple,
                        f"Q published vs. Q calculated (universal GFE + Churchill, eps/D={eps_over_d:.3f})")
            out_full = os.path.join(out_dir, f"{filename_prefix}_capacity_comparison_full_{gas}_{safe}.png")
            plot_Q_vs_L(cmp_subset, full_series(eps_over_d), source_name, gas, i, cond_label, out_full,
                        "Q published vs. Q Churchill (own eps) vs. Q Polyflo vs. Q calculated (universal)")


def plot_parker(data, eps_over_d):
    out_dir = _fig_dir(PLOTS_ROOT, BRAND_FOLDER["Parker Hannifin"], "capacity_comparison_universal_model")
    labels = list(data.keys())
    ref_pipe = data[labels[0]]["pipe"]
    cmp_subset = {lbl: d["cmp"] for lbl, d in data.items()}
    n_cond = cmp_subset[labels[0]]["Q_pub"].shape[0]
    for i in range(n_cond):
        cond_label = f"{ref_pipe['conditions'][i][3]:g} inWC drop"
        safe = re.sub(r"[^\w.\-]+", "_", cond_label)
        out_simple = os.path.join(out_dir, f"parker_hannifin_capacity_comparison_{safe}.png")
        plot_Q_vs_L(cmp_subset, simple_series(eps_over_d), "Parker Hannifin CSST", "natural_gas", i, cond_label,
                    out_simple,
                    f"Q published vs. Q calculated (universal GFE + Churchill, eps/D={eps_over_d:.3f})")
        out_full = os.path.join(out_dir, f"parker_hannifin_capacity_comparison_full_{safe}.png")
        plot_Q_vs_L(cmp_subset, full_series(eps_over_d), "Parker Hannifin CSST", "natural_gas", i, cond_label,
                    out_full,
                    "Q published vs. Q Churchill (own eps) vs. Q Polyflo vs. Q calculated (universal)")


# STEP 5 (plot) - all manufacturers on the same axes, natural gas only
# (propane exists for only three sources, at very different flow scales).
BRAND_COLOURS = {
    "Gastite":            "#2a78d6",
    "Parker Hannifin":    "#eb6834",
    "TracPipe":           "#1baf7a",
    "Parker Parflex":     "#800080",
    "CSA/ANSI LC 1:23":   "#555555",
}


# STEP 7 (plot) - dP/ft = a*Q^b characteristic-curve comparison, grouped by

# EHD -> {brand: (pipe, arr, module)}, natural-gas pipes only.
def build_ehd_pipe_map(gastite_data, parker_data, tracpipe_data, parflex_data, csa_data):
    sources = [
        ("Gastite", gastite_data, md),
        ("Parker Hannifin", parker_data, md),
        ("TracPipe", tracpipe_data, md),
        ("Parker Parflex", parflex_data, md),
        ("CSA/ANSI LC 1:23", csa_data, md),
    ]
    out = {}
    for brand, data, mod in sources:
        for label, d in data.items():
            if d["pipe"].get("gas", "natural_gas") != "natural_gas":
                continue
            # Parker Hannifin's own parser stores ehd as a float (the other
            ehd = int(d["pipe"]["ehd"])
            out.setdefault(ehd, {})[brand] = (d["pipe"], d["arr"], mod)
    return out


# (Q_lo, Q_hi) after each successive window constraint for one EHD.
def ehd_window_stages(fits_ehd, ehd_map_ehd):
    brands = [b for b in fits_ehd if XLIM_SOURCES is None or b in XLIM_SOURCES]
    mins, maxs = [], []
    for brand in brands:
        Q_v = fits_ehd[brand]["Q"]
        Q_v = Q_v[np.isfinite(Q_v) & (Q_v > 0)]
        if Q_v.size == 0:
            continue
        mins.append(Q_v.min())
        maxs.append(Q_v.max())
    if not mins:
        return dict(raw=None, intersect=None, h2=None, floor=None)

    raw = (min(mins), max(maxs))
    Q_lo, Q_hi = max(mins), min(maxs)
    intersect = (Q_lo, Q_hi) if Q_lo < Q_hi else None

    h2 = intersect
    if intersect is not None and APPLY_H2_RE_WINDOW:
        Q_lo_h2 = max([Q_lo] + [Q_min / RE_SCALE_H2 for Q_min in mins])
        h2 = (Q_lo_h2, Q_hi) if Q_lo_h2 < Q_hi else None

    floor = h2
    if h2 is not None and RE_WINDOW_FLOOR is not None:
        ratios = [Q_to_Re_ratio(ehd_map_ehd[b][1]) for b in brands if b in ehd_map_ehd]
        floor_los = [RE_WINDOW_FLOOR / (RE_SCALE_H2 * r) for r in ratios if r > 0]
        Q_lo_floor = max([h2[0]] + floor_los)
        floor = (Q_lo_floor, Q_hi) if Q_lo_floor < Q_hi else None

    return dict(raw=raw, intersect=intersect, h2=h2, floor=floor)


# (Q_lo, Q_hi) to plot for one EHD, or None if the window came out empty.
def ehd_plot_window(fits_ehd, ehd_map_ehd):
    return ehd_window_stages(fits_ehd, ehd_map_ehd)["floor"]


# y-range spanned by every drawn source's fitted curve at the window edges.
def _window_edge_ylim(fits_ehd, Q_lo, Q_hi, log):
    ys = [fit["a"] * Q ** fit["b"] for fit in fits_ehd.values() for Q in (Q_lo, Q_hi)]
    y_lo, y_hi = min(ys), max(ys)
    if log:
        return y_lo / 1.15, y_hi * 1.15
    pad = 0.08 * (y_hi - y_lo)
    return max(y_lo - pad, 0.0), y_hi + pad


# dP/ft = a*Q^b, fitted per (source, EHD) on that source's own cells.
def plot_characteristic_curves_by_ehd(ehd_map, out_dir, show_linear=False):
    ehds = sorted(ehd_map.keys())
    rows, cols = grid_shape(len(ehds))

    fits = {}
    for ehd in ehds:
        fits[ehd] = {}
        for brand, (pipe, arr, mod) in ehd_map[ehd].items():
            Q, dP = arr["Q_scfh"], mod.dP_per_ft(pipe, arr)
            a, b, r2 = mod.fit_power_law(Q.ravel(), dP.ravel())
            fits[ehd][brand] = dict(a=a, b=b, r2=r2, Q=Q, dP=dP)

    # -- log-log, with the underlying scatter shown -------------------------
    # 2026-08-13: disabled - only the linear-axes figure below is wanted right
    # now (kept as a sanity check). Commented out, not deleted.
    # fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    # axes = axes.flatten()
    # for ax, ehd in zip(axes, ehds):
    #     window = ehd_plot_window(fits[ehd], ehd_map[ehd])
    #     if window is None:
    #         ax.set_visible(False)
    #         continue
    #     Q_lo, Q_hi = window
    #     Q_line = np.logspace(np.log10(Q_lo), np.log10(Q_hi), 200)
    #
    #     for brand, fit in fits[ehd].items():
    #         col = BRAND_COLOURS.get(brand, "grey")
    #         ax.scatter(fit["Q"], fit["dP"], s=10, alpha=0.3, color=col)
    #         ax.plot(Q_line, fit["a"] * Q_line ** fit["b"], color=col, linewidth=2.0,
    #                label=f"{brand}  b={fit['b']:.3f}")
    #
    #     y_lo, y_hi = _window_edge_ylim(fits[ehd], Q_lo, Q_hi, log=True)
    #     ax.set_xscale("log")
    #     ax.set_yscale("log")
    #     ax.set_xlim(Q_lo, Q_hi)
    #     ax.set_ylim(y_lo, y_hi)
    #     ax.set_title(f"EHD {ehd}", fontsize=10, fontweight="bold")
    #     ax.set_xlabel("Flow (SCFH-equiv.)", fontsize=8)
    #     ax.set_ylabel("dP/ft (in.WC/ft)", fontsize=8)
    #     ax.grid(True, which="both", linestyle="--", alpha=0.4)
    #     ax.legend(fontsize=6.5)
    # for ax in axes[len(ehds):]:
    #     ax.set_visible(False)
    # fig.suptitle("Characteristic Flow Curves by EHD - All Sources (natural gas, log-log)\n"
    #             "dP/ft = a*Q^b, fitted per source over its own full range, plotted only where "
    #             "every governing source has data and Re_H2 = 0.489*Re_NG stays in band",
    #             fontsize=13, fontweight="bold")
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    # out_path = os.path.join(out_dir, "all_manufacturers_by_ehd.png")
    # fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # plt.close(fig)
    # print(f"Saved: {out_path}")

    # -- linear scale, fitted curves only ------------------------------------
    if show_linear:
        matplotlib.use("TkAgg")
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    axes = axes.flatten()
    for ax, ehd in zip(axes, ehds):
        window = ehd_plot_window(fits[ehd], ehd_map[ehd])
        if window is None:
            ax.set_visible(False)
            continue
        Q_lo, Q_hi = window
        Q_line = np.linspace(Q_lo, Q_hi, 200)

        for brand, fit in fits[ehd].items():
            col = BRAND_COLOURS.get(brand, "grey")
            ax.plot(Q_line, fit["a"] * Q_line ** fit["b"], color=col, linewidth=2.2,
                   label=f"{brand}  y={fit['a']:.3g}*Q^{fit['b']:.3f}")

        y_lo, y_hi = _window_edge_ylim(fits[ehd], Q_lo, Q_hi, log=False)
        ax.set_title(f"EHD {ehd}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Flow (SCFH-equiv.)", fontsize=8)
        ax.set_ylabel("dP/ft (in.WC/ft)", fontsize=8)
        ax.set_xlim(Q_lo, Q_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=6.5)
    for ax in axes[len(ehds):]:
        ax.set_visible(False)
    fig.suptitle("Characteristic Flow Curves by EHD - All Sources (natural gas, linear)\n"
                "dP/ft = a*Q^b, fitted per source over its own full range, plotted only where "
                "every governing source has data and Re_H2 = 0.489*Re_NG stays in band",
                fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, "all_manufacturers_by_ehd_linear.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # print(f"Saved: {out_path}")  # 2026-08-13: console restricted to f_vs_Re-related prints
    if show_linear:
        print("Opening interactive window for the linear-by-EHD figure...")
        plt.show()
    else:
        plt.close(fig)

    return fits


# Figure 14, restricted per Olga's OA1 comment to CSA/ANSI LC 1 plus the two
# manufacturers actually carried into the report (Manufacturer C = Parker
# Parflex's published sizing table, Manufacturer D = Parker Hannifin's
# measured engineering data for that same tubing). Gastite and TracPipe
# (Manufacturers A and B) are dropped, not just hidden, so their data does
# not enter the fitted window either.
FIG14_BRANDS = {
    "Parker Parflex":   ("Manufacturer C (published sizing tables)", "#1baf7a", ":", "^"),
    "Parker Hannifin":  ("Manufacturer C (measurement data)", "#c23b8f", "--", "D"),
    "CSA/ANSI LC 1:23": ("CSA/ANSI LC 1 reference", "#333333", "-.", None),
}


def plot_figure14_csd(ehd_map, out_dir, require_all=True, layout=None, panel_ratio=1.1):
    # panel_ratio: each panel's own width/height ratio (e.g. 1.1 for panels
    # slightly wider than tall, an 11:10 box). 1.0 = perfect square.
    # layout: "row" (single row, 1xN) or "grid" (auto near-square, e.g. 2x2
    # for 4 panels). None picks the default for each version: row for the
    # 4-EHD (require_all) version, grid for the 6-EHD (all_ehd) version.
    if require_all:
        # Only EHDs where all three governing sources (Manufacturers C, D
        # and the CSA/ANSI LC 1 reference) have data, per the figure's own
        # caption rule of plotting only where every shown source has data.
        ehds = sorted(ehd for ehd in ehd_map
                     if set(FIG14_BRANDS).issubset(ehd_map[ehd]))
        out_name = "fig14_manufacturer_c_d_vs_csa.png"
    else:
        # Every EHD where at least one of Manufacturer C or D has data next
        # to the CSA/ANSI LC 1 reference; a panel with only C (no matching D
        # measurement) plots C and CSA alone rather than being dropped.
        ehds = sorted(ehd for ehd in ehd_map
                     if "CSA/ANSI LC 1:23" in ehd_map[ehd]
                     and ({"Parker Parflex", "Parker Hannifin"} & set(ehd_map[ehd])))
        out_name = "fig14_manufacturer_c_d_vs_csa_all_ehd.png"
    if layout is None:
        layout = "row" if require_all else "grid"
    if layout == "grid" and require_all:
        out_name = out_name.replace(".png", "_2x2.png")
    rows, cols = (1, len(ehds)) if layout == "row" else grid_shape(len(ehds))

    fits = {}
    for ehd in ehds:
        fits[ehd] = {}
        for brand, (pipe, arr, mod) in ehd_map[ehd].items():
            if brand not in FIG14_BRANDS:
                continue
            Q, dP = arr["Q_scfh"], mod.dP_per_ft(pipe, arr)
            a, b, r2 = mod.fit_power_law(Q.ravel(), dP.ravel())
            fits[ehd][brand] = dict(a=a, b=b, r2=r2, Q=Q, dP=dP)

    PANEL_H = 4.0  # inches; PANEL_W follows panel_ratio, so panels stay this shape
    PANEL_W = PANEL_H * panel_ratio
    fig, axes = plt.subplots(rows, cols, figsize=(PANEL_W * cols, PANEL_H * rows),
                             squeeze=False)
    axes = axes.flatten()
    legend_handles = {}
    for ax, ehd in zip(axes, ehds):
        ehd_map_filtered = {b: v for b, v in ehd_map[ehd].items() if b in FIG14_BRANDS}
        window = ehd_plot_window(fits[ehd], ehd_map_filtered)
        if window is None:
            ax.set_visible(False)
            continue
        Q_lo, Q_hi = window
        Q_line = np.linspace(Q_lo, Q_hi, 200)

        for brand, fit in fits[ehd].items():
            label, color, ls, marker = FIG14_BRANDS[brand]
            line, = ax.plot(Q_line, fit["a"] * Q_line ** fit["b"], color=color,
                            linestyle=ls, linewidth=2.0, marker=marker,
                            markevery=25, markersize=6, label=label)
            legend_handles[label] = line

        y_lo, y_hi = _window_edge_ylim(fits[ehd], Q_lo, Q_hi, log=False)
        ax.set_title(f"EHD {ehd}", fontsize=10, fontweight="bold")
        ax.set_xlim(Q_lo, Q_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_box_aspect(1 / panel_ratio)  # fixed width/height frame, data span aside
        ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes[len(ehds):]:
        ax.set_visible(False)

    # Reserve a fixed strip, in inches (not a fraction of figure height, which
    # varies with rows), for the shared x-label and legend below the panels.
    fig_h_in = fig.get_size_inches()[1]
    xlabel_y_in, legend_y_in, bottom_margin_in = 0.70, 0.05, 1.05
    fig.tight_layout(rect=[0.015, bottom_margin_in / fig_h_in, 1, 1])
    fig.subplots_adjust(wspace=0.10, hspace=0.30)

    order = ["Manufacturer C (published sizing tables)", "Manufacturer C (measurement data)", "CSA/ANSI LC 1 reference"]
    handles = [legend_handles[l] for l in order if l in legend_handles]
    fig.text(0.52, xlabel_y_in / fig_h_in, "Flow (standard cubic feet per hour)",
             fontsize=11, ha="center")
    fig.text(0.002, 0.5, "Pressure drop per foot (in. w.c./ft)",
             fontsize=11, va="center", rotation="vertical")
    fig.legend(handles, [h.get_label() for h in handles], title="Data source",
              loc="lower center", ncol=len(handles), fontsize=10, title_fontsize=11,
              frameon=True, bbox_to_anchor=(0.52, legend_y_in / fig_h_in))

    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return fits


# Figure 15 - capacity at a matched pressure gradient, across every EHD any
# source publishes, one point per (brand, EHD). Rebuilt from scratch since
# the original generating script could not be located; matched to the
# embedded image's own conventions as stated in its caption: gradient formed
# from each cell's own bare tabulated length (no fitting allowance added),
# capacity read at 0.02 in. WC/ft by inverting a least-squares power-law fit
# to that source's own published points, matched gradient inside every
# source's tested range so nothing is extrapolated.
FIG15_MATCHED_GRADIENT = 0.02  # in. WC/ft

FIG15_BRANDS = {
    "Gastite":          ("Manufacturer A", "#1f6fb2", "-", "o"),
    "TracPipe":         ("Manufacturer B", "#d9761f", "--", "s"),
    "Parker Parflex":   ("Manufacturer C (published sizing tables)", "#1baf7a", ":", "^"),
    "Parker Hannifin":  ("Manufacturer C (measurement data)", "#c23b8f", "-.", "D"),
    "CSA/ANSI LC 1:23": ("CSA/ANSI LC 1 reference", "#333333", "-.", "*"),
}
FIG15_ORDER = ["Manufacturer A", "Manufacturer B",
              "Manufacturer C (published sizing tables)",
              "Manufacturer C (measurement data)", "CSA/ANSI LC 1 reference"]


def plot_figure15_capacity_vs_ehd(ehd_map, out_dir, brands=None,
                                  out_name="fig15_capacity_vs_ehd_matched_gradient.png"):
    # brands: subset of FIG15_BRANDS keys to include (default: all five).
    # One (a, b) power-law fit per (brand, EHD), then Q at the matched
    # gradient by inverting dP/ft = a*Q^b, i.e. Q = (grad/a)^(1/b). Every
    # source's own published cells are used for that source's own fit, so
    # the matched point never extrapolates past what that source tested.
    wanted = set(brands) if brands is not None else set(FIG15_BRANDS)
    points = {}   # brand -> {ehd: Q_at_grad}
    for ehd in sorted(ehd_map):
        for brand, (pipe, arr, mod) in ehd_map[ehd].items():
            if brand not in wanted:
                continue
            Q, dP = arr["Q_scfh"], mod.dP_per_ft(pipe, arr)
            a, b, r2 = mod.fit_power_law(Q.ravel(), dP.ravel())
            Q_match = (FIG15_MATCHED_GRADIENT / a) ** (1.0 / b)
            points.setdefault(brand, {})[ehd] = Q_match

    all_ehds = sorted({ehd for brand_pts in points.values() for ehd in brand_pts})
    # Real EHD spacing, not a categorical index - the gap between 46 and 48
    # should read as narrower than the gap between 48 and 60.
    x_pos = {ehd: ehd for ehd in all_ehds}
    ref_pts = points.get("CSA/ANSI LC 1:23", {})

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.3], hspace=0.06))

    ax_bot.axhspan(-10, 10, color="#f5e97a", alpha=0.45, zorder=1)
    ax_bot.axhline(0, color="black", linewidth=1.0, zorder=2)

    for brand, brand_pts in points.items():
        label, color, ls, marker = FIG15_BRANDS[brand]
        ehds = sorted(brand_pts)
        xs = [x_pos[e] for e in ehds]
        ys = [brand_pts[e] for e in ehds]
        ax_top.plot(xs, ys, color=color, linestyle=ls, marker=marker,
                   markersize=8, linewidth=2.0, label=label, zorder=4)

        if brand == "CSA/ANSI LC 1:23":
            continue
        diff_x, diff_y = [], []
        for e in ehds:
            if e in ref_pts:
                diff_x.append(x_pos[e])
                diff_y.append(100.0 * (brand_pts[e] - ref_pts[e]) / ref_pts[e])
        ax_bot.plot(diff_x, diff_y, color=color, linestyle=ls, marker=marker,
                   markersize=8, linewidth=2.0, zorder=4)

    ax_top.set_yscale("log")
    ax_top.set_ylabel("Capacity at 0.02 inch water column per foot\n"
                      "(standard cubic feet per hour)", fontsize=11)
    ax_top.grid(True, which="both", linestyle="--", alpha=0.35)
    handles, labels = ax_top.get_legend_handles_labels()
    order_idx = [labels.index(l) for l in FIG15_ORDER if l in labels]
    ax_top.legend([handles[i] for i in order_idx], [labels[i] for i in order_idx],
                 title="Data source", loc="upper left", fontsize=10, title_fontsize=11,
                 frameon=True)

    ax_bot.set_ylabel("Difference from the CSA/ANSI LC 1 reference\n"
                      "(percent; above zero = higher capacity)", fontsize=11)
    ax_bot.set_xlabel("Equivalent hydraulic diameter index EHD (thirty-seconds of an inch),\n"
                      "increasing to the right", fontsize=11)
    ax_bot.set_xticks(all_ehds)
    ax_bot.set_xticklabels(all_ehds, rotation=0, fontsize=9)
    ax_bot.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.align_ylabels([ax_top, ax_bot])
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.14)
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return points


# Per-EHD raw union Q range and the window after each clipping stage.
def print_ehd_window_table(ehd_map, fits):
    rows = []
    for ehd in sorted(ehd_map.keys()):
        stages = ehd_window_stages(fits[ehd], ehd_map[ehd])
        raw, intersect, h2, floor = stages["raw"], stages["intersect"], stages["h2"], stages["floor"]

        if intersect is None:
            log_span = None
        else:
            log_span = np.log10(intersect[1]) - np.log10(intersect[0])

        def pct_of_intersect(window):
            if window is None or log_span is None or log_span <= 0:
                return 0.0
            return 100.0 * (np.log10(window[1]) - np.log10(window[0])) / log_span

        rows.append([
            ehd,
            f"{raw[0]:.1f}-{raw[1]:.1f}" if raw else "-",
            f"{intersect[0]:.1f}-{intersect[1]:.1f}" if intersect else "empty",
            f"{h2[0]:.1f}-{h2[1]:.1f}" if h2 else "empty",
            f"{floor[0]:.1f}-{floor[1]:.1f}" if floor else "empty",
            pct_of_intersect(h2),
            pct_of_intersect(floor),
        ])

    df = pd.DataFrame(rows, columns=[
        "EHD", "raw union Q", "intersected window", "+ H2 Re trim", "+ turbulent floor",
        "% log span after H2 trim", "% log span after floor",
    ])
    print("\n--- Plotted window per EHD (SCFH-equiv.), and cost of each clipping stage "
          "vs. the intersected window's log span ---")
    print(df.round(1).to_string(index=False))


# f_moody vs.
def plot_moody_all_manufacturers(ehd_map, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    Re_lam = np.linspace(100, 2300, 200)
    ax.plot(Re_lam, 64.0 / Re_lam, color="black", linewidth=1.5,
           label="Laminar (f = 64/Re)", zorder=3)

    Re_turb = np.logspace(np.log10(2300), np.log10(2e5), 400)
    mask_b = Re_turb <= 1e5
    ax.plot(Re_turb[mask_b], 0.316 / Re_turb[mask_b] ** 0.25,
           color="dimgray", linewidth=1.5, linestyle="--",
           label="Smooth pipe - Blasius", zorder=3)

    all_D = [arr["D_mm"] for brands in ehd_map.values() for (_, arr, _) in brands.values()]
    D_ref = float(np.mean(all_D)) if all_D else 12.7
    ax.plot(Re_turb, colebrook_f(Re_turb, 1e-8, D_ref),
           color="dimgray", linewidth=1.5, linestyle=":",
           label="Smooth pipe - Colebrook", zorder=3)
    for eps, col in ((0.1, "#3c7d9b"), (0.5, "#99885c")):
        ax.plot(Re_turb, colebrook_f(Re_turb, eps, D_ref),
               color=col, linewidth=1.2, linestyle="-.", alpha=0.75,
               label=f"Colebrook e={eps:g} mm / {D_ref:.1f} mm D", zorder=2)

    ax.axvspan(2300, 4000, color="lightyellow", alpha=0.4, zorder=1)

    seen = set()
    for brands in ehd_map.values():
        for brand, (pipe, arr, mod) in brands.items():
            col = BRAND_COLOURS.get(brand, "grey")
            lbl = brand if brand not in seen else None
            seen.add(brand)
            ax.scatter(arr["Re"], arr["f_moody"], s=28, color=col,
                      edgecolors="white", linewidths=0.4, alpha=0.8,
                      label=lbl, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(500, 2e5)
    ax.set_ylim(0.005, 2.0)
    ax.set_xlabel("Reynolds Number Re (-)", fontsize=11)
    ax.set_ylabel("Darcy-Weisbach Friction Factor f (-)", fontsize=11)
    ax.set_title("Moody Diagram - All Sources (natural gas)\n"
                "Each source's own f_moody back-calculated from its own published data",
                fontsize=11, fontweight="bold")
    ax.grid(True, which="major", linestyle="--", alpha=0.45, zorder=0)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, zorder=0)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "all_manufacturers.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# STEP 8 - f = a_f*Re^-n, fit per (EHD, brand) over that pipe's FULL published
# Re range, straight off arr["f_moody"]/arr["Re"] (Darcy-Weisbach back-
# calculated from each cell's own P1, P2, L, Q - no Colebrook/Churchill
# closure anywhere in this step). No laminar/turbulent split, no cross-brand
# Q intersection, no RE_SCALE_H2 trim - some pipes come back with n < 0
# (friction rising with Re), reported as found. n is the NEGATIVE of the raw
# log-log regression slope, matching Polyflo's own n = 0.151571 sign
# convention; fit_fRe_by_ehd_brand() is the single place this fit is
# computed, and print_fRe_fits/plot_fRe_by_brand/plot_fRe_all_brands all
# consume its output rather than re-fitting.

# (EHD, brand) -> a_f, n, r2, f_bar, plus the linear and quadratic fits.
def fit_fRe_by_ehd_brand(ehd_map):
    fits = {}
    for ehd, brands in ehd_map.items():
        for brand, (pipe, arr, mod) in brands.items():
            Re, f = arr["Re"], arr["f_moody"]
            mask = np.isfinite(Re) & np.isfinite(f) & (Re > 0) & (f > 0)
            Re_v, f_v = Re[mask], f[mask]
            if Re_v.size == 0:
                continue

            log_Re, log_f = np.log(Re_v), np.log(f_v)
            slope, intercept = np.polyfit(log_Re, log_f, 1)
            n = -slope
            a_f = np.exp(intercept)
            pred = slope * log_Re + intercept
            resid = log_f - pred
            # rms_pow = float(np.sqrt(np.mean(resid ** 2)))  # 2026-08-13: dropped from
            # the table - r2 below already summarizes power-law fit quality.
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((log_f - log_f.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

            # f_bar (geometric-mean f) is kept only for the pooled f_bar~D^s
            f_bar = float(np.exp(np.mean(log_f)))
            # rms_con = float(np.sqrt(np.mean((log_f - np.log(f_bar)) ** 2)))

            # Linear fit f = m*Re + b - a straight line in LINEAR (Re, f) space,
            lin_m, lin_b = np.polyfit(Re_v, f_v, 1)
            lin_resid = f_v - (lin_m * Re_v + lin_b)
            f_ss_tot = np.sum((f_v - f_v.mean()) ** 2)
            lin_r2 = float(1 - np.sum(lin_resid ** 2) / f_ss_tot) if f_ss_tot > 0 else np.nan

            # Quadratic polynomial fit f = c2*Re^2 + c1*Re + c0 - also curves on
            # log-log axes, independent of the linear fit's curvature.
            poly_c = np.polyfit(Re_v, f_v, 2)
            poly_resid = f_v - np.polyval(poly_c, Re_v)
            poly_r2 = float(1 - np.sum(poly_resid ** 2) / f_ss_tot) if f_ss_tot > 0 else np.nan

            lam_mask = Re_v < RE_LAMINAR
            n_lam_pts = int(lam_mask.sum())
            pct_lam_pts = float(100.0 * n_lam_pts / Re_v.size)
            if n_lam_pts >= 3:
                # f_over_64Re = float(np.median(f_v[lam_mask] / (64.0 / Re_v[lam_mask])))
                slope_lam_raw, intercept_lam = np.polyfit(log_Re[lam_mask], log_f[lam_mask], 1)
                slope_lam = float(slope_lam_raw)
                # Same power-law form/sign convention as a_f/n above (n = -slope), restricted
                # to just the laminar-range (Re < RE_LAMINAR) points - used by
                # plot_fRe_by_ehd_brand_individual()'s laminar-only best-fit line.
                a_f_lam = float(np.exp(intercept_lam))
                n_lam = float(-slope_lam_raw)
                Re_lam_min = float(Re_v[lam_mask].min())
                Re_lam_max = float(Re_v[lam_mask].max())
            else:
                slope_lam = np.nan
                a_f_lam = np.nan
                n_lam = np.nan
                Re_lam_min = np.nan
                Re_lam_max = np.nan

            Re_min, Re_max = float(Re_v.min()), float(Re_v.max())
            Re_H2_lo, Re_H2_hi = RE_SCALE_H2 * Re_min, RE_SCALE_H2 * Re_max
            log_lo, log_hi, log_min = np.log(Re_H2_lo), np.log(Re_H2_hi), np.log(Re_min)
            pct_H2_below_NG_min = float(100.0 * np.clip((min(log_min, log_hi) - log_lo)
                                                          / (log_hi - log_lo), 0.0, 1.0))

            fits[(ehd, brand)] = dict(
                D_mm=float(arr["D_mm"]), n_pts=int(Re_v.size), Re_min=Re_min, Re_max=Re_max,
                a_f=float(a_f), n=float(n), r2=r2, f_bar=f_bar,
                n_lam_pts=n_lam_pts, pct_lam_pts=pct_lam_pts, slope_lam=slope_lam,
                a_f_lam=a_f_lam, n_lam=n_lam, Re_lam_min=Re_lam_min, Re_lam_max=Re_lam_max,
                Re_H2_lo=Re_H2_lo, Re_H2_hi=Re_H2_hi, pct_H2_below_NG_min=pct_H2_below_NG_min,
                lin_m=float(lin_m), lin_b=float(lin_b), lin_r2=lin_r2,
                poly_c=poly_c, poly_r2=poly_r2,
            )
    return fits


# 2026-08-13: table trimmed per feedback - rms_pow/rms_con/f_over_64Re removed
FRE_FIT_COLUMNS = ["EHD", "brand", "D_mm", "n_pts", "Re_min", "Re_max", "a_f", "n", "r2",
                   "pct_lam_pts", "slope_lam", "Re_H2_lo", "Re_H2_hi", "pct_H2_below_NG_min"]

GENERATOR_D_EXPONENTS = {"Gastite": 2.863, "TracPipe": 2.892, "Parker Hannifin": 2.901, "Parflex": 2.677}


def print_fRe_fits(fits, out_dir):
    rows = [[ehd, brand] + [v[c] for c in FRE_FIT_COLUMNS[2:]]
            for (ehd, brand), v in sorted(fits.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    df = pd.DataFrame(rows, columns=FRE_FIT_COLUMNS)

    disp = df.copy()
    for c in ("Re_min", "Re_max", "Re_H2_lo", "Re_H2_hi"):
        disp[c] = disp[c].round(0)
    for c in ("D_mm", "a_f", "n", "r2", "pct_lam_pts", "slope_lam", "pct_H2_below_NG_min"):
        disp[c] = disp[c].round(4)
    disp = disp.rename(columns={"n_pts": "n_pts_total"})

    print("\n--- STEP 8: f = a_f*Re^-n, fit per (EHD, brand), full published Re range "
          "(no laminar/turbulent split, no RE_SCALE_H2 trim) ---")
    print(disp.to_string(index=False))

    out_path = os.path.join(out_dir, "csst_fRe_fits_by_ehd.csv")
    disp.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    n_v = df["n"].to_numpy()
    print(f"\nn: min = {n_v.min():.4f}, median = {np.median(n_v):.4f}, max = {n_v.max():.4f}, "
          f"count with n < 0 = {int((n_v < 0).sum())} of {n_v.size}")

    print(f"median slope_lam (regression slope of log(f) vs. log(Re) restricted to the "
          f"Re < {RE_LAMINAR:g} points) = {np.nanmedian(df['slope_lam']):.4f} "
          f"(-1.0 would indicate those points carry real laminar physics, f ~ 1/Re)")

    # f_bar (geometric-mean f per EHD/brand, computed in fit_fRe_by_ehd_brand
    f_bar_v = np.array([v["f_bar"] for v in fits.values()])
    D_v = np.array([v["D_mm"] for v in fits.values()])
    s, _ = np.polyfit(np.log(D_v), np.log(f_bar_v), 1)
    q_exp = 2.5 - s / 2.0
    print(f"pooled fit: f_bar ~ D^{s:.4f}  =>  Q ~ D^{q_exp:.4f}  "
          f"(published pooled generator diameter exponents: "
          + ", ".join(f"{brand} {exp:g}" for brand, exp in GENERATOR_D_EXPONENTS.items()) + ")")


def _fRe_ref_lines(ax, show_64re=True):
    Re_lam = np.linspace(50.0, RE_LAMINAR, 200)
    if show_64re:
        ax.plot(Re_lam, 64.0 / Re_lam, color="black", linewidth=1.6, label="f = 64/Re", zorder=5)
    ax.axvline(RE_LAMINAR, color="grey", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvline(RE_TURBULENT, color="grey", linestyle="--", linewidth=1.0, zorder=1)
    ax.axvspan(50.0, RE_LAMINAR, color="lightyellow", alpha=0.3, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", alpha=0.45, zorder=0)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, zorder=0)


_FRE_TITLE = ("f and Re are back-calculated from the published cells by Darcy-Weisbach - "
             "no friction closure assumed. Fitted form f = a_f*Re^-n. "
             "Black line is f = 64/Re for reference.")


# Three curve forms fitted per (EHD, brand) in fit_fRe_by_ehd_brand() above,
FIT_KINDS = ("power", "linear", "poly", "scatter")
FIT_FILENAME_SUFFIX = {"power": "powerfit", "linear": "linfit", "poly": "polyfit", "scatter": "scatteronly"}
FIT_KIND_TITLE = {
    "power": "power-law fit: f = a*Re^-n (straight line on log-log axes)",
    "linear": "linear fit: f = m*Re + b (straight line on LINEAR axes, curved here)",
    "poly": "quadratic fit: f = c2*Re^2 + c1*Re + c0 (curved on log-log axes)",
    "scatter": "raw scatter only, no fitted curve",
}


def _fit_curve(fit, Re_arr, kind):
    if kind == "power":
        return fit["a_f"] * Re_arr ** -fit["n"]
    if kind == "linear":
        return fit["lin_m"] * Re_arr + fit["lin_b"]
    return np.polyval(fit["poly_c"], Re_arr)


def _fit_legend_label(fit, prefix, kind):
    if kind == "power":
        return f"{prefix}  n={fit['n']:.3f}, R2={fit['r2']:.3f}"
    if kind == "linear":
        return f"{prefix}  R2={fit['lin_r2']:.3f}"
    if kind == "poly":
        return f"{prefix}  R2={fit['poly_r2']:.3f}"
    return f"{prefix}"


# One figure per (brand, fit kind) - that brand's own EHDs overlaid on one f vs.
def plot_fRe_by_brand(fits, ehd_map, out_dir):
    brands = sorted({b for (_, b) in fits})
    for brand in brands:
        ehds = sorted(e for (e, b) in fits if b == brand)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=min(ehds), vmax=max(ehds)) if len(ehds) > 1 else None
        brand_dir = _fig_dir(out_dir, BRAND_FOLDER[brand], "f_vs_Re")

        for kind in FIT_KINDS:
            fig, ax = plt.subplots(figsize=(8, 6))
            for ehd in ehds:
                fit = fits[(ehd, brand)]
                color = cmap(norm(ehd)) if norm else cmap(0.5)
                pipe, arr, mod = ehd_map[ehd][brand]
                Re, f = arr["Re"], arr["f_moody"]
                mask = np.isfinite(Re) & np.isfinite(f) & (Re > 0) & (f > 0)
                ax.scatter(Re[mask], f[mask], s=18, color=color, alpha=0.6, zorder=3,
                           label=_fit_legend_label(fit, f"EHD {ehd}", kind) if kind == "scatter" else None)

                if kind != "scatter":
                    Re_fit = np.logspace(np.log10(fit["Re_min"]), np.log10(fit["Re_max"]), 150)
                    ax.plot(Re_fit, _fit_curve(fit, Re_fit, kind), color=color, linewidth=2.0,
                            label=_fit_legend_label(fit, f"EHD {ehd}", kind), zorder=4)

                    Re_h2 = np.logspace(np.log10(fit["Re_H2_lo"]), np.log10(fit["Re_min"]), 80)
                    ax.plot(Re_h2, _fit_curve(fit, Re_h2, kind), color=color, linewidth=1.6,
                            linestyle=":", alpha=0.85, zorder=4)

            _fRe_ref_lines(ax, show_64re=(kind != "scatter"))
            ax.set_xlabel("Reynolds number Re (-)", fontsize=10)
            ax.set_ylabel("Darcy-Weisbach friction factor f (-)", fontsize=10)
            if kind != "scatter":
                ax.set_title(f"{brand} - f vs. Re by EHD\n{FIT_KIND_TITLE[kind]}\n{_FRE_TITLE}",
                            fontsize=9, fontweight="bold")
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

            fig.tight_layout()
            out_path = os.path.join(brand_dir, f"f_vs_Re_by_ehd_{FIT_FILENAME_SUFFIX[kind]}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


# One subplot per EHD, every brand at that EHD overlaid in its own colour.
def plot_fRe_all_brands(fits, ehd_map, out_dir):
    ehds = sorted(ehd_map.keys())
    rows, cols = grid_shape(len(ehds))

    for kind in FIT_KINDS:
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        for ax, ehd in zip(axes, ehds):
            for brand in sorted(ehd_map[ehd].keys()):
                key = (ehd, brand)
                if key not in fits:
                    continue
                fit = fits[key]
                col = BRAND_COLOURS.get(brand, "grey")
                pipe, arr, mod = ehd_map[ehd][brand]
                Re, f = arr["Re"], arr["f_moody"]
                mask = np.isfinite(Re) & np.isfinite(f) & (Re > 0) & (f > 0)
                ax.scatter(Re[mask], f[mask], s=14, color=col, alpha=0.6, zorder=3,
                           label=_fit_legend_label(fit, brand, kind) if kind == "scatter" else None)

                if kind != "scatter":
                    Re_fit = np.logspace(np.log10(fit["Re_min"]), np.log10(fit["Re_max"]), 150)
                    ax.plot(Re_fit, _fit_curve(fit, Re_fit, kind), color=col, linewidth=1.8,
                            label=_fit_legend_label(fit, brand, kind), zorder=4)

            _fRe_ref_lines(ax)
            ax.set_title(f"EHD {ehd}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Re (-)", fontsize=8)
            ax.set_ylabel("f (-)", fontsize=8)
            ax.legend(fontsize=6.5)

        for ax in axes[len(ehds):]:
            ax.set_visible(False)

        fig.suptitle(f"Friction Factor vs. Re by EHD - All Manufacturers\n"
                    f"{FIT_KIND_TITLE[kind]}\n{_FRE_TITLE}",
                    fontsize=11.5, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.91])
        out_path = os.path.join(out_dir, f"all_manufacturers_by_ehd_{FIT_FILENAME_SUFFIX[kind]}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


# One figure per (EHD, brand) - that single pipe's own Re/f scatter only, no fitted curve at all.
def plot_fRe_by_ehd_brand_individual(fits, ehd_map, out_dir):
    for ehd in sorted(ehd_map.keys()):
        for brand in sorted(ehd_map[ehd].keys()):
            pipe, arr, mod = ehd_map[ehd][brand]
            fit = fits.get((ehd, brand))
            Re, f = arr["Re"], arr["f_moody"]
            mask = np.isfinite(Re) & np.isfinite(f) & (Re > 0) & (f > 0)
            Re_v, f_v = Re[mask], f[mask]
            if Re_v.size == 0:
                continue
            col = BRAND_COLOURS.get(brand, "grey")
            brand_dir = _fig_dir(out_dir, BRAND_FOLDER[brand], "f_vs_Re")

            # -- plain scatter, no fitted curve --------------------------------
            fig, ax = plt.subplots(figsize=(7, 5.5))
            ax.scatter(Re_v, f_v, s=22, color=col, alpha=0.65, zorder=3)
            _fRe_ref_lines(ax)
            ax.set_xlabel("Reynolds number Re (-)", fontsize=10)
            ax.set_ylabel("Darcy-Weisbach friction factor f (-)", fontsize=10)
            ax.set_title(f"{brand} - EHD {ehd} - f vs. Re\nraw scatter only, no fitted curve\n{_FRE_TITLE}",
                        fontsize=9, fontweight="bold")
            fig.tight_layout()
            out_path = os.path.join(brand_dir, f"f_vs_Re_ehd{ehd}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")

            # -- same scatter, + best-fit line through laminar points only ----
            fig, ax = plt.subplots(figsize=(7, 5.5))
            ax.scatter(Re_v, f_v, s=22, color=col, alpha=0.65, zorder=3)
            if fit is not None and np.isfinite(fit.get("n_lam", np.nan)):
                Re_lam_fit = np.linspace(fit["Re_lam_min"], fit["Re_lam_max"], 60)
                ax.plot(Re_lam_fit, fit["a_f_lam"] * Re_lam_fit ** -fit["n_lam"], color=col,
                        linewidth=2.4, linestyle="-", zorder=4,
                        label=f"laminar fit (Re < {RE_LAMINAR:g})  n={fit['n_lam']:.3f}")
                ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
            _fRe_ref_lines(ax)
            ax.set_xlabel("Reynolds number Re (-)", fontsize=10)
            ax.set_ylabel("Darcy-Weisbach friction factor f (-)", fontsize=10)
            ax.set_title(f"{brand} - EHD {ehd} - f vs. Re\n"
                        f"best-fit line through laminar-range points only (Re < {RE_LAMINAR:g}) - "
                        f"turbulent-range points not fitted\n{_FRE_TITLE}",
                        fontsize=9, fontweight="bold")
            fig.tight_layout()
            out_path = os.path.join(brand_dir, f"f_vs_Re_ehd{ehd}_laminarfit.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


# STEP 6 (plot) - pressure / local pressure gradient / Mach number vs.
# distance along the pipe, folded in from the standalone pipe_script.py's
# plot_profile(). A brand can only be meaningfully overlaid against another
# brand at the SAME pressure schedule (same P1, same P2) - plotting each
# brand's own "highest drop" condition (an earlier version of this script did
# that) silently overlays DIFFERENT schedules on the same axes, which is not
# a valid comparison. So: find every (inlet_psi, drop_inWC) schedule shared
# by >=2 sources, and for each EHD that ALSO has >=2 sources sharing that
# exact schedule, plot them together at their own longest tested length
# within that condition. Same physics as each brand script's own STEP
# 15/14: for fixed Q, f, D along the pipe, the isothermal compressible GFE
# makes P1^2-P2^2 linear in x/L, so
# P(x) = sqrt(P1^2 - (P1^2-P2^2)*x/L) follows exactly from that source's own
# P1, P2, L alone - no separate friction-factor assumption needed.

# (inlet_psi, drop_inWC), rounded - detects two sources on the same schedule.
def _condition_key(brand, cond):
    return (round(cond[0], 3), round(cond[2], 3))


# schedule_key -> {ehd: {brand: ...}}, kept only where >=2 sources share it.
def build_schedule_groups(ehd_map):
    raw = {}
    for ehd, brands in ehd_map.items():
        for brand, (pipe, arr, mod) in brands.items():
            for i, cond in enumerate(pipe["conditions"]):
                key = _condition_key(brand, cond)
                raw.setdefault(key, {}).setdefault(ehd, {})[brand] = (pipe, arr, mod, i)

    groups = {}
    for key, ehds_at_key in raw.items():
        kept = {ehd: b for ehd, b in ehds_at_key.items() if len(b) >= 2}
        if kept:
            groups[key] = kept
    return groups


def _schedule_label(key):
    inlet_psi, drop_inWC = key
    return f"{inlet_psi:g} psi inlet, {drop_inWC:g} inWC drop"


def _schedule_slug(key):
    inlet_psi, drop_inWC = key
    return f"inlet_{inlet_psi:g}psi_drop_{drop_inWC:g}inWC".replace(".", "p")


# P1, P2 for this matched condition, at this brand's own longest tested length.
def _profile_at_condition(pipe, arr, mod, cond_index):
    P1, P2 = mod.get_pressures_kPa(pipe["conditions"][cond_index])
    Q_row = arr["Q_scfh"][cond_index, :]
    valid = np.isfinite(arr["L_ft"]) & np.isfinite(Q_row)
    L_ft = arr["L_ft"][valid].max()
    L_m = L_ft * 0.3048
    j = np.where(valid & (arr["L_ft"] == L_ft))[0][0]
    Q_base = Q_row[j] * SCFH_TO_M3S
    return P1, P2, L_m, Q_base


# P(x), |dP/dx|(x), Mach(x) vs.
def plot_pressure_profile_matched_schedules(ehd_map, out_dirs):
    groups = build_schedule_groups(ehd_map)
    if not groups:
        print("No matching pressure schedules found across brands - "
              "skipping pressure/gradient/Mach profile comparison.")
        return

    S = fm.GAS_PROPERTIES["natural_gas"]["S"]
    GAMMA = fm.GAS_PROPERTIES["natural_gas"]["GAMMA"]
    a_sound = np.sqrt(GAMMA * (fm.R_air / S) * fm.Tb)

    titles = {"P": "Pressure vs. Distance", "dPdx": "Local Pressure Gradient vs. Distance",
              "M": "Mach Number vs. Distance"}
    ylabels = {"P": "P (kPa, gauge)", "dPdx": "|dP/dx| (kPa/m)", "M": "Mach number"}

    for key, ehds_at_key in groups.items():
        ehd_list = sorted(ehds_at_key.keys())
        rows, cols = grid_shape(len(ehd_list))
        figs = {q: plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
                for q in ("P", "dPdx", "M")}
        axes = {q: figs[q][1].flatten() for q in figs}

        for idx, ehd in enumerate(ehd_list):
            for brand, (pipe, arr, mod, cond_i) in ehds_at_key[ehd].items():
                col = BRAND_COLOURS.get(brand, "grey")
                D_m = arr["D_mm"] / 1000.0
                area = np.pi / 4 * D_m ** 2

                P1, P2, L_m, Q_base = _profile_at_condition(pipe, arr, mod, cond_i)
                x = np.linspace(0, L_m, 60)
                Px = np.sqrt(np.maximum(P1 ** 2 - (P1 ** 2 - P2 ** 2) * (x / L_m), 0))
                dPdx = (P1 ** 2 - P2 ** 2) / (2.0 * L_m * np.maximum(Px, 1e-9))
                V = (Q_base * (fm.Pb / Px)) / area
                M = V / a_sound

                axes["P"][idx].plot(x, Px - fm.Pb, color=col, linewidth=1.8, label=brand)
                axes["dPdx"][idx].plot(x, dPdx, color=col, linewidth=1.8, label=brand)
                axes["M"][idx].plot(x, M, color=col, linewidth=1.8, label=brand)

            for q in ("P", "dPdx", "M"):
                ax = axes[q][idx]
                ax.set_title(f"EHD {ehd}", fontsize=10, fontweight="bold")
                ax.set_xlabel("Distance along pipe (m)", fontsize=8)
                ax.set_ylabel(ylabels[q], fontsize=8)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7)

        for q in figs:
            for ax in axes[q][len(ehd_list):]:
                ax.set_visible(False)

        sched_lbl = _schedule_label(key)
        for q, (fig, _) in figs.items():
            fig.suptitle(f"{titles[q]} - Matching Pressure Schedule ({sched_lbl})\n"
                        "one subplot per EHD shared by >=2 sources, sources overlaid",
                        fontsize=12, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            out_path = os.path.join(out_dirs[q], f"pressure_schedule_{_schedule_slug(key)}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


def main():
    pd.set_option("display.width", 220)

    # 2026-08-13: console output restricted to f_vs_Re-related prints only.
    alpha, beta = fit_universal_alpha_beta()
    eps_over_d = beta / alpha
    # print(f"alpha = {alpha:.3f}, beta = {beta:.3f}, eps/D = {eps_over_d:.3f}")

    gastite_data = build_source_data("gastite", alpha, beta)
    parker_data = build_source_data("parker", alpha, beta)
    tracpipe_data = build_source_data("tracpipe", alpha, beta)
    parflex_data = build_source_data("parflex", alpha, beta)
    csa_data = build_source_data("csa", alpha, beta)

    # all_ehds = ({d["pipe"]["ehd"] for d in gastite_data.values()}
    #             | {d["pipe"]["ehd"] for d in parker_data.values()}
    #             | {d["pipe"]["ehd"] for d in tracpipe_data.values()}
    #             | {d["pipe"]["ehd"] for d in parflex_data.values()})
    # print_verification_table(alpha, beta, all_ehds)
    #
    # print("\n\n========== GASTITE ==========")
    # print_tables("Gastite", gastite_data, md._idx_cols, alpha, beta)
    # print("\n\n========== PARKER HANNIFIN ==========")
    # print_tables("Parker Hannifin", parker_data, md._idx_cols, alpha, beta)
    # print("\n\n========== TRACPIPE ==========")
    # print_tables("TracPipe", tracpipe_data, md._idx_cols, alpha, beta)
    # print("\n\n========== PARKER PARFLEX ==========")
    # print_tables("Parker Parflex", parflex_data, md._idx_cols, alpha, beta)

    # ==========================================================================
    # >>> COMMENT OUT FROM HERE ... (down to the END marker below) to skip all
    # figure generation/saving and just run the tables above quickly. >>>
    # ==========================================================================
    # 2026-08-13: per-brand Q-vs-L capacity_comparison_universal_model figures
    # disabled - only f_vs_Re output is wanted right now. Commented out, not
    # deleted, so they can be switched back on later.
    # print("\n\n--- generating per-brand figures (comparison_re/plots/<brand>/) ---")
    # plot_gastite_or_tracpipe("Gastite CSST", gastite_data, eps_over_d, "gastite", "Gastite")
    # plot_parker(parker_data, eps_over_d)
    # plot_gastite_or_tracpipe("TracPipe CSST", tracpipe_data, eps_over_d, "tracpipe", "TracPipe")
    # plot_gastite_or_tracpipe("Parker Parflex CSST", parflex_data, eps_over_d, "parflex", "Parker Parflex")

    # print("\n--- generating all-manufacturers-in-one-figure comparisons (comparison_re/plots/combined/) ---")
    ehd_map = build_ehd_pipe_map(gastite_data, parker_data, tracpipe_data, parflex_data, csa_data)

    print("\n--- STEP 8: fitting f = a_f*Re^-n (power), f = m*Re+b (linear), and a quadratic "
          "polynomial per (EHD, brand) ---")
    fits = fit_fRe_by_ehd_brand(ehd_map)
    print_fRe_fits(fits, OUTPUT_DIR)
    plot_fRe_by_brand(fits, ehd_map, PLOTS_ROOT)
    plot_fRe_all_brands(fits, ehd_map, _fig_dir(COMBINED_DIR, "f_vs_Re"))
    plot_fRe_by_ehd_brand_individual(fits, ehd_map, PLOTS_ROOT)

    # 2026-08-13: only f_vs_Re output is wanted right now, plus the linear
    fits_cc = plot_characteristic_curves_by_ehd(ehd_map, _fig_dir(COMBINED_DIR, "characteristic_curves"), show_linear=False)
    plot_figure14_csd(ehd_map, _fig_dir(COMBINED_DIR, "characteristic_curves"), require_all=True)
    plot_figure14_csd(ehd_map, _fig_dir(COMBINED_DIR, "characteristic_curves"), require_all=True, layout="grid")
    plot_figure14_csd(ehd_map, _fig_dir(COMBINED_DIR, "characteristic_curves"), require_all=False)
    # print_ehd_window_table(ehd_map, fits_cc)  # 2026-08-13: console restricted to f_vs_Re-related prints
    # plot_moody_all_manufacturers(ehd_map, _fig_dir(COMBINED_DIR, "moody"))
    # plot_pressure_profile_matched_schedules(ehd_map, {
    #     "P":    _fig_dir(COMBINED_DIR, "pressure_profile"),
    #     "dPdx": _fig_dir(COMBINED_DIR, "pressure_gradient_profile"),
    #     "M":    _fig_dir(COMBINED_DIR, "mach_profile"),
    # })
    # ==========================================================================
    # <<< END: figure generation block <<<
    # ==========================================================================


if __name__ == "__main__":
    main()
