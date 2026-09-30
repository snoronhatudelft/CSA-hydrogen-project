# CSST manufacturer flow-capacity data: one parser per source file layout,
# then the shared 15-step analysis. Run: python csst_manufacturer_data.py gastite 2,4,5
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from friction import colebrook_f, churchill_f, polyflo_f, eps_from_churchill

HERE = os.path.dirname(os.path.abspath(__file__))

INWC_TO_PSI = 0.0361273
PROPANE_BTU_PER_SCF = 2520.0
SCFH_TO_M3S = 0.0000078658

R_air = 287.05
Pb = 101.325                       # kPa, base pressure
Tb = (65 - 32) * 5 / 9 + 273.15    # 65 F, isothermal CSA LC-1 test

a0, n = 0.140370, 0.151571         # Polyflo f = a0*Re^-n
alpha_poly = 2.0 / (2.0 - n)

# Published tables already include 4 bends + 2 end fittings, but the cross-table
# identities hold on bare L, so no equivalent length is added.
APPLY_BASELINE_FITTING_LENGTH = False
BASELINE_FITTING_LENGTH_FT = (1.3 * 6) if APPLY_BASELINE_FITTING_LENGTH else 0.0

GAS_PROPERTIES = {
    "natural_gas": dict(S=0.60, mu_Pas=1.2e-5, GAMMA=1.31),
    "propane":     dict(S=1.52, mu_Pas=8.0e-6, GAMMA=1.13),
}

GASTITE_EHD = {13: "3/8 inch", 19: "1/2 inch", 23: "3/4 inch", 31: "1 inch",
               37: "1-1/4 inch", 48: "1-1/2 inch", 60: "2 inch"}
PARFLEX_EHD = {13: "3/8 inch", 18: "1/2 inch", 23: "3/4 inch", 31: "1 inch",
               39: "1-1/4 inch", 62: "2 inch"}
TRACPIPE_EHD = {15: "3/8 inch", 19: "1/2 inch", 25: "3/4 inch", 31: "1 inch",
                37: "1-1/4 inch", 46: "1-1/2 inch", 62: "2 inch"}
CSA_EHD = {13: "3/8 in", 15: "~7/16 in", 18: "1/2 in (sm)", 19: "1/2 in",
           23: "3/4 in", 25: "3/4 in (lg)", 30: "1 in (sm)", 31: "1 in",
           37: "1-1/4 in", 39: "1-1/4 in (lg)", 46: "1-1/2 in (sm)",
           48: "1-1/2 in", 60: "2 in", 62: "2 in (lg)"}

# EHD is a dimensionless 32nds-of-an-inch size index; the bore is EHD*25.4/32 mm.
EHD_TO_MM = 25.4 / 32


# ---------------------------------------------------------------- parsing

# Conditions are normalised to (P_inlet_psi, drop_psi, drop_inWC, label).
def _condition(block):
    p_inlet = float(re.search(r"P_inlet_psi\s*=\s*([\d.]+)", block).group(1))
    m = re.search(r"dP_psi\s*=\s*([\d.]+)", block)
    if m is not None:
        drop_psi = float(m.group(1))
        label = f"{drop_psi:g} psi"
    else:
        drop_inwc = float(re.search(r"dP_inWC\s*=\s*([\d.]+)", block).group(1))
        drop_psi = drop_inwc * INWC_TO_PSI
        label = f"{drop_inwc:g} inWC"
    return (p_inlet, drop_psi, drop_psi / INWC_TO_PSI, label)


def _table_lines(block, tag="TABLE1"):
    m = re.search(tag + r"(.*?)END_" + tag, block, re.S)
    if m is None:
        raise ValueError(f"Missing {tag}")
    for line in m.group(1).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield [x.strip() for x in line.split(",")]


def _assemble(raw, ehd_label, label_fmt):
    pipes = {}
    for (gas, ehd), e in raw.items():
        pipes[label_fmt(ehd_label[ehd], ehd, gas)] = {
            "gas": gas, "ehd": ehd, "id_mm": ehd * EHD_TO_MM,
            "lengths_ft": e["lengths_ft"], "conditions": e["conditions"],
            "flows": e["flows"],
        }
    return pipes


# Gastite / Parflex / CSA layout: one block per condition, rows = length, cols = EHD.
def _parse_grid(path, ehd_label, label_fmt):
    text = open(path).read()
    blocks = re.split(r"\[TABLE ([\w\-]+)\]", text)[1:]
    if not blocks:
        raise ValueError("No TABLE blocks found in " + path)

    raw = {}
    for i in range(0, len(blocks), 2):
        block = blocks[i + 1]
        gas = re.search(r"gas\s*=\s*(\S+)", block).group(1)
        cond = _condition(block)
        flow_units = re.search(r"flow_units\s*=\s*(\S+)", block).group(1)
        ehds = [int(x) for x in re.search(r"EHDS\s*=\s*([\d,\s]+)", block).group(1).split(",")]

        lengths_ft, rows = [], []
        for parts in _table_lines(block):
            lengths_ft.append(float(parts[0]))
            rows.append([float(v) for v in parts[1:]])
        rows = np.array(rows)
        if flow_units == "MBtu_per_h":
            rows = rows * 1000.0 / PROPANE_BTU_PER_SCF

        for col, ehd in enumerate(ehds):
            e = raw.setdefault((gas, ehd), {"lengths_ft": lengths_ft, "conditions": [], "flows": []})
            e["conditions"].append(cond)
            e["flows"].append(rows[:, col])

    return _assemble(raw, ehd_label, label_fmt)


# TracPipe layout: one block per condition, rows = EHD, cols = length.
def _parse_tracpipe(path, ehd_label, label_fmt):
    text = open(path).read()
    blocks = re.split(r"\[TABLE ([\w\-]+)\]", text)[1:]
    if not blocks:
        raise ValueError("No TABLE blocks found in " + path)

    raw = {}
    for i in range(0, len(blocks), 2):
        block = blocks[i + 1]
        gas = re.search(r"gas\s*=\s*(\S+)", block).group(1)
        cond = _condition(block)
        lengths_ft = [float(x) for x in
                      re.search(r"LENGTHS\s*=\s*([\d,\s]+)", block).group(1).split(",")]
        m = re.search(r"flow_units\s*=\s*(\S+)", block)
        flow_units = m.group(1) if m else "CFH"

        for parts in _table_lines(block):
            ehd = int(parts[0])
            flows = np.array([float(v) for v in parts[3:]])   # parts[1:3] = nom size, id_in
            if flow_units == "MBtu_per_h":
                flows = flows * 1000.0 / PROPANE_BTU_PER_SCF
            e = raw.setdefault((gas, ehd), {"lengths_ft": lengths_ft, "conditions": [], "flows": []})
            e["conditions"].append(cond)
            e["flows"].append(flows)

    return _assemble(raw, ehd_label, label_fmt)


# Parker layout: one block per pipe, rows = condition, cols = length, plus a
# TABLE2 straight-tube flow spliced in as the "30S" column.
def _parse_parker(path, ehd_label, label_fmt):
    text = open(path).read()
    blocks = re.split(r"\[PIPE ([\d/\.]+ inch)\]", text)[1:]
    if not blocks:
        raise ValueError("No PIPE blocks found in " + path)

    pipes = {}
    for i in range(0, len(blocks), 2):
        name, block = blocks[i].strip(), blocks[i + 1]
        ehd = int(float(re.search(r"ehd\s*=\s*([\d.]+)", block).group(1)))
        lengths_ft = [float(x) for x in
                      re.search(r"lengths_ft\s*=\s*([\d,\s]+)", block).group(1).split(",")]

        conditions, flows = [], []
        for parts in _table_lines(block):
            if len(parts) < 5:
                continue
            inlet, drop_psi, drop_inwc = float(parts[0]), float(parts[2]), float(parts[3])
            conditions.append((inlet, drop_psi, drop_inwc, f"{drop_inwc:g} inWC"))
            flows.append([np.nan if v in ("NaN", "nan", "-", "—", "") else float(v)
                          for v in parts[4:]])

        flows_30s = [np.nan] * len(conditions)
        if re.search(r"TABLE2(.*?)END_TABLE2", block, re.S) is not None:
            t2 = [np.nan if p[4] in ("NaN", "nan", "-", "—", "") else float(p[4])
                  for p in _table_lines(block, "TABLE2") if len(p) >= 5]
            if len(t2) == len(conditions):
                flows_30s = t2

        pipes[name] = {
            "gas": "natural_gas", "ehd": ehd, "id_mm": ehd * EHD_TO_MM,
            "lengths_ft": [lengths_ft[0], "30S"] + lengths_ft[1:],
            "conditions": conditions,
            "flows": [row[:1] + [q] + row[1:] for row, q in zip(flows, flows_30s)],
        }
    return pipes


SOURCES = {
    "gastite": dict(
        title="Gastite CSST", file="gastite_csst.txt", parse=_parse_grid,
        ehd_label=GASTITE_EHD, label_fmt=lambda s, e, g: f"{s} ({g})"),
    "parflex": dict(
        title="Parflex CSST", file="csst_parflex.txt", parse=_parse_grid,
        ehd_label=PARFLEX_EHD, label_fmt=lambda s, e, g: f"{s} ({g})"),
    "tracpipe": dict(
        title="TracPipe CSST", file="csst_tracpipe.txt", parse=_parse_tracpipe,
        ehd_label=TRACPIPE_EHD, label_fmt=lambda s, e, g: f"{s} ({g})"),
    "parker": dict(
        title="Parker Hannifin CSST", file="flow_capacity_data.txt", parse=_parse_parker,
        ehd_label=None, label_fmt=None),
    "csa": dict(
        title="CSA/ANSI LC 1:23", file="csa_ansi_tables.txt", parse=_parse_grid,
        ehd_label=CSA_EHD, label_fmt=lambda s, e, g: f"{s} EHD{e} ({g})"),
}


# Parse one source by key; returns pipes[label] = {gas, ehd, id_mm, lengths_ft, conditions, flows}.
def load(source, path=None):
    src = SOURCES[source]
    return src["parse"](path or os.path.join(HERE, src["file"]),
                        src["ehd_label"], src["label_fmt"])


def _fig_dir(source, name):
    d = os.path.join(HERE, "output", "plots", source, name)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- physics

def get_pressures_kPa(cond):
    P1 = cond[0] * 6.89476 + Pb
    return P1, P1 - cond[1] * 6.89476


def _numeric_L_ft(pipe):
    return np.array([30.0 if v == "30S" else float(v) for v in pipe["lengths_ft"]])


def _P1_P2(pipe):
    P1 = np.array([get_pressures_kPa(c)[0] for c in pipe["conditions"]])[:, None]
    P2 = np.array([get_pressures_kPa(c)[1] for c in pipe["conditions"]])[:, None]
    return P1, P2


# Steps 2-3: flow, velocity, Re, and f_moody back-calculated from Darcy-Weisbach.
def pipe_arrays(pipe):
    D_mm = pipe["id_mm"]
    D_m = D_mm / 1000.0
    area = np.pi / 4 * D_m ** 2
    props = GAS_PROPERTIES[pipe["gas"]]

    L_ft = _numeric_L_ft(pipe)
    L_m = L_ft[None, :] * 0.3048
    Q_scfh = np.array(pipe["flows"])
    P1, P2 = _P1_P2(pipe)
    P_avg = (P1 + P2) / 2
    dP_Pa = (P1 - P2) * 1000

    with np.errstate(invalid="ignore", divide="ignore"):
        Q_base = Q_scfh * SCFH_TO_M3S
        V = Q_base * (Pb / P_avg) / area          # isothermal expansion to actual
        rho_b = props["S"] * Pb * 1000 / (R_air * Tb)
        Re = rho_b * Q_base * D_m / (area * props["mu_Pas"])
        rho_act = props["S"] * P_avg * 1000 / (R_air * Tb)
        f_moody = dP_Pa / ((L_m / D_m) * 0.5 * rho_act * V ** 2)

    return dict(D_mm=D_mm, L_ft=L_ft, Q_scfh=Q_scfh, Q_base=Q_base, V=V, Re=Re,
                f_moody=f_moody, P1=P1, P2=P2)


def _fit_eps(points, closure):
    masked = []
    for Re_v, f_v, D_mm in points:
        turb = Re_v > 4000                        # CW/Churchill valid only above ~4000
        if turb.sum() < 3:
            print(f"WARNING: fewer than 3 turbulent points for pipe {D_mm:.1f} mm, using all Re")
            turb = np.full(Re_v.shape, True)
        masked.append((Re_v[turb], f_v[turb], D_mm))

    def ssr(log_eps):
        return sum(np.sum((np.log(f_v) - np.log(closure(Re_v, 10 ** log_eps, D_mm))) ** 2)
                   for Re_v, f_v, D_mm in masked)

    return 10 ** minimize_scalar(ssr, bounds=(-3, 1), method="bounded").x


# Step 4: single equivalent roughness per pipe.
def fit_eps(Re, f_obs, D_mm, closure=colebrook_f):
    mask = np.isfinite(Re) & np.isfinite(f_obs)
    return _fit_eps([(Re[mask], f_obs[mask], D_mm)], closure)


# Step 4: single roughness pooled over all pipes.
def fit_eps_global(points, closure=colebrook_f):
    return _fit_eps(points, closure)


def rms_log_resid(Re, f_obs, f_pred):
    mask = np.isfinite(Re) & np.isfinite(f_obs) & np.isfinite(f_pred)
    if not mask.any():
        return np.nan
    return np.sqrt(np.mean((np.log(f_obs[mask]) - np.log(f_pred[mask])) ** 2))


# Step 8: closed-form Colebrook-White inversion for eps at one (Re, f) point.
def eps_direct(Re, f, D_mm):
    return 3.7 * D_mm * (10 ** (-1 / (2 * np.sqrt(f))) - 2.51 / (Re * np.sqrt(f)))


def _polyflo_terms(pipe):
    props = GAS_PROPERTIES[pipe["gas"]]
    L_km = (_numeric_L_ft(pipe) + BASELINE_FITTING_LENGTH_FT)[None, :] * 0.3048 / 1000.0
    P1, P2 = _P1_P2(pipe)
    K = 1250 * props["S"] * Pb / (27 * np.pi * R_air * Tb * props["mu_Pas"])   # Re = K*Q_day/D_mm
    b1 = 1.1494e-3 * (Tb / Pb) * a0 ** -0.5 * K ** (n / 2)
    b2 = (P1 ** 2 - P2 ** 2) / (props["S"] * Tb * L_km)
    return props, L_km, P1, P2, K, b1, b2


# Step 9: closed-form Polyflo forward solve for Q, no measured flow used.
def polyflo_Q_fwd_scfh(pipe):
    _, _, _, _, _, b1, b2 = _polyflo_terms(pipe)
    with np.errstate(invalid="ignore", divide="ignore"):
        Q_day = b1 ** alpha_poly * b2 ** (alpha_poly / 2) * \
            pipe["id_mm"] ** (alpha_poly * (2.5 - n / 2))
        return Q_day / 86400.0 / SCFH_TO_M3S


# Step 10: the same equation solved for the diameter that reproduces reported Q.
def polyflo_D_from_Q(pipe):
    _, _, _, _, _, b1, b2 = _polyflo_terms(pipe)
    Q_day = np.array(pipe["flows"]) * SCFH_TO_M3S * 86400.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Q_day / (b1 ** alpha_poly * b2 ** (alpha_poly / 2))) ** \
            (1 / (alpha_poly * (2.5 - n / 2)))


# Step 11 helper: same GFE closed by Churchill, fixed-point iterated (under-relaxed).
def churchill_Q_fwd_scfh(pipe, eps_mm, max_iters=200, tol=1e-10):
    props, L_km, P1, P2, K, _, _ = _polyflo_terms(pipe)
    D_mm = pipe["id_mm"]
    C = 1.1494e-3 * (Tb / Pb) * np.sqrt((P1 ** 2 - P2 ** 2) * D_mm ** 5 /
                                        (props["S"] * Tb * L_km))
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.full((P1.shape[0], L_km.shape[1]), 0.03)
        for _ in range(max_iters):
            f_new = churchill_f(K * (C / np.sqrt(f)) / D_mm, eps_mm, D_mm)
            step = np.nanmax(np.abs(f_new / f - 1))
            f = f + 0.5 * (f_new - f)
            if step < tol:
                break
        else:
            print(f"WARNING: churchill_Q_fwd_scfh did not converge in {max_iters} "
                  f"iterations (final step={step:.2e})")
        return (C / np.sqrt(f)) / 86400.0 / SCFH_TO_M3S


# Runs steps 2-11 once; the print_/plot_ functions below only format slices of it.
def compute(pipes):
    data = {label: dict(pipe=pipe, arr=pipe_arrays(pipe)) for label, pipe in pipes.items()}

    points = []
    for d in data.values():
        arr = d["arr"]
        d["eps"] = fit_eps(arr["Re"], arr["f_moody"], arr["D_mm"])
        d["eps_churchill"] = fit_eps(arr["Re"], arr["f_moody"], arr["D_mm"], churchill_f)
        mask = np.isfinite(arr["Re"]) & np.isfinite(arr["f_moody"])
        points.append((arr["Re"][mask], arr["f_moody"][mask], arr["D_mm"]))
    eps_global = fit_eps_global(points)
    eps_global_churchill = fit_eps_global(points, churchill_f)

    for d in data.values():
        arr = d["arr"]
        d["f_cw"] = colebrook_f(arr["Re"], d["eps"], arr["D_mm"])
        d["f_poly"] = polyflo_f(arr["Re"])
        with np.errstate(invalid="ignore", divide="ignore"):
            d["err_cw"] = (d["f_cw"] - arr["f_moody"]) / arr["f_moody"] * 100
            d["err_poly"] = (d["f_poly"] - arr["f_moody"]) / arr["f_moody"] * 100
        d["Q_fwd"] = polyflo_Q_fwd_scfh(d["pipe"])
        d["D_req_mm"] = polyflo_D_from_Q(d["pipe"])
        d["Q_churchill_fwd"] = churchill_Q_fwd_scfh(d["pipe"], d["eps"])

    return data, eps_global, eps_global_churchill


# ---------------------------------------------------------------- printing

def _idx_cols(pipe):
    idx = pd.Index([c[3] for c in pipe["conditions"]], name="dP")
    cols = pd.Index([str(v) if v == "30S" else f"{v:g}" for v in pipe["lengths_ft"]],
                    name="L (ft)")
    return idx, cols


def print_step1(pipes):
    rows = [[p["gas"], p["ehd"], round(p["id_mm"], 3), len(p["conditions"]),
             len(p["lengths_ft"])] for p in pipes.values()]
    print("\n--- STEP 1: parsed pipes ---")
    print(pd.DataFrame(rows, index=list(pipes),
                       columns=["gas", "EHD", "ID (mm)", "n conditions", "n lengths"]))


def print_step2(data):
    for label, d in data.items():
        idx, cols = _idx_cols(d["pipe"])
        arr = d["arr"]
        print(f"\n--- {label}: STEP 2 - Q / velocity / Reynolds (ID = {arr['D_mm']:.2f} mm) ---")
        for name, vals, dp in (("Q (SCFH-equivalent)", arr["Q_scfh"], 2),
                               ("Q_base (m3/s)", arr["Q_base"], 6),
                               ("Velocity (m/s)", arr["V"], 3),
                               ("Reynolds", arr["Re"], 0)):
            print(f"\n{name}:")
            print(pd.DataFrame(np.round(vals, dp), idx, cols))


def print_step3(data):
    for label, d in data.items():
        idx, cols = _idx_cols(d["pipe"])
        print(f"\n--- {label}: STEP 3 - f_moody (back-calculated) ---")
        print(pd.DataFrame(np.round(d["arr"]["f_moody"], 5), idx, cols))


def print_step4(data, eps_global, eps_global_churchill):
    rows = []
    for label, d in data.items():
        arr = d["arr"]
        eps_D = d["eps"] / arr["D_mm"]
        rows.append([arr["D_mm"], d["eps"], eps_D, d["eps_churchill"],
                     d["eps_churchill"] / arr["D_mm"],
                     rms_log_resid(arr["Re"], arr["f_moody"], d["f_cw"]),
                     rms_log_resid(arr["Re"], arr["f_moody"], d["f_poly"])])
        if eps_D > 0.05:
            print(f"WARNING: {label} fitted eps/D = {eps_D:.3f} > 0.05 - "
                  f"Colebrook is extrapolating past its calibration range.")

    print("\n--- STEP 4: best-fit roughness per pipe ---")
    print("(residual = sqrt(mean((ln f_obs - ln f_pred)^2)), not a percentage)")
    print(pd.DataFrame(rows, index=list(data),
                       columns=["ID (mm)", "eps CW (mm)", "eps/D (CW)",
                                "eps Churchill (mm)", "eps/D (Churchill)",
                                "rms resid (CW)", "rms resid (Polyflo)"]).round(5))
    print(f"\nGlobal best-fit eps = {eps_global:.5f} mm (CW), "
          f"{eps_global_churchill:.5f} mm (Churchill)")
    for label, d in data.items():
        if eps_global / d["arr"]["D_mm"] > 0.05:
            print(f"WARNING: global eps/D = {eps_global / d['arr']['D_mm']:.3f} > 0.05 for {label}")


def print_step5(data):
    for label, d in data.items():
        idx, cols = _idx_cols(d["pipe"])
        print(f"\n--- {label}: STEP 5 - f_CW vs f_moody (eps = {d['eps']:.5f} mm) ---")
        print("\nf_CW:")
        print(pd.DataFrame(np.round(d["f_cw"], 5), idx, cols))
        print("\nerr_CW% = (f_CW - f_moody)/f_moody*100:")
        print(pd.DataFrame(np.round(d["err_cw"], 2), idx, cols))


def print_step6(data):
    for label, d in data.items():
        idx, cols = _idx_cols(d["pipe"])
        print(f"\n--- {label}: STEP 6 - f_polyflo vs f_moody ---")
        print("\nf_polyflo:")
        print(pd.DataFrame(np.round(d["f_poly"], 5), idx, cols))
        print("\nerr_Polyflo% = (f_polyflo - f_moody)/f_moody*100:")
        print(pd.DataFrame(np.round(d["err_poly"], 2), idx, cols))


# Positive err% = model over-predicts f (conservative on capacity).
def print_step7(data):
    rows = []
    for d in data.values():
        cw = d["err_cw"][np.isfinite(d["err_cw"])]
        poly = d["err_poly"][np.isfinite(d["err_poly"])]
        closer = int(np.sum(np.abs(cw) < np.abs(poly)))
        rows.append([len(cw), closer, len(cw) - closer,
                     np.mean(np.abs(cw)), np.mean(np.abs(poly))])
    print("\n--- STEP 7: model comparison summary ---")
    print(pd.DataFrame(rows, index=list(data),
                       columns=["n points", "n CW closer", "n Polyflo closer",
                                "MAE CW (%)", "MAE Polyflo (%)"]).round(2))


def print_step8(data):
    for label, d in data.items():
        pipe, arr = d["pipe"], d["arr"]
        D_mm = arr["D_mm"]
        rows = []
        for i, cond in enumerate(pipe["conditions"]):
            for j, L_ft in enumerate(arr["L_ft"]):
                Re, f = arr["Re"][i, j], arr["f_moody"][i, j]
                if not (np.isfinite(Re) and np.isfinite(f)) or Re <= 2300:
                    continue
                eps = eps_direct(Re, f, D_mm)
                flag = "  <-- negative eps (f below CW's floor at this Re)" if eps < 0 else ""
                if Re < 2000:
                    flag += "  <-- Churchill eps ill-conditioned below Re~2000"
                rows.append([cond[3], L_ft, round(Re), round(eps, 5),
                             round(eps_from_churchill(Re, f, D_mm), 5), flag])
        print(f"\n--- {label}: STEP 8 - direct per-point eps (ID = {D_mm:.2f} mm) ---")
        print(pd.DataFrame(rows, columns=["dP", "L_ft", "Re", "eps CW (mm)",
                                          "eps Churchill (mm)", ""]).to_string(index=False))


def print_step9(data):
    for label, d in data.items():
        Q_fwd, Q_rep = d["Q_fwd"], d["arr"]["Q_scfh"]
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = (Q_fwd - Q_rep) / Q_rep * 100
        entries = np.full(Q_fwd.shape, "-", dtype=object)
        for i in range(Q_fwd.shape[0]):
            for j in range(Q_fwd.shape[1]):
                if not np.isnan(Q_rep[i, j]):
                    entries[i, j] = f"{Q_fwd[i, j]:.1f} ({pct[i, j]:+.1f}%)"
        idx, cols = _idx_cols(d["pipe"])
        print(f"\n--- {label}: STEP 9 - Q_poly_fwd SCFH-equiv (% diff vs reported), "
              f"ID = {d['pipe']['id_mm']:.2f} mm ---")
        print(pd.DataFrame(entries, idx, cols).to_string())


def print_step10(data):
    for label, d in data.items():
        pipe, D_req = d["pipe"], d["D_req_mm"]
        idx, cols = _idx_cols(pipe)
        print(f"\n--- {label}: STEP 10 - required D to match reported Q "
              f"(actual ID = {pipe['id_mm']:.2f} mm, EHD = {pipe['ehd']:g}) ---")
        for name, vals, dp in (("Required D (mm)", D_req, 3),
                               ("Required D (in)", D_req / 25.4, 4),
                               ("Required D (in*32, approx EHD)", D_req / 25.4 * 32, 2)):
            print(f"\n{name}:")
            print(pd.DataFrame(np.round(vals, dp), idx, cols))


# ---------------------------------------------------------------- plotting

SRC_STYLE = {
    "pub":  dict(color="#2a78d6", ls="-",  label="published"),
    "cw":   dict(color="#eb6834", ls=":",  label="Churchill"),
    "poly": dict(color="#1baf7a", ls="--", label="Polyflo"),
}
GAS_MARKERS = {"natural_gas": "o", "propane": "^"}
L_XLIM_FT = (30, 250)


def _grid(n):
    cols = int(np.ceil(np.sqrt(n)))
    return int(np.ceil(n / cols)), cols


def _by_gas(data):
    groups = {}
    for label, d in data.items():
        groups.setdefault(d["pipe"]["gas"], {})[label] = d
    return groups


def _subplots(labels):
    rows, cols = _grid(len(labels))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
    return fig, axes.flatten()


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Re is proportional to Q for a fixed pipe/gas; recover the ratio from the data.
def _Q_to_Re_ratio(arr):
    mask = np.isfinite(arr["Re"]) & np.isfinite(arr["Q_scfh"]) & (arr["Q_scfh"] != 0)
    return np.median(arr["Re"][mask] / arr["Q_scfh"][mask])


def _re_crossings(L, Re, thresholds=(2300, 4000)):
    mask = np.isfinite(L) & np.isfinite(Re)
    order = np.argsort(L[mask])
    L_v, Re_v = L[mask][order], Re[mask][order]
    out = {}
    for thresh in thresholds:
        crossings = []
        for k in np.where(np.diff(np.sign(Re_v - thresh)))[0]:
            crossings.append(L_v[k] + (thresh - Re_v[k]) * (L_v[k + 1] - L_v[k]) /
                             (Re_v[k + 1] - Re_v[k]))
        if crossings:
            out[thresh] = crossings
    return out


# Step 11: published vs Churchill vs Polyflo Q against length, one figure per condition.
def plot_step11(data, source, title):
    out_dir = _fig_dir(source, "Q_vs_L")
    for gas, subset in _by_gas(data).items():
        ref = subset[next(iter(subset))]
        for ci in range(ref["arr"]["Q_scfh"].shape[0]):
            cond_label = ref["pipe"]["conditions"][ci][3]
            labels = list(subset)
            fig, axes = _subplots(labels)
            for ax, label in zip(axes, labels):
                d, arr = subset[label], subset[label]["arr"]
                L_ft = arr["L_ft"]
                ratio = _Q_to_Re_ratio(arr)
                series = {"pub": arr["Q_scfh"][ci, :], "cw": d["Q_churchill_fwd"][ci, :],
                          "poly": d["Q_fwd"][ci, :]}
                for key, sty in SRC_STYLE.items():
                    y = series[key]
                    m = (np.isfinite(L_ft) & np.isfinite(y) & (y > 0)
                         & (L_ft >= L_XLIM_FT[0]) & (L_ft <= L_XLIM_FT[1]))
                    if not m.any():
                        continue
                    order = np.argsort(L_ft[m])
                    ax.plot(L_ft[m][order], y[m][order], color=sty["color"],
                            linestyle=sty["ls"], linewidth=1.8, marker="o",
                            markersize=4, alpha=0.9)
                    for thresh, ls in ((2300, ":"), (4000, "-.")):
                        for xc in _re_crossings(L_ft, y * ratio).get(thresh, []):
                            ax.axvline(xc, color=sty["color"], linewidth=1.1,
                                       linestyle=ls, alpha=0.8)
                ax.set_title(label, fontsize=9)
                ax.set_xlabel("L (ft)", fontsize=8)
                ax.set_ylabel("Q (SCFH-equiv.)", fontsize=8)
                ax.set_xlim(*L_XLIM_FT)
                ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=7)
                ax.grid(True, linestyle="--", alpha=0.4)
            for ax in axes[len(labels):]:
                ax.set_visible(False)

            handles = [plt.Line2D([0], [0], color=s["color"], linestyle=s["ls"], marker="o",
                                  markersize=4, linewidth=1.8, label=f"Q {s['label']}")
                       for s in SRC_STYLE.values()]
            handles += [plt.Line2D([0], [0], color="grey", linewidth=1.1, linestyle=ls,
                                   label=f"Re = {t} (color = source)")
                        for t, ls in ((2300, ":"), (4000, "-."))]
            fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, framealpha=0.9)
            fig.suptitle(f"{title} - Q published vs Churchill vs Polyflo\n"
                         f"({gas}, {cond_label}, by pipe size, x = tube length)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=[0, 0.06, 1, 0.96])
            safe = re.sub(r"[^\w.\-]+", "_", str(cond_label))
            _save(fig, os.path.join(out_dir, f"{source}_Q_vs_L_{gas}_{safe}.png"))


# Step 12: published Q vs length, raw data only.
def plot_flow_vs_length(data, source, title):
    out_dir = _fig_dir(source, "flow_vs_length")
    for gas, subset in _by_gas(data).items():
        labels = list(subset)
        fig, axes = _subplots(labels)
        for ax, label in zip(axes, labels):
            pipe, arr = subset[label]["pipe"], subset[label]["arr"]
            L_ft = arr["L_ft"]
            for i, cond in enumerate(pipe["conditions"]):
                y = arr["Q_scfh"][i, :]
                m = (np.isfinite(L_ft) & np.isfinite(y)
                     & (L_ft >= L_XLIM_FT[0]) & (L_ft <= L_XLIM_FT[1]))
                if not m.any():
                    continue
                order = np.argsort(L_ft[m])
                ax.plot(L_ft[m][order], y[m][order], marker="o", markersize=4,
                        linewidth=1.6, label=cond[3])
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("L (ft)", fontsize=8)
            ax.set_ylabel("Q (SCFH-equiv.)", fontsize=8)
            ax.set_xlim(*L_XLIM_FT)
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=6.5, loc="upper right", title="Test Condition", title_fontsize=7)
        for ax in axes[len(labels):]:
            ax.set_visible(False)
        fig.suptitle(f"{title} - Published Flow vs Tube Length ({gas})",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, os.path.join(out_dir, f"{source}_flow_vs_length_{gas}.png"))


def fit_power_law(Q, dP):
    mask = np.isfinite(Q) & np.isfinite(dP) & (Q > 0) & (dP > 0)
    logQ, logdP = np.log(Q[mask]), np.log(dP[mask])
    b, log_a = np.polyfit(logQ, logdP, 1)
    ss_res = np.sum((logdP - (log_a + b * logQ)) ** 2)
    ss_tot = np.sum((logdP - logdP.mean()) ** 2)
    return np.exp(log_a), b, (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


# dP/ft [inWC/ft] at every cell, same shape as Q_scfh.
def dP_per_ft(pipe, arr):
    return np.array([c[2] for c in pipe["conditions"]])[:, None] / arr["L_ft"][None, :]


# Step 13: characteristic curves dP/ft = a*Q^b, fitted per pipe.
def plot_characteristic_curves(data, source, title):
    out_dir = _fig_dir(source, "characteristic_curves")
    cmap = plt.get_cmap("tab10")

    for gas, subset in _by_gas(data).items():
        labels = list(subset)
        fits = {}
        for label in labels:
            pipe, arr = subset[label]["pipe"], subset[label]["arr"]
            Q, dP = arr["Q_scfh"], dP_per_ft(pipe, arr)
            a, b, r2 = fit_power_law(Q.ravel(), dP.ravel())
            fits[label] = dict(a=a, b=b, r2=r2, Q=Q, dP=dP)

        fig, axes = _subplots(labels)
        for ax, label in zip(axes, labels):
            pipe, fit = subset[label]["pipe"], fits[label]
            for i, cond in enumerate(pipe["conditions"]):
                m = np.isfinite(fit["Q"][i]) & np.isfinite(fit["dP"][i])
                ax.scatter(fit["Q"][i][m], fit["dP"][i][m], s=25, alpha=0.8, label=cond[3])
            Q_fit = np.linspace(1, np.nanmax(fit["Q"]) * 1.05, 300)
            ax.plot(Q_fit, fit["a"] * Q_fit ** fit["b"], color="black", linewidth=2.0,
                    label=f"Fit: y={fit['a']:.4g}*Q^{fit['b']:.3f}")
            ax.text(0.97, 0.97, f"a = {fit['a']:.5g}\nb = {fit['b']:.4f}\nR2 = {fit['r2']:.4f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.8))
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Flow (SCFH-equiv.)", fontsize=8)
            ax.set_ylabel("Pressure Loss (in.WC/ft)", fontsize=8)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=6, loc="upper left")
        for ax in axes[len(labels):]:
            ax.set_visible(False)
        fig.suptitle(f"{title} - Characteristic Flow Curves ({gas}, dP/ft = a*Q^b)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _save(fig, os.path.join(out_dir, f"{source}_characteristic_individual_{gas}.png"))

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, label in enumerate(labels):
            fit, col = fits[label], cmap(i % 10)
            ax.scatter(fit["Q"], fit["dP"], s=18, alpha=0.5, color=col)
            Q_v = fit["Q"][np.isfinite(fit["Q"]) & (fit["Q"] > 0)]
            if Q_v.size == 0:
                continue
            Q_fit = np.logspace(np.log10(Q_v.min()), np.log10(Q_v.max() * 1.1), 200)
            ax.plot(Q_fit, fit["a"] * Q_fit ** fit["b"], color=col, linewidth=2.2,
                    label=f"{label}  b={fit['b']:.3f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Flow (SCFH-equiv.)", fontsize=12)
        ax.set_ylabel("Pressure Loss Gradient (in.WC/ft)", fontsize=12)
        ax.set_title(f"{title} - Characteristic Flow Curves ({gas}, log-log)",
                     fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, title="Pipe size (dots = data, line = fit)")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"{source}_characteristic_combined_{gas}.png"))

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, label in enumerate(labels):
            fit = fits[label]
            Q_v = fit["Q"][np.isfinite(fit["Q"]) & (fit["Q"] > 0)]
            Q_fit = np.linspace(0.5, Q_v.max() * 1.05 if Q_v.size else 500.0, 300)
            ax.plot(Q_fit, fit["a"] * Q_fit ** fit["b"], color=cmap(i % 10), linewidth=2.5,
                    label=f"{label}  y={fit['a']:.4g}*Q^{fit['b']:.3f}")
        ax.set_xlabel("Flow (SCFH-equiv.)", fontsize=12)
        ax.set_ylabel("Pressure Loss Gradient (in.WC/ft)", fontsize=12)
        ax.set_title(f"{title} - Fitted Characteristic Curves ({gas}, linear)",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        _save(fig, os.path.join(out_dir, f"{source}_characteristic_linear_{gas}.png"))


# Step 14: Moody diagram, all pipes and gases on one axes.
def plot_moody(data, source, title):
    out_dir = _fig_dir(source, "moody")
    fig, ax = plt.subplots(figsize=(9, 6))

    Re_lam = np.linspace(100, 2300, 200)
    ax.plot(Re_lam, 64.0 / Re_lam, color="black", linewidth=1.5,
            label="Laminar (f = 64/Re)", zorder=3)
    Re_turb = np.logspace(np.log10(2300), np.log10(2e5), 400)
    m_b = Re_turb <= 1e5
    ax.plot(Re_turb[m_b], 0.316 / Re_turb[m_b] ** 0.25, color="dimgray", linewidth=1.5,
            linestyle="--", label="Smooth pipe - Blasius", zorder=3)

    D_ref = float(np.mean([d["arr"]["D_mm"] for d in data.values()]))
    ax.plot(Re_turb, colebrook_f(Re_turb, 1e-8, D_ref), color="dimgray", linewidth=1.5,
            linestyle=":", label="Smooth pipe - Colebrook", zorder=3)
    for eps, col in ((0.1, "#3c7d9b"), (0.5, "#99885c")):
        ax.plot(Re_turb, colebrook_f(Re_turb, eps, D_ref), color=col, linewidth=1.2,
                linestyle="-.", alpha=0.75,
                label=f"Colebrook e={eps:g} mm / {D_ref:.1f} mm D", zorder=2)
    for eps, col in ((0.1, "#8e44ad"), (0.5, "#c0392b")):
        ax.plot(Re_turb, churchill_f(Re_turb, eps, D_ref), color=col, linewidth=1.2,
                linestyle="--", alpha=0.75,
                label=f"Churchill e={eps:g} mm / {D_ref:.1f} mm D", zorder=2)

    ax.axvspan(2300, 4000, color="lightyellow", alpha=0.4, zorder=1)
    cmap = plt.get_cmap("tab20")
    for i, (label, d) in enumerate(data.items()):
        ax.scatter(d["arr"]["Re"], d["arr"]["f_moody"], s=32, color=cmap(i % 20),
                   marker=GAS_MARKERS.get(d["pipe"]["gas"], "o"), edgecolors="white",
                   linewidths=0.5, alpha=0.85, label=label, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(500, 2e5)
    ax.set_ylim(0.005, 2.0)
    ax.set_xlabel("Reynolds Number Re (-)", fontsize=11)
    ax.set_ylabel("Darcy-Weisbach Friction Factor f (-)", fontsize=11)
    ax.set_title(f"Moody Diagram - {title}\nf_moody back-calculated from "
                 f"Darcy-Weisbach vs Re (o = natural gas, ^ = propane)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, which="major", linestyle="--", alpha=0.45, zorder=0)
    ax.grid(True, which="minor", linestyle=":", alpha=0.25, zorder=0)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9, ncol=2)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"{source}_moody.png"))


def _speed_of_sound(gas):
    p = GAS_PROPERTIES[gas]
    return np.sqrt(p["GAMMA"] * (R_air / p["S"]) * Tb)


# Step 15: P, |dP/dx| and Mach along the pipe, per condition at its longest length.
def plot_pressure_profile(data, source, title):
    quantities = {"P": ("Pressure vs Distance", "P (kPa, gauge)", "pressure_profile"),
                  "dPdx": ("Local Pressure Gradient vs Distance", "|dP/dx| (kPa/m)",
                           "pressure_gradient_profile"),
                  "M": ("Mach Number vs Distance", "Mach number", "mach_profile")}

    for gas, subset in _by_gas(data).items():
        labels = list(subset)
        a_sound = _speed_of_sound(gas)
        figs = {q: _subplots(labels) for q in quantities}

        for idx, label in enumerate(labels):
            pipe, arr = subset[label]["pipe"], subset[label]["arr"]
            area = np.pi / 4 * (arr["D_mm"] / 1000.0) ** 2
            for i, cond in enumerate(pipe["conditions"]):
                Q_row = arr["Q_scfh"][i, :]
                valid = np.isfinite(arr["L_ft"]) & np.isfinite(Q_row)
                if not valid.any():
                    continue
                L_ft = arr["L_ft"][valid].max()
                L_m = L_ft * 0.3048
                j = np.where(valid & (arr["L_ft"] == L_ft))[0][0]
                Q_base = Q_row[j] * SCFH_TO_M3S
                P1, P2 = get_pressures_kPa(cond)
                x = np.linspace(0, L_m, 60)
                Px = np.sqrt(np.maximum(P1 ** 2 - (P1 ** 2 - P2 ** 2) * (x / L_m), 0))
                curves = {"P": Px - Pb,
                          "dPdx": (P1 ** 2 - P2 ** 2) / (2.0 * L_m * np.maximum(Px, 1e-9)),
                          "M": (Q_base * (Pb / Px)) / area / a_sound}
                for q in quantities:
                    figs[q][1][idx].plot(x, curves[q], linewidth=1.6, label=cond[3])

            for q, (_, ylab, _) in quantities.items():
                ax = figs[q][1][idx]
                ax.set_title(label, fontsize=9)
                ax.set_xlabel("Distance along pipe (m)", fontsize=8)
                ax.set_ylabel(ylab, fontsize=8)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(fontsize=6, loc="best", title="Test Condition", title_fontsize=6.5)

        for q, (qtitle, _, slug) in quantities.items():
            fig, axes = figs[q]
            for ax in axes[len(labels):]:
                ax.set_visible(False)
            fig.suptitle(f"{title} - {qtitle} ({gas})\n"
                         "(per condition, at its own longest tested length)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            _save(fig, os.path.join(_fig_dir(source, slug), f"{source}_{slug}_{gas}.png"))


STEP_FUNCS = {
    1:  lambda pipes, data, eg, egc, src, title: print_step1(pipes),
    2:  lambda pipes, data, eg, egc, src, title: print_step2(data),
    3:  lambda pipes, data, eg, egc, src, title: print_step3(data),
    4:  lambda pipes, data, eg, egc, src, title: print_step4(data, eg, egc),
    5:  lambda pipes, data, eg, egc, src, title: print_step5(data),
    6:  lambda pipes, data, eg, egc, src, title: print_step6(data),
    7:  lambda pipes, data, eg, egc, src, title: print_step7(data),
    8:  lambda pipes, data, eg, egc, src, title: print_step8(data),
    9:  lambda pipes, data, eg, egc, src, title: print_step9(data),
    10: lambda pipes, data, eg, egc, src, title: print_step10(data),
    11: lambda pipes, data, eg, egc, src, title: plot_step11(data, src, title),
    12: lambda pipes, data, eg, egc, src, title: plot_flow_vs_length(data, src, title),
    13: lambda pipes, data, eg, egc, src, title: plot_characteristic_curves(data, src, title),
    14: lambda pipes, data, eg, egc, src, title: plot_moody(data, src, title),
    15: lambda pipes, data, eg, egc, src, title: plot_pressure_profile(data, src, title),
}


def run(source, steps=None):
    pipes = load(source)
    data, eg, egc = compute(pipes)
    for step in sorted(steps or STEP_FUNCS):
        STEP_FUNCS[step](pipes, data, eg, egc, source, SOURCES[source]["title"])
    return pipes, data


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    args = sys.argv[1:]
    source = next((a for a in args if a in SOURCES), None)
    steps_arg = next((a for a in args if re.fullmatch(r"\d+(,\d+)*", a)), None)
    if source is None:
        print(f"usage: python {os.path.basename(__file__)} <{'|'.join(SOURCES)}> [steps]")
        sys.exit(1)
    run(source, {int(x) for x in steps_arg.split(",")} if steps_arg else None)
